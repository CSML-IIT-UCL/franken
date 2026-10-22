import abc
import json
import logging
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.utils.data

from franken.config import asdict_with_classvar
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics, compute_dataset_statistics
from franken.trainers.log_utils import (
    DataSplit,
    LogCollection,
    LogEntry,
    dtypeJSONEncoder,
)
from franken.utils.linalg.tri import pack_upper, unpack_symm
from franken.utils.misc import are_dicts_equal

logger = logging.getLogger("franken")


class BaseTrainer(abc.ABC):
    """Base trainer class. Requires :meth:`~BaseTrainer.fit` and :meth:`~BaseTrainer.evaluate` methods."""

    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        log_dir: Path | None = None,  # If None, logging is disabled
        save_every_model: bool = True,
        device: Union[torch.device, str, int] = "cpu",
        dtype: Union[str, torch.dtype] = torch.float32,
        save_cho_factor: bool = True,
    ):
        self.log_dir = log_dir
        self.save_every_model = save_every_model
        self.save_cho_factor = save_cho_factor
        self.train_dataloader = train_dataloader
        self.statistics_ = None
        if isinstance(dtype, str):
            if dtype.lower() == "float64" or dtype.lower == "double":
                dtype = torch.float64
            elif dtype.lower() == "float32" or dtype.lower == "float":
                dtype = torch.float32
            else:
                raise ValueError(
                    f"Invalid dtype {dtype}. Allowed values are 'float64', 'double', 'float32', 'single'."
                )
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError(
                f"Invalid dtype {dtype}. torch.float32 or torch.float64 are allowed."
            )
        self.buffer_dt = dtype
        self.device = torch.device(device)
        self._covs_cache = None
        self._coeffs_cache = None

    def _offload_covs_and_coeffs(self, covs, coeffs):
        """Cache packed covariances and coefficients on CPU until serialization."""
        self._covs_cache, self._coeffs_cache = (
            {
                target: pack_upper(cov.detach().to("cpu"))
                for target, cov in covs.items()
            },
            {target: coeff.detach().to("cpu") for target, coeff in coeffs.items()},
        )

    def _restore_covs_and_coeffs(self, n_features: int):
        """Restore the latest fit's system on the trainer device for the winning solve."""
        if self._covs_cache is None or self._coeffs_cache is None:
            raise RuntimeError("Covariance and coefficient caches are not available")
        return (
            {
                target: unpack_symm(packed, n_features).to(self.device)
                for target, packed in self._covs_cache.items()
            },
            {
                target: coeff.to(self.device)
                for target, coeff in self._coeffs_cache.items()
            },
        )

    def _purge_covs_and_coeffs_cache(self):
        """Release cached systems after serialization or when abandoning a fit."""
        self._covs_cache = None
        self._coeffs_cache = None

    def psd_solve(
        self,
        cov: torch.Tensor,
        rhs: torch.Tensor,
        penalty: float,
        return_cho_factor: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Solve ridge regression via Cholesky factorization, overwriting :attr:`cov` and :attr:`rhs`.

        Multiple right-hand sides are supported.
        Instead of providing the data matrix (commonly :math:`X` in ridge-regression notation),
        and labels (commonly :math:`y`), we are given directly :math:`\text{cov} = X^{\top} X`
        and :math:`\text{rhs} = X^{\top} y`.
        Since :attr:`cov` is symmetric only its **upper triangle** will be accessed.

        To limit memory usage, the :attr:`cov` matrix **may be overwritten**, and :math:`rhs`
        may also be overwritten (depending on its memory layout).

        Args:
            cov (Tensor): covariance of the linear system
            rhs (Tensor): right hand side (one or more) of the linear system
            penalty (float): Tikhonov l2 penalty
            return_cho_factor (bool): Also return the packed upper Cholesky factor.

        Returns:
            The ridge regression coefficients, or (coefficients, packed factor)
            when return_cho_factor is True.
        """
        # Add diagonal without copies
        cov.diagonal().add_(penalty)
        # Solve with cholesky on GPU
        L = torch.linalg.cholesky(cov, upper=True)
        rhs_shape = rhs.shape
        solution = torch.cholesky_solve(rhs.view(cov.shape[0], -1), L, upper=True)
        solution = solution.view(rhs_shape)
        if return_cho_factor:
            return solution, pack_upper(L).detach()
        return solution

    def _best_cho_factor(self, model: FrankenPotential, best_model: LogEntry):
        """Repeat the winning solve for the latest fitted model, capturing its packed factor."""
        if not self.save_cho_factor:
            return None
        if self._covs_cache is None and self._coeffs_cache is None:
            return None  # Manual serialization without a preceding fit.
        solver_hps = best_model.to_dict()["hyperparameters"]["solver"]
        solver_hps = {k: v for k, v in solver_hps.items() if k != "dtype"}
        covs, coeffs = self._restore_covs_and_coeffs(model.rf.total_random_features)
        _, factor = self.solve(
            covs=covs, coeffs=coeffs, return_cho_factor=True, **solver_hps
        )
        return factor

    @torch.no_grad()
    def get_statistics(self, model: FrankenPotential) -> Tuple[Statistics, dict]:
        """Compute statistics on the training dataset with the provided model

        Args:
            model (FrankenPotential): Franken model from which the attached GNN
                is used to compute the features on atomic configurations.

        Returns:
            A tuple containing an object of type :class:`franken.rf.scaler.Statistics` containing
            the dataset statistics, and a dictionary containing the GNN-backbone hyperparameters
            used when computing dataset features.
        """
        if self.statistics_ is None or not are_dicts_equal(
            self.statistics_[1], asdict_with_classvar(model.gnn_config)
        ):
            stat = compute_dataset_statistics(
                dataset=self.train_dataloader.dataset,  # type: ignore
                gnn=model.gnn,
                device=self.device,
            )
            stat_dict = asdict_with_classvar(model.gnn_config)
            self.statistics_ = (stat, stat_dict)

        return self.statistics_

    @abc.abstractmethod
    def fit(
        self,
        model: FrankenPotential,
    ) -> tuple[LogCollection, torch.Tensor]:
        """Fit a given franken model on the training set.

        Args:
            model (FrankenPotential): The model which defines GNN and random features.
            solver_params (dict): Parameters for the solver which actually
                performs the fit.

        Returns:
            tuple[LogCollection, torch.Tensor]:
            - Logs which contain all parameters related to the fitting, as well as timings.
            - Weights which were learned during the fit.
        """
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
        log_collection: LogCollection,
        all_weights: torch.Tensor,
    ) -> LogCollection:
        """Evaluate a fitted model by computing metrics on a validation dataset.

        Args:
            model: The model which defines GNN and random features.
            dataloader (torch.utils.data.DataLoader): Evaluation will run the model
                on each configuration in the dataloader, computing averaged metrics.
            log_collection: Log object as output by the :meth:`fit`
                method. Metric values will be added to the logs and the same object will
                be returned by this method.
            all_weights (torch.Tensor): The weights as output by the :meth:`fit` method.

        Returns:
            logs (LogCollection): Logs which contain all parameters related
            to the fitting, as well as timings and metrics.
        """
        pass

    def serialize_logs(
        self,
        model: FrankenPotential,
        log_collection: LogCollection,
        all_weights: torch.Tensor,
        best_model_selection: list[str],
        best_model_split: DataSplit = DataSplit.TRAIN,
    ):
        """Serialize the latest fitted model and release its cached system, even on failure."""
        try:
            assert self.log_dir is not None, "Log directory is not set"
            model_hash_set = set(log.checkpoint_hash for log in log_collection)
            assert len(model_hash_set) == 1
            model_hash = model_hash_set.pop()
            log_collection.save_json(self.log_dir / "log.json")

            # Save the model checkpoint
            if self.save_every_model:
                ckpt_dir = self.log_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model_save_path = ckpt_dir / f"{model_hash}.pt"
                model.save(model_save_path, multi_weights=all_weights)
                logger.debug(
                    f"Saved multiple models (hash={model_hash}) to {model_save_path}"
                )
            # Log the best model
            self.serialize_best_model(
                model,
                all_weights,
                split=best_model_split,
                best_model_selection=best_model_selection,
            )
        finally:
            self._purge_covs_and_coeffs_cache()

    def serialize_best_model(
        self,
        model: FrankenPotential,
        all_weights: torch.Tensor,
        best_model_selection: list[str],
        split: DataSplit = DataSplit.TRAIN,
    ) -> None:
        """Save the best checkpoint using the latest fit's system, then release the cache."""
        try:
            assert self.log_dir is not None, "Log directory is not set"
            log_collection = LogCollection.from_json(self.log_dir / "log.json")
            best_model = log_collection.get_best_model(
                split=split, metrics_to_minimize=best_model_selection
            )

            best_model_file = self.log_dir / "best.json"
            should_save = True
            if best_model_file.exists():
                with open(best_model_file, "r") as f:
                    current_best = LogEntry.from_dict(json.load(f))
                if best_model == current_best:
                    should_save = False

            if should_save:
                logger.debug(f"Identified new best model: {best_model}")
                model._cho_factor = self._best_cho_factor(model, best_model)
                weights = all_weights[best_model.checkpoint_rf_weight_id]
                model.rf.weights = weights.reshape_as(model.rf.weights)
                model.save(self.log_dir / "best_ckpt.pt")
                with open(best_model_file, "w") as f:
                    json.dump(best_model.to_dict(), f, indent=4, cls=dtypeJSONEncoder)
                logger.debug(
                    f"Saved best model (within-experiment ID={best_model.checkpoint_rf_weight_id}) "
                    f"to {self.log_dir / 'best_ckpt.pt'}"
                )
        finally:
            self._purge_covs_and_coeffs_cache()
