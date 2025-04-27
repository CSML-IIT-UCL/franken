import hashlib
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Any, List, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.utils.data
from torch import Tensor

import franken.metrics
from franken.metrics.base import BaseMetric
import franken.utils.distributed as dist_utils
from franken.data.base import Target
from franken.rf.model import FrankenPotential
from franken.trainers import BaseTrainer
from franken.trainers.log_utils import (
    DataSplit,
    HyperParameterGroup,
    LogCollection,
    LogEntry,
)
from franken.utils.linalg.cov import (
    lowmem_normalize_leading_eig,
    rank1_update,
    rankk_update,
)
from franken.utils.linalg.psdsolve import psd_ridge
from franken.utils.linalg.tri import triangular_lerp
from franken.utils.misc import params_grid, throughput


logger = logging.getLogger("franken")


class RandomFeaturesTrainer(BaseTrainer):
    """Main class which groups training and evaluation functionality for franken models."""

    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        random_features_normalization: Literal["leading_eig"] | None = "leading_eig",
        log_dir: Path | None = None,
        save_every_model: bool = True,
        device: torch.device | str | int = "cuda:0",
        dtype: str | torch.dtype = torch.float32,
        save_fmaps: bool = True,
    ):
        """Initialize random-feature trainer class

        Args:
            train_dataloader (torch.utils.data.DataLoader):
                Dataloader which iterates over the training set.
            random_features_normalization (Literal["leading_eig"] | None, optional):
                How to normalize the covariance matrices formed by random-features. Defaults to "leading_eig".
            log_dir (Path | None, optional):
                Directory where to save logs and models. If not specified, no logs will be saved.
                Defaults to None.
            save_every_model (bool, optional):
                Model fitting with this class is done simultaneously for a list
                of solver parameters. This argument controls the behavior of model saving:
                if set to True, the models corresponding to all solver parameters will be saved,
                otherwise only the 'best' model among them (according to some validation set) will
                be saved. Defaults to True.
            device (_type_, optional):
                PyTorch device on which computations are performed. Defaults to "cuda:0".
            dtype (str | torch.dtype, optional):
                Data-type for solver operations. Random features will be computed in float32, and
                then converted to float64 if requested. Defaults to torch.float32.
            save_fmaps (bool, optional):
                Whether or not to save feature-maps for the training set. Saving them
                requires extra memory (linear in the training-set size), but speeds up
                the :meth:`~franken.trainers.FrankenPotential.evaluate` method on training
                data. Defaults to True.
        """
        super().__init__(
            train_dataloader,
            log_dir=log_dir,
            save_every_model=save_every_model,
            device=device,
            dtype=dtype,
        )
        self.random_features_normalization = random_features_normalization
        self.save_fmaps = save_fmaps

    def on_fit_start(self, model: FrankenPotential):
        # initialize input scaler based on statistics property
        model.input_scaler.set_from_statistics(self.get_statistics(model)[0])
        # initialize energy shift based on atomic energies
        if not model.energy_shift.is_initialized:
            model.energy_shift.set_from_atomic_energies(
                self.train_dataloader.dataset.atomic_energies
            )

    def fit(
        self,
        model: FrankenPotential,
        solver_params: Mapping[str, Sequence[Any]],
    ) -> tuple[LogCollection, torch.Tensor]:
        """Fit a given franken model on the training set.

        Args:
            model (FrankenPotential): The model which defines GNN and random features.
            solver_params (dict): Parameters for the solver which actually
                performs the fit. This argument allows to specify multiple parameters,
                for each of which we will perform a fit. For example

                >>> solver_params = {
                >>>     "l2_penalty": [1e-6, 1e-4],
                >>>     "force_weight": [0.5]
                >>> }

                will result in two different models, one with :code:`l2_penalty=1e-6, force_weight=0.5`
                and one with :code:`l2_penalty=1e-4, force_weight=0.5`. This way of specifying solver
                parameters allows to easily perform a grid-search.

        Returns:
            logs (LogCollection): Logs which contain all parameters related
                to the fitting, as well as timings.
            weights (torch.Tensor): Weights which were learned during the fit.

        Note:
            More information about the available solver parameters can be found under the
            :meth:`solve` method.
        """
        if self.device.type == "cuda":
            # Patch E3NN for batched jacobians!
            from franken.backbones.wrappers.common_patches import patch_e3nn

            patch_e3nn()

        model = model.to(self.device)
        self.on_fit_start(model)
        model_hash = hashlib.md5(str(model.hyperparameters).encode())
        model_hash = model_hash.hexdigest()

        t_cov_coeffs_start = perf_counter()
        self._compute_covs_and_coeffs(model, self.train_dataloader)
        t_cov_coeffs = perf_counter() - t_cov_coeffs_start

        solver_grid_size = math.prod([len(v) for v in solver_params.values()])
        all_weights = torch.zeros(
            solver_grid_size,
            model.rf.total_random_features,
            dtype=self.buffer_dt,
            device=self.device,
        )

        solver_iter = throughput(
            params_grid(solver_params, split_distributed=True),
            desc="least-squares",
            units="models",
            device=self.device,
        )

        local_logs = dict()
        num_failed = torch.zeros((1,), device=self.device, dtype=torch.int)
        for hp_idx, hp_val in solver_iter:
            t_solve_start = perf_counter()
            try:
                weights = self.solve(**hp_val)
            except torch.linalg.LinAlgError as e:
                weights = torch.full_like(all_weights[hp_idx], torch.inf)
                num_failed += 1
                logger.debug(f"Hyperparameter {hp_val} failed. Error: {e}")
            finally:
                t_solve = perf_counter() - t_solve_start
            # Update all weights
            all_weights[hp_idx].copy_(weights.view(-1))

            # Logging
            solver_hps = hp_val | {"dtype": self.buffer_dt}
            hp_groups = model.hyperparameters | {"solver": solver_hps}
            hyperparameters = []
            for group_name, hps in hp_groups.items():
                hyperparameters.append(HyperParameterGroup.from_dict(group_name, hps))

            local_logs[hp_idx] = LogEntry(
                checkpoint_hash=model_hash,
                checkpoint_rf_weight_id=hp_idx,
                timings_cov_coeffs=t_cov_coeffs,
                timings_solve=t_solve,
                hyperparameters=hyperparameters,
            )

        # Broadcast weights and logs
        dist_utils.all_sum(all_weights)
        log_collection = LogCollection.gather_from_ranks(local_logs)
        assert len(log_collection) == solver_grid_size

        # Reporting failed runs
        dist_utils.all_sum(num_failed)

        if num_failed.item() > 0:
            logger.warning(
                f"Solver failed in {num_failed.item()}/{solver_grid_size} cases."
            )

        return log_collection, all_weights

    def evaluate(
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
        log_collection: LogCollection,
        all_weights: torch.Tensor | None,
        metrics: Sequence[str] = ("energy_MAE", "forces_MAE", "forces_cosim"),
    ) -> LogCollection:
        if self.device.type == "cuda":
            # Patch E3NN for batched jacobians!
            from franken.backbones.wrappers.common_patches import patch_e3nn

            patch_e3nn()
        tot_dset_size = len(dataloader.dataset)  # type: ignore

        metric_objects: List[BaseMetric] = []
        for name in metrics:
            try:
                metric_obj = franken.metrics.init_metric(
                    name, device=self.device, dtype=self.buffer_dt
                )
                metric_objects.append(metric_obj)
            except KeyError:
                logger.warning(
                    f"Unknown metric {name}. Skipping. Available metrics: {franken.metrics.available_metrics()}"
                )

        split_name = dataloader.dataset.split
        try:
            split = DataSplit[split_name.upper()]
        except KeyError:
            logger.warning(f"Unrecognized split '{split_name}' in dataloader")
            split = DataSplit.UNDEFINED

        progress_bar = throughput(
            dataloader,
            desc=f"{split.name.lower()} evaluation",
            total=tot_dset_size,
            device=self.device,
        )
        for i, (data, targets) in enumerate(progress_bar):
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            if split == DataSplit.TRAIN and self.save_fmaps:
                # Shortcut to compute predictions for the training-set, for which
                # we already have computed energy and force feature maps
                assert len(self.forces_fmap) == len(self.energy_fmap)
                assert len(self.forces_fmap) == len(dataloader)
                predictions = Target(
                    *model.energy_and_forces_from_fmaps(
                        data,
                        energy_fmap=self.energy_fmap[i],
                        forces_fmap=self.forces_fmap[i],
                        weights=all_weights,
                        add_energy_shift=False,  # since it's always train here
                    )
                )
            else:
                if all_weights is None or all_weights.shape[0] <= 100:
                    forces_mode = "torch.autograd"
                else:
                    forces_mode = "torch.func"
                predictions = Target(
                    *model.energy_and_forces(
                        data,
                        weights=all_weights,
                        forces_mode=forces_mode,
                        add_energy_shift=(False if split == DataSplit.TRAIN else True),
                    )
                )
            if torch.any(torch.isnan(predictions.energy)):
                logger.warning(
                    f"Configuration {i} - {split_name} has NaNs in energy predictions"
                )
            if predictions.forces is not None and torch.any(
                torch.isnan(predictions.forces)
            ):
                logger.warning(
                    f"Configuration {i} - {split_name} has NaNs in force predictions"
                )
            for metric in metric_objects:
                metric.update(predictions, targets)

        num_models = (
            all_weights.shape[0]
            if all_weights is not None
            else model.rf.weights.shape[0]
        )
        # Sync metrics across GPUs
        metric_values = {}
        for metric in metric_objects:
            value = metric.compute(reset=False)
            if metric.samples_counter.item() != tot_dset_size:
                logger.warning(
                    f"Metric {metric.name} has {metric.samples_counter.item()} samples "
                    f"while the dataset has {tot_dset_size} configurations."
                )
            assert value.ndim == 1
            assert value.shape[0] == num_models
            metric_values[metric.name] = value

        # Explode metric values into list of MetricLog
        raw_logs = []
        for idx in range(num_models):
            results = []
            for name, value in metric_values.items():
                float_value = value[idx].item()
                results.append(dict(name=name, value=float_value))
            raw_logs.append(results)

        assert len(raw_logs) == len(log_collection)

        for log_entry, results in zip(log_collection, raw_logs):
            for metric in results:
                try:
                    log_entry.add_metric(metric["name"], metric["value"], split)
                except ValueError as e:
                    logger.warning(f"Could not add metric because: {str(e)}")
        return log_collection

    def warn_save_fmaps(
        self, energy_fmap: Tensor, forces_fmap: Tensor, num_maps: int
    ) -> None:
        fmap_size = np.prod(energy_fmap.shape) + np.prod(forces_fmap.shape)
        tot_bytes = fmap_size * forces_fmap.element_size() * num_maps
        if self.device.type == "cuda":
            avail_bytes = torch.cuda.mem_get_info(self.device)[0]
            if tot_bytes > 0.8 * avail_bytes:
                logger.warning(
                    f"Saved feature maps require {tot_bytes / 2**30:.2f}GB of device memory. "
                    f"Device has {avail_bytes / 2**30:.2f}GB of available memory, a crash may occur. "
                    f"Use 'trainer.save_fmaps=False' in your config to stop saving feature maps."
                )
            else:
                logger.info(
                    f"Saved feature maps require {tot_bytes / 2**30:.2f}GB of device memory."
                )

    @torch.no_grad()
    def _compute_covs_and_coeffs(
        self, model: FrankenPotential, dataloader: torch.utils.data.DataLoader
    ) -> None:
        tot_dset_size = len(dataloader.dataset)  # type: ignore
        n_rf = model.rf.total_random_features
        # Initialize Buffers: `covariance` will have
        # - covariance of forces on LOWER triangle
        # - covariance of energies on UPPER triangle
        # with diagonals stored separately, and linsys coefficients.
        self.covariance = torch.zeros(
            (n_rf, n_rf), device=self.device, dtype=self.buffer_dt
        )
        self.diag_energy = torch.zeros(
            (n_rf,), device=self.device, dtype=self.buffer_dt
        )
        self.diag_forces = torch.zeros(
            (n_rf,), device=self.device, dtype=self.buffer_dt
        )
        self.coeffs_energy = torch.zeros(
            (n_rf,), device=self.device, dtype=self.buffer_dt
        )
        self.coeffs_forces = torch.zeros(
            (n_rf,), device=self.device, dtype=self.buffer_dt
        )
        # NOTE: the feature maps are never synced between devices.
        #       They can only used correctly by iterating through the
        #       same dataloader as here, using the same method. Otherwise
        #       they may not be in the correct order
        self.energy_fmap = []
        self.forces_fmap = []

        progress_bar = throughput(
            dataloader,
            desc="covs+coeffs",
            total=tot_dset_size,
            device=self.device,
        )

        for i, (data, targets) in enumerate(progress_bar):
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)

            energy_per_atom = targets.energy / data.natoms
            forces_per_atom = targets.forces / data.natoms

            forces_fmap, energy_fmap = model.grad_feature_map(data)
            energy_fmap = energy_fmap.to(dtype=self.buffer_dt)

            rank1_update(self.covariance, self.diag_energy, energy_fmap, upper=True)
            self.coeffs_energy.add_(energy_fmap, alpha=energy_per_atom.item())

            forces_fmap = (-forces_fmap).to(dtype=self.buffer_dt)
            forces_fmap = forces_fmap.view(forces_fmap.shape[0], -1)
            rankk_update(self.covariance, self.diag_forces, forces_fmap, upper=False)
            self.coeffs_forces.addmv_(
                forces_fmap, forces_per_atom.view(-1).to(dtype=self.buffer_dt)
            )
            if self.save_fmaps:
                if i == 0:
                    self.warn_save_fmaps(energy_fmap, forces_fmap, len(dataloader))
                self.energy_fmap.append(energy_fmap)
                self.forces_fmap.append(forces_fmap)

        # Sync covariance matrices & coefficients
        dist_utils.all_sum(self.covariance)
        dist_utils.all_sum(self.diag_forces)
        dist_utils.all_sum(self.diag_energy)
        dist_utils.all_sum(self.coeffs_energy)
        dist_utils.all_sum(self.coeffs_forces)

        if self.random_features_normalization == "leading_eig":
            logger.warning(
                "`leading_eig` normalization has high memory usage. If you encounter OOM errors try to disable it."
            )
            lowmem_normalize_leading_eig(
                self.covariance, self.diag_energy, self.coeffs_energy, upper=True
            )
            lowmem_normalize_leading_eig(
                self.covariance, self.diag_forces, self.coeffs_forces, upper=False
            )
        elif self.random_features_normalization is not None:
            raise ValueError(
                f"Covariance normalization {self.random_features_normalization} is not implemented."
            )

    @torch.no_grad()
    def solve(self, force_weight: float, l2_penalty: float = 1e-6) -> Tensor:
        # This is the 2nd copy of the covariance matrix that we need to store.
        lerped_cov, lerped_diag = triangular_lerp(
            self.covariance,
            diag_upper=self.diag_energy,
            diag_lower=self.diag_forces,
            weight=force_weight,
            inplace=False,
        )
        lerped_cov.diagonal().copy_(lerped_diag)
        rhs = torch.lerp(self.coeffs_energy, self.coeffs_forces, force_weight)
        solution = psd_ridge(lerped_cov, rhs, l2_penalty)
        return solution
