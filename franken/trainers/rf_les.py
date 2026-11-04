import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Literal, Mapping, Sequence, cast
import warnings

import torch
import torch.utils.data
from torch import Tensor
import tqdm

from franken.data.dataset import FrankenAtomsDataset
from franken.metrics.base import BaseMetric
from franken.rf.les_model import LESFrankenPotential
from franken.trainers.rf_trainer import RandomFeaturesTrainer
import franken.utils.distributed as dist_utils
from franken.data.base import (
    TARGET_UNITS,
    Configuration,
    Target,
    TargetType,
    all_target_keys,
    is_scalar_target,
)
from franken.trainers.log_utils import (
    DataSplit,
    HyperParameterGroup,
    LogCollection,
    LogEntry,
)
from franken.utils.misc import no_jit, throughput

logger = logging.getLogger("franken")


class RandomFeaturesEwaldsTrainer(RandomFeaturesTrainer):
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        training_targets: list[TargetType],
        l2_penalty: float,
        target_weight: Mapping[TargetType, float],
        optimizer: Literal["adam", "lbfgs"],
        num_cycles: int,
        epochs_per_cycle: int,
        learning_rate: float,
        lr_decay: float,
        lbfgs_max_iter: int,
        lbfgs_max_eval: int,
        lbfgs_learning_rate: float,
        lbfgs_history_size: int,
        lbfgs_tolerance_grad: float,
        lbfgs_tolerance_change: float,
        mode: Literal["alternating", "variable_projection", "joint"],
        random_features_normalization: Literal["leading_eig"] | None = "leading_eig",
        log_dir: Path | None = None,
        save_every_model: bool = True,
        device: torch.device | str | int = "cuda:0",
        dtype: str | torch.dtype = torch.float32,
        save_fmaps: bool = True,
        metrics: list[str] | None = None,
    ):
        super().__init__(
            train_dataloader,
            training_targets=training_targets,
            l2_penalty=l2_penalty,
            target_weight=target_weight,
            random_features_normalization=random_features_normalization,
            log_dir=log_dir,
            save_every_model=save_every_model,
            device=device,
            dtype=dtype,
            save_fmaps=save_fmaps,
            metrics=metrics,
        )
        # ensure no multi-weight
        for k, v in self.solver_hps.items():
            if len(v) > 1:
                raise ValueError(
                    f"RandomFeaturesEwaldsTrainer does not support grid-search over hyperparameters. "
                    f"Multiple values were found for hyperparameter {k}, but only a single value is supported."
                )
        if not save_fmaps and mode == "variable_projection":
            warnings.warn(
                "save_fmaps is False. This requires many extra passes through the GNN backbone. "
                "If GPU memory allows it, set `save_fmaps=True`."
            )
        self.mode: Literal["alternating", "variable_projection", "joint"] = mode
        self.val_dataloader: torch.utils.data.DataLoader | None = None
        self.l2_penalty = l2_penalty
        self.rel_tgt_weight = self._get_relative_weights(target_weight)
        self.optimizer = optimizer
        self.num_cycles = num_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_max_eval = lbfgs_max_eval
        self.lbfgs_learning_rate = lbfgs_learning_rate
        self.lbfgs_history_size = lbfgs_history_size
        self.lbfgs_tolerance_grad = lbfgs_tolerance_grad
        self.lbfgs_tolerance_change = lbfgs_tolerance_change
        self.training_history: list[dict] = []
        self._les_optimizer = None
        self.fmaps: dict[TargetType, list[Tensor]] = {}

    def create_log_entry(self, model):
        solver_hps = {
            "dtype": self.buffer_dt,
            "l2_penalty": self.l2_penalty,
            "optimizer": self.optimizer,
            "num_cycles": self.num_cycles,
            "epochs_per_cycle": self.epochs_per_cycle,
            "learning_rate": self.learning_rate,
            "lr_decay": self.lr_decay,
            "lbfgs_max_iter": self.lbfgs_max_iter,
            "lbfgs_learning_rate": self.lbfgs_learning_rate,
            "lbfgs_history_size": self.lbfgs_history_size,
            "lbfgs_tolerance_grad": self.lbfgs_tolerance_grad,
            "lbfgs_tolerance_change": self.lbfgs_tolerance_change,
            "mode": self.mode,
        } | {f"{k}_weight": v[0] for k, v in self.target_weight.items()}
        # NOTE: the LES HPs will be duplicated (present both here and in the solver
        # due to the LESConfig containing them).
        hp_groups = model.hyperparameters | {"solver": solver_hps}
        # Note: LES fits each solver configuration as a separate model (weight ID 0).
        # Include solver settings so best-model comparison distinguishes trials.
        model_hash = hashlib.md5(str(hp_groups).encode()).hexdigest()
        hyperparameters = []
        for group_name, hps in hp_groups.items():
            hyperparameters.append(HyperParameterGroup.from_dict(group_name, hps))
        local_log = LogEntry(
            checkpoint_hash=model_hash,
            checkpoint_rf_weight_id=0,
            timings_cov_coeffs=0,  # TODO: Fix timings in logs (need to allow arbitrary timings)
            timings_solve=0,
            hyperparameters=hyperparameters,
        )
        return local_log

    def eval_summary(
        self,
        log: LogEntry,
        cycle: int,
        splits: Sequence[DataSplit],
        title="",
        prev_metrics: dict | None = None,
    ) -> str:
        def _get_first_available_metric(
            candidates: list[str], split: DataSplit
        ) -> tuple[float, str] | tuple[None, None]:
            for name in candidates:
                try:
                    return log.get_metric(name, split), name
                except KeyError:
                    pass
            return None, None

        lines = ["=" * 60, f"[Cycle {cycle:3}] {title.upper()}"]
        for split in splits:
            entries = []
            for tgt in all_target_keys():
                error, metric_name = _get_first_available_metric(
                    [f"{tgt}_RMSE", f"{tgt}_MAE"], split
                )
                if error is None:
                    continue
                delta_str = ""
                if prev_metrics is not None:
                    split_key = split.name.lower()
                    key = f"{split_key}:{metric_name}"
                    if key in prev_metrics:
                        delta = error - prev_metrics[key]
                        delta_str = f" ({delta:+.2f})"
                entries.append(
                    f"{metric_name}={error:.2f} {TARGET_UNITS[tgt]}{delta_str}"
                )
            split_name = "VAL" if split == DataSplit.VAL else split.name
            lines.append(f"{split_name:<6}: " + " | ".join(entries))
        lines.append("=" * 60)
        return "\n".join(lines)

    def _get_previous_metrics(self, step: str):
        """store metrics of previous cycle to compare with the new ones and check improvements"""
        for entry in reversed(self.training_history):
            if entry["step"] != step:
                continue
            prev = {}
            for split_name, split_metrics in entry["metrics"].items():
                for metric_name, value in split_metrics.items():
                    prev[f"{split_name}:{metric_name}"] = value
            return prev
        return None

    def _print_eval(self, model, cycle, step: Literal["rff", "les", "joint"]):
        logc = LogCollection([self.create_log_entry(model)])
        for split, loader in (
            (DataSplit.TRAIN, self.train_dataloader),
            (DataSplit.VAL, self.val_dataloader),
        ):
            if loader is None:
                continue
            self.evaluate(model, loader, log_collection=logc, all_weights=None)
        if dist_utils.get_rank() == 0:
            splits = [DataSplit.TRAIN]
            if self.val_dataloader is not None:
                splits.append(DataSplit.VAL)
            summary = self.eval_summary(
                log=logc[0],
                cycle=cycle + 1,
                splits=splits,
                title=f"after {step.upper()}",
                prev_metrics=self._get_previous_metrics(step),
            )
            logger.info(summary)

        # Store history / best model
        if dist_utils.get_rank() == 0:
            self.training_history.append(
                {
                    "cycle": cycle + 1,
                    "step": step,
                    "metrics": logc[0].to_dict()["metrics"],
                }
            )
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)

                history_path = self.log_dir / "training_history.json"
                temp_path = history_path.with_suffix(".json.tmp")

                temp_path.write_text(json.dumps(self.training_history, indent=2))
                temp_path.replace(history_path)

    def _get_relative_weights(
        self, hps: Mapping[TargetType, float]
    ) -> dict[TargetType, float]:
        weights: dict[TargetType, float] = {
            target: hps.get(target, 1.0) for target in self.training_targets
        }
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Target weights must be finite and nonnegative")
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("At least one training target must have positive weight")
        return {t: w / total_weight for t, w in weights.items()}

    def _full_gradient(self, model, normalization):
        """Accumulate a mean gradient, freeing each structure's graph immediately."""
        n_samples = len(self.train_dataloader.dataset)
        total = torch.zeros((), device=self.device, dtype=self.buffer_dt)
        # Each device will process its own subset of the dataset.
        for i, (data, targets) in enumerate(self.train_dataloader):
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            if data.natoms.numel() != 1:
                raise ValueError("LES accumulation expects individual configurations")
            if self.mode == "variable_projection" or (
                self.fmaps and self.mode != "joint"
            ):
                # RFF is fixed in this derivative. Use the same linear maps as
                # its solve, in solver precision. Differentiating the combined
                # energy instead rounds the RFF force back to descriptor dtype,
                # which can spoil a projected line search through cancellation.
                fmaps = (
                    {target: self.fmaps[target][i] for target in self.training_targets}
                    if self.fmaps
                    else model.grad_feature_map(data, list(self.training_targets))
                )
                les_predictions = model.predict_les(data, list(self.training_targets))
                predictions = {}
                for target in self.training_targets:
                    short_range = (
                        model.rf.weights.detach().to(self.buffer_dt)
                        @ fmaps[target].to(self.buffer_dt)
                    ) * data.natoms
                    predictions[target] = short_range.reshape_as(
                        les_predictions[target]
                    ) + les_predictions[target].to(self.buffer_dt)
            else:
                predictions = model.predict(
                    targets=list(self.training_targets),
                    data=data,
                    add_energy_shift=False,
                )
            loss = total.new_zeros(())
            for target, weight in self.rel_tgt_weight.items():
                residual = (
                    predictions[target].to(self.buffer_dt)
                    - targets[target].to(self.buffer_dt)
                ) / data.natoms.to(self.buffer_dt)
                squared_error = residual.square().sum()
                if normalization is not None:
                    squared_error = squared_error / normalization[target].squeeze()
                loss = loss + weight * squared_error
            loss = loss / n_samples
            loss.backward()
            total += loss.detach()
        dist_utils.all_sum(total)
        # Sum gradients over all devices
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        dist_utils.all_sum_coalesced(grads)
        return total

    def _fit_les(
        self,
        model,
        epoch,
        normalization: dict[TargetType, Tensor] | None,
        proj_mat: Tensor | None = None,
    ):
        local_n_samples = len(self._train_dataset_indices)

        for target in self.training_targets:
            assert len(self.fmaps[target]) == local_n_samples, (
                f"Inconsistent fmap length for {target}: "
                f"{len(self.fmaps[target])} != {local_n_samples}"
            )
        if local_n_samples == 0:
            raise ValueError("LES requires a nonempty training dataset")

        params = list(model.les.parameters())
        if self.mode == "joint":
            params += list(model.rf.parameters())
        if self.optimizer == "adam":
            lr = self.learning_rate * self.lr_decay**epoch
            if self._les_optimizer is None:
                self._les_optimizer = torch.optim.Adam(params, lr=lr, eps=1e-8)
            optim = self._les_optimizer
            for group in optim.param_groups:
                group["lr"] = lr
            for _ in tqdm.trange(self.epochs_per_cycle, desc="LES epochs"):
                optim.zero_grad(set_to_none=True)
                self._full_gradient(model, normalization)
                optim.step()
        else:
            # RFF refits change this objective: do not reuse curvature history.
            lr = self.lbfgs_learning_rate * self.lr_decay**epoch
            if self._les_optimizer is None or self.mode != "variable_projection":
                self._les_optimizer = torch.optim.LBFGS(
                    params,
                    lr=lr,
                    max_eval=self.lbfgs_max_eval,
                    max_iter=self.lbfgs_max_iter,
                    history_size=self.lbfgs_history_size,
                    tolerance_grad=self.lbfgs_tolerance_grad,
                    tolerance_change=self.lbfgs_tolerance_change,
                    line_search_fn="strong_wolfe",
                )
            optim = self._les_optimizer
            for group in optim.param_groups:
                group["lr"] = lr

            def closure():
                if self.mode == "variable_projection":
                    assert proj_mat is not None
                    self._fit_projected_rff(model, proj_mat, normalization)
                optim.zero_grad(set_to_none=True)
                loss = self._full_gradient(model, normalization)
                if self.mode == "variable_projection":
                    # Envelope theorem: no derivative through the exact RFF
                    # minimizer is needed. Its ridge cost is still needed by
                    # the line search, since it changes between trial points.
                    loss = loss + (
                        self.l2_penalty
                        * model.rf.weights.detach().square().sum()
                        / n_samples
                    )
                return loss

            optim.step(closure)
            if self.mode == "variable_projection":
                # The last closure may have evaluated a rejected trial.
                assert proj_mat is not None
                # GIAC: Test without this last call
                # self._refit_projected_rff(model, proj_mat, normalization)

        self._print_eval(
            model,
            cycle=epoch,
            step="joint" if self.mode == "joint" else "les",
        )

    @torch.no_grad()
    def _prepare_projection(self, covs: dict[TargetType, Tensor]) -> Tensor:
        l2_penalty = self.l2_penalty
        target_weights = self.rel_tgt_weight
        matrix: torch.Tensor = cast(
            Tensor, sum(target_weights[t] * covs[t] for t in self.training_targets)
        )
        matrix.diagonal().add_(l2_penalty)
        return torch.linalg.cholesky(matrix, upper=True)

    @torch.no_grad()
    def _fit_projected_rff(
        self,
        model: LESFrankenPotential,
        proj_mat: Tensor,
        normalization: dict[TargetType, Tensor] | None,
        direct_coeffs: dict[TargetType, Tensor] | None = None,
    ) -> Tensor:
        if direct_coeffs is not None:
            coeffs = direct_coeffs
        else:
            coeffs = self.residual_coeffs(
                model, self.train_dataloader, normalization=normalization
            )
        target_weights = self.rel_tgt_weight
        coeffs = self.residual_coeffs(model, self.train_dataloader, normalization)
        weighted_coeffs: Tensor = cast(
            Tensor, sum(target_weights[t] * coeffs[t] for t in self.training_targets)
        )
        solution = torch.cholesky_solve(weighted_coeffs[:, None], proj_mat, upper=True)
        # weights from [n_rf, 1] to [1, n_rf]
        solution = solution.unsqueeze(0).squeeze(-1)
        model.rf.weights = torch.nn.Parameter(solution)
        return solution

    @no_jit()
    def fit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, model: LESFrankenPotential
    ) -> tuple[LogCollection, torch.Tensor]:
        """Fit a given franken model on the training set.

        Args:
            model (LESFrankenPotential): The model which defines GNN, random features and LES module.

        Returns:
            tuple[LogCollection, torch.Tensor]:
                The fitting logs, together with the learned weights.
        """
        self.patch_e3nn()
        self.training_history = []
        self._les_optimizer = None
        self.on_fit_start(model)

        self._train_dataset_indices = list(self.train_dataloader.sampler)
        covs, coeffs, normalization, proj_mat = None, None, None, None
        if self.mode != "joint":  # Joint doesn't need covariance!
            t_cov = perf_counter()
            covs, coeffs = self._covs_and_coeffs(model, self.train_dataloader)
            if self.fmaps and len(self._train_dataset_indices) != len(
                self.fmaps[self.training_targets[0]]
            ):
                raise RuntimeError(
                    "Training sampler indices and cached feature maps have different lengths"
                )
            covs, coeffs, normalization = self._calc_cov_normalization(covs, coeffs)
            proj_mat = self._prepare_projection(covs)
            t_cov = perf_counter() - t_cov

        if self.mode == "variable_projection":
            assert covs is not None
            assert proj_mat is not None
            rf_weights = self._fit_projected_rff(
                model,
                proj_mat,
                normalization,
                direct_coeffs=coeffs,
            )
            for outer_it in range(self.num_cycles):
                self._fit_les(
                    model,
                    epoch=outer_it,
                    normalization=normalization,
                    proj_mat=proj_mat,
                )
        else:
            for outer_it in range(self.num_cycles):
                # 1. Fit the residual of the current LES head, including its
                #    nonzero random initialization in the first cycle.
                if self.mode == "alternating":
                    assert covs is not None
                    assert proj_mat is not None
                    rf_weights = self._fit_projected_rff(
                        model,
                        proj_mat,
                        normalization,
                        direct_coeffs=coeffs if outer_it == 0 else None,
                    )
                    if torch.any(torch.isnan(rf_weights)):
                        logger.warning(
                            f"NaNs encountered in training after RFF step. Stopping at cycle {outer_it}."
                        )
                        break
                    self._print_eval(model, outer_it, step="rff")
                # 2. Train LES on residuals from RFF training
                #    or in joint mode, train also RFF coefficients jointly.
                self._fit_les(
                    model,
                    epoch=outer_it,
                    normalization=normalization,
                    proj_mat=proj_mat,
                )

        self._les_optimizer = None
        log_collection = LogCollection([self.create_log_entry(model)])
        return log_collection, model.rf.weights.detach()

    @no_jit()
    @torch.no_grad()
    def evaluate(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: LESFrankenPotential,
        dataloader: torch.utils.data.DataLoader,
        log_collection: LogCollection,
        all_weights: torch.Tensor | None,
    ) -> LogCollection:
        self.patch_e3nn()
        was_training = model.training
        model.eval()
        model.gnn.franken_val()

        try:
            assert isinstance(dataloader.dataset, FrankenAtomsDataset)
            tot_dset_size = len(dataloader.dataset)
            metric_objects: list[BaseMetric] = self.get_metrics()
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
                predictions = model.predict(
                    targets=self.training_targets,  # type: ignore
                    data=data,
                    weights=all_weights,
                    differential_mode="torch.autograd",
                    add_energy_shift=(False if split == DataSplit.TRAIN else True),
                )
                for tt, val in predictions.items():
                    if torch.any(torch.isnan(val)):
                        logger.warning(
                            f"Configuration {i} - {split_name} has NaNs in {tt} predictions"
                        )
                for metric in metric_objects:
                    metric.update(Target.from_types(predictions), targets, data)  # type: ignore

            num_models = (
                all_weights.shape[0]
                if all_weights is not None
                else model.rf.weights.shape[0]
            )

            # list with one element for each model trained
            for metric in metric_objects:
                metric_values = metric.compute()
                for metric_name, metric_value in metric_values:
                    assert metric_value.shape == (
                        num_models,
                    ), f"Incorrect metric shape: {metric_value.shape=}, {num_models=}"
                    for model_idx in range(metric_value.shape[0]):
                        log_entry = log_collection[model_idx]
                        try:
                            log_entry.add_metric(
                                name=metric_name,
                                value=metric_value[model_idx].item(),
                                split=split,
                            )
                        except ValueError as e:
                            logger.warning(f"Could not add metric: {str(e)}")
            return log_collection
        finally:
            model.train(was_training)

    @no_jit()
    @torch.no_grad()
    def residual_coeffs(
        self,
        model: LESFrankenPotential,
        dataloader: torch.utils.data.DataLoader,
        normalization: dict[TargetType, Tensor] | None,
    ):
        # n_samples = len(dataloader.dataset)  # type: ignore
        local_n_samples = len(self._train_dataset_indices)
        n_rf = model.rf.total_random_features

        coeffs = {
            t: torch.zeros((n_rf,), device=self.device, dtype=self.buffer_dt)
            for t in self.training_targets
        }
        progress_bar = throughput(
            dataloader, "coeffs", total=local_n_samples, device=self.device
        )
        for i, (data, targets) in enumerate(progress_bar):
            assert isinstance(data, Configuration)
            data = data.to(device=self.device)
            assert data.natoms.numel() == 1, "Batched training is not supported"
            targets: Target = targets.to(device=self.device)

            model.eval()
            les_preds = model.predict_les(data, self.training_targets)
            model.train()
            target_fmaps = (
                {t: self.fmaps[t][i] for t in self.training_targets}
                if self.fmaps
                else model.grad_feature_map(data, self.training_targets)
            )
            for tgt_name in self.training_targets:
                try:
                    tgt = targets[tgt_name] - les_preds[tgt_name]
                except KeyError:
                    raise RuntimeError(
                        f"Target {i} does not contain any values for {tgt_name}."
                    )
                tgt_per_atom = (tgt / data.natoms).to(dtype=self.buffer_dt)
                fmap = target_fmaps[tgt_name].to(self.buffer_dt)
                if is_scalar_target(tgt_name):
                    coeffs[tgt_name].add_(fmap.view(-1), alpha=tgt_per_atom.item())
                else:
                    coeffs[tgt_name].addmv_(fmap, tgt_per_atom.view(-1))
        # Sync coefficients
        for tgt_name in self.training_targets:
            dist_utils.all_sum(coeffs[tgt_name])
        # Normalize using covariance coefficients
        if normalization is not None:
            for tgt_name in self.training_targets:
                if tgt_name not in normalization:
                    continue
                coeffs[tgt_name].div_(normalization[tgt_name])
        return coeffs

    @no_jit()
    @torch.no_grad()
    def _calc_cov_normalization(
        self, covs: dict[TargetType, Tensor], coeffs: dict[TargetType, Tensor]
    ) -> tuple[
        dict[TargetType, Tensor],
        dict[TargetType, Tensor],
        dict[TargetType, Tensor] | None,
    ]:
        # RF normalization
        norm_coefs = None
        if self.random_features_normalization == "leading_eig":
            norm_coefs = {}
            for tgt_name in self.training_targets:
                norm, _ = torch.lobpcg(covs[tgt_name], k=1, largest=True)
                norm_coefs[tgt_name] = norm
                covs[tgt_name].div_(norm)
                coeffs[tgt_name].div_(norm)
        elif self.random_features_normalization is not None:
            raise NotImplementedError(
                f"Covariance normalization {self.random_features_normalization} is not implemented."
            )
        return covs, coeffs, norm_coefs

    def serialize_best_model(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: LESFrankenPotential,
        all_weights: torch.Tensor,
        best_model_selection: list[str],
        split: DataSplit = DataSplit.TRAIN,
    ) -> bool:
        should_save = super().serialize_best_model(
            model, all_weights, best_model_selection, split
        )
        assert self.log_dir is not None, "Log directory is not set"
        if should_save:
            history_file = self.log_dir / "training_history.json"
            if history_file.exists():
                shutil.copy2(
                    history_file,
                    self.log_dir / "best_training_history.json",
                )
        return should_save
