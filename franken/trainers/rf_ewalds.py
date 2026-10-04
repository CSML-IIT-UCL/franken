import hashlib
import json
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping

import torch
import torch.utils.data
from torch import Tensor

from franken.metrics.base import BaseMetric
from franken.rf.les_model import LESFrankenPotential
from franken.trainers.rf_trainer import RandomFeaturesTrainer
import franken.utils.distributed as dist_utils
from franken.data.base import (
    Configuration,
    Target,
    TargetType,
    is_scalar_target,
)
from franken.rf.model import FrankenPotential
from franken.trainers.log_utils import (
    DataSplit,
    HyperParameterGroup,
    LogCollection,
    LogEntry,
)
from franken.utils.linalg.psdsolve import psd_ridge
from franken.utils.misc import no_jit, throughput

logger = logging.getLogger("franken")


# NOTES:
#  - does not support hyperparameter tuning internally (only one penalty, weight)
#    -> no multi-weight
class RandomFeaturesEwaldsTrainer(RandomFeaturesTrainer):
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        training_targets: list[TargetType],
        l2_penalty: float,
        target_weight: Mapping[TargetType, float],
        num_cycles: int,
        lbfgs_max_iter: int,
        lbfgs_lr: float,
        lbfgs_lr_decay: float,
        lbfgs_history_size: int,
        lbfgs_tolerance_grad: float,
        lbfgs_tolerance_change: float,
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
        self.mode: Literal["alternating", "joint"] = "alternating"
        self.val_dataloader: torch.utils.data.DataLoader | None = None
        self.training_history: list[dict] = []

        self.l2_penalty: float = l2_penalty
        self.rel_tgt_weight = self._get_relative_weights(target_weight)
        self.num_cycles = num_cycles
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_lr = lbfgs_lr
        self.lbfgs_lr_decay = lbfgs_lr_decay
        self.lbfgs_history_size = lbfgs_history_size
        self.lbfgs_tolerance_grad = lbfgs_tolerance_grad
        self.lbfgs_tolerance_change = lbfgs_tolerance_change

    def create_log_entry(self, model):
        model_hash = hashlib.md5(str(model.hyperparameters).encode())
        model_hash = model_hash.hexdigest()
        solver_hps = {
            "dtype": self.buffer_dt,
            "l2_penalty": self.l2_penalty,
            "num_cycles": self.num_cycles,
            "lbfgs_max_iter": self.lbfgs_max_iter,
            "lbfgs_lr": self.lbfgs_lr,
            "lbfgs_lr_decay": self.lbfgs_lr_decay,
            "lbfgs_history_size": self.lbfgs_history_size,
            "lbfgs_tolerance_grad": self.lbfgs_tolerance_grad,
            "lbfgs_tolerance_change": self.lbfgs_tolerance_change,
        } | {f"{k}_relative_weight": v for k, v in self.rel_tgt_weight.items()}
        hp_groups = model.hyperparameters | {"solver": solver_hps}
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
        self, log: LogEntry, epoch: int, split: DataSplit, title=""
    ) -> str:
        hp_summary = f"[Epoch {epoch:3}] {title} {split.name}"

        def _get_first_available_metric(
            candidates: list[str],
        ) -> tuple[float, str] | tuple[None, None]:
            for name in candidates:
                try:
                    return log.get_metric(name, split), name
                except KeyError:
                    pass
            return None, None

        energy_error, energy_metric = _get_first_available_metric(
            ["energy_RMSE", "energy_MAE"],
        )
        forces_error, forces_metric = _get_first_available_metric(
            ["forces_RMSE", "forces_MAE"],
        )
        stress_error, stress_metric = _get_first_available_metric(
            ["stress_RMSE", "stress_MAE"],
        )
        if energy_error is None:
            energy_error = float("nan")
        hp_summary += f" ({energy_metric} {energy_error:.2f} meV/atom)"
        if forces_error is None:
            forces_error = float("nan")
        hp_summary += f" ({forces_metric} {forces_error:.2f} meV/Ang)"
        if stress_error is not None:
            hp_summary += f" ({stress_metric} {stress_error:.2f} meV/Ang^3)"
        return hp_summary

    def _print_eval(self, model, epoch, step: Literal["rff", "les", "joint"]):
        logc = LogCollection([self.create_log_entry(model)])
        for split, loader in (
            (DataSplit.TRAIN, self.train_dataloader),
            (DataSplit.VAL, self.val_dataloader),
        ):
            if loader is None:
                continue
            self.evaluate(
                model,
                loader,
                log_collection=logc,
                all_weights=None,
            )
            if dist_utils.get_rank() == 0:
                summary = self.eval_summary(
                    log=logc[0],
                    epoch=epoch,
                    split=split,
                    title=f"after {step.upper()} training",
                )
                print(summary)

        if dist_utils.get_rank() == 0:
            self.training_history.append(
                {
                    "cycle": epoch + 1,
                    "step": step,
                    "metrics": logc[0].to_dict()["metrics"],
                }
            )
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                history_path = self.log_dir / "training_history.json"
                # Preserve completed stages even if a later training step fails.
                temp_path = history_path.with_suffix(".json.tmp")
                temp_path.write_text(json.dumps(self.training_history, indent=2))
                temp_path.replace(history_path)
            print()

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

    def _fit_rff(
        self,
        model: LESFrankenPotential,
        covs: dict[TargetType, Tensor],
        normalization: dict[str, Tensor] | None,
        direct_coeffs: dict[TargetType, Tensor] | None = None,
    ) -> Tensor:
        if direct_coeffs is not None:
            coeffs = direct_coeffs
        else:
            coeffs = self.residual_coeffs(
                model, self.train_dataloader, normalization=normalization
            )
        rf_weights = self.solve(
            covs=covs,
            coeffs=coeffs,  # type: ignore
        )
        # weights from [n_rf] to [1, n_rf]
        rf_weights = rf_weights.unsqueeze(0)
        return rf_weights

    def _backward_batch(self, model, indices, do_bwd=True):
        """Accumulate a mean gradient, freeing each structure's graph immediately."""
        target_weights = self.rel_tgt_weight
        per_tgt_losses = {
            tgt_name: torch.zeros((), device=self.device, dtype=self.buffer_dt)
            for tgt_name in self.training_targets
        }
        for index in indices:
            data, targets = self.train_dataloader.dataset[index]
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            if data.natoms.numel() != 1:
                raise ValueError("LES accumulation expects individual configurations")
            predictions = model.predict(
                targets=self.training_targets,
                data=data,
                is_training=do_bwd,
                add_energy_shift=False,
            )
            per_index_loss = torch.zeros((), device=self.device, dtype=self.buffer_dt)
            for tgt_name in self.training_targets:
                try:
                    tgt = targets[tgt_name]
                except KeyError:
                    raise RuntimeError(
                        f"Target {index} does not contain any values for {tgt_name}."
                    )
                loss = target_weights[tgt_name] * torch.mean(
                    (predictions[tgt_name] - tgt) ** 2
                )
                loss = loss / len(indices)  # normalize gradient by the batch size
                per_tgt_losses[tgt_name] += loss.detach()
                per_index_loss += loss
            if do_bwd:
                per_index_loss.backward()
        return {k: v.item() for k, v in per_tgt_losses.items()}

    def _fit_les(self, model, epoch: int):
        n_samples = len(self.train_dataloader.dataset)  # type: ignore

        params = list(model.les.parameters())
        if self.mode == "joint":
            params += list(model.rf.parameters())
        # RFF refits change this objective: do not reuse curvature history.
        optim = torch.optim.LBFGS(
            params,
            lr=self.lbfgs_lr * self.lbfgs_lr_decay**epoch,
            max_iter=self.lbfgs_max_iter,
            history_size=self.lbfgs_history_size,
            tolerance_grad=self.lbfgs_tolerance_grad,
            tolerance_change=self.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe",
        )
        indices = list(range(n_samples))  # full-batch

        def closure():
            optim.zero_grad()
            batch_loss = self._backward_batch(model, indices)
            return sum(batch_loss.values())

        optim.step(closure)

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

        if dist_utils.get_world_size() != 1:
            raise NotImplementedError(
                "LES optimization currently supports one process only"
            )
        self.training_history = []
        model = model.to(self.device)
        model.train()
        self.on_fit_start(model)

        covs, coeffs, normalization = None, None, None
        if self.mode != "joint":
            # Joint doesn't need covariance!
            t_cov = perf_counter()
            covs, coeffs, normalization = self.covariances(model, self.train_dataloader)
            t_cov = perf_counter() - t_cov

        for outer_it in range(self.num_cycles):
            # 1. Train RFF on full targets (original coefficients)
            #    or on residual coefficients depending on the iteration
            if self.mode == "alternating":
                assert covs is not None
                assert coeffs is not None
                if outer_it == 0:
                    # 1st iteration has no valid LES residual: train against full target
                    rf_weights = self._fit_rff(
                        model,
                        covs,  # type: ignore
                        normalization,
                        direct_coeffs=coeffs,  # type: ignore
                    )
                else:
                    # pass
                    # From 2nd iteration, train against y - y_les
                    rf_weights = self._fit_rff(
                        model, covs, normalization  # type: ignore
                    )
                model.rf.weights = torch.nn.Parameter(rf_weights)
                self._print_eval(model, outer_it, step="rff")
            # 2. Train LES on residuals from RFF training
            #    or in joint mode, train also RFF coefficients jointly.
            self._fit_les(model, epoch=outer_it)
            step_name = "joint" if self.mode == "joint" else "les"
            self._print_eval(model, epoch=outer_it, step=step_name)
        # Logging
        log_collection = LogCollection([self.create_log_entry(model)])
        return log_collection, model.rf.weights.detach()

    @no_jit()
    def evaluate(
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
        log_collection: LogCollection,
        all_weights: torch.Tensor | None,
    ) -> LogCollection:
        self.patch_e3nn()
        tot_dset_size = len(dataloader.dataset)  # type: ignore

        metric_objects: list[BaseMetric] = self.get_metrics()

        split_name = dataloader.dataset.split  # type: ignore
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
        model.gnn.franken_val()
        for i, (data, targets) in enumerate(progress_bar):
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            if all_weights is None or all_weights.shape[0] <= 100:
                forces_mode = "torch.autograd"
            else:
                forces_mode = "torch.func"  # FIXME: interaction between torch.func and franken_val is unclear!
            predictions = model.predict(
                targets=self.training_targets,  # type: ignore
                data=data,
                weights=all_weights,
                differential_mode=forces_mode,
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

    @no_jit()
    @torch.no_grad()
    def residual_coeffs(
        self,
        model: LESFrankenPotential,
        dataloader: torch.utils.data.DataLoader,
        normalization: dict[str, Tensor] | None,
    ):
        n_samples = len(dataloader.dataset)  # type: ignore
        n_rf = model.rf.total_random_features

        coeffs = {
            t: torch.zeros((n_rf,), device=self.device, dtype=self.buffer_dt)
            for t in self.training_targets
        }
        progress_bar = throughput(
            dataloader, "coeffs", total=n_samples, device=self.device
        )
        for i, (data, targets) in enumerate(progress_bar):
            assert isinstance(data, Configuration)
            data = data.to(device=self.device)
            assert data.natoms.numel() == 1, "Batched training is not supported"
            targets: Target = targets.to(device=self.device)

            les_preds = model.predict_les(
                data, self.training_targets, is_training=False  # type: ignore
            )
            target_fmaps = model.grad_feature_map(data, self.training_targets)
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
    def covariances(
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
    ):
        n_samples = len(dataloader.dataset)  # type: ignore
        n_rf = model.rf.total_random_features

        covs = {
            t: torch.zeros((n_rf, n_rf), device=self.device, dtype=self.buffer_dt)
            for t in self.training_targets
        }
        coeffs = {
            t: torch.zeros((n_rf,), device=self.device, dtype=self.buffer_dt)
            for t in self.training_targets
        }

        progress_bar = throughput(
            dataloader, "covs", total=n_samples, device=self.device
        )
        for i, (data, targets) in enumerate(progress_bar):
            assert isinstance(data, Configuration)
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            assert data.natoms.numel() == 1, "Batched training is not supported"

            target_fmaps = model.grad_feature_map(data, self.training_targets)
            for tgt_name in self.training_targets:
                try:
                    tgt = targets[tgt_name]
                except KeyError:
                    raise RuntimeError(
                        f"Target {i} does not contain any values for {tgt_name}."
                    )
                tgt_per_atom = (tgt / data.natoms).to(dtype=self.buffer_dt)
                fmap = target_fmaps[tgt_name].to(self.buffer_dt)
                if is_scalar_target(tgt_name):
                    covs[tgt_name].addmm_(fmap, fmap.T)
                    coeffs[tgt_name].add_(fmap.reshape(-1), alpha=tgt_per_atom.item())
                else:
                    covs[tgt_name].addmm_(fmap, fmap.T)
                    coeffs[tgt_name].addmv_(fmap, tgt_per_atom.reshape(-1))
        # Sync covariance matrices & coefficients
        for tgt_name in self.training_targets:
            dist_utils.all_sum(covs[tgt_name])
            dist_utils.all_sum(coeffs[tgt_name])
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

    @torch.no_grad()
    def solve(
        self,
        covs: dict[TargetType, Tensor],
        coeffs: dict[TargetType, Tensor],
        l2_penalty: float = 1e-6,
        **kwargs,  # only for compatibility with super
    ) -> Tensor:
        target_weights = self.rel_tgt_weight
        solve_cov, solve_coeff = None, None
        for tt in self.training_targets:
            if solve_cov is None or solve_coeff is None:
                solve_cov = covs[tt] * target_weights[tt]
                solve_coeff = coeffs[tt] * target_weights[tt]
            else:
                solve_cov.add_(covs[tt], alpha=target_weights[tt])
                solve_coeff.add_(coeffs[tt], alpha=target_weights[tt])
        assert solve_cov is not None and solve_coeff is not None
        return psd_ridge(solve_cov, solve_coeff, l2_penalty)
