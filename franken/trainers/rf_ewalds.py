from dataclasses import asdict
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
import tqdm

from franken.config import LESConfig
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
from franken.utils.misc import no_jit, params_grid, throughput

logger = logging.getLogger("franken")


# NOTES:
#  - does not support hyperparameter tuning internally (only one penalty, weight)
#    -> no multi-weight
class RandomFeaturesEwaldsTrainer(RandomFeaturesTrainer):
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        training_targets: list[TargetType],
        l2_penalty: float | list[float],
        target_weight: Mapping[TargetType, float | list[float]],
        random_features_normalization: Literal["leading_eig"] | None = "leading_eig",
        log_dir: Path | None = None,
        save_every_model: bool = True,
        device: torch.device | str | int = "cuda:0",
        dtype: str | torch.dtype = torch.float32,
        save_fmaps: bool = True,
        metrics: list[str] | None = None,
        les_config: LESConfig | None = None,
        best_model_selection: list[str] | None = None,
        seed: int = 1337,
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
        self.les_config = les_config or LESConfig()
        self.mode: Literal["alternating", "variable_projection", "joint"] = (
            self.les_config.mode
        )
        self.best_model_selection = best_model_selection or [
            f"{target}_MAE" for target in self.training_targets
        ]
        self.seed = seed
        self.val_dataloader = None
        self.training_history: list[dict] = []
        self.best_stage: dict | None = None
        self._best_state = None
        self._les_optimizer = None
        self._loss_normalization: dict[str, Tensor] | None = None
        self._projection_factor: Tensor | None = None
        self.fmaps: dict[TargetType, list[Tensor]] = {}
        self._shuffle_rng = torch.Generator().manual_seed(seed)

    def create_log_entry(self, rf_hps, model):
        model_hash = hashlib.md5(str(model.hyperparameters).encode())
        model_hash = model_hash.hexdigest()
        solver_hps = rf_hps | {
            "dtype": self.buffer_dt,
            **{f"les_{key}": value for key, value in asdict(self.les_config).items()},
        }
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
            ["energy_MAE", "energy_RMSE"],
        )
        forces_error, forces_metric = _get_first_available_metric(
            ["forces_MAE", "forces_RMSE"],
        )
        stress_error, stress_metric = _get_first_available_metric(
            ["stress_MAE", "stress_RMSE"],
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

    def _print_eval(
        self, rf_hps, model, weights, epoch, step: Literal["rff", "les", "joint"]
    ):
        logc = LogCollection([self.create_log_entry(rf_hps, model)])
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
                all_weights=weights,
            )
            if dist_utils.get_rank() == 0:
                print(
                    self.eval_summary(
                        log=logc[0],
                        epoch=epoch,
                        split=split,
                        title=f"after {step.upper()} training",
                    )
                )

        if dist_utils.get_rank() == 0:
            if self.les_config.restore_best:
                self._remember_best(model, weights, logc[0], epoch + 1, step)
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

    def _remember_best(self, model, weights, log, cycle, step):
        split = DataSplit.VAL if self.val_dataloader is not None else DataSplit.TRAIN
        values = [log.get_metric(name, split) for name in self.best_model_selection]
        if not all(math.isfinite(value) for value in values):
            logger.warning(
                "Skipping non-finite model-selection metrics at cycle %s", cycle
            )
            return
        # For nonnegative error metrics this is the same minimum-L1 selection
        # used by LogCollection.get_best_model across trials.
        score = sum(abs(value) for value in values)
        if self.best_stage is not None and score >= self.best_stage["score"]:
            return
        self.best_stage = {
            "cycle": cycle,
            "step": step,
            "split": split.name.lower(),
            "score": score,
            "metrics": dict(zip(self.best_model_selection, values)),
        }
        self._best_state = {
            "rf_weights": (model.rf.weights if weights is None else weights)
            .detach()
            .cpu()
            .clone(),
            "les": {
                key: value.detach().cpu().clone()
                for key, value in model.les.state_dict().items()
            },
        }
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "best_stage.json").write_text(
                json.dumps(self.best_stage, indent=2)
            )

    def _fit_rff(
        self,
        model: LESFrankenPotential,
        covs: dict[TargetType, Tensor],
        normalization: dict[str, Tensor] | None,
        rf_hps: dict,
        epoch: int,
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
            coeffs=coeffs,
            **rf_hps,
        )
        # weights from [n_rf] to [1, n_rf]
        rf_weights = rf_weights.unsqueeze(0)
        return rf_weights

    def _backward_batch(self, model, indices, target_weights):
        """Accumulate a mean gradient, freeing each structure's graph immediately."""
        total = torch.zeros((), device=self.device, dtype=self.buffer_dt)
        for index in indices:
            data, targets = self.train_dataloader.dataset[index]
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            if data.natoms.numel() != 1:
                raise ValueError("LES accumulation expects individual configurations")
            # Zero-weight targets are still evaluated for reporting, but need not
            # create costly force/stress derivative graphs during optimization.
            if self.mode == "variable_projection" or (
                self.fmaps and self.mode != "joint"
            ):
                # RFF is fixed in this derivative. Use the same linear maps as
                # its solve, in solver precision. Differentiating the combined
                # energy instead rounds the RFF force back to descriptor dtype,
                # which can spoil a projected line search through cancellation.
                fmaps = (
                    {target: self.fmaps[target][index] for target in target_weights}
                    if self.fmaps
                    else model.grad_feature_map(data, list(target_weights))
                )
                les_predictions = model.predict_les(
                    data, list(target_weights), is_training=True
                )
                predictions = {}
                for target in target_weights:
                    short_range = (
                        model.rf.weights.detach().to(self.buffer_dt)
                        @ fmaps[target].to(self.buffer_dt)
                    ) * data.natoms
                    predictions[target] = short_range.reshape_as(
                        les_predictions[target]
                    ) + les_predictions[target].to(self.buffer_dt)
            else:
                predictions = model.predict(
                    targets=list(target_weights),
                    data=data,
                    is_training=True,
                    add_energy_shift=False,
                )
            # Match the RFF normal equations: all feature maps and targets
            # represent the observable divided by N, and vector components
            # are summed (not averaged). Covariance normalization must also
            # be shared by the two alternating subproblems.
            loss = total.new_zeros(())
            for target, weight in target_weights.items():
                residual = (
                    predictions[target].to(self.buffer_dt)
                    - targets[target].to(self.buffer_dt)
                ) / data.natoms.to(self.buffer_dt)
                squared_error = residual.square().sum()
                if self._loss_normalization is not None:
                    squared_error = (
                        squared_error / self._loss_normalization[target].squeeze()
                    )
                loss = loss + weight * squared_error
            loss = loss / len(indices)
            loss.backward()
            total += loss.detach()
        return total

    def _fit_les(self, model, epoch, rf_hps):
        cfg = self.les_config
        n_samples = len(self.train_dataloader.dataset)
        if n_samples == 0:
            raise ValueError("LES requires a nonempty training dataset")
        weights = {
            target: rf_hps[f"{target}_weight"] for target in self.training_targets
        }
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Target weights must be finite and nonnegative")
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("At least one training target must have positive weight")
        target_weights = {t: w / total_weight for t, w in weights.items() if w > 0}

        params = list(model.les.parameters())
        if self.mode == "joint":
            params += list(model.rf.parameters())
        trainable_ids = {id(p) for p in params}
        previous_flags = [(p, p.requires_grad) for p in model.parameters()]
        # Position gradients remain enabled for forces; only parameter gradients
        # outside the selected head(s) are disabled.
        for p, _ in previous_flags:
            p.requires_grad_(id(p) in trainable_ids)
        try:
            if cfg.optimizer == "adam":
                lr = cfg.learning_rate * cfg.lr_decay**epoch
                if self._les_optimizer is None:
                    self._les_optimizer = torch.optim.Adam(params, lr=lr, eps=1e-8)
                optim = self._les_optimizer
                for group in optim.param_groups:
                    group["lr"] = lr
                batch_size = cfg.batch_size or n_samples
                for _ in tqdm.trange(cfg.epochs_per_cycle, desc="LES epochs"):
                    indices = (
                        torch.randperm(n_samples, generator=self._shuffle_rng).tolist()
                        if batch_size < n_samples
                        else list(range(n_samples))
                    )
                    for start in range(0, n_samples, batch_size):
                        optim.zero_grad(set_to_none=True)
                        self._backward_batch(
                            model, indices[start : start + batch_size], target_weights
                        )
                        optim.step()
            else:
                # RFF refits change this objective: do not reuse curvature history.
                optim = torch.optim.LBFGS(
                    params,
                    lr=cfg.lbfgs_learning_rate * cfg.lr_decay**epoch,
                    max_iter=cfg.lbfgs_max_iter,
                    history_size=cfg.lbfgs_history_size,
                    tolerance_grad=cfg.lbfgs_tolerance_grad,
                    tolerance_change=cfg.lbfgs_tolerance_change,
                    line_search_fn="strong_wolfe",
                )
                indices = list(range(n_samples))

                def closure():
                    if self.mode == "variable_projection":
                        self._refit_projected_rff(model, rf_hps)
                    optim.zero_grad(set_to_none=True)
                    loss = self._backward_batch(model, indices, target_weights)
                    if self.mode == "variable_projection":
                        # Envelope theorem: no derivative through the exact RFF
                        # minimizer is needed. Its ridge cost is still needed by
                        # the line search, since it changes between trial points.
                        loss = loss + (
                            rf_hps["l2_penalty"]
                            * model.rf.weights.detach().square().sum()
                            / n_samples
                        )
                    return loss

                optim.step(closure)
                if self.mode == "variable_projection":
                    # The last closure may have evaluated a rejected trial.
                    self._refit_projected_rff(model, rf_hps)
        finally:
            for p, requires_grad in previous_flags:
                p.requires_grad_(requires_grad)

        self._print_eval(
            rf_hps,
            model,
            weights=None,
            epoch=epoch,
            step="joint" if self.mode == "joint" else "les",
        )

    @torch.no_grad()
    def _prepare_projection(self, covs, rf_hps):
        """Factor the fixed regularized RFF matrix once for all LES trial points."""
        total_weight = sum(rf_hps[f"{t}_weight"] for t in self.training_targets)
        matrix = sum(
            (rf_hps[f"{t}_weight"] / total_weight) * covs[t]
            for t in self.training_targets
        )
        matrix.diagonal().add_(rf_hps["l2_penalty"])
        self._projection_factor = torch.linalg.cholesky(matrix)

    @torch.no_grad()
    def _refit_projected_rff(self, model, rf_hps):
        if self._projection_factor is None:
            raise RuntimeError("Variable projection requires prepared RFF covariances")
        coeffs = self.residual_coeffs(
            model, self.train_dataloader, self._loss_normalization
        )
        total_weight = sum(rf_hps[f"{t}_weight"] for t in self.training_targets)
        rhs = sum(
            (rf_hps[f"{t}_weight"] / total_weight) * coeffs[t]
            for t in self.training_targets
        )
        solution = torch.cholesky_solve(rhs[:, None], self._projection_factor).T
        model.rf.weights.copy_(solution)

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
        self.best_stage = None
        self._best_state = None
        self._les_optimizer = None
        self._loss_normalization = None
        self._projection_factor = None
        self._shuffle_rng = torch.Generator().manual_seed(self.seed)
        model = model.to(self.device)
        model.train()
        self.on_fit_start(model)

        _, rf_hps = next(params_grid(self.solver_hps))

        covs, coeffs, normalization = None, None, None
        if self.mode != "joint":
            # Joint doesn't need covariance!
            t_cov = perf_counter()
            covs, coeffs, normalization = self.covariances(model, self.train_dataloader)
            self._loss_normalization = normalization
            if self.mode == "variable_projection":
                self._prepare_projection(covs, rf_hps)
            t_cov = perf_counter() - t_cov

        for outer_it in range(self.les_config.num_cycles):
            # 1. Fit the residual of the current LES head, including its
            # nonzero random initialization in the first cycle.
            if self.mode in {"alternating", "variable_projection"}:
                assert covs is not None
                assert coeffs is not None
                rf_weights = self._fit_rff(
                    model,
                    covs,
                    normalization,
                    rf_hps,
                    epoch=outer_it,  # , direct_coeffs=coeffs if outer_it == 0 else None ## use direct_coeffs only for the first cycle, otherwise use residuals from previous LES head
                )
                if torch.any(torch.isnan(rf_weights)):
                    logger.warning(
                        f"NaNs encountered in training after RFF step. Stopping at cycle {outer_it}."
                    )
                    break

                model.rf.weights = torch.nn.Parameter(rf_weights)
                self._print_eval(rf_hps, model, rf_weights, outer_it, step="rff")
            # 2. Train LES on residuals from RFF training
            #    or in joint mode, train also RFF coefficients jointly.
            self._fit_les(model, epoch=outer_it, rf_hps=rf_hps)

        if self.les_config.restore_best:
            if self._best_state is None:
                raise RuntimeError("No stage has finite model-selection metrics")
            with torch.no_grad():
                model.rf.weights.copy_(self._best_state["rf_weights"])
            model.les.load_state_dict(self._best_state["les"])
        # Optimizer moments correspond to the last stage, not the restored model.
        self._les_optimizer = None
        self._projection_factor = None
        self._best_state = None

        # Logging
        log_collection = LogCollection([self.create_log_entry(rf_hps, model)])

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
        model.gnn.franken_val()
        for i, (data, targets) in enumerate(progress_bar):
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            if all_weights is None or all_weights.shape[0] <= 100:
                forces_mode = "torch.autograd"
            else:
                forces_mode = "torch.func"  # FIXME: interaction between torch.func and franken_val is unclear!
            predictions = model.predict(
                targets=self.training_targets,
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
                metric.update(Target.from_types(predictions), targets, data)

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
                data, self.training_targets, is_training=False
            )
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
    def covariances(
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
    ):
        n_samples = len(dataloader.dataset)  # type: ignore
        n_rf = model.rf.total_random_features
        self.fmaps = {t: [] for t in self.training_targets} if self.save_fmaps else {}

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
            if self.save_fmaps and i == 0:
                self.warn_save_fmaps(list(target_fmaps.values()), len(dataloader))
            for tgt_name in self.training_targets:
                try:
                    tgt = targets[tgt_name]
                except KeyError:
                    raise RuntimeError(
                        f"Target {i} does not contain any values for {tgt_name}."
                    )
                tgt_per_atom = (tgt / data.natoms).to(dtype=self.buffer_dt)
                fmap = target_fmaps[tgt_name].to(self.buffer_dt)
                if self.save_fmaps:
                    self.fmaps[tgt_name].append(fmap)
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
        **weights,
    ) -> Tensor:
        target_weight = {}
        for k, v in weights.items():
            target_weight[k.split("_")[0]] = v
        weights_norm_factor = sum(target_weight.values())
        solve_cov, solve_coeff = None, None
        for tt in self.training_targets:
            normalized_weight = target_weight[tt] / weights_norm_factor
            if solve_cov is None or solve_coeff is None:
                solve_cov = covs[tt] * normalized_weight
                solve_coeff = coeffs[tt] * normalized_weight
            else:
                solve_cov.add_(covs[tt], alpha=normalized_weight)
                solve_coeff.add_(coeffs[tt], alpha=normalized_weight)
        assert solve_cov is not None and solve_coeff is not None
        return psd_ridge(solve_cov, solve_coeff, l2_penalty)
