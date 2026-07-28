from collections import defaultdict
import hashlib
import logging
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping, cast

import numpy as np
import torch
import torch.utils.data
from torch import Tensor
import tqdm

from franken.metrics.base import BaseMetric
from franken.rf.les_model import LESFrankenPotential
from franken.trainers.rf_trainer import RandomFeaturesTrainer
import franken.utils.distributed as dist_utils
from franken.data.base import Configuration, Target, TargetType, is_scalar_target
from franken.rf.model import FrankenPotential
from franken.trainers.log_utils import (
    DataSplit,
    HyperParameterGroup,
    LogCollection,
    LogEntry,
)
from franken.utils.linalg.cgsolve import conjugate_gradient
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
        self.solver = "cg"
        self.num_les_iterations = 3000
        self.num_outer_iterations = 10
        self.les_lr = 1e-4
        self.lr_scheduling_gamma = 0.5
        self.val_dataloader = None

    def create_log_entry(self, rf_hps, model):
        model_hash = hashlib.md5(str(model.hyperparameters).encode())
        model_hash = model_hash.hexdigest()
        solver_hps = rf_hps | {
            "dtype": self.buffer_dt,
            "les_lr": self.les_lr,
            "les_optim": "adam",
            "num_outer": self.num_outer_iterations,
            "num_inner": self.num_les_iterations,
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

    def _fit_rff(self, model, weights, covs, norm_coeffs, rf_hps, epoch):
        coeffs = self.residual_coeffs(
            model, self.train_dataloader, normalization=norm_coeffs
        )
        rf_weights = self.solve(
            covs=covs, coeffs=coeffs, x0=weights, cg_maxiter=1, cg_tol=1e-6, **rf_hps
        )
        # Evaluate on training data
        logc = LogCollection([self.create_log_entry(rf_hps, model)])
        self.evaluate(
            model,
            self.train_dataloader,
            log_collection=logc,
            all_weights=rf_weights.unsqueeze(0),
        )
        print(
            self.eval_summary(
                log=logc[0], epoch=epoch, split=DataSplit.TRAIN, title="after Franken"
            )
        )
        logc = LogCollection([self.create_log_entry(rf_hps, model)])
        self.evaluate(
            model,
            self.val_dataloader,
            log_collection=logc,
            all_weights=rf_weights.unsqueeze(0),
        )
        print(
            self.eval_summary(
                log=logc[0], epoch=epoch, split=DataSplit.VAL, title="after Franken"
            )
        )
        print()
        return rf_weights

    def _fit_les(self, model, weights, epoch, rf_hps):
        # Determine and normalize target weights
        target_weights = {}
        for k, v in rf_hps.items():
            if k.split("_")[1] == "weight":
                target_weights[k.split("_")[0]] = v
        weights_norm_factor = sum(target_weights.values())

        cur_lr = self.les_lr * (self.lr_scheduling_gamma**epoch)

        inner_data = iter(
            self.train_dataloader
        )  # TODO: Make sure this is randomized, otherwise every epoch may pick the same data. Currently NOT RANDOMIZED.
        # TODO: Fix for multi-process
        print(f"LES has {sum(p.numel() for p in model.les.parameters())} parameters")
        params = list(model.les.parameters()) + list(model.rf.parameters())
        optim = torch.optim.Adam(params, cur_lr, eps=1e-8)
        avg_losses = defaultdict(list)
        for inner_it in (pb := tqdm.tqdm(range(self.num_les_iterations), desc="LES")):
            try:
                data, targets = next(inner_data)
            except StopIteration:
                inner_data = iter(self.train_dataloader)
                data, targets = next(inner_data)

            # 1. compute predictions of the joint model
            preds = model.predict(
                targets=self.training_targets,  # type: ignore
                data=data,
                weights=None,
                is_training=True,
                add_energy_shift=False,
            )

            # 2. compute LES loss
            losses = {}
            for tt in self.training_targets:
                try:
                    tgt = targets[tt].to(dtype=self.buffer_dt)
                except KeyError:
                    raise RuntimeError(f"Target does not contain any values for {tt}.")
                # TODO: This is only correct for batch-size=1
                normalized_weight = target_weights[tt] / weights_norm_factor
                losses[tt] = normalized_weight * torch.mean(
                    torch.square(preds[tt] - tgt)
                )
                avg_losses[tt].append(losses[tt].item())
            loss = cast(torch.Tensor, sum(losses.values()))

            # 3. Optimize LES parameters
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Limited loss reporting
            loss_str = f"[{epoch}/{inner_it}] LES loss " + ", ".join(
                [f"{k}={np.mean(v):.2e}" for k, v in avg_losses.items()]
            )
            pb.set_description(loss_str)
        # Evaluate on training and validation data
        logc = LogCollection([self.create_log_entry(rf_hps, model)])
        if weights is not None:
            weights = weights.unsqueeze(0)
        self.evaluate(
            model,
            self.train_dataloader,
            log_collection=logc,
            all_weights=weights,
        )
        print(
            self.eval_summary(
                log=logc[0], epoch=epoch, split=DataSplit.TRAIN, title="after LES"
            )
        )
        logc = LogCollection([self.create_log_entry(rf_hps, model)])
        self.evaluate(
            model,
            self.val_dataloader,
            log_collection=logc,
            all_weights=weights,
        )
        print(
            self.eval_summary(
                log=logc[0], epoch=epoch, split=DataSplit.VAL, title="after LES"
            )
        )
        print()

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

        model = model.to(self.device)
        model.train()
        self.on_fit_start(model)

        _, rf_hps = next(params_grid(self.solver_hps))

        t_cov = perf_counter()
        covs, norm_coeffs = self.covariances(model, self.train_dataloader)
        t_cov = perf_counter() - t_cov

        rf_weights = None

        for outer_it in range(self.num_outer_iterations):
            # Compute LES predictions and update targets
            # recompute coefficients and solve RFF problem
            self._fit_les(model, rf_weights, outer_it, rf_hps)
            # rf_weights = self._fit_rff(model, rf_weights, covs, norm_coeffs, rf_hps, outer_it)

        # Logging
        log_collection = LogCollection([self.create_log_entry(rf_hps, model)])

        return log_collection, rf_weights.unsqueeze(0)

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

        progress_bar = throughput(
            dataloader, "covs", total=n_samples, device=self.device
        )
        for data, targets in progress_bar:
            assert isinstance(data, Configuration)
            data = data.to(device=self.device)
            assert data.natoms.numel() == 1, "Batched training is not supported"

            target_fmaps = model.grad_feature_map(data, self.training_targets)
            for tgt_name in self.training_targets:
                fmap = target_fmaps[tgt_name].to(self.buffer_dt)
                if is_scalar_target(tgt_name):
                    covs[tgt_name].addmm_(fmap, fmap.T)
                else:
                    covs[tgt_name].addmm_(fmap, fmap.T)
        # Sync covariance matrices & coefficients
        for tgt_name in self.training_targets:
            dist_utils.all_sum(covs[tgt_name])
        # RF normalization
        norm_coefs = None
        if self.random_features_normalization == "leading_eig":
            norm_coefs = {}
            for tgt_name in self.training_targets:
                norm, _ = torch.lobpcg(covs[tgt_name], k=1, largest=True)
                norm_coefs[tgt_name] = norm
                covs[tgt_name].div_(norm)
        elif self.random_features_normalization is not None:
            raise NotImplementedError(
                f"Covariance normalization {self.random_features_normalization} is not implemented."
            )
        return covs, norm_coefs

    @torch.no_grad()
    def solve(
        self,
        covs: dict[TargetType, Tensor],
        coeffs: dict[TargetType, Tensor],
        l2_penalty: float = 1e-6,
        x0: Tensor | None = None,
        cg_maxiter: int = 50,
        cg_tol: float = 1e-4,
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
        if self.solver == "cg":
            solve_cov.diagonal().add_(l2_penalty)
            return conjugate_gradient(
                A=solve_cov, b=solve_coeff, x0=x0, max_iter=cg_maxiter, tol=cg_tol
            )
        else:
            return psd_ridge(solve_cov, solve_coeff, l2_penalty)
