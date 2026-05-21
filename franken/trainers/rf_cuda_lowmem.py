import hashlib
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Literal, Mapping

import numpy as np
import torch
import torch.utils.data
from torch import Tensor

from franken.metrics.base import BaseMetric
import franken.utils.distributed as dist_utils
from franken.data.base import Configuration, Target, TargetType, is_scalar_target
from franken.rf.model import FrankenPotential
from franken.trainers import BaseTrainer
from franken.trainers.log_utils import (
    DataSplit,
    HyperParameterGroup,
    LogCollection,
    LogEntry,
)
from franken.utils.linalg.cov import normalize_leading_eig
from franken.utils.linalg.psdsolve import psd_ridge
from franken.utils.misc import ensure_list, no_jit, params_grid, throughput
from franken.metrics import metric_registry

logger = logging.getLogger("franken")


class RandomFeaturesTrainer(BaseTrainer):
    """Main class which groups training and evaluation functionality for franken models.

    Args:
        train_dataloader (torch.utils.data.DataLoader):
            Dataloader which iterates over the training set.
        random_features_normalization (Literal["leading_eig"] | None):
            How to normalize the covariance matrices formed by random-features. Defaults to "leading_eig".
        log_dir (Path | None):
            Directory where to save logs and models. If not specified, no logs will be saved.
            Defaults to None.
        save_every_model (bool):
            Model fitting with this class is done simultaneously for a list
            of solver parameters. This argument controls the behavior of model saving:
            if set to True, the models corresponding to all solver parameters will be saved,
            otherwise only the 'best' model among them (according to some validation set) will
            be saved. Defaults to True.
        device:
            PyTorch device on which computations are performed. Defaults to "cuda:0".
            Note that this class is multi-GPU aware. Users can create a RandomFeaturesTrainer
            in a distributed setting and it will handle synchronization across its replicas.
        dtype (str | torch.dtype):
            Data-type for solver operations. Random features will be computed in float32, and
            then converted to float64 if requested. Defaults to torch.float32.
        save_fmaps (bool):
            Whether or not to save feature-maps for the training set. Saving them
            requires extra memory (linear in the training-set size), but speeds up
            the ``evaluate()`` path on training data. Defaults to True.
    """

    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        l2_penalty: float | list[float],
        training_targets: list[TargetType],
        target_weight: Mapping[TargetType, float | list[float]],
        random_features_normalization: Literal["leading_eig"] | None = "leading_eig",
        log_dir: Path | None = None,
        save_every_model: bool = True,
        device: torch.device | str | int = "cuda:0",
        dtype: str | torch.dtype = torch.float32,
        save_fmaps: bool = True,
    ):
        super().__init__(
            train_dataloader,
            log_dir=log_dir,
            save_every_model=save_every_model,
            device=device,
            dtype=dtype,
        )
        self.random_features_normalization = random_features_normalization
        self.save_fmaps = save_fmaps
        if len(training_targets) == 0:
            raise ValueError(
                "Cannot initialize trainer with no targets. "
                "Please pass a non-empty list as the `training_targets` parameter."
            )
        self.training_targets = training_targets
        self.l2_penalty = ensure_list(l2_penalty)
        self.target_weight = process_tgt_weights(target_weight, self.training_targets)
        self.solver_hps = dict(l2_penalty=self.l2_penalty) | {
            f"{k}_weight": v for k, v in self.target_weight.items()
        }

    def on_fit_start(self, model: FrankenPotential):
        # initialize input scaler based on statistics property
        model.input_scaler.set_from_statistics(self.get_statistics(model)[0])
        # initialize energy shift based on atomic energies
        if not model.energy_shift.is_initialized:
            model.energy_shift.set_from_atomic_energies(
                self.train_dataloader.dataset.atomic_energies  # type: ignore
            )
        model.gnn.franken_train()

    def patch_e3nn(self):
        if self.device.type == "cuda":
            # Patch E3NN for batched jacobians!
            from franken.backbones.wrappers.common_patches import patch_e3nn

            patch_e3nn()

    @no_jit()
    def fit(self, model: FrankenPotential) -> tuple[LogCollection, torch.Tensor]:
        """Fit a given franken model on the training set.

        Args:
            model (FrankenPotential): The model which defines GNN and random features.
            solver_params (dict): Parameters for the solver which actually
                performs the fit. This argument allows to specify multiple parameters,
                for each of which we will perform a fit. For example, passing
                ``{"l2_penalty": [1e-6, 1e-4], "force_weight": [0.5]}``
                will result in two different models, one with :code:`l2_penalty=1e-6, force_weight=0.5`
                and one with :code:`l2_penalty=1e-4, force_weight=0.5`. This way of specifying solver
                parameters allows to easily perform a grid-search.

        Returns:
            tuple[LogCollection, torch.Tensor]:
                The fitting logs, together with the learned weights.

        Note:
            More information about the available solver parameters can be found under the
            ``solve()`` method.
        """
        self.patch_e3nn()

        model = model.to(self.device)
        self.on_fit_start(model)
        model_hash = hashlib.md5(str(model.hyperparameters).encode())
        model_hash = model_hash.hexdigest()

        t_cov_coeffs_start = perf_counter()
        covs, coeffs = self._covs_and_coeffs(model, self.train_dataloader)
        t_cov_coeffs = perf_counter() - t_cov_coeffs_start

        solver_grid_size = math.prod([len(v) for v in self.solver_hps.values()])
        all_weights = torch.zeros(
            solver_grid_size,
            model.rf.total_random_features,
            dtype=self.buffer_dt,
            device=self.device,
        )
        solver_iter = throughput(
            params_grid(self.solver_hps, split_distributed=True),
            desc="least-squares",
            units="models",
            device=self.device,
        )

        local_logs = dict()
        num_failed = torch.zeros((1,), device=self.device, dtype=torch.int)
        for hp_idx, hp_val in solver_iter:
            t_solve_start = perf_counter()
            try:
                weights = self.solve(covs=covs, coeffs=coeffs, **hp_val)
            except torch.linalg.LinAlgError as e:  # type: ignore
                weights = torch.zeros_like(all_weights[hp_idx])
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

    def get_metrics(self) -> list[BaseMetric]:
        dev = self.device
        dt = self.buffer_dt
        return [
            metric_registry.init_metric(metric, device=dev, dtype=dt)
            for tt in self.training_targets
            for metric in metric_registry.available_metrics_for_target(tt)
        ]

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
            if split == DataSplit.TRAIN and self.save_fmaps:
                # Shortcut to compute predictions for the training-set, for which
                # we already have computed feature maps. No energy shift since this
                # is always the training set.
                predictions = model.predict_from_fmaps(
                    data,
                    fmaps={k: v[i] for k, v in self.fmaps.items()},
                    weights=all_weights,
                    add_energy_shift=False,
                )
            else:
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
                assert metric_value.shape == (num_models,)
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

    def warn_save_fmaps(self, fmaps: list[Tensor], num_maps: int) -> None:
        fmap_size = sum(np.prod(fmap.shape) for fmap in fmaps)
        tot_bytes = fmap_size * fmaps[0].element_size() * num_maps
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

    @no_jit()
    @torch.no_grad()
    def _covs_and_coeffs(
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
    ):
        tot_dset_size = len(dataloader.dataset)  # type: ignore
        n_rf = model.rf.total_random_features

        covs = {
            t: torch.zeros((n_rf, n_rf), device=self.device, dtype=self.buffer_dt)
            for t in self.training_targets
        }
        coeffs = {
            t: torch.zeros((n_rf,), device=self.device, dtype=self.buffer_dt)
            for t in self.training_targets
        }
        self.fmaps: dict[TargetType, list[Tensor]] = {
            t: [] for t in self.training_targets
        }

        progress_bar = throughput(
            dataloader,
            desc="covs+coeffs",
            total=tot_dset_size,
            device=self.device,
        )
        for i, (data, targets) in enumerate(progress_bar):
            assert isinstance(data, Configuration)
            data = data.to(device=self.device)
            assert data.natoms.numel() == 1, "Batched training is not supported"
            targets: Target = targets.to(device=self.device)

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
                if is_scalar_target(tgt_name):
                    covs[tgt_name].addmm_(fmap, fmap.T)
                    coeffs[tgt_name].add_(fmap.view(-1), alpha=tgt_per_atom.item())
                else:
                    covs[tgt_name].addmm_(fmap, fmap.T)
                    coeffs[tgt_name].addmv_(fmap, tgt_per_atom.view(-1))
                if self.save_fmaps:
                    self.fmaps[tgt_name].append(fmap)
        # Sync covariance matrices & coefficients
        for tgt_name in self.training_targets:
            dist_utils.all_sum(covs[tgt_name])
            dist_utils.all_sum(coeffs[tgt_name])
        # RF normalization
        if self.random_features_normalization == "leading_eig":
            for tgt_name in self.training_targets:
                normalize_leading_eig(covs[tgt_name], coeffs[tgt_name])
        elif self.random_features_normalization is not None:
            raise NotImplementedError(
                f"Covariance normalization {self.random_features_normalization} is not implemented."
            )
        return covs, coeffs

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
            print(f"{tt=} {normalized_weight=}")
            if solve_cov is None or solve_coeff is None:
                solve_cov = covs[tt] * normalized_weight
                solve_coeff = coeffs[tt] * normalized_weight
            else:
                solve_cov.add_(covs[tt], alpha=normalized_weight)
                solve_coeff.add_(coeffs[tt], alpha=normalized_weight)
        assert solve_cov is not None and solve_coeff is not None
        return psd_ridge(solve_cov, solve_coeff, l2_penalty)


def process_tgt_weights(
    tgt_weights: Mapping[TargetType, float | list[float]], tgts: list[TargetType]
) -> dict[TargetType, list[float]]:
    """
    Make sure all weights in tgts are present (default value for
    non-provided weights is 1.0).
    """
    out_tgt_weights = {}
    for tgt in tgts:
        out_tgt_weights[tgt] = ensure_list(tgt_weights.get(tgt, [1.0]))
    return out_tgt_weights
