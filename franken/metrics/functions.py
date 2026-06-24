import typing

import numpy as np
import torch

from franken.data.base import (
    ENERGY_TARGET_KEY,
    FORCES_TARGET_KEY,
    STRESS_TARGET_KEY,
    Configuration,
    Target,
    TargetType,
)
from franken.metrics.base import BaseMetric
from franken.metrics.registry import metric_registry
from franken.utils import distributed

__all__ = [
    "EnergyPerAtomMAE",
    "EnergyPerAtomRMSE",
    "ForceMAE",
    "ForcePerSpeciesMAE",
    "ForceRMSE",
    "ForcePerSpeciesRMSE",
    "ForceCosineSimilarity",
    "StressMAE",
    "StressRMSE",
    "is_pareto_efficient",
]


def get_tgt_pred(targets: Target, predictions: Target, target_type: TargetType):
    pred_t = predictions[target_type]
    tgt_t = targets[target_type]
    if pred_t.ndim == tgt_t.ndim:
        pred_t = pred_t.unsqueeze(0)
    tgt_t = tgt_t.reshape(-1)
    pred_t = pred_t.reshape(pred_t.shape[0], -1)
    return tgt_t, pred_t


def check_single_system(data):
    if data.batch_ids is None:
        return 1
    return len(data.batch_ids.unique())
    # assert data.batch_ids is None or  == 1, "Multiple systems not supported."


class MAEPerAtomMetric(BaseMetric):
    def __init__(
        self,
        units: dict[str, str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        n_sys = check_single_system(data)
        num_atoms = torch.atleast_1d(data.natoms)
        tgt_t, pred_t = get_tgt_pred(targets, predictions, self.target_type)
        error = (1000 * torch.abs(tgt_t[None, :] - pred_t) / num_atoms[None, :]).mean(
            1
        ) * n_sys
        self.buffer_add(error, num_samples=n_sys)


class RMSEPerAtomMetric(BaseMetric):
    def __init__(
        self,
        units: dict[str, str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        n_sys = check_single_system(data)
        num_atoms = torch.atleast_1d(data.natoms)
        tgt_t, pred_t = get_tgt_pred(targets, predictions, self.target_type)
        error = (
            torch.square((tgt_t[None, :] - pred_t) / num_atoms[None, :]).mean(1) * n_sys
        )
        self.buffer_add(error, num_samples=n_sys)

    def compute(self) -> list[tuple[str, torch.Tensor]]:
        sq_error = super()._compute_val()
        error = torch.sqrt(sq_error) * 1000
        self.reset()
        return [(self.name, error)]


class MAEMetric(BaseMetric):
    def __init__(
        self,
        units: dict[str, str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        n_sys = check_single_system(data)
        tgt_t, pred_t = get_tgt_pred(targets, predictions, self.target_type)
        error = 1000 * torch.abs(tgt_t[None, :] - pred_t).mean(1) * n_sys
        self.buffer_add(error, num_samples=n_sys)


class RMSEMetric(BaseMetric):
    def __init__(self, units, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__(device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        n_sys = check_single_system(data)
        tgt_t, pred_t = get_tgt_pred(targets, predictions, self.target_type)
        error = torch.square(tgt_t[None, :] - pred_t).mean(1) * n_sys
        self.buffer_add(error, num_samples=n_sys)

    def compute(self) -> list[tuple[str, torch.Tensor]]:
        sq_error = super()._compute_val()
        error = torch.sqrt(sq_error) * 1000
        self.reset()
        return [(self.name, error)]


class CosineSimilarityMetric(BaseMetric):
    def __init__(
        self,
        units: dict[str, str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        pred_systems = predictions.iter_individual_systems(data)
        tgt_systems = targets.iter_individual_systems(data)
        for pred_sys, tgt_sys in zip(pred_systems, tgt_systems):
            tgt_t, pred_t = get_tgt_pred(tgt_sys, pred_sys, self.target_type)
            cos_similarity = torch.nn.functional.cosine_similarity(
                pred_t, tgt_t[None, ...], dim=-1
            )
            self.buffer_add(cos_similarity, num_samples=1)


class PerSpeciesMAEMetric(BaseMetric):
    Z_MAX = 90  # upper bound

    def __init__(
        self,
        units: dict[str, str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype, units)
        self.samples_counter = torch.zeros(
            self.Z_MAX + 1, device=device, dtype=torch.int64
        )

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        check_single_system(data)
        tgt_t, pred_t = get_tgt_pred(targets, predictions, self.target_type)
        atomic_numbers = data.atomic_numbers  # A
        assert atomic_numbers.max() <= self.Z_MAX
        n_models = pred_t.shape[0]
        error = torch.abs(tgt_t[None, ...] - pred_t)  # N, A*?
        error = error.reshape(error.shape[0], len(atomic_numbers), -1)  # N, A, ?
        error = error.mean(-1)  # N, A

        if self.buffer is None:
            self.buffer = torch.zeros(
                n_models, self.Z_MAX + 1, device=self.device, dtype=self.dtype
            )
        self.buffer.scatter_add_(
            dim=1, index=atomic_numbers.repeat(n_models, 1), src=error.to(self.dtype)
        )
        self.samples_counter.scatter_add_(
            dim=0,
            index=atomic_numbers,
            src=torch.ones_like(atomic_numbers, dtype=torch.int64),
        )

    def compute(self) -> list[tuple[str, torch.Tensor]]:
        if self.buffer is None:
            raise ValueError(
                f"Cannot compute value for metric '{self.name}' "
                "because it was never updated."
            )
        distributed.all_sum(self.buffer)
        distributed.all_sum(self.samples_counter)
        # MAE per model, per species
        mae = torch.zeros_like(self.buffer)
        mask = self.samples_counter > 0
        mae[:, mask] = self.buffer[:, mask] / self.samples_counter[mask]
        # unit conversion: eV/Å → meV/Å
        mae = mae * 1000
        self.reset()
        return [(f"{self.name}_{z}", mae[:, z]) for z in mask.nonzero().view(-1)]


class PerSpeciesRMSEMetric(BaseMetric):
    Z_MAX = 90  # upper bound

    def __init__(
        self,
        units: dict[str, str],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device, dtype, units)
        self.samples_counter = torch.zeros(
            self.Z_MAX + 1, device=device, dtype=torch.int64
        )

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        tgt_t, pred_t = get_tgt_pred(targets, predictions, self.target_type)
        atomic_numbers = data.atomic_numbers
        assert atomic_numbers.max() <= self.Z_MAX
        n_models = pred_t.shape[0]
        error = torch.square(tgt_t[None, ...] - pred_t)  # N, A*?
        error = error.reshape(error.shape[0], len(atomic_numbers), -1)  # N, A, ?
        error = error.mean(-1)  # N, A

        if self.buffer is None:
            self.buffer = torch.zeros(
                n_models, self.Z_MAX + 1, device=self.device, dtype=self.dtype
            )
        self.buffer.scatter_add_(
            dim=1, index=atomic_numbers.repeat(n_models, 1), src=error.to(self.dtype)
        )
        self.samples_counter.scatter_add_(
            dim=0,
            index=atomic_numbers,
            src=torch.ones_like(atomic_numbers, dtype=torch.int64),
        )

    def compute(self) -> list[tuple[str, torch.Tensor]]:
        if self.buffer is None:
            raise ValueError(
                f"Cannot compute value for metric '{self.name}' "
                "because it was never updated."
            )
        distributed.all_sum(self.buffer)
        distributed.all_sum(self.samples_counter)
        # MAE per model, per species
        mse = torch.zeros_like(self.buffer)
        mask = self.samples_counter > 0
        mse[:, mask] = self.buffer[:, mask] / self.samples_counter[mask]
        # unit conversion: eV/Å → meV/Å
        rmse = torch.sqrt(mse) * 1000
        self.reset()
        return [(f"{self.name}_{z}", rmse[:, z]) for z in mask.nonzero().view(-1)]


"""Energy"""


@metric_registry.register()
class EnergyPerAtomMAE(MAEPerAtomMetric):
    target_type: typing.ClassVar[TargetType] = ENERGY_TARGET_KEY
    name: typing.ClassVar[str] = "energy_MAE"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV", "outputs": "meV/atom"}, device, dtype)


@metric_registry.register()
class EnergyPerAtomRMSE(RMSEPerAtomMetric):
    target_type: typing.ClassVar[TargetType] = ENERGY_TARGET_KEY
    name: typing.ClassVar[str] = "energy_RMSE"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV", "outputs": "meV/atom"}, device, dtype)


"""Force"""


@metric_registry.register()
class ForceMAE(MAEMetric):
    target_type: typing.ClassVar[TargetType] = FORCES_TARGET_KEY
    name: typing.ClassVar[str] = "forces_MAE"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV/ang", "outputs": "meV/ang"}, device, dtype)


@metric_registry.register()
class ForcePerSpeciesMAE(PerSpeciesMAEMetric):
    target_type: typing.ClassVar[TargetType] = FORCES_TARGET_KEY
    name: typing.ClassVar[str] = "forces_MAE_species"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV/ang", "outputs": "meV/ang"}, device, dtype)


@metric_registry.register()
class ForceRMSE(RMSEMetric):
    target_type: typing.ClassVar[TargetType] = FORCES_TARGET_KEY
    name: typing.ClassVar[str] = "forces_RMSE"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV/ang", "outputs": "meV/ang"}, device, dtype)


@metric_registry.register()
class ForcePerSpeciesRMSE(PerSpeciesRMSEMetric):
    target_type: typing.ClassVar[TargetType] = FORCES_TARGET_KEY
    name: typing.ClassVar[str] = "forces_RMSE_species"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV/ang", "outputs": "meV/ang"}, device, dtype)


@metric_registry.register()
class ForceCosineSimilarity(CosineSimilarityMetric):
    target_type: typing.ClassVar[TargetType] = FORCES_TARGET_KEY
    name: typing.ClassVar[str] = "forces_cosim"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV/ang", "outputs": ""}, device, dtype)


"""Stress"""


@metric_registry.register()
class StressMAE(MAEMetric):
    target_type: typing.ClassVar[TargetType] = STRESS_TARGET_KEY
    name: typing.ClassVar[str] = "stress_MAE"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV/ang", "outputs": "meV/ang"}, device, dtype)


@metric_registry.register()
class StressRMSE(RMSEMetric):
    target_type: typing.ClassVar[TargetType] = STRESS_TARGET_KEY
    name: typing.ClassVar[str] = "stress_RMSE"

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__({"inputs": "eV/ang", "outputs": "meV/ang"}, device, dtype)


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient
