import numpy as np
import torch

from franken.data.base import Target
from franken.metrics.base import BaseMetric
from franken.metrics.registry import registry
from franken.utils import distributed


__all__ = [
    "EnergyMAE",
    "EnergyRMSE",
    "ForcesMAE",
    "ForcesRMSE",
    "ForcesCosineSimilarity",
    "is_pareto_efficient",
]


class EnergyMAE(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV",
            "outputs": "meV/atom",
        }
        super().__init__("energy_MAE", device, dtype, units)

    def update(self, predictions: Target, targets: Target) -> None:
        if targets.forces is None:
            raise NotImplementedError(
                "At the moment, target's forces are required to get the number of atoms in the configuration."
            )
        num_atoms = targets.forces.shape[-2]
        num_samples = 1
        if targets.energy.ndim > 0:
            num_samples = targets.energy.shape[0]

        error = 1000 * torch.abs(targets.energy - predictions.energy) / num_atoms

        self.buffer_add(error, num_samples=num_samples)


class EnergyRMSE(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV",
            "outputs": "meV/atom",
        }
        super().__init__("energy_RMSE", device, dtype, units)

    def update(self, predictions: Target, targets: Target) -> None:
        if targets.forces is None:
            raise NotImplementedError(
                "At the moment, target's forces are required to get the number of atoms in the configuration."
            )
        num_atoms = targets.forces.shape[-2]
        num_samples = 1
        if targets.energy.ndim > 0:
            num_samples = targets.energy.shape[0]

        error = torch.square((targets.energy - predictions.energy) / num_atoms)

        self.buffer_add(error, num_samples=num_samples)

    def compute(self, reset: bool = True) -> torch.Tensor:
        if self.buffer is None:
            raise ValueError(
                f"Cannot compute value for metric '{self.name}' "
                "because it was never updated."
            )
        distributed.all_sum(self.buffer)
        distributed.all_sum(self.samples_counter)
        error = self.buffer / self.samples_counter
        # square-root and fix units
        error = torch.sqrt(error) * 1000
        if reset:
            self.reset()
        return error


class ForcesMAE(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": "meV/ang",
        }
        super().__init__("forces_MAE", device, dtype, units)

    def update(self, predictions: Target, targets: Target) -> None:
        if targets.forces is None or predictions.forces is None:
            raise AttributeError("Forces must be specified to compute the MAE.")
        num_samples = 1
        if targets.forces.ndim > 2:
            num_samples = targets.forces.shape[0]
        elif targets.forces.ndim < 2:
            raise ValueError("Forces must be a 2D tensor or a batch of 2D tensors.")

        error = 1000 * torch.abs(targets.forces - predictions.forces)
        error = error.mean(dim=(-1, -2))  # Average over atoms and components

        self.buffer_add(error, num_samples=num_samples)


class ForcesRMSE(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": "meV/ang",
        }
        super().__init__("forces_RMSE", device, dtype, units)

    def update(self, predictions: Target, targets: Target) -> None:
        if targets.forces is None or predictions.forces is None:
            raise AttributeError("Forces must be specified to compute the MAE.")
        num_samples = 1
        if targets.forces.ndim > 2:
            num_samples = targets.forces.shape[0]
        elif targets.forces.ndim < 2:
            raise ValueError("Forces must be a 2D tensor or a batch of 2D tensors.")

        error = torch.square(targets.forces - predictions.forces)
        error = error.mean(dim=(-1, -2))  # Average over atoms and components

        self.buffer_add(error, num_samples=num_samples)

    def compute(self, reset: bool = True) -> torch.Tensor:
        if self.buffer is None:
            raise ValueError(
                f"Cannot compute value for metric '{self.name}' "
                "because it was never updated."
            )
        distributed.all_sum(self.buffer)
        distributed.all_sum(self.samples_counter)
        error = self.buffer / self.samples_counter
        # square-root and fix units
        error = torch.sqrt(error) * 1000
        if reset:
            self.reset()
        return error


class ForcesRMSE2(BaseMetric):
    """Average of RMSE along individual structures"""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": "meV/ang",
        }
        super().__init__("forces_RMSE", device, dtype, units)

    def update(self, predictions: Target, targets: Target) -> None:
        if targets.forces is None or predictions.forces is None:
            raise AttributeError("Forces must be specified to compute the MAE.")
        num_samples = 1
        if targets.forces.ndim > 2:
            num_samples = targets.forces.shape[0]
        elif targets.forces.ndim < 2:
            raise ValueError("Forces must be a 2D tensor or a batch of 2D tensors.")

        error = torch.square(targets.forces - predictions.forces)
        error = error.mean(dim=(-1, -2))  # Average over atoms and components
        error = torch.sqrt(error) * 1000
        self.buffer_add(error, num_samples=num_samples)


class ForcesCosineSimilarity(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": None,
        }
        super().__init__("forces_cosim", device, dtype, units)

    def update(
        self,
        predictions: Target,
        targets: Target,
    ) -> None:
        num_samples = 1
        assert targets.forces is not None
        assert predictions.forces is not None
        if targets.forces.ndim > 2:
            num_samples = targets.forces.shape[0]
        elif targets.forces.ndim < 2:
            raise ValueError("Forces must be a 2D tensor or a batch of 2D tensors.")

        cos_similarity = torch.nn.functional.cosine_similarity(
            predictions.forces, targets.forces, dim=-1
        )
        cos_similarity = cos_similarity.mean(dim=-1)
        self.buffer_add(cos_similarity, num_samples=num_samples)


def is_pareto_efficient(costs):
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


registry.register("energy_MAE", EnergyMAE)
registry.register("energy_RMSE", EnergyRMSE)
registry.register("forces_MAE", ForcesMAE)
registry.register("forces_RMSE", ForcesRMSE)
registry.register("forces_RMSE2", ForcesRMSE2)
registry.register("forces_cosim", ForcesCosineSimilarity)
