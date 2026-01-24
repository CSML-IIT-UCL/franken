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


class ForcesMAEWeighted(BaseMetric):
    """Average of MAE per species."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": "meV/ang",
        }
        super().__init__(
            "forces_MAE_weighted", device, dtype, units, requires_species=True
        )

    def update(
        self,
        predictions: Target,
        targets: Target,
        atomic_numbers: torch.Tensor,
    ) -> None:

        if targets.forces is None or predictions.forces is None:
            raise AttributeError("Forces must be specified to compute the MAE.")

        assert atomic_numbers.ndim == 1
        assert atomic_numbers.shape[0] == targets.forces.shape[-2]

        # Same num_samples logic as ForcesMAE
        num_samples = 1
        if targets.forces.ndim > 2:
            num_samples = targets.forces.shape[0]
        elif targets.forces.ndim < 2:
            raise ValueError("Forces must be a 2D tensor or a batch of 2D tensors.")

        # abs error in meV/Å
        # shape: (M, N, 3) or (N, 3)
        error = 1000 * torch.abs(targets.forces - predictions.forces)

        # mean over force components
        # -> (M, N) or (N,)
        error = error.mean(dim=-1)

        species = torch.unique(atomic_numbers)

        # per-species mean over atoms
        species_errors = []
        for z in species:
            mask = atomic_numbers == z  # (N,)
            species_errors.append(error[..., mask].mean(dim=-1))

        # mean over species
        # -> (M,) or ()
        error = torch.stack(species_errors, dim=0).mean(dim=0)

        self.buffer_add(error, num_samples=num_samples)


class ForcesMAESpecies(BaseMetric):
    """
    Returns force MAE computed for each species.
    """

    Z_MAX = 90  # upper bound

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": "meV/ang",
        }
        super().__init__(
            name="forces_MAE_species",
            device=device,
            dtype=dtype,
            units=units,
            requires_species=True,
        )

        # buffers will be initialized later once we know n_models
        self.buffer = None
        self.samples_counter = torch.zeros(self.Z_MAX + 1, device=device, dtype=dtype)

    def reset(self) -> None:
        if self.buffer is not None:
            self.buffer.zero_()
        self.samples_counter.zero_()

    def update(
        self,
        predictions: Target,
        targets: Target,
        atomic_numbers: torch.Tensor,
    ) -> None:

        # ---- assertions on species ----
        assert atomic_numbers.ndim == 1
        assert atomic_numbers.shape[0] == targets.forces.shape[-2]
        assert atomic_numbers.max() <= self.Z_MAX

        # |ΔF| averaged over Cartesian components
        # shapes:
        #   single model: (N,)
        #   ensemble:     (M, N)
        error = torch.abs(targets.forces - predictions.forces).mean(dim=-1)

        # ensure model dimension exists
        if error.ndim == 1:
            error = error.unsqueeze(0)  # (1, N)

        n_models = error.shape[0]

        # lazy buffer initialization
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.Z_MAX + 1,
                n_models,
                device=self.device,
                dtype=self.dtype,
            )

        # accumulate per species
        for z in torch.unique(atomic_numbers):
            z_int = int(z)
            mask = atomic_numbers == z  # (N,)

            # sum over atoms, keep models
            # (M, N_z) → (M,)
            self.buffer[z_int] += error[:, mask].sum(dim=1)

            # count atoms (same for all models)
            self.samples_counter[z_int] += mask.sum()

    def compute(self, reset: bool = True) -> torch.Tensor:
        if self.buffer is None:
            raise ValueError(
                f"Cannot compute value for metric '{self.name}' "
                "because it was never updated."
            )

        # sync across ranks
        distributed.all_sum(self.buffer)
        distributed.all_sum(self.samples_counter)

        # buffer shape: (Z, M) → transpose to (M, Z)
        buffer = self.buffer.transpose(0, 1)  # (M, Z)

        # MAE per model, per species
        mae = torch.zeros_like(buffer)

        mask = self.samples_counter > 0
        mae[:, mask] = buffer[:, mask] / self.samples_counter[mask]

        # unit conversion: eV/Å → meV/Å
        mae = mae * 1000

        if reset:
            self.reset()

        return mae


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
registry.register("forces_MAE_weighted", ForcesMAEWeighted)
registry.register("forces_MAE_species", ForcesMAESpecies)
