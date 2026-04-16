import numpy as np
import torch

from franken.data.base import Configuration, Target
from franken.metrics.base import BaseMetric
from franken.metrics.registry import registry
from franken.utils import distributed


__all__ = [
    "EnergyMAE",
    "EnergyRMSE",
    "ForcesMAE",
    "ForcesMAESpecies",
    "ForcesRMSE",
    "ForcesRMSESpecies",
    "ForcesCosineSimilarity",
    "is_pareto_efficient",
]


def add_batch_dim(t: torch.Tensor, expected_dims: int) -> torch.Tensor:
    if t.ndim == expected_dims:
        return t
    if t.ndim == expected_dims - 1:
        return t[None, ...]
    raise ValueError(
        f"Tensor has too few dimensions. Expected at least {expected_dims - 1} but found {t.ndim}"
    )


def check_energy_sizes(pred: Target, tgt: Target):
    # energy. preds: [n_models, n_configs], targets: [n_configs]
    e_pred = add_batch_dim(pred.energy, 2)
    e_tgt = torch.atleast_1d(tgt.energy)
    if e_tgt.ndim != 1:
        raise ValueError(f"Energy target has invalid shape {e_tgt.shape}")
    n_models, n_configs = e_pred.shape
    if e_tgt.shape[0] != n_configs:
        raise ValueError(
            f"Energy target has invalid shape {e_tgt.shape} because predictions have shape {e_pred.shape}"
        )
    return e_pred, e_tgt


def check_force_sizes(pred: Target, tgt: Target):
    if tgt.forces is None or pred.forces is None:
        raise AttributeError(
            "Forces must be specified to compute a force-based metric."
        )
    f_pred = add_batch_dim(pred.forces, 3)
    f_tgt = tgt.forces
    if f_tgt.ndim != 2:
        raise ValueError(f"Energy target has invalid shape {f_tgt.shape}")
    return f_pred, f_tgt


class EnergyMAE(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV",
            "outputs": "meV/atom",
        }
        super().__init__("energy_MAE", device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        num_atoms = torch.atleast_1d(data.natoms)
        # energy. preds: [n_models, n_configs], targets: [n_configs]
        e_pred, e_tgt = check_energy_sizes(predictions, targets)

        error = (1000 * torch.abs(e_tgt[None, :] - e_pred) / num_atoms[None, :]).sum(1)

        self.buffer_add(error, num_samples=e_tgt.shape[0])


class EnergyRMSE(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV",
            "outputs": "meV/atom",
        }
        super().__init__("energy_RMSE", device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        num_atoms = torch.atleast_1d(data.natoms)
        # energy. preds: [n_models, n_configs], targets: [n_configs]
        e_pred, e_tgt = check_energy_sizes(predictions, targets)

        error = torch.square((e_tgt[None, :] - e_pred) / num_atoms[None, :]).sum(1)

        self.buffer_add(error, num_samples=e_tgt.shape[0])

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

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        # forces (predicted): [n_models, n_atoms, 3]  (target): [n_atoms, 3]
        f_pred, f_tgt = check_force_sizes(predictions, targets)

        error = 1000 * torch.abs(f_tgt[None, ...] - f_pred)
        error = error.mean(dim=(-1, -2))  # Average over atoms and components

        self.buffer_add(error, num_samples=1)


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
        )

        # buffers will be initialized later once we know n_models
        self.buffer = None
        self.samples_counter = torch.zeros(self.Z_MAX + 1, device=device, dtype=dtype)

    def reset(self) -> None:
        if self.buffer is not None:
            self.buffer.zero_()
        self.samples_counter.zero_()

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        # forces (predicted): [n_models, n_atoms, 3]  (target): [n_atoms, 3]
        f_pred, f_tgt = check_force_sizes(predictions, targets)
        atomic_numbers = data.atomic_numbers
        assert atomic_numbers.ndim == 1
        assert atomic_numbers.shape[0] == f_tgt.shape[-2]
        assert atomic_numbers.max() <= self.Z_MAX

        # |ΔF| in eV/Å, averaged over Cartesian components
        error = torch.abs(f_tgt[None, ...] - f_pred).mean(dim=-1)
        n_models = error.shape[0]

        # lazy buffer initialization
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.Z_MAX + 1, n_models, device=self.device, dtype=self.dtype
            )

        species = torch.unique(atomic_numbers)

        # accumulate per species
        for z in species:
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

        # store average across present species at index 0
        species_mask = mask.clone()
        species_mask[0] = False
        if species_mask.any():
            mae[:, 0] = mae[:, species_mask].mean(dim=1)

        if reset:
            self.reset()

        return mae


class ForcesRMSESpecies(BaseMetric):
    """
    Returns force RMSE computed for each species.
    """

    Z_MAX = 90  # upper bound

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": "meV/ang",
        }
        super().__init__(
            name="forces_RMSE_species",
            device=device,
            dtype=dtype,
            units=units,
        )

        # buffers will be initialized later once we know n_models
        self.buffer = None
        self.samples_counter = torch.zeros(self.Z_MAX + 1, device=device, dtype=dtype)

    def reset(self) -> None:
        if self.buffer is not None:
            self.buffer.zero_()
        self.samples_counter.zero_()

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        # forces (predicted): [n_models, n_atoms, 3]  (target): [n_atoms, 3]
        f_pred, f_tgt = check_force_sizes(predictions, targets)
        atomic_numbers = data.atomic_numbers
        assert atomic_numbers.ndim == 1
        assert atomic_numbers.shape[0] == f_tgt.shape[-2]
        assert atomic_numbers.max() <= self.Z_MAX
        # ΔF^2 in (eV/Å)^2, averaged over Cartesian components
        error = torch.square(f_tgt[None, ...] - f_pred).mean(dim=-1)
        n_models = error.shape[0]

        # lazy buffer initialization
        if self.buffer is None:
            self.buffer = torch.zeros(
                self.Z_MAX + 1,
                n_models,
                device=self.device,
                dtype=self.dtype,
            )

        species = torch.unique(atomic_numbers)

        # accumulate per species
        for z in species:
            z_int = int(z)
            mask = atomic_numbers == z  # (N,)

            # sum over atoms, keep models
            # (M, N_z) -> (M,)
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

        # buffer shape: (Z, M) -> transpose to (M, Z)
        buffer = self.buffer.transpose(0, 1)  # (M, Z)

        # mean squared error per model, per species
        mse = torch.zeros_like(buffer)

        mask = self.samples_counter > 0
        mse[:, mask] = buffer[:, mask] / self.samples_counter[mask]

        # RMSE and unit conversion: eV/Å -> meV/Å
        rmse = torch.sqrt(mse) * 1000

        # store average across present species at index 0
        species_mask = mask.clone()
        species_mask[0] = False
        if species_mask.any():
            rmse[:, 0] = rmse[:, species_mask].mean(dim=1)

        if reset:
            self.reset()

        return rmse


class ForcesRMSE(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": "meV/ang",
        }
        super().__init__("forces_RMSE", device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        # forces (predicted): [n_models, n_atoms, 3]  (target): [n_atoms, 3]
        f_pred, f_tgt = check_force_sizes(predictions, targets)

        error = torch.square(f_tgt[None, ...] - f_pred)
        error = error.mean(dim=(-1, -2))  # Average over atoms and components

        self.buffer_add(error, num_samples=1)

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

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        # forces (predicted): [n_models, n_atoms, 3]  (target): [n_atoms, 3]
        f_pred, f_tgt = check_force_sizes(predictions, targets)
        n_models = f_pred.shape[0]

        # initial averaging over XYZ
        error = torch.square(f_tgt[None, ...] - f_pred).mean(-1)  # [M, A]
        if data.batch_ids is not None:
            # Compute RMSE within each structure
            n_configs = len(torch.atleast_1d(data.natoms))
            batch_ids = data.batch_ids.unsqueeze(0).expand(n_models, -1)
            error = torch.zeros(
                (n_models, n_configs), device=self.device, dtype=self.dtype
            ).scatter_reduce_(1, batch_ids, error, "mean", include_self=False)
            error = torch.sqrt(error) * 1000
            # Average over structures
            error = error.mean(dim=-1)
        else:
            # Average over atoms in structure
            error = error.mean(dim=-1)
            error = torch.sqrt(error) * 1000
        self.buffer_add(error, num_samples=1)


class ForcesCosineSimilarity(BaseMetric):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
        units = {
            "inputs": "eV/ang",
            "outputs": None,
        }
        super().__init__("forces_cosim", device, dtype, units)

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        # forces (predicted): [n_models, n_atoms, 3]  (target): [n_atoms, 3]
        f_pred, f_tgt = check_force_sizes(predictions, targets)

        cos_similarity = torch.nn.functional.cosine_similarity(
            f_pred, f_tgt[None, ...], dim=-1
        )
        cos_similarity = cos_similarity.mean(dim=-1)
        self.buffer_add(cos_similarity, num_samples=1)


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
registry.register("forces_MAE_species", ForcesMAESpecies)
registry.register("forces_RMSE_species", ForcesRMSESpecies)
