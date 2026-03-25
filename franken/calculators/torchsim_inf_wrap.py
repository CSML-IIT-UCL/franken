"""torch-sim interface for FrankenPotential."""

from pathlib import Path
import traceback
import warnings
from typing import Any, Callable

import torch

from franken.backbones.wrappers.base import AtomisticModelWrapper
from franken.data.base import Configuration
from franken.rf.model import FrankenPotential


try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
except ImportError as exc:
    warnings.warn(
        f"torch-sim import failed: {traceback.format_exc()}",
        stacklevel=2,
    )

    class FrankenTorchSimModel(torch.nn.Module):
        """Placeholder class when torch-sim is not installed."""

        def __init__(self, err: ImportError = exc, *_args: Any, **_kwargs: Any) -> None:
            raise err

else:

    class FrankenTorchSimModel(ModelInterface):
        """Wrap a FrankenPotential model with the torch-sim ``ModelInterface`` API.

        This adapter returns per-system energies and per-atom forces. Stress is not
        supported in this first version.
        """

        def __init__(
            self,
            franken_model: FrankenPotential | str | Path,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype = torch.float32,
            rf_weight_id: int | None = None,
            neighbor_list_fn: Callable = torchsim_nl,
            compute_forces: bool = True,
            compute_stress: bool = False,
        ) -> None:
            super().__init__()

            if compute_stress:
                raise NotImplementedError(
                    "FrankenTorchSimModel does not support stress in this version."
                )

            resolved_device: torch.device
            if device is None:
                resolved_device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            elif isinstance(device, str):
                resolved_device = torch.device(device)
            else:
                resolved_device = device

            self._device = resolved_device
            self._dtype = dtype
            self._compute_forces = compute_forces
            self._compute_stress = False
            self._memory_scales_with = "n_atoms_x_density"
            self.neighbor_list_fn = neighbor_list_fn

            if isinstance(franken_model, FrankenPotential):
                self.model = franken_model
            elif isinstance(franken_model, (str, Path)):
                self.model = FrankenPotential.load(
                    franken_model,
                    map_location=self._device,
                    rf_weight_id=rf_weight_id,
                )
            else:
                raise TypeError(
                    "franken_model must be a FrankenPotential instance or a checkpoint path"
                )

            family = getattr(self.model.gnn_config, "family", None)
            if family not in ("mace", "pet"):
                raise NotImplementedError(
                    f"FrankenTorchSimModel supports only MACE/PET backbones, found {family!r}."
                )
            if not isinstance(self.model.gnn, AtomisticModelWrapper):
                raise NotImplementedError(
                    "Underlying Franken backbone does not implement AtomisticModelWrapper."
                )

            self.model = self.model.to(device=self._device, dtype=self._dtype).eval()
            self.model.gnn.franken_val()

        def forward(self, state: ts.SimState, **_kwargs: Any) -> dict[str, torch.Tensor]:
            """Compute energies and forces for one or more systems."""
            sim_state = state
            if sim_state.device != self._device or sim_state.dtype != self._dtype:
                sim_state = sim_state.to(device=self._device, dtype=self._dtype)

            system_idx = sim_state.system_idx.to(dtype=torch.long)
            pbc = sim_state.pbc
            pbc_batched = (
                pbc.unsqueeze(0).expand(sim_state.n_systems, -1)
                if pbc.ndim == 1
                else pbc
            )

            wrapped_positions = (
                ts.transforms.pbc_wrap_batched(
                    sim_state.positions,
                    sim_state.cell,
                    system_idx,
                    pbc,
                )
                if pbc.any()
                else sim_state.positions
            ).detach().clone()

            cutoff = torch.tensor(
                self.model.gnn.cutoff_radius(),
                dtype=self._dtype,
                device=self._device,
            )
            edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
                positions=wrapped_positions,
                cell=sim_state.row_vector_cell,
                pbc=pbc_batched,
                cutoff=cutoff,
                system_idx=system_idx,
            )

            shifts = ts.transforms.compute_cell_shifts(
                sim_state.row_vector_cell,
                unit_shifts,
                mapping_system,
            )

            data = Configuration(
                atom_pos=wrapped_positions,
                atomic_numbers=sim_state.atomic_numbers,
                natoms=torch.bincount(system_idx, minlength=sim_state.n_systems).to(
                    dtype=torch.int64
                ),
                edge_index=edge_index.transpose(0, 1),
                shifts=shifts,
                unit_shifts=unit_shifts,
                cell=sim_state.row_vector_cell,
                batch_ids=system_idx,
                pbc=sim_state.pbc,
            )

            energy, forces = self.model.energy_and_forces_batched(
                data,
                compute_forces=self._compute_forces,
                add_energy_shift=True,
            )

            results: dict[str, torch.Tensor] = {"energy": energy.detach()}
            if self._compute_forces:
                assert forces is not None
                results["forces"] = forces.detach()
            return results
