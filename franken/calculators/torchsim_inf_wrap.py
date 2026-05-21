from pathlib import Path
import traceback
from typing import Any, Callable
import warnings

import torch

from franken.backbones.wrappers.base import AtomisticModelWrapper
from franken.data.base import (
    ENERGY_TARGET_KEY,
    FORCES_TARGET_KEY,
    STRESS_TARGET_KEY,
    Configuration,
)
from franken.rf.model import FrankenPotential

try:
    import torch_sim.state
    import torch_sim.typing
    import torch_sim.transforms
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
except ImportError:
    warnings.warn(
        f"torch-sim import failed: {traceback.format_exc()}",
        stacklevel=2,
    )

    class FrankenTorchSimModel:
        # dummy class in case imports failed
        def __init__(self):
            raise

else:

    class FrankenTorchSimModel(ModelInterface):  # type: ignore
        """Wrap a :class:`franken.rf.model.FrankenPotential` model with the torch-sim ``ModelInterface`` API.

        This adapter returns per-system energies, per-atom forces and per-system stress tensors.
        """

        def __init__(
            self,
            franken_model: FrankenPotential | str | Path,
            *,
            device: torch.device | str,
            dtype: torch.dtype = torch.float32,
            rf_weight_id: int | None = None,
            neighbor_list_fn: Callable = torchsim_nl,
            compute_forces: bool = True,
            compute_stress: bool = False,
        ) -> None:
            super().__init__()
            if isinstance(device, str):
                self._device = torch.device(device)
            else:
                self._device = device
            self._dtype = dtype
            self._compute_forces = compute_forces
            self._compute_stress = compute_stress
            self._memory_scales_with = "n_atoms_x_density"
            self.neighbor_list_fn = neighbor_list_fn

            if isinstance(franken_model, (str, Path)):
                self.model = FrankenPotential.load(
                    franken_model,
                    map_location=self._device,
                    rf_weight_id=rf_weight_id,
                )
            else:
                self.model = franken_model

            if not isinstance(self.model.gnn, AtomisticModelWrapper):
                raise NotImplementedError(
                    f"Underlying Franken backbone does not implement AtomisticModelWrapper. Found type {type(self.model.gnn)}"
                )

            self.model = self.model.to(device=self._device, dtype=self._dtype).eval()
            self.model.gnn.franken_val()

        def forward(
            self,
            state: torch_sim.state.SimState | torch_sim.typing.StateLike,
            **_kwargs: Any,
        ) -> dict[str, torch.Tensor]:
            """Compute energies and forces for one or more systems."""

            sim_state = (
                state
                if isinstance(state, torch_sim.state.SimState)
                else torch_sim.state.SimState(
                    **state, masses=torch.ones_like(state["positions"])
                )
            )

            if sim_state.device != self._device or sim_state.dtype != self._dtype:
                sim_state = sim_state.to(device=self._device, dtype=self._dtype)

            # Wrap positions into the unit cell
            wrapped_positions = (
                torch_sim.transforms.pbc_wrap_batched(
                    sim_state.positions,
                    sim_state.cell,
                    sim_state.system_idx,
                    sim_state.pbc,
                )
                if sim_state.pbc.any()
                else sim_state.positions
            )

            cutoff = torch.tensor(
                self.model.gnn.cutoff_radius(),
                dtype=self._dtype,
                device=self._device,
            )
            # Batched neighbor list using linked-cell algorithm
            edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
                positions=wrapped_positions,
                cell=sim_state.row_vector_cell,
                pbc=sim_state.pbc,
                cutoff=cutoff,
                system_idx=sim_state.system_idx,
            )
            # Convert unit cell shift indices to Cartesian shifts
            shifts = torch_sim.transforms.compute_cell_shifts(
                sim_state.row_vector_cell, unit_shifts, mapping_system
            )

            data = Configuration(
                atom_pos=wrapped_positions,
                atomic_numbers=sim_state.atomic_numbers,
                natoms=torch.bincount(
                    sim_state.system_idx, minlength=sim_state.n_systems
                ).long(),
                edge_index=edge_index.transpose(0, 1),
                shifts=shifts,
                unit_shifts=unit_shifts,
                cell=sim_state.row_vector_cell,
                batch_ids=sim_state.system_idx,
                pbc=sim_state.pbc,
            )
            targets = [ENERGY_TARGET_KEY]
            if self.compute_forces:
                targets.append(FORCES_TARGET_KEY)
            if self.compute_stress:
                targets.append(STRESS_TARGET_KEY)
            out = self.model(targets, data, weights=None, add_energy_shift=True)
            for k, v in out.items():
                print(f"{k}: {v.shape=}")
            results: dict[str, torch.Tensor] = {
                "energy": out[ENERGY_TARGET_KEY].detach()
            }
            if self.compute_forces:
                results["forces"] = out[FORCES_TARGET_KEY].detach()
            if self.compute_stress:
                results["stress"] = out[STRESS_TARGET_KEY].detach()
            return results
