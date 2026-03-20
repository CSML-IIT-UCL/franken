import dataclasses
import logging
from typing import Optional, Sequence

import torch
from torch import Tensor
import numpy as np

logger = logging.getLogger("franken")


@torch.jit.script
class Configuration:
    """Container for a single configuration (molecule or crystal).

    The set of attributes which are non empty depends on the GNN backbone which the
    :class:`Configuration` object will be passed to.
    """

    def __init__(
        self,
        atom_pos: Tensor,
        atomic_numbers: Tensor,
        natoms: Tensor,
        edge_index: Optional[Tensor] = None,
        shifts: Optional[Tensor] = None,
        unit_shifts: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
        batch_ids: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ):
        assert (
            atom_pos.dim() == 2 and atom_pos.shape[1] == 3
        ), f"Incorrect atom position shape {atom_pos.shape}"
        n_atoms = atom_pos.shape[0]
        self.atom_pos = atom_pos
        assert (
            atomic_numbers.dim() == 1 and len(atomic_numbers) == n_atoms
        ), f"Incorrect atomic numbers shape {atomic_numbers.shape}"
        self.atomic_numbers = atomic_numbers
        self.natoms = natoms
        if edge_index is not None:
            assert (
                edge_index.dim() == 2 and edge_index.shape[1] == 2
            ), f"Incorrect edge_index shape {edge_index.shape}"
        self.edge_index = edge_index
        if shifts is not None:
            assert (
                shifts.dim() == 2 and shifts.shape[1] == 3
            ), f"Incorrect shifts shape {shifts.shape}"
        self.shifts = shifts
        if unit_shifts is not None:
            assert (
                unit_shifts.dim() == 2 and unit_shifts.shape[1] == 3
            ), f"Incorrect unit shifts shape {unit_shifts.shape}"
        self.unit_shifts = unit_shifts
        if cell is not None:
            assert cell.shape == (3, 3), f"Incorrect cell shape {cell.shape}"
        self.cell = cell
        self.batch_ids = batch_ids
        self.pbc = pbc

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> "Configuration":
        # optional-type refinement must be on local variables (torch.jit.script)
        edge_index = self.edge_index
        if edge_index is not None:
            edge_index = edge_index.to(device=device)
        shifts = self.shifts
        if shifts is not None:
            shifts = shifts.to(device=device, dtype=dtype)
        unit_shifts = self.unit_shifts
        if unit_shifts is not None:
            unit_shifts = unit_shifts.to(device=device, dtype=dtype)
        cell = self.cell
        if cell is not None:
            cell = cell.to(device=device, dtype=dtype)
        batch_ids = self.batch_ids
        if batch_ids is not None:
            batch_ids = batch_ids.to(device=device, dtype=dtype)
        pbc = self.pbc
        if pbc is not None:
            pbc = pbc.to(device=device, dtype=dtype)
        return Configuration(
            atom_pos=self.atom_pos.to(device=device, dtype=dtype),
            atomic_numbers=self.atomic_numbers.to(device=device),
            natoms=self.natoms.to(device=device),
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell,
            batch_ids=batch_ids,
            pbc=pbc,
        )

    @staticmethod
    def concatenate(configs: Sequence["Configuration"]) -> "Configuration":
        positions: list[torch.Tensor] = []
        edge_indices: list[Tensor] = []
        species: list[torch.Tensor] = []
        unit_shifts: list[torch.Tensor] = []
        shifts: list[torch.Tensor] = []
        cells: list[torch.Tensor] = []
        all_batch_ids: list[torch.Tensor] = []
        pbc: torch.Tensor | None = None
        node_counter = 0

        # Check that all are consistently None or not None
        pbc_none = np.asarray([c.pbc is None for c in configs])
        if not np.all(pbc_none == pbc_none[0]):
            raise ValueError("PBC inconsistent")
        edge_index_none = np.asarray([c.edge_index is None for c in configs])
        if not np.all(edge_index_none == edge_index_none[0]):
            raise ValueError("Edge index inconsistent")
        shifts_none = np.asarray([c.shifts is None for c in configs])
        if not np.all(shifts_none == shifts_none[0]):
            raise ValueError("Shifts inconsistent")
        unit_shifts_none = np.asarray([c.unit_shifts is None for c in configs])
        if not np.all(unit_shifts_none == unit_shifts_none[0]):
            raise ValueError("Unit shifts inconsistent")
        cell_none = np.asarray([c.cell is None for c in configs])
        if not np.all(cell_none == cell_none[0]):
            raise ValueError("Cell inconsistent")

        for i, config in enumerate(configs):
            system_size = len(config.atom_pos)
            positions.append(config.atom_pos)
            species.append(config.atomic_numbers)
            # All PBCs must be equal across configs! They will not be stacked.
            if config.pbc is not None:
                if pbc is None:
                    pbc = config.pbc
                else:
                    assert torch.all(pbc == config.pbc)
            if config.edge_index is not None:
                edge_indices.append(config.edge_index + node_counter)
            if config.unit_shifts is not None:
                unit_shifts.append(config.unit_shifts)
            if config.cell is not None:
                cells.append(config.cell)
            if config.shifts is not None:
                shifts.append(config.shifts)
            # Check batch IDs: they must not be present in the input configurations
            if config.batch_ids is not None:
                assert len(config.batch_ids.unique()) == 1
            all_batch_ids.append(
                torch.full((system_size,), i, device=config.atom_pos.device)
            )
            node_counter += system_size

        batch_ids = torch.cat(all_batch_ids)
        return Configuration(
            atom_pos=torch.cat(positions),
            edge_index=torch.cat(edge_indices) if len(edge_indices) > 0 else None,
            natoms=torch.bincount(batch_ids, minlength=len(configs)).to(
                dtype=torch.int64
            ),
            atomic_numbers=torch.cat(species),
            cell=torch.stack(cells, dim=0) if len(cells) > 0 else None,
            unit_shifts=torch.cat(unit_shifts) if len(unit_shifts) > 0 else None,
            shifts=torch.cat(shifts) if len(shifts) > 0 else None,
            batch_ids=batch_ids,
            pbc=pbc,
        )


@dataclasses.dataclass
class Target:
    """Container class for the target variables of a single configuration."""

    energy: Tensor
    forces: Optional[Tensor]

    def to(self, device=None, dtype=None) -> "Target":
        return Target(
            energy=self.energy.to(device=device, dtype=dtype),
            forces=(
                self.forces.to(device=device, dtype=dtype)
                if self.forces is not None
                else None
            ),
        )

    def detach(self) -> "Target":
        return Target(
            energy=self.energy.detach(),
            forces=self.forces.detach() if self.forces is not None else None,
        )

    # @staticmethod
    # def concatenate(targets: Sequence['Target']):
    #     energies: list[Tensor] = []
    #     forcess: list[Tensor] = []

    #     forces_none = np.asarray([t.forces is None for t in targets])
    #     if not np.all(forces_none == forces_none[0]):
    #         raise ValueError("Forces inconsistent")

    #     for target in targets:
    #         energies.append(target.energy)
    #         if target.forces is not None:
    #             forcess.append(target.forces)

    #     return Configuration(
    #         atom_pos=torch.cat(positions),
    #         edge_index=torch.cat(edge_indices) if len(edge_indices) > 0 else None,

    #     torch.cat([prd1.energy, prd2.energy], dim=1), torch.cat([prd1.forces, prd2.forces], dim=1)
