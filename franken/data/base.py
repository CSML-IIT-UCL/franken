import dataclasses
import logging
from typing import Generator, Literal, Optional, Sequence, get_args

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
        n_batches = 1 if batch_ids is None else len(torch.unique(batch_ids))
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
            if n_batches == 1 and cell.ndim == 2:
                assert cell.shape == (3, 3), f"Incorrect cell shape {cell.shape}"
            else:
                assert cell.shape == (
                    n_batches,
                    3,
                    3,
                ), f"Incorrect cell shape {cell.shape}"
        self.cell = cell
        if batch_ids is not None:
            assert (
                batch_ids.dim() == 1 and len(batch_ids) == n_atoms
            ), f"Incorrect batch_ids shape {batch_ids.shape}"
        self.batch_ids = batch_ids
        self.pbc = pbc

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
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
                assert len(torch.unique(config.batch_ids)) == 1
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

    @torch.jit.unused
    def __str__(self) -> str:
        attrs = {
            "atom_pos": self.atom_pos,
            "atomic_numbers": self.atomic_numbers,
            "natoms": self.natoms,
            "edge_index": self.edge_index,
            "shifts": self.shifts,
            "unit_shifts": self.unit_shifts,
            "cell": self.cell,
            "batch_ids": self.batch_ids,
            "pbc": self.pbc,
        }

        def fmt(value):
            if value is None:
                return "None"

            # Torch tensors
            if hasattr(value, "shape"):
                shape = tuple(value.shape)
                dtype = getattr(value, "dtype", None)
                device = getattr(value, "device", None)
                return f"Tensor(shape={shape}, dtype={dtype}, device={device})"

            return repr(value)

        formatted = ",\n    ".join(
            f"{name}={fmt(value)}" for name, value in attrs.items()
        )
        return f"{self.__class__.__name__}(\n    {formatted}\n)"

    def __repr__(self) -> str:
        return str(self)


# NOTE: TargetType should be an enum, but instances of
#       dict[Enum, Any] do not work with torch jit. We
#       use aliases of strings such that when scripting
#       str can be used.
TargetType = Literal["energy", "forces", "stress"]
"""TargetType describes the range of possible targets for :class:`franken.rf.model.FrankenPotential`"""
ENERGY_TARGET_KEY: TargetType = "energy"
FORCES_TARGET_KEY: TargetType = "forces"
STRESS_TARGET_KEY: TargetType = "stress"


def is_target_key(s: str):
    return s in get_args(TargetType)


def all_target_keys() -> tuple[str]:
    return get_args(TargetType)


def is_scalar_target(s: str):
    if not is_target_key(s):
        raise ValueError(f"{s} is not a valid target type")
    return s == ENERGY_TARGET_KEY


@dataclasses.dataclass
class Target:
    """Container class for the target variables of a single configuration."""

    energy: Tensor | None
    forces: Tensor | None
    stress: Tensor | None = None

    def __getitem__(self, key) -> Tensor:
        if not is_target_key(key):
            raise KeyError(key)
        elif key == ENERGY_TARGET_KEY:
            val = self.energy
        elif key == FORCES_TARGET_KEY:
            val = self.forces
        elif key == STRESS_TARGET_KEY:
            val = self.stress
        else:
            raise KeyError(key)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key) -> bool:
        try:
            return self[key] is not None
        except KeyError:
            return False

    def to(self, device=None, dtype=None) -> "Target":
        return Target(
            energy=(
                self.energy.to(device=device, dtype=dtype)
                if self.energy is not None
                else None
            ),
            forces=(
                self.forces.to(device=device, dtype=dtype)
                if self.forces is not None
                else None
            ),
            stress=(
                self.stress.to(device=device, dtype=dtype)
                if self.stress is not None
                else None
            ),
        )

    def detach(self) -> "Target":
        return Target(
            energy=self.energy.detach() if self.energy is not None else None,
            forces=self.forces.detach() if self.forces is not None else None,
            stress=self.stress.detach() if self.stress is not None else None,
        )

    @staticmethod
    def from_types(tt_dict: dict[TargetType, Tensor]):
        return Target(
            energy=tt_dict.get(ENERGY_TARGET_KEY),
            forces=tt_dict.get(FORCES_TARGET_KEY),
            stress=tt_dict.get(STRESS_TARGET_KEY),
        )

    def iter_individual_systems(
        self, data: Configuration
    ) -> Generator["Target", None, None]:
        if data.batch_ids is None:
            yield self
        else:
            sys_ids = data.batch_ids.unique()
            for i, sys_id in enumerate(sys_ids):
                cfg_energy, cfg_forces, cfg_stress = None, None, None
                if self.energy is not None:
                    cfg_energy = self.energy[..., i]
                if self.forces is not None:
                    cfg_forces = self.forces[..., data.batch_ids == sys_id, :]
                if self.stress is not None:
                    has_batch = self.stress.ndim > 2
                    stress = self.stress.reshape(-1, len(sys_ids), 3, 3)
                    cfg_stress = stress[:, i]
                    if not has_batch:
                        cfg_stress = cfg_stress.squeeze(0)
                yield Target(cfg_energy, cfg_forces, cfg_stress)

    @staticmethod
    def concatenate(targets: Sequence["Target"]):
        e_lst: list[Tensor] = []
        f_lst: list[Tensor] = []
        s_lst: list[Tensor] = []

        energy_none = np.asarray([t.energy is None for t in targets])
        if not np.all(energy_none == energy_none[0]):
            raise ValueError("Energies inconsistent")

        forces_none = np.asarray([t.forces is None for t in targets])
        if not np.all(forces_none == forces_none[0]):
            raise ValueError("Forces inconsistent")

        stress_none = np.asarray([t.stress is None for t in targets])
        if not np.all(stress_none == stress_none[0]):
            raise ValueError("Stresses inconsistent")

        e_has_batch, f_has_batch, s_has_batch = False, False, False
        for target in targets:
            if target.energy is not None:
                e_lst.append(target.energy)
                if target.energy.ndim != e_lst[0].ndim:
                    raise ValueError(
                        f"Energy shapes inconsistent. Found {target.energy.ndim} and {e_lst[0].ndim} dimensions"
                    )
                if target.energy.ndim > 1:
                    e_has_batch = True
            if target.forces is not None:
                f_lst.append(target.forces)
                if target.forces.ndim != f_lst[0].ndim:
                    raise ValueError(
                        f"Forces shapes inconsistent. Found {target.forces.ndim} and {f_lst[0].ndim} dimensions"
                    )
                if target.forces.ndim > 2:
                    f_has_batch = True
            if target.stress is not None:
                s_lst.append(target.stress)
                if target.stress.ndim != s_lst[0].ndim:
                    raise ValueError(
                        f"Stress shapes inconsistent. Found {target.stress.ndim} and {s_lst[0].ndim} dimensions"
                    )
                if target.stress.ndim > 2:
                    s_has_batch = True
        return Target(
            energy=(
                torch.cat(e_lst, 1 if e_has_batch else 0) if len(e_lst) > 0 else None
            ),
            forces=(
                torch.cat(f_lst, 1 if f_has_batch else 0) if len(f_lst) > 0 else None
            ),
            stress=(
                torch.cat(s_lst, 1 if s_has_batch else 0) if len(s_lst) > 0 else None
            ),
        )
