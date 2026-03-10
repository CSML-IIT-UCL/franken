from pathlib import Path
from typing import Optional

import ase
import torch
import metatrain.utils.io
import metatomic.torch as mta
import vesin.metatomic

from franken.backbones.utils import get_checkpoint_path
from franken.data.base import BaseAtomsDataset, Configuration, Target


def convert(ase_frames: list[ase.Atoms], neighbor_list_opt: mta.NeighborListOptions):
    systems = mta.systems_to_torch(ase_frames)  # pyright: ignore[reportArgumentType]
    assert isinstance(systems, list)
    vesin.metatomic.compute_requested_neighbors_from_options(
        systems,
        options=[neighbor_list_opt],
        system_length_unit="Angstrom",
        check_consistency=True,  # pyright: ignore[reportArgumentType]
    )
    for system in systems:
        neighbor_list = system.get_neighbor_list(neighbor_list_opt)
        nl_values = neighbor_list.samples.values
        # nl_values contains: center index, neighbor index, cell shifts
        yield Configuration(
            atom_pos=system.positions,
            atomic_numbers=system.types,
            natoms=torch.tensor(len(system.types)).view(1),
            pbc=system.pbc,
            cell=system.cell,
            shifts=nl_values[:, 2:],
            edge_index=nl_values[:, :2],
        )


class PETAtomsDataset(BaseAtomsDataset):
    def __init__(
        self,
        data_path: str | Path | None,
        split: str,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
        gnn_backbone_id: str | torch.nn.Module | None = None,
        neighbor_list_options: Optional[mta.NeighborListOptions] = None,
        precompute=True,
    ):
        super().__init__(data_path, split, num_random_subsamples, subsample_rng)
        if gnn_backbone_id is not None:
            neighbor_list_options = self.load_info_from_gnn_config(gnn_backbone_id)
        else:
            assert neighbor_list_options is not None
        self.neighbor_list_options = neighbor_list_options

        self.systems: list[Configuration] | None = None
        if precompute and len(self.ase_atoms) > 0:
            self.systems = list(convert(self.ase_atoms, self.neighbor_list_options))

    def load_info_from_gnn_config(
        self, gnn_backbone_id: str | torch.nn.Module
    ) -> mta.NeighborListOptions:
        if isinstance(gnn_backbone_id, str):
            ckpt_path = get_checkpoint_path(gnn_backbone_id)
            loaded_model = metatrain.utils.io.load_model(ckpt_path)
            loaded_model = loaded_model.export()  # no metadata?
        else:
            loaded_model = gnn_backbone_id
        assert isinstance(loaded_model, mta.AtomisticModel)
        all_options = loaded_model.requested_neighbor_lists()[0]
        del loaded_model
        return all_options

    def __getitem__(self, idx, no_targets: bool = False):
        """Returns an array of (inputs, outputs) with inputs being a configuration
        and outputs being the target (energy and forces).
        Note: ONLY for the 'train' split, the energy_shift is removed from the target.
        """
        data: Configuration
        if self.systems is None:
            data = list(convert([self.ase_atoms[idx]], self.neighbor_list_options))[0]
        else:
            data = self.systems[idx]

        if no_targets:
            return data

        energy = torch.tensor(
            self.ase_atoms[idx].get_potential_energy(apply_constraint=False)
        )
        if self.split == "train":
            energy = energy - self.energy_shifts[idx]

        target = Target(
            energy=energy,
            forces=torch.Tensor(self.ase_atoms[idx].get_forces(apply_constraint=False)),
        )
        return data, target
