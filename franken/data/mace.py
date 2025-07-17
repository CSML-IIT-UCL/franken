from pathlib import Path
from typing import Optional

import torch
from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools.utils import AtomicNumberTable
from tqdm.auto import tqdm

import franken.utils.distributed as dist_utils
from franken.backbones.utils import get_checkpoint_path
from franken.data import BaseAtomsDataset, Configuration, Target
from franken.utils.misc import torch_load_maybejit


class MACEAtomsToGraphs:
    def __init__(self, z_table: AtomicNumberTable, cutoff: float):
        self.cutoff = cutoff
        self.z_table = z_table

    def convert(self, atoms):
        mace_config = config_from_atoms(atoms)
        graph = AtomicData.from_config(
            mace_config, z_table=self.z_table, cutoff=self.cutoff
        )
        graph.atomic_numbers = torch.tensor(atoms.get_atomic_numbers()).int()
        return graph

    def convert_all(
        self,
        atoms_list,
        process_rank: Optional[int] = None,
        split_name: Optional[str] = None,
    ):
        graphs = []
        atoms_iter = atoms_list
        if process_rank is None:
            process_rank = dist_utils.get_rank()
        if process_rank == 0:
            desc = "ASE -> MACE"
            if split_name is not None:
                desc += f" ({split_name})"
            atoms_iter = tqdm(atoms_list, desc=desc)
        for atoms in atoms_iter:
            graphs.append(self.convert(atoms))
        return graphs


class MACEAtomsDataset(BaseAtomsDataset):
    def __init__(
        self,
        data_path: str | Path | None,
        split: str,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
        gnn_backbone_id: str | torch.nn.Module | None = None,
        z_table: AtomicNumberTable | None = None,
        cutoff=6.0,
        precompute=True,
    ):
        super().__init__(data_path, split, num_random_subsamples, subsample_rng)
        if gnn_backbone_id is not None:
            z_table, cutoff = self.load_info_from_gnn_config(gnn_backbone_id)
        else:
            assert z_table is not None

        self.a2g = MACEAtomsToGraphs(z_table, cutoff)
        self.graphs = None
        if precompute and len(self.ase_atoms) > 0:
            self.graphs = self.a2g.convert_all(
                self.ase_atoms,
                split_name=self.split,
            )

    def load_info_from_gnn_config(self, gnn_backbone_id: str | torch.nn.Module):
        if isinstance(gnn_backbone_id, str):
            ckpt_path = get_checkpoint_path(gnn_backbone_id)
            mace_gnn = torch_load_maybejit(
                ckpt_path, map_location="cpu", weights_only=False
            )
        else:
            mace_gnn = gnn_backbone_id
        z_table = AtomicNumberTable([z.item() for z in mace_gnn.atomic_numbers])
        cutoff = mace_gnn.r_max.item()
        del mace_gnn
        return z_table, cutoff

    def __getitem__(self, idx, no_targets: bool = False):
        """Returns an array of (inputs, outputs) with inputs being a configuration
        and outputs being the target (energy and forces).
        Note: ONLY for the 'train' split, the energy_shift is removed from the target.
        """
        if self.graphs is None:
            graph = self.a2g.convert(self.ase_atoms[idx])
        else:
            graph = self.graphs[idx]

        data = Configuration(
            atom_pos=graph.positions,
            atomic_numbers=graph.atomic_numbers,
            natoms=torch.tensor(len(graph.atomic_numbers)).view(1),
            node_attrs=graph.node_attrs,
            edge_index=graph.edge_index,
            shifts=graph.shifts,
            unit_shifts=graph.unit_shifts,
        )
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
