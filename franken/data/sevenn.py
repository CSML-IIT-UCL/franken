from pathlib import Path
from typing import Optional

import ase
import numpy as np
import torch
from tqdm.auto import tqdm
import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataload import atoms_to_graph

import franken.utils.distributed as dist_utils
from franken.backbones.utils import get_checkpoint_path
from franken.data import BaseAtomsDataset, Configuration, Target


class SevennAtomsToGraphs:
    """
    Args:
        cutoff (float): Cutoff in Angstrom to build atoms graph from positions
        transfer_info (bool): if True, copy info from atoms to graph
    """

    def __init__(self, cutoff: float, transfer_info: bool, y_from_calc: bool):
        self.cutoff = cutoff
        self.transfer_info = transfer_info
        self.y_from_calc = y_from_calc

    def convert(self, atoms: ase.Atoms):
        if not self.y_from_calc:
            # It means we're not interested in forces and energies.
            # workaround is to set the attributes to invalid and then
            # remove the attributes
            atoms.info["y_energy"] = np.nan
            atoms.arrays["y_force"] = np.full(atoms.arrays["positions"].shape, np.nan)
        graph = atoms_to_graph(
            atoms,
            cutoff=self.cutoff,
            transfer_info=self.transfer_info,
            y_from_calc=self.y_from_calc,
            with_shift=True,
        )
        if not self.y_from_calc:
            del graph[KEY.ENERGY]
            del graph[KEY.FORCE]
            del atoms.info["y_energy"]
            del atoms.arrays["y_force"]
        atom_graph_data = AtomGraphData.from_numpy_dict(graph)
        return atom_graph_data

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
            desc = "ASE -> SEVENN"
            if split_name is not None:
                desc += f" ({split_name})"
            atoms_iter = tqdm(atoms_list, desc=desc)
        for atoms in atoms_iter:
            graphs.append(self.convert(atoms))
        return graphs


class SevennAtomsDataset(BaseAtomsDataset):
    def __init__(
        self,
        data_path: str | Path | None,
        split: str,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
        gnn_backbone_id: str | torch.nn.Module | None = None,
        cutoff: float = 6.0,
        precompute=True,
    ):
        super().__init__(data_path, split, num_random_subsamples, subsample_rng)
        if gnn_backbone_id is not None:
            cutoff = self.load_info_from_gnn_config(gnn_backbone_id)
        else:
            assert cutoff is not None

        if split == "md":
            self.a2g = SevennAtomsToGraphs(
                cutoff, transfer_info=False, y_from_calc=False
            )
        else:
            self.a2g = SevennAtomsToGraphs(
                cutoff, transfer_info=False, y_from_calc=True
            )
        self.graphs = None
        if precompute and len(self.ase_atoms) > 0:
            self.graphs = self.a2g.convert_all(
                self.ase_atoms,
                split_name=self.split,
            )

    def load_info_from_gnn_config(self, gnn_backbone_id: str | torch.nn.Module):
        if not isinstance(gnn_backbone_id, str):
            raise ValueError(
                "Backbone path must be provided instead of the preloaded model."
            )
        ckpt_path = get_checkpoint_path(gnn_backbone_id)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        cutoff = config["cutoff"]
        del checkpoint, config
        return cutoff

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
            atom_pos=graph.pos,
            atomic_numbers=graph[KEY.ATOMIC_NUMBERS],
            natoms=torch.tensor(len(graph[KEY.ATOMIC_NUMBERS])).view(1),
            edge_index=graph.edge_index,
            shifts=graph[KEY.CELL_SHIFT],
            cell=graph[KEY.CELL],
            batch_ids=(
                graph.batch
                if graph.batch is not None
                else torch.zeros(graph[KEY.NUM_ATOMS], dtype=torch.int64)
            ),
        )
        if no_targets:
            return data

        energy = graph[KEY.ENERGY]
        if self.split == "train":
            energy = energy - self.energy_shifts[idx]
        target = Target(
            energy=energy,
            forces=graph[KEY.FORCE],
        )
        return data, target
