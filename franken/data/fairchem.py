from pathlib import Path

import torch
from fairchem.core.preprocessing import AtomsToGraphs as FairchemAtomsToGraphs

import franken.utils.distributed as dist_utils
from franken.backbones.utils import get_checkpoint_path
from franken.data import BaseAtomsDataset, Configuration, Target


class FairchemAtomsDataset(BaseAtomsDataset):
    def __init__(
        self,
        data_path: str | Path | None,
        split: str,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
        gnn_backbone_id: str | torch.nn.Module | None = None,
        cutoff=6.0,
        max_num_neighbors=200,
        precompute=True,
    ):
        super().__init__(data_path, split, num_random_subsamples, subsample_rng)

        if gnn_backbone_id is not None:
            cutoff, max_num_neighbors = self.load_info_from_gnn_config(gnn_backbone_id)

        if split == "md":
            # Cannot get energy and forces in MD mode (the calculator fails)
            self.a2g = FairchemAtomsToGraphs(
                max_neigh=max_num_neighbors,
                radius=cutoff,  # type: ignore
            )
        else:
            self.a2g = FairchemAtomsToGraphs(
                max_neigh=max_num_neighbors,
                radius=cutoff,
                r_energy=True,
                r_forces=True,  # type: ignore
            )
        self.graphs = None
        if precompute and len(self.ase_atoms) > 0:
            self.graphs = self.a2g.convert_all(
                self.ase_atoms, disable_tqdm=dist_utils.get_rank() != 0
            )

    def load_info_from_gnn_config(self, gnn_backbone_id: str | torch.nn.Module):
        if not isinstance(gnn_backbone_id, str):
            raise ValueError(
                "Backbone path must be provided instead of the preloaded model."
            )
        ckpt_path = get_checkpoint_path(gnn_backbone_id)
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = model["config"]["model"]
        cutoff = getattr(model_cfg, "cutoff", 6.0)
        max_num_neighbors = getattr(model_cfg, "max_num_neighbors", 200)
        del model, model_cfg
        return cutoff, max_num_neighbors

    def graph_to_inputs(self, graph):
        return Configuration(
            atom_pos=graph.pos,  # type: ignore
            atomic_numbers=graph.atomic_numbers.int(),
            natoms=torch.tensor(graph.natoms).view(1),
            batch_ids=(
                graph.batch
                if graph.batch is not None
                else torch.zeros(graph.natoms, dtype=torch.int64)
            ),
            cell=graph.cell,
            pbc=getattr(graph, "pbc", None),
        )

    def graph_to_targets(self, graph):
        energy = torch.tensor(graph.energy)
        return Target(energy=energy, forces=graph.forces)

    def __getitem__(self, idx, no_targets: bool = False):
        """Returns an array of (inputs, outputs) with inputs being a configuration
        and outputs being the target (energy and forces).
        Note: ONLY for the 'train' split, the energy_shift is removed from the target.
        """
        if self.graphs is None:
            graph = self.a2g.convert(self.ase_atoms[idx])
        else:
            graph = self.graphs[idx]

        data = self.graph_to_inputs(graph)
        if no_targets:
            return data
        target = self.graph_to_targets(graph)

        if self.split == "train":
            target.energy -= self.energy_shifts[idx]

        return data, target
