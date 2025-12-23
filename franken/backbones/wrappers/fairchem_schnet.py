from collections import namedtuple

import torch
from fairchem.core.models.schnet import SchNetWrap
import fairchem.core.common.utils

from franken.data import Configuration


def segment_coo_patch(src, index, dim_size=None):
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim=0, index=index, src=src)
    return out


def segment_csr_patch(src, indptr):
    out = torch.zeros(indptr.size(0) - 1, dtype=src.dtype, device=src.device)
    for i in range(len(indptr) - 1):
        out[i] = src[indptr[i] : indptr[i + 1]].sum()
    return out


def patch_fairchem():
    """
    The `segment_coo` and `segment_csr` patches are necessary to allow
    forward-mode autodiff through the network, which is not implemented
    in the original torch-scatter functions.
    """
    fairchem.core.common.utils.segment_coo = segment_coo_patch
    fairchem.core.common.utils.segment_csr = segment_csr_patch


FairchemCompatData = namedtuple(
    "FairchemCompatData", ["pos", "cell", "batch", "natoms", "atomic_numbers"]
)


class FrankenSchNetWrap(SchNetWrap):
    def __init__(self, *args, interaction_block, gnn_backbone_id, **kwargs):
        patch_fairchem()
        super().__init__(*args, **kwargs)

        self.interaction_block = interaction_block
        self.gnn_backbone_id = gnn_backbone_id

    def descriptors(
        self,
        data: Configuration,
    ):
        """
        Forward pass for the SchNet model to get the embedded representations of the input data
        """
        fairchem_compat_data = FairchemCompatData(
            data.atom_pos, data.cell, data.batch_ids, data.natoms, data.atomic_numbers
        )
        # fairchem checks if the attribute exists, not whether it's None.
        if data.pbc is not None:
            fairchem_compat_data.pbc = data.pbc  # type: ignore
        # Get the atomic numbers of the input data
        z = data.atomic_numbers.long()
        assert z.dim() == 1
        # Get the edge index, edge weight and other attributes of the input data
        graph = self.generate_graph(fairchem_compat_data)

        edge_attr = self.distance_expansion(graph.edge_distance)

        # Get the embedded representations of the input data
        h = self.embedding(z)
        for interaction in self.interactions[: self.interaction_block]:
            h = h + interaction(h, graph.edge_index, graph.edge_distance, edge_attr)

        return h

    def feature_dim(self):
        return self.hidden_channels

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_args(self):
        return {
            "gnn_backbone_id": self.gnn_backbone_id,
            "interaction_block": self.interaction_block,
        }

    @staticmethod
    def load_from_checkpoint(
        trainer_ckpt, gnn_backbone_id, interaction_block
    ) -> "FrankenSchNetWrap":
        ckpt_data = torch.load(
            trainer_ckpt, map_location=torch.device("cpu"), weights_only=False
        )

        model_config = ckpt_data["config"]["model_attributes"]
        model_config["otf_graph"] = True

        model = FrankenSchNetWrap(
            **model_config,
            interaction_block=interaction_block,
            gnn_backbone_id=gnn_backbone_id,
        )
        # Before we can load state, need to fix state-dict keys:
        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."
        ckpt_key_count = next(iter(ckpt_data["state_dict"])).count("module")
        mod_key_count = next(iter(model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count
        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in ckpt_data["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in ckpt_data["state_dict"].items()
            }
        else:
            new_dict = ckpt_data["state_dict"]
        model.load_state_dict(new_dict)
        return model
