from typing import Final

import torch
from mace.modules.models import MACE
from mace.modules.utils import get_edge_vectors_and_lengths
from e3nn.util.jit import compile_mode

from franken.data import Configuration
from franken.utils.misc import torch_load_maybejit


@compile_mode("script")
class FrankenMACE(torch.nn.Module):
    interaction_block: Final[int]

    def __init__(self, base_model: MACE, interaction_block, gnn_backbone_id):
        super().__init__()
        self.gnn_backbone_id = gnn_backbone_id
        self.interaction_block = interaction_block
        # Copy things from base model
        self.atomic_numbers = base_model.atomic_numbers
        self.r_max = base_model.r_max
        self.num_interactions = self.register_buffer(
            "num_interactions", torch.tensor(interaction_block, dtype=torch.int64)
        )
        self.node_embedding = base_model.node_embedding
        self.spherical_harmonics = base_model.spherical_harmonics
        self.radial_embedding = base_model.radial_embedding

        # Load the interactions and products from the torchscript hell
        # that this module becomes.
        module_dict = {nm[0]: nm[1] for nm in base_model.named_modules()}
        try:
            self.interactions = torch.nn.ModuleList(
                [
                    module_dict[f"interactions.{i}"]
                    for i in range(0, self.interaction_block)
                ]
            )
            self.products = torch.nn.ModuleList(
                [module_dict[f"products.{i}"] for i in range(0, self.interaction_block)]
            )
        except KeyError:
            # only for printing helpful message.
            max_interaction = max(
                [
                    int(k.split(".")[1])
                    for k in module_dict.keys()
                    if k.startswith("interactions")
                ]
            )
            raise ValueError(
                f"This model has {max_interaction} gnn layers, while descriptors "
                f"have been required for the {self.interaction_block} layer"
            )
        # This would be the equivalent code if the model were not torchscripted!
        # self.interactions = self.base_model.interactions[: self.interaction_block]
        # self.products = self.base_model.products[: self.interaction_block]

    def init_args(self):
        return {
            "gnn_backbone_id": self.gnn_backbone_id,
            "interaction_block": self.interaction_block,
        }

    def descriptors(self, data: Configuration) -> torch.Tensor:
        # assert on local variables to make torchscript happy
        edge_index = data.edge_index
        shifts = data.shifts
        node_attrs = data.node_attrs
        assert edge_index is not None
        assert shifts is not None
        assert node_attrs is not None
        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.atom_pos,
            edge_index=edge_index,
            shifts=shifts,
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )

        node_feats_list = []
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )

            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs)
            # Extract only scalars. Use `irreps_out` attribute to figure out which features correspond to scalars.
            # irreps_out is an `Irreps` object: a 2-tuple of multiplier and `Irrep` objects.
            # Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]
            # The `Irrep` object is a tuple consisting of parameters `l` and `p`.
            # The scalar irrep is the first in `irreps_out`. Its dimension is computed
            # as `mul * ir.dim` where `ir.dim == 2 * ir.l  + 1`
            # Note this is equivalent code, which does not support TorchScript.
            # invariant_slices = product.linear.irreps_out.slices()[0]
            irreps = product.linear.irreps_out
            invariant_slices = slice(0, irreps[0][0] * (2 * irreps[0][1][0] + 1))
            node_feats_list.append(node_feats[..., invariant_slices])
        return torch.cat(node_feats_list, dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def feature_dim(self):
        nfeat = 0
        for p in self.products:
            nfeat += p.linear.irreps_out[0][0] * (2 * p.linear.irreps_out[0][1][0] + 1)
        return nfeat

    @staticmethod
    def load_from_checkpoint(
        trainer_ckpt, gnn_backbone_id: str, interaction_block: int, map_location=None
    ) -> "FrankenMACE":
        mace = torch_load_maybejit(
            trainer_ckpt, map_location=map_location, weights_only=False
        ).to(dtype=torch.float32)
        return FrankenMACE(
            base_model=mace,
            gnn_backbone_id=gnn_backbone_id,
            interaction_block=interaction_block,
        )
