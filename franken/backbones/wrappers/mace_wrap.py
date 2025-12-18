from typing import Final, Optional, Tuple
import mace
from packaging.version import Version

import torch
from mace.modules.models import MACE
from mace.modules.utils import get_edge_vectors_and_lengths
from e3nn.util.jit import compile_mode
from e3nn import o3

from franken.data import Configuration
from franken.utils.misc import torch_load_maybejit


def undo_script_mace(base: torch.jit.ScriptModule) -> MACE:
    from mace.modules.models import ScaleShiftMACE
    from mace.modules.blocks import (
        RealAgnosticResidualInteractionBlock,
        RealAgnosticInteractionBlock,
    )

    if base.original_name != "ScaleShiftMACE":
        raise ValueError("Only ScaleShiftMACE supported")

    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        else:
            raise ValueError(f"Unrecognised bessel function {radial_type}")

    def radial_to_transform(radial):
        if not hasattr(radial, "distance_transform"):
            return None
        if radial.distance_transform.original_name == "AgnesiTransform":
            return "Agnesi"
        if radial.distance_transform.original_name == "SoftTransform":
            return "Soft"
        else:
            raise ValueError(
                f"Unrecognised distance transform {radial.distance_transform.original_name}"
            )

    def interaction_cls(module):
        if module.original_name == "RealAgnosticResidualInteractionBlock":
            return RealAgnosticResidualInteractionBlock
        elif module.original_name == "RealAgnosticInteractionBlock":
            return RealAgnosticInteractionBlock
        else:
            raise ValueError(f"Unrecognised interaction class {module.original_name}")

    def gate_fn(non_linearity):
        nonlin = non_linearity._modules["acts"]._modules["0"]
        if hasattr(nonlin, "f"):
            return nonlin.f
        code = nonlin.code
        if "silu" in code:
            return torch.nn.functional.silu
        elif "torch.abs" in code:
            return torch.abs
        else:
            raise ValueError(f"Unrecognized gating function. Code is {code}")

    module_dict = {nm[0]: nm[1] for nm in base.named_modules()}
    heads = base.heads if hasattr(base, "heads") else ["default"]
    num_interactions = int(base.num_interactions.item())
    model_mlp_irreps = (
        o3.Irreps(module_dict[f"readouts.{num_interactions - 1}"].hidden_irreps)
        if base.num_interactions.item() > 1
        else 1
    )
    try:
        correlation = (
            len(
                list(
                    module_dict[
                        "products.0.symmetric_contractions.contractions.0"
                    ].weights.parameters()
                )
            )
            + 1
        )
    except (KeyError, AttributeError):
        correlation = module_dict[
            "products.0.symmetric_contractions"
        ].contraction_degree

    args = {}
    args["r_max"] = base.r_max.item()
    args["num_bessel"] = len(base.radial_embedding.bessel_fn.bessel_weights)
    args["num_polynomial_cutoff"] = base.radial_embedding.cutoff_fn.p.item()
    args["max_ell"] = base.spherical_harmonics._lmax
    args["interaction_cls"] = interaction_cls(module_dict["interactions.1"])
    args["interaction_cls_first"] = interaction_cls(module_dict["interactions.0"])
    args["num_interactions"] = num_interactions
    args["num_elements"] = len(base.atomic_numbers)
    args["hidden_irreps"] = o3.Irreps(module_dict["products.0.linear"].irreps_out)
    # args["edge_irreps"] = base.edge_irreps if hasattr(base, "edge_irreps") else None
    args["MLP_irreps"] = (
        o3.Irreps(f"{model_mlp_irreps.count((0, 1)) // len(heads)}x0e")  # type: ignore
        if num_interactions > 1
        else 1
    )
    args["atomic_energies"] = base.atomic_energies_fn.atomic_energies.cpu().numpy()
    args["avg_num_neighbors"] = module_dict["interactions.0"].avg_num_neighbors
    args["atomic_numbers"] = base.atomic_numbers.cpu().numpy()
    args["correlation"] = correlation
    args["gate"] = (
        gate_fn(module_dict[f"readouts.{num_interactions - 1}"].non_linearity)
        if num_interactions > 1
        else None
    )
    args["pair_repulsion"] = hasattr(base, "pair_repulsion")
    args["distance_transform"] = radial_to_transform(base.radial_embedding)
    args["radial_MLP"] = module_dict["interactions.0"].conv_tp_weights.hs[1:-1]
    # args["radial_MLP"] = module_dict["interactions.0"].radial_MLP
    args["radial_type"] = radial_to_name(base.radial_embedding.bessel_fn.original_name)
    args["heads"] = base.heads
    # args["use_reduced_cg"] = base.use_reduced_cg if hasattr(base, "use_reduced_cg") else False
    # args["use_so3"] = base.use_so3 if hasattr(base, "use_so3") else False
    # args["cueq_config"] = base.cueq_config if hasattr(base, "cueq_config") else None
    args["atomic_inter_scale"] = base.scale_shift.scale.cpu().numpy()
    args["atomic_inter_shift"] = base.scale_shift.shift.cpu().numpy()
    model = ScaleShiftMACE(**args)
    load_result = model.load_state_dict(base.state_dict(), strict=False)
    if len(load_result.unexpected_keys) > 0:
        raise RuntimeError(
            f"Failed to load state-dict. There were unexpected keys: {load_result.unexpected_keys}"
        )
    for k in load_result.missing_keys:
        if (
            "weights_0_zeroed" not in k
            and "weights_1_zeroed" not in k
            and "weights_max_zeroed" not in k
        ):
            raise RuntimeError(
                f"Failed to load state-dict. There were missing keys: {load_result.missing_keys}"
            )
    return model


@compile_mode("script")
class FrankenMACE(torch.nn.Module):
    """Wrap MACE module.

    Most of the complication here is to support torch-scripted modules, which don't play well
    with `torch.func.jacfwd` and other transformations. In particular
     - the spherical harmonics of e3nn don't work as they are jitted by default. We have a patch
        in :module:`franken.backbones.wrappers.common_patches` which fixes this, but does not get
        called when the full model is scripted.
     - the radial embedding doesn't work (more investigation needed to figure out exactly which
        part of it fails), so we have to rebuild it discarding the jitted version.
     - In order to extract features at a specific layer, we also have to pull apart the interactions
        and products module-lists which are not easily indexable when jitted.
    """

    interaction_block: Final[int]

    def __init__(self, base_model: MACE, interaction_block, gnn_backbone_id):
        super().__init__()
        self.gnn_backbone_id = gnn_backbone_id
        self.interaction_block = interaction_block
        # Copy things from base model
        if isinstance(base_model, torch.jit.ScriptModule):
            base_model = undo_script_mace(base_model)
        self.register_buffer(
            "atomic_numbers",
            torch.as_tensor(base_model.atomic_numbers, dtype=torch.int64),
        )
        self.r_max = base_model.r_max
        self.register_buffer(
            "num_interactions", torch.tensor(interaction_block, dtype=torch.int64)
        )
        self.node_embedding = base_model.node_embedding
        self.spherical_harmonics = base_model.spherical_harmonics
        self.radial_embedding = base_model.radial_embedding
        self.interactions = base_model.interactions[: self.interaction_block]
        self.products = base_model.products[: self.interaction_block]
        self.is_mace_v3_14 = Version(mace.__version__) >= Version("0.3.14")  # type: ignore
        self.is_mace_v3_13 = Version(mace.__version__) >= Version("0.3.13")  # type: ignore

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
        node_feats = self.node_embedding(node_attrs)  # type: ignore
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.atom_pos,
            edge_index=edge_index,
            shifts=shifts,
        )
        edge_attrs = self.spherical_harmonics(vectors)  # type: ignore
        rad_emb = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )  # type: ignore
        if torch.jit.isinstance(rad_emb, torch.Tensor):
            edge_feats, cutoff = rad_emb, None
        elif torch.jit.isinstance(rad_emb, Tuple[torch.Tensor, torch.Tensor]):
            edge_feats, cutoff = rad_emb
        elif torch.jit.isinstance(rad_emb, Tuple[torch.Tensor, Optional[float]]):
            edge_feats, cutoff = rad_emb
        else:  # Python fallback
            edge_feats, cutoff = rad_emb

        # lammps_class, lammps_natoms are only set when using LAMMPS MLIAP
        # the defaults are used otherwise and set here
        lammps_class = None
        lammps_natoms = (0, 0)

        node_feats_list = []
        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            if self.is_mace_v3_14:
                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                    cutoff=cutoff,
                    first_layer=(i == 0),
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                )  # type: ignore
            elif self.is_mace_v3_13:
                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                    first_layer=(i == 0),
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                )  # type: ignore
            else:
                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )  # type: ignore

            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )  # type: ignore
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
