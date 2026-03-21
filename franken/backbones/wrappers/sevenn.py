from pathlib import Path
from typing import Union

import sevenn._keys as KEY
import torch
from sevenn.util import model_from_checkpoint
from sevenn.train.dataload import _graph_build_f

from franken.backbones.wrappers.base import AtomisticModelWrapper
from franken.data.base import Configuration


class FrankenSevenn(torch.nn.Module, AtomisticModelWrapper):
    def __init__(
        self,
        base_model: torch.nn.Module,
        gnn_backbone_id,
        interaction_layer: int,
        extract_after_act: bool = True,
        append_layers: bool = True,
    ):
        super().__init__()

        self.base_model = base_model
        self.gnn_backbone_id = gnn_backbone_id
        self.interaction_layer = interaction_layer
        self.extract_after_act = extract_after_act
        self.append_layers = append_layers
        # Save useful hyperparameters here
        self.cutoff: float = float(
            self.base_model.cutoff
        )  # pyright: ignore[reportArgumentType]
        self.atom_types = torch.tensor(
            list(self.base_model.type_map.keys()), dtype=torch.int64
        )  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]

    def init_args(self):
        return {
            "gnn_backbone_id": self.gnn_backbone_id,
            "interaction_layer": self.interaction_layer,
            "extract_after_act": self.extract_after_act,
            "append_layers": self.append_layers,
        }

    def descriptors(self, data: Configuration) -> torch.Tensor:
        # Convert data to sevenn
        assert data.cell is not None
        sevenn_data = {
            KEY.NODE_FEATURE: data.atomic_numbers,
            KEY.ATOMIC_NUMBERS: data.atomic_numbers,
            KEY.POS: data.atom_pos,
            KEY.EDGE_IDX: data.edge_index.transpose(0, 1),
            KEY.CELL: data.cell,
            KEY.CELL_SHIFT: data.shifts,  # TODO: Check this correct?
            KEY.CELL_VOLUME: torch.einsum(
                "i,i",
                data.cell[0, :],
                torch.linalg.cross(data.cell[1, :], data.cell[2, :]),
            ),
            KEY.NUM_ATOMS: len(data.atomic_numbers),
            KEY.BATCH: torch.zeros(
                (data.atomic_numbers.shape[0],),
                dtype=torch.int32,
                device=data.atomic_numbers.device,
            ),
        }

        # From v0.9.3 to v10 sevenn introduced some changes in how models are built
        # (`build_E3_equivariant_model`), removing the EdgePreprocess class before the
        # network itself. The main purpose of EdgePreprocess was to initialize the
        # KEY.EDGE_VEC (r_ij: the vector between atom positions) and KEY.EDGE_LENGTH.
        # We replace that functionality here.
        # NOTE: the original preprocess had some special handling of the PBC cell
        #       when self.is_stress was set to True. We're ignoring all that.
        # NOTE: as comparison to the original EdgePreprocess we assume `is_batch_data`
        #       to be False.
        idx_src = sevenn_data[KEY.EDGE_IDX][0]
        idx_dst = sevenn_data[KEY.EDGE_IDX][1]
        pos = sevenn_data[KEY.POS]
        edge_vec = pos[idx_dst] - pos[idx_src]
        edge_vec = edge_vec + torch.einsum(
            "ni,ij->nj", sevenn_data[KEY.CELL_SHIFT], sevenn_data[KEY.CELL].view(3, 3)
        )
        sevenn_data[KEY.EDGE_VEC] = edge_vec
        sevenn_data[KEY.EDGE_LENGTH] = torch.linalg.norm(edge_vec, dim=-1)

        # Iterate through the model's layers
        # the sanest way to figure out which layer we're at is through the
        # `_modules` attribute of `nn.Sequential` (which the Sevenn network
        # inherits from), which exposes key-value pairs.
        layer_idx = 0
        scalar_features_list = []
        for i, (name, module) in enumerate(self.base_model._modules.items()):
            if "self_connection_intro" in name:
                layer_idx += 1

            new_sevenn_data = module(sevenn_data)
            if "equivariant_gate" in name:
                if self.extract_after_act:
                    scalar_features = extract_scalar_irrep(
                        new_sevenn_data, module.gate.irreps_out
                    )
                else:
                    scalar_features = extract_scalar_irrep(
                        sevenn_data, module.gate.irreps_in
                    )
                if self.append_layers:
                    scalar_features_list.append(scalar_features)
                else:
                    scalar_features_list[0] = scalar_features
                if layer_idx == self.interaction_layer:
                    break
            sevenn_data = new_sevenn_data
        return torch.cat(scalar_features_list, dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.base_model.parameters())

    def feature_dim(self) -> int:
        layer_idx = 0
        tot_feat_dim = 0
        for i, (name, module) in enumerate(self.base_model._modules.items()):
            if "self_connection_intro" in name:
                layer_idx += 1
            if "equivariant_gate" in name:
                if self.extract_after_act:
                    new_feat_dim = module.gate.irreps_out.count("0e")
                else:
                    new_feat_dim = module.gate.irreps_in.count("0e")
                if self.append_layers:
                    tot_feat_dim += new_feat_dim
                else:
                    tot_feat_dim = new_feat_dim
                if layer_idx == self.interaction_layer:
                    break
        return tot_feat_dim

    def cutoff_radius(self) -> float:
        return self.cutoff

    def num_interaction_layers(self) -> int:
        return self.interaction_layer

    def supported_atomic_types(self) -> torch.Tensor:
        return self.atom_types

    @torch.jit.unused
    def get_neighbors(self, partial_config: Configuration) -> Configuration:
        # 1. config to system
        assert partial_config.cell is not None
        assert partial_config.pbc is not None

        edge_src, edge_dst, edge_vec, shift = _graph_build_f(
            self.cutoff,
            partial_config.pbc.numpy(force=True),
            partial_config.cell.numpy(force=True),
            partial_config.atom_pos.numpy(force=True),
        )

        dtype = partial_config.atom_pos.dtype
        device = partial_config.atom_pos.device

        shift = torch.from_numpy(shift).to(dtype=dtype, device=device)
        edge_index = torch.stack(
            [
                torch.from_numpy(edge_src).to(dtype=torch.int64, device=device),
                torch.from_numpy(edge_dst).to(dtype=torch.int64, device=device),
            ],
            dim=1,
        )
        return Configuration(
            atom_pos=partial_config.atom_pos,
            atomic_numbers=partial_config.atomic_numbers.long(),
            natoms=partial_config.natoms,
            edge_index=edge_index,
            shifts=shift,
            cell=partial_config.cell,
            pbc=partial_config.pbc,
        )

    @staticmethod
    def load_from_checkpoint(
        trainer_ckpt: Union[str, Path],
        gnn_backbone_id: str,
        interaction_block: int,
        extract_after_act: bool = True,
        append_layers: bool = True,
    ):
        sevenn, config = model_from_checkpoint(str(trainer_ckpt))
        return FrankenSevenn(
            base_model=sevenn,
            gnn_backbone_id=gnn_backbone_id,
            interaction_layer=interaction_block,
            extract_after_act=extract_after_act,
            append_layers=append_layers,
        )


def extract_scalar_irrep(data, irreps):
    node_features = data[KEY.NODE_FEATURE]
    scalar_slice = irreps.slices()[0]
    scalar_features = node_features[..., scalar_slice]
    return scalar_features
