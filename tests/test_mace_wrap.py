

import pytest
import torch
from mace.data import Configuration

from mace.tools import AtomicNumberTable, atomic_numbers_to_indices, to_one_hot

from franken.backbones.utils import load_checkpoint
from franken.backbones.wrappers.mace_wrap import atom_numbers_to_node_attrs
from franken.config import MaceBackboneConfig
from franken.data.base import Configuration
from franken.data.dataset import FrankenAtomsDataset
from franken.datasets.registry import DATASET_REGISTRY


def test_batched_inference():
    gnn_cfg = MaceBackboneConfig(path_or_id="mace_mp/small", interaction_block=2)
    mace_gnn = load_checkpoint(gnn_cfg)
    
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = FrankenAtomsDataset(
        data_path=data_path,
        split="train",
        gnn_config=gnn_cfg,
    )
    data1, data2 = dataset[0], dataset[1]
    assert isinstance(data1, tuple)
    assert isinstance(data2, tuple)
    cfg1 = data1[0]
    cfg2 = data2[0]
    cfg1.atom_pos.requires_grad_(True)
    cfg2.atom_pos.requires_grad_(True)
    data_concat = Configuration.concatenate([cfg1, cfg2])

    desc_actual = mace_gnn.descriptors(data_concat)
    desc_expected = torch.cat(
        [mace_gnn.descriptors(cfg1), mace_gnn.descriptors(cfg2)], dim=0
    )
    torch.testing.assert_close(desc_actual, desc_expected, msg="Batched inference values not equal")

    grad_actual = torch.autograd.grad(desc_actual[0].sum(), cfg1.atom_pos, retain_graph=True)
    grad_expected = torch.autograd.grad(desc_expected[0].sum(), cfg1.atom_pos, retain_graph=True)
    torch.testing.assert_close(grad_actual, grad_expected, msg="Batched inference gradients (1) not equal")
    grad_actual = torch.autograd.grad(desc_actual[1].sum(), cfg1.atom_pos)
    grad_expected = torch.autograd.grad(desc_expected[1].sum(), cfg1.atom_pos)
    torch.testing.assert_close(grad_actual, grad_expected, msg="Batched inference gradients (2) not equal")


def node_attrs_ref_impl(atomic_numbers: torch.Tensor, all_atomic_numbers: torch.Tensor) -> torch.Tensor:
    z_table = AtomicNumberTable([int(z.item()) for z in all_atomic_numbers])
    indices = atomic_numbers_to_indices(atomic_numbers.numpy(), z_table=z_table)
    one_hot = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )
    return one_hot


class TestNodeAttrs:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test1(self, dtype):
        frame_nums = torch.tensor([1, 4])
        all_nums = torch.tensor([1, 2, 4, 5, 6, 7, 8])
        ref_out = node_attrs_ref_impl(frame_nums, all_nums).to(dtype)
        our_out = atom_numbers_to_node_attrs(frame_nums, all_nums, dtype)
        torch.testing.assert_close(our_out, ref_out)
        torch.testing.assert_close(ref_out, torch.tensor([[1., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 1., 0., 0., 0., 0.]], dtype=dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test2(self, dtype):
        frame_nums = torch.tensor([1, 4, 1, 8])
        all_nums = torch.tensor([1, 2, 4, 5, 6, 7, 8])
        ref_out = node_attrs_ref_impl(frame_nums, all_nums).to(dtype)
        our_out = atom_numbers_to_node_attrs(frame_nums, all_nums, dtype)
        torch.testing.assert_close(our_out, ref_out)
        torch.testing.assert_close(ref_out, torch.tensor([[1., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 1., 0., 0., 0., 0.],
                                                          [1., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 1.]], dtype=dtype))

    def test_non_existing(self):
        frame_nums = torch.tensor([1, 3, 4])
        all_nums = torch.tensor([1, 2, 4, 5, 6, 7, 8])
        with pytest.raises(ValueError):
            node_attrs_ref_impl(frame_nums, all_nums)
        with pytest.raises(ValueError):
            atom_numbers_to_node_attrs(frame_nums, all_nums, torch.float32)

    def test_empty(self):
        # Different behavior in case of empty frame atom numbers
        frame_nums = torch.tensor([])
        all_nums = torch.tensor([1, 2, 4, 5, 6, 7, 8])
        with pytest.raises(ValueError):
            node_attrs_ref_impl(frame_nums, all_nums)
        llm_out = atom_numbers_to_node_attrs(frame_nums, all_nums, torch.float32)
        torch.testing.assert_close(llm_out, torch.zeros((0, len(all_nums)), dtype=torch.float32))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_scriptable(self, dtype):
        frame_nums = torch.tensor([1, 4])
        all_nums = torch.tensor([1, 2, 4, 5, 6, 7, 8])

        scripted = torch.jit.script(atom_numbers_to_node_attrs)

        eager_out = atom_numbers_to_node_attrs(frame_nums, all_nums, dtype)
        script_out = scripted(frame_nums, all_nums, dtype)

        torch.testing.assert_close(script_out, eager_out)
