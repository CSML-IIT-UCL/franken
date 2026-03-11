

import pytest
import torch

from mace.tools import AtomicNumberTable, atomic_numbers_to_indices, to_one_hot

from franken.backbones.wrappers.mace_wrap import atom_numbers_to_node_attrs


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
