
import torch

from franken.backbones.utils import load_checkpoint
from franken.config import PETBackboneConfig
from franken.data.base import BaseAtomsDataset, Configuration
from franken.datasets.registry import DATASET_REGISTRY


def test_batched_inference():
    gnn_cfg = PETBackboneConfig(path_or_id="PET_MAD/xs_1.5")
    pet_gnn = load_checkpoint(gnn_cfg)
    
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = BaseAtomsDataset.from_path(
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

    desc_actual = pet_gnn.descriptors(data_concat)
    desc_expected = torch.cat(
        [pet_gnn.descriptors(cfg1), pet_gnn.descriptors(cfg2)], dim=0
    )
    torch.testing.assert_close(desc_actual, desc_expected, msg="Batched inference values not equal")

    grad_actual = torch.autograd.grad(desc_actual[0].sum(), cfg1.atom_pos, retain_graph=True)
    grad_expected = torch.autograd.grad(desc_expected[0].sum(), cfg1.atom_pos, retain_graph=True)
    torch.testing.assert_close(grad_actual, grad_expected, msg="Batched inference gradients (1) not equal")
    grad_actual = torch.autograd.grad(desc_actual[1].sum(), cfg1.atom_pos)
    grad_expected = torch.autograd.grad(desc_expected[1].sum(), cfg1.atom_pos)
    torch.testing.assert_close(grad_actual, grad_expected, msg="Batched inference gradients (2) not equal")