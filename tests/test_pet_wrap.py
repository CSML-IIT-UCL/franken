
from copy import deepcopy

import torch
import metatomic.torch

from franken.backbones.utils import load_checkpoint
from franken.config import PETBackboneConfig
from franken.data.base import BaseAtomsDataset, Configuration
from franken.datasets.registry import DATASET_REGISTRY
from franken.backbones.wrappers.pet_wrap import systems_to_batch


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
    # setup data-concat
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
    
    for i in range(desc_actual.shape[0]):
        grad_actual = torch.autograd.grad(desc_actual[i].sum(), cfg1.atom_pos, retain_graph=True)
        grad_expected = torch.autograd.grad(desc_expected[i].sum(), cfg1.atom_pos, retain_graph=True)
        torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-4, atol=1e-4, 
            msg=f"Batched inference gradients cfg1, index {i} not equal. {grad_actual=} {grad_expected=}")
        if i >= cfg1.atom_pos.shape[0]:
            torch.testing.assert_close(grad_actual[0].sum().item(), 0.0)
        grad_actual = torch.autograd.grad(desc_actual[i].sum(), cfg2.atom_pos, retain_graph=True)
        grad_expected = torch.autograd.grad(desc_expected[i].sum(), cfg2.atom_pos, retain_graph=True)
        torch.testing.assert_close(grad_actual, grad_expected, rtol=1e-4, atol=1e-4, 
            msg=f"Batched inference gradients cfg2, index {i} not equal. {grad_actual=} {grad_expected=}")
        if i < cfg1.atom_pos.shape[0]:
            torch.testing.assert_close(grad_actual[0].sum().item(), 0.0)


def test_pet_systems_to_batch_accepts_precomputed_cartesian_shifts() -> None:
    """LLM generated, unclear what's being tested."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [2, 3], [3, 2]],
        dtype=torch.long,
    )
    cell = torch.stack(
        [
            torch.eye(3, dtype=torch.float32) * 10.0,
            torch.eye(3, dtype=torch.float32) * 8.0,
        ]
    )
    unit_shifts = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=torch.long,
    )
    batch_ids = torch.tensor([0, 0, 1, 1])

    cartesian_shifts = torch.einsum(
        "ni,nij->nj", 
        unit_shifts.to(positions.dtype), 
        cell[batch_ids[edge_index[:, 0]]]
    )
    species = torch.tensor([1, 8, 1, 8], dtype=torch.long)

    species_to_species_index = torch.zeros(9, dtype=torch.long)
    species_to_species_index[1] = 0
    species_to_species_index[8] = 1

    nl_options = metatomic.torch.NeighborListOptions(
        cutoff=6.0,
        full_list=True,
        strict=True,
    )

    out_from_unit = systems_to_batch(
        positions,
        edge_index,
        cell,
        unit_shifts,
        species,
        nl_options,
        species_to_species_index,
        cutoff_function="cosine",
        cutoff_width=0.5,
        num_neighbors_adaptive=None,
        cartesian_shifts=None,
        batch_ids=batch_ids,
    )
    out_from_cart = systems_to_batch(
        positions,
        edge_index,
        cell,
        unit_shifts,
        species,
        nl_options,
        species_to_species_index,
        cutoff_function="cosine",
        cutoff_width=0.5,
        num_neighbors_adaptive=None,
        cartesian_shifts=cartesian_shifts,
        batch_ids=batch_ids
    )

    for a, b in zip(out_from_unit, out_from_cart):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
