import traceback

import pytest
import torch

from franken.backbones.wrappers.pet_wrap import systems_to_batch
from franken.calculators.torchsim_inf_wrap import FrankenTorchSimModel
from franken.config import GaussianRFConfig, MaceBackboneConfig, PETBackboneConfig
from franken.data.base import Configuration
from franken.rf.model import FrankenPotential
from tests.utils import mocked_gnn


try:
    import metatomic.torch
    import torch_sim as ts
    from ase.build import molecule
    from torch_sim.models.interface import validate_model_outputs
except ImportError:
    pytest.skip(
        f"torch-sim/metatomic not installed: {traceback.format_exc()}",
        allow_module_level=True,
    )


def _build_mock_franken_model(
    family: str = "mace",
    *,
    device: str | torch.device = "cpu",
) -> FrankenPotential:
    if family == "mace":
        gnn_cfg = MaceBackboneConfig("mace_mp/small")
    elif family == "pet":
        gnn_cfg = PETBackboneConfig("PET_OMat/xs_1.0")
    else:
        raise ValueError(family)

    rf_cfg = GaussianRFConfig(num_random_features=16, length_scale=1.0)
    with mocked_gnn(device=device, dtype=torch.float32, feature_dim=12):
        model = FrankenPotential(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
            scale_by_Z=False,
            num_species=1,
        )
    return model.to(device=device, dtype=torch.float32)


def _demo_state(dtype: torch.dtype = torch.float32) -> ts.SimState:
    h2o = molecule("H2O")
    h2o.center(vacuum=5.0)
    ch4 = molecule("CH4")
    ch4.center(vacuum=5.0)
    return ts.io.atoms_to_state([h2o, ch4], device=torch.device("cpu"), dtype=dtype)


def test_batched_energy_force_matches_single_system_calls() -> None:
    model = _build_mock_franken_model("mace")

    atom_pos = torch.randn(6, 3, dtype=torch.float32)
    atomic_numbers = torch.tensor([1, 8, 1, 6, 1, 8], dtype=torch.int64)
    batch_ids = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)
    natoms = torch.bincount(batch_ids, minlength=2).to(dtype=torch.int64)
    batched = Configuration(
        atom_pos=atom_pos,
        atomic_numbers=atomic_numbers,
        natoms=natoms,
        batch_ids=batch_ids,
    )

    fmap = model.feature_map_batched(batched)
    assert fmap.shape[0] == 2

    energy_b, forces_b = model.energy_and_forces_batched(batched)
    assert forces_b is not None

    single_energies = []
    single_forces = []
    for sid in range(2):
        mask = batch_ids == sid
        single = Configuration(
            atom_pos=atom_pos[mask].clone(),
            atomic_numbers=atomic_numbers[mask].clone(),
            natoms=torch.tensor([int(mask.sum())], dtype=torch.int64),
        )
        e, f = model.energy_and_forces(single)
        assert f is not None
        single_energies.append(e.reshape(-1)[0])
        single_forces.append(f.reshape(-1, 3))

    torch.testing.assert_close(
        energy_b,
        torch.stack(single_energies),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        forces_b,
        torch.cat(single_forces, dim=0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_torchsim_adapter_forward_shapes_detached_and_no_state_mutation() -> None:
    model = _build_mock_franken_model("mace")
    adapter = FrankenTorchSimModel(model, device="cpu", dtype=torch.float32)

    state = _demo_state(torch.float32)
    pos0 = state.positions.clone()
    cell0 = state.cell.clone()
    sys0 = state.system_idx.clone()
    z0 = state.atomic_numbers.clone()

    out = adapter(state)
    assert set(out.keys()) == {"energy", "forces"}
    assert out["energy"].shape == (state.n_systems,)
    assert out["forces"].shape == state.positions.shape
    assert not out["energy"].requires_grad
    assert not out["forces"].requires_grad

    torch.testing.assert_close(state.positions, pos0)
    torch.testing.assert_close(state.cell, cell0)
    torch.testing.assert_close(state.system_idx, sys0)
    torch.testing.assert_close(state.atomic_numbers, z0)


def test_torchsim_adapter_supports_mace_and_pet_families() -> None:
    state = _demo_state(torch.float32)

    mace_model = _build_mock_franken_model("mace")
    pet_model = _build_mock_franken_model("pet")

    mace_adapter = FrankenTorchSimModel(mace_model, device="cpu", dtype=torch.float32)
    pet_adapter = FrankenTorchSimModel(pet_model, device="cpu", dtype=torch.float32)

    out_mace = mace_adapter(state)
    out_pet = pet_adapter(state)

    assert out_mace["energy"].shape == (state.n_systems,)
    assert out_pet["energy"].shape == (state.n_systems,)


def test_torchsim_adapter_construction_from_checkpoint_path(tmp_path) -> None:
    model = _build_mock_franken_model("mace")
    ckpt_path = tmp_path / "franken_mock.pt"
    model.save(ckpt_path)

    with mocked_gnn(device="cpu", dtype=torch.float32, feature_dim=12):
        adapter = FrankenTorchSimModel(ckpt_path, device="cpu", dtype=torch.float32)

    out = adapter(_demo_state(torch.float32))
    assert out["energy"].ndim == 1
    assert out["forces"].ndim == 2


def test_torchsim_model_interface_contract() -> None:
    model = _build_mock_franken_model("mace")
    adapter = FrankenTorchSimModel(model, device=torch.device("cpu"), dtype=torch.float32)
    validate_model_outputs(
        adapter,
        device=torch.device("cpu"),
        dtype=torch.float32,
        check_detached=True,
    )


def test_pet_systems_to_batch_accepts_precomputed_cartesian_shifts() -> None:
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
    edge_system_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    cartesian_shifts = torch.einsum(
        "ni,nij->nj",
        unit_shifts.to(cell.dtype),
        cell[edge_system_idx],
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
        edge_system_idx=edge_system_idx,
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
        edge_system_idx=edge_system_idx,
    )

    for a, b in zip(out_from_unit, out_from_cart):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torchsim_adapter_cuda_batched_smoke() -> None:
    model = _build_mock_franken_model("mace", device="cuda")
    adapter = FrankenTorchSimModel(model, device="cuda", dtype=torch.float32)

    state = _demo_state(torch.float32).to(device=torch.device("cuda"), dtype=torch.float32)
    out = adapter(state)

    assert out["energy"].shape == (state.n_systems,)
    assert out["forces"].shape == state.positions.shape
    assert out["energy"].device.type == "cuda"
    assert out["forces"].device.type == "cuda"
    assert not out["energy"].requires_grad
    assert not out["forces"].requires_grad
