import pytest
import torch
from ase.build import molecule

ts = pytest.importorskip("torch_sim")
torch_sim_interface = pytest.importorskip("torch_sim.models.interface")

from franken.calculators.torchsim_inf_wrap import FrankenTorchSimModel
from franken.config import GaussianRFConfig, MaceBackboneConfig, PETBackboneConfig
from franken.data.base import Configuration
from franken.rf.model import FrankenPotential
from tests.utils import mocked_gnn

validate_model_outputs = torch_sim_interface.validate_model_outputs


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
    adapter = FrankenTorchSimModel(
        model, device=torch.device("cpu"), dtype=torch.float32, 
        compute_forces=True, compute_stress=True
    )
    validate_model_outputs(
        adapter,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


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


@pytest.mark.parametrize("from_checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64, "float32", "float64", "default"])
def test_torchsim_adapter_precision(tmp_path, dtype, from_checkpoint):
    model = FrankenPotential(
        MaceBackboneConfig("mace_mp/small"),
        GaussianRFConfig(num_random_features=16, length_scale=1.0),
    ).to(dtype=torch.float32)
    weights = torch.full_like(model.rf.weights, 1.000000000123, dtype=torch.float64)
    model.rf.weights = weights.clone()
    source = model
    if from_checkpoint:
        source = tmp_path / "model.pt"
        model.save(source)
    kwargs = {} if dtype == "default" else {"dtype": dtype}
    adapter = FrankenTorchSimModel(source, device="cpu", **kwargs)
    expected_dtype = torch.float64 if dtype in (torch.float64, "float64") else torch.float32
    expected_weights = weights if dtype in (None, "default") else weights.to(expected_dtype)
    assert adapter.dtype == expected_dtype
    assert next(adapter.model.gnn.parameters()).dtype == expected_dtype
    torch.testing.assert_close(adapter.model.rf.weights, expected_weights, rtol=0, atol=0)

    # The adapter converts the incoming state's precision without mutating it.
    input_dtype = torch.float64 if expected_dtype == torch.float32 else torch.float32
    state = _demo_state(input_dtype)
    positions = state.positions.clone()
    out = adapter(state)
    assert out["energy"].dtype == torch.float64
    assert out["forces"].dtype == expected_dtype
    assert torch.isfinite(out["energy"]).all()
    assert torch.isfinite(out["forces"]).all()
    torch.testing.assert_close(state.positions, positions)
    assert state.dtype == input_dtype


@pytest.mark.parametrize("dtype", [torch.float16, "float16", "invalid"])
def test_torchsim_adapter_invalid_precision(dtype):
    with pytest.raises(ValueError, match="dtype must be"):
        FrankenTorchSimModel("unused.pt", device="cpu", dtype=dtype)
