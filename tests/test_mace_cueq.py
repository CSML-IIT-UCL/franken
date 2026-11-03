"""cuEquivariance inference preserves the model fitted with e3nn."""

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule

from franken.backbones.wrappers.common_patches import patch_e3nn
from franken.calculators.ase_calc import FrankenCalculator
from franken.config import GaussianRFConfig, MaceBackboneConfig
from franken.data import Configuration
from franken.rf.model import FrankenPotential
from franken.trainers import RandomFeaturesTrainer


def graph(atoms, model):
    data = Configuration(
        atom_pos=torch.tensor(atoms.positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atoms.numbers),
        natoms=torch.tensor([len(atoms)]),
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
        pbc=torch.tensor(atoms.pbc),
    )
    return model.gnn.get_neighbors(data).to(device=torch.device("cuda"))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native cuEquivariance requires CUDA"
)
@pytest.mark.parametrize("model_id", ["mace_mp/small", "mace_mp/medium"])
def test_fitted_model_cueq_inference_and_reload(model_id, tmp_path):
    pytest.importorskip("cuequivariance_torch")
    if torch.cuda.get_device_capability() == (7, 0):
        pytest.skip("cuEquivariance 0.10 wheels do not support Volta")
    from cuequivariance_torch.primitives.segmented_polynomial import HAS_CUE_OPS

    assert HAS_CUE_OPS
    patch_e3nn()
    classic = FrankenPotential(
        MaceBackboneConfig(model_id),
        rf_config=GaussianRFConfig(
            num_random_features=32, length_scale=2.0, rng_seed=17
        ),
        scale_by_Z=False,
        num_species=1,
        jac_chunk_size=2,
    ).cuda()
    classic.rf.weights = torch.linspace(
        -0.4, 0.7, 32, device="cuda", dtype=torch.float64
    ).reshape(1, -1)
    checkpoint = tmp_path / "fitted.pt"
    classic.save(checkpoint)
    native = FrankenPotential.load(checkpoint, map_location="cuda", use_cueq=True)
    assert native.gnn.use_cueq and native.gnn_config.use_cueq
    assert native.rf.weights.dtype == torch.float64
    systems = [molecule("H2O"), bulk("Si", "diamond", a=5.43)]
    systems[0].set_cell(np.eye(3) * 12)
    for atoms in systems:
        atoms.positions += np.random.default_rng(31).normal(
            0, 0.03, atoms.positions.shape
        )
        reference = classic.predict(
            ["energy", "forces", "stress"], graph(atoms, classic)
        )
        actual = native.predict(["energy", "forces", "stress"], graph(atoms, native))
        for key in reference:
            torch.testing.assert_close(
                actual[key], reference[key], atol=3e-5, rtol=3e-4
            )
        weights = torch.stack(
            [native.rf.weights.flatten(), native.rf.weights.flatten() * 0.3]
        )
        multiple = native.predict(
            ["energy", "forces", "stress"],
            graph(atoms, native),
            weights=weights,
            differential_mode="torch.func",
        )
        for key in actual:
            torch.testing.assert_close(
                multiple[key][:1], actual[key], atol=3e-5, rtol=3e-4
            )
    with pytest.raises(ValueError, match="inference-only"):
        native.grad_feature_map(graph(atoms, native), ["energy", "forces"])
    trainer = RandomFeaturesTrainer(
        None,
        training_targets=["energy", "forces"],
        l2_penalty=0.01,
        target_weight={},
        device="cuda",
    )
    with pytest.raises(ValueError, match="inference-only"):
        trainer.fit(native)
    infer_checkpoint = tmp_path / "native.pt"
    native.save(infer_checkpoint)
    loaded = FrankenPotential.load(infer_checkpoint, map_location="cuda")
    assert loaded.gnn.use_cueq
    portable = FrankenPotential.load(
        infer_checkpoint, map_location="cuda", use_cueq=False
    )
    assert not portable.gnn.use_cueq
    for model in [loaded, portable]:
        prediction = model.predict(["energy", "forces"], graph(atoms, model))
        for key in prediction:
            torch.testing.assert_close(
                prediction[key], actual[key], atol=3e-5, rtol=3e-4
            )
    # Switching an existing instance preserves the fitted head and can be undone.
    state = {key: value.clone() for key, value in classic.rf.state_dict().items()}
    calc = FrankenCalculator(classic, device="cuda", use_cueq=True)
    assert calc.franken is classic and classic.gnn.use_cueq
    for key, value in classic.rf.state_dict().items():
        torch.testing.assert_close(value, state[key])
    calc.calculate(atoms, properties=["energy", "forces"])
    assert np.isclose(
        calc.results["energy"], actual["energy"].item(), atol=3e-5, rtol=3e-4
    )
    np.testing.assert_allclose(
        calc.results["forces"], actual["forces"][0].cpu().numpy(), atol=3e-5, rtol=3e-4
    )
    classic.enable_cueq(False)
    assert not classic.gnn.use_cueq
    maps = classic.grad_feature_map(graph(atoms, classic), ["energy", "forces"])
    assert maps["forces"].norm() > 1e-5
    # A normal fitted checkpoint can also select cuEquivariance directly in ASE.
    calculator = FrankenCalculator(checkpoint, device="cuda", use_cueq=True)
    calculator.calculate(atoms, properties=["energy", "forces"])
    np.testing.assert_allclose(
        calculator.results["forces"],
        actual["forces"][0].cpu().numpy(),
        atol=3e-5,
        rtol=3e-4,
    )


def test_training_cli_has_no_cueq_flag():
    from franken.autotune.cli import parse_cli

    args = [
        "--train-path",
        "/tmp/train.xyz",
        "--backbone",
        "mace",
        "--mace.path-or-id",
        "mace_mp/small",
        "--rf",
        "gaussian",
        "--gaussian.num-rf",
        "32",
        "--gaussian.length-scale",
        "1",
    ]
    assert parse_cli(args).backbone.use_cueq is False
    with pytest.raises(SystemExit):
        parse_cli(args + ["--mace.use-cueq"])


def test_inference_checkpoint_can_override_to_cpu_without_cuda(tmp_path, monkeypatch):
    classic = FrankenPotential(
        MaceBackboneConfig("mace_mp/small"),
        rf_config=GaussianRFConfig(
            num_random_features=16, length_scale=2.0, rng_seed=17
        ),
        scale_by_Z=False,
        num_species=1,
    )
    filename = tmp_path / "inference.pt"
    classic.save(filename)
    state = torch.load(filename, map_location="cpu", weights_only=False)
    # The saved head is backend-independent; the inference choice is metadata.
    state["gnn"]["config"]["use_cueq"] = True
    torch.save(state, filename)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        FrankenPotential.load(filename, map_location="cpu")
    portable = FrankenPotential.load(filename, map_location="cpu", use_cueq=False)
    assert not portable.gnn.use_cueq
    assert portable.rf.weights.device.type == "cpu"
    for key, value in classic.rf.state_dict().items():
        torch.testing.assert_close(portable.rf.state_dict()[key], value)
