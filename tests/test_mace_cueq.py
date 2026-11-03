"""Numerical checks for the opt-in native cuEquivariance backend."""

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule

from franken.backbones.wrappers.common_patches import patch_e3nn
from franken.config import GaussianRFConfig, MaceBackboneConfig
from franken.data import Configuration
from franken.rf.model import FrankenPotential


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
@pytest.mark.parametrize("num_rf", [4, 32])
def test_cueq_feature_derivatives_and_reload(model_id, num_rf, tmp_path):
    pytest.importorskip("cuequivariance_torch")
    from cuequivariance_torch.primitives.segmented_polynomial import HAS_CUE_OPS

    assert HAS_CUE_OPS, "installed cuEquivariance kernels cannot import"
    patch_e3nn()
    kwargs = dict(
        rf_config=GaussianRFConfig(
            num_random_features=num_rf, length_scale=2.0, rng_seed=17
        ),
        scale_by_Z=False,
        num_species=1,
        jac_chunk_size=2,
    )
    classic = FrankenPotential(MaceBackboneConfig(model_id), **kwargs).cuda()
    native = FrankenPotential(
        MaceBackboneConfig(model_id, use_cueq=True), **kwargs
    ).cuda()
    native.rf.load_state_dict(classic.rf.state_dict())
    with torch.no_grad():
        native.rf.weights.copy_(
            torch.linspace(-0.4, 0.7, native.rf.weights.numel(), device="cuda")
        )
        classic.rf.weights.copy_(native.rf.weights)
    systems = [molecule("H2O"), bulk("Si", "diamond", a=5.43)]
    systems[0].set_cell(np.eye(3) * 12)
    for atoms in systems:
        atoms.positions += np.random.default_rng(31).normal(
            0, 0.03, atoms.positions.shape
        )
        data = graph(atoms, native)
        old_pos, old_shifts = data.atom_pos, data.shifts
        actual = native.grad_feature_map(data, ["energy", "forces", "stress"])
        assert data.atom_pos is old_pos and data.shifts is old_shifts
        reference = classic.grad_feature_map(
            graph(atoms, classic), ["energy", "forces", "stress"]
        )
        again = native.grad_feature_map(data, ["energy", "forces", "stress"])
        force_only = native.grad_feature_map(graph(atoms, native), ["energy", "forces"])
        for key in actual:
            torch.testing.assert_close(
                actual[key], reference[key], atol=3e-5, rtol=3e-4
            )
            torch.testing.assert_close(again[key], actual[key])
        torch.testing.assert_close(
            force_only["forces"], actual["forces"], atol=3e-5, rtol=3e-4
        )
        assert actual["forces"].norm() > 1e-5
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[1, 0] += 0.002
        minus.positions[1, 0] -= 0.002
        fd = (
            -(
                native.grad_feature_map(graph(plus, native), ["energy"])["energy"]
                - native.grad_feature_map(graph(minus, native), ["energy"])["energy"]
            )
            / 0.004
        )
        torch.testing.assert_close(
            actual["forces"][:, 3], fd[:, 0], atol=1e-4, rtol=0.01
        )
        strain = np.eye(3)
        strain[0, 0] += 0.002
        plus, minus = atoms.copy(), atoms.copy()
        plus.set_cell(atoms.cell.array @ strain, scale_atoms=True)
        strain[0, 0] -= 0.004
        minus.set_cell(atoms.cell.array @ strain, scale_atoms=True)
        fd = (
            (
                native.grad_feature_map(graph(plus, native), ["energy"])["energy"]
                - native.grad_feature_map(graph(minus, native), ["energy"])["energy"]
            )
            / 0.004
            / atoms.get_volume()
        )
        torch.testing.assert_close(
            actual["stress"][:, 0], fd[:, 0], atol=1e-6, rtol=0.015
        )
    checkpoint = tmp_path / "native.pt"
    native.save(checkpoint)
    loaded = FrankenPotential.load(checkpoint, map_location="cuda")
    assert loaded.gnn_config.use_cueq and loaded.gnn.use_cueq
    for key, expected in native.predict(
        ["energy", "forces", "stress"], graph(atoms, native)
    ).items():
        actual = loaded.predict(["energy", "forces", "stress"], graph(atoms, loaded))[
            key
        ]
        torch.testing.assert_close(actual, expected)
    heads = torch.stack(
        [native.rf.weights.flatten(), native.rf.weights.flatten() * 0.3]
    )
    multiple = native.predict(
        ["energy", "forces", "stress"],
        graph(atoms, native),
        weights=heads,
        differential_mode="torch.func",
    )
    single = native.predict(["energy", "forces", "stress"], graph(atoms, native))
    for key in single:
        torch.testing.assert_close(multiple[key][:1], single[key], atol=3e-5, rtol=3e-4)


def test_cueq_is_opt_in_and_survives_configuration_roundtrip():
    from franken.autotune.cli import parse_cli
    from franken.config import BackboneConfig

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
    config = parse_cli(args + ["--mace.use-cueq"]).backbone
    assert config.use_cueq is True
    assert BackboneConfig.from_ckpt(config.to_ckpt()).use_cueq is True
    assert (
        BackboneConfig.from_ckpt(
            {"family": "mace", "path_or_id": "mace_mp/small"}
        ).use_cueq
        is False
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native cuEquivariance requires CUDA"
)
def test_preconverted_checkpoint_selects_safe_derivatives(tmp_path):
    pytest.importorskip("cuequivariance_torch")
    from mace.cli.convert_e3nn_cueq import run as convert
    from franken.backbones.utils import get_checkpoint_path, load_checkpoint

    patch_e3nn()
    model = torch.load(
        get_checkpoint_path("mace_mp/small"), map_location="cpu", weights_only=False
    ).float()
    converted = convert(model, device="cpu", conv_fusion=False)
    path = tmp_path / "converted.model"
    torch.save(converted, path)
    wrapper = load_checkpoint(MaceBackboneConfig(str(path)))
    assert wrapper.use_cueq
    atoms = molecule("H2O")
    data = Configuration(
        atom_pos=torch.tensor(atoms.positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atoms.numbers),
        natoms=torch.tensor([3]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False] * 3),
    )
    data = wrapper.get_neighbors(data)
    with pytest.raises(RuntimeError, match="requires CUDA inputs"):
        wrapper.descriptors(data)
