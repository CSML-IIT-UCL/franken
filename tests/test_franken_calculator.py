import importlib.util
import os
import subprocess
from unittest.mock import patch

import ase
import ase.md
import ase.md.npt
import numpy as np
import pytest
import torch
from ase import units
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from franken.backbones.wrappers.common_patches import unpatch_e3nn
from franken.config import (
    GaussianRFConfig,
    MaceBackboneConfig,
    MultiscaleGaussianRFConfig,
    PETBackboneConfig,
)
from franken.calculators.ase_calc import FrankenCalculator
from franken.rf.model import FrankenPotential
from franken.data.base import Configuration
from franken.utils.linalg.tri import pack_upper
from franken.datasets.registry import DATASET_REGISTRY

from .conftest import DEFAULT_GNN_CONFIGS, DEVICES
from .utils import mocked_gnn


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


HAS_MACE = has_module("mace")
HAS_PET = (
    has_module("metatomic.torch")
    and has_module("metatrain.pet")
    and has_module("vesin.metatomic")
)


def compiler_supports_cxx20():
    compiler = os.environ.get("CXX", "g++")
    try:
        subprocess.run(
            [compiler, "-std=c++20", "-x", "c++", "-", "-fsyntax-only"],
            input="int main() { return 0; }",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


GNN_CONFIGS = [
    pytest.param(
        MaceBackboneConfig("mace_mp/small"),
        marks=pytest.mark.skipif(not HAS_MACE, reason="MACE dependencies not installed"),
    ),
    pytest.param(
        PETBackboneConfig("PET_MAD/xs_1.5"),
        marks=pytest.mark.skipif(not HAS_PET, reason="PET dependencies not installed"),
    ),
]

# Energy baselines use the explicit RF configuration below so changing production
# defaults does not change the potential used in these regression tests.
EXPECTED_ENERGIES = {
    "mace_mp/small": -0.614428,
    "PET_MAD/xs_1.5": -1.19013,
}
EXPECTED_ENERGIES_NPT = {
    "mace_mp/small": -0.752493,
    "PET_MAD/xs_1.5": -1.32933,
}



def init_npt_md(calc):
    data_path = DATASET_REGISTRY.get_path("test", "md", None, False)
    init_traj_atoms = read(data_path, index=0)
    assert isinstance(init_traj_atoms, ase.Atoms)
    init_traj_atoms.calc = calc
    MaxwellBoltzmannDistribution(init_traj_atoms, temperature_K=300)
    dyn = ase.md.npt.NPT(
        init_traj_atoms,
        timestep=1.0 * units.fs,
        temperature_K=300,
        externalstress=0.0,  # Or set a 3x3/Voigt-style stress tensor
        ttime=25 * units.fs,  # Thermostat timescale
        pfactor=75 * units.fs,  # Barostat timescale
    )
    return dyn


def init_langevin_md(calc):
    # Molecular dynamics
    # 1. Get the initial configuration
    # 2. Set some attribute on the configuration with MaxwellBoltzmannDistribution
    # 3. Create and run the MD
    data_path = DATASET_REGISTRY.get_path("test", "md", None, False)
    init_traj_atoms = read(data_path, index=0)
    assert isinstance(init_traj_atoms, ase.Atoms)
    init_traj_atoms.calc = calc
    MaxwellBoltzmannDistribution(init_traj_atoms, temperature_K=500)
    md = ase.md.Langevin(
        init_traj_atoms,
        timestep=1 * units.fs,
        friction=0.01 / units.fs,
        temperature_K=500,
        logfile="-",
        trajectory=None,
        loginterval=1,
        rng=np.random.default_rng(1),
    )
    return md

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gnn_cfg", GNN_CONFIGS)
def test_calculator_in_md(device, gnn_cfg):
    np.random.seed(1)
    torch.manual_seed(1)
    rf_cfg = MultiscaleGaussianRFConfig(
        num_random_features=128,
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6,
    )
    # Define the rng_seed and initialize the model
    model = FrankenPotential(gnn_cfg, rf_cfg).to(device)
    num_lin_models = 1  # only a single weight for MD
    rf_weights = torch.randn(
        (num_lin_models, model.rf.total_random_features)
    ).to(device=device)  # random number gen on CPU for consistency
    model.rf.weights = rf_weights
    calculator = FrankenCalculator(model, device=device, forces_mode="torch.autograd")
    md = init_langevin_md(calculator)
    md.run(2)
    energy = md.atoms.get_total_energy()
    np.testing.assert_allclose(energy, EXPECTED_ENERGIES[gnn_cfg.path_or_id], rtol=1e-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gnn_cfg", GNN_CONFIGS)
def test_calculator_in_npt_md(device, gnn_cfg):
    np.random.seed(1)
    torch.manual_seed(1)
    rf_cfg = MultiscaleGaussianRFConfig(
        num_random_features=128,
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6,
    )
    # Define the rng_seed and initialize the model
    model = FrankenPotential(gnn_cfg, rf_cfg).to(device)
    num_lin_models = 1  # only a single weight for MD
    rf_weights = torch.randn(
        (num_lin_models, model.rf.total_random_features)
    ).to(device=device)  # random number gen on CPU for consistency
    model.rf.weights = rf_weights
    calculator = FrankenCalculator(model, device=device, forces_mode="torch.autograd")
    md = init_npt_md(calculator)
    md.run(2)
    energy = md.atoms.get_total_energy()
    np.testing.assert_allclose(energy, EXPECTED_ENERGIES_NPT[gnn_cfg.path_or_id], rtol=1e-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gnn_cfg", GNN_CONFIGS)
def test_calculator_jitscript(device, gnn_cfg):
    # unpatching is needed for MACE models in case the patch has already been applied.
    unpatch_e3nn()
    np.random.seed(1)
    torch.manual_seed(1)
    rf_cfg = MultiscaleGaussianRFConfig(
        num_random_features=128,
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6,
    )
    # Define the rng_seed and initialize the model
    model = FrankenPotential(gnn_cfg, rf_cfg).to(device)
    num_lin_models = 1  # only a single weight for MD
    rf_weights = torch.randn(
        (num_lin_models, model.rf.total_random_features)
    ).to(device=device)
    model.rf.weights = rf_weights
    jit_model = torch.jit.script(model)
    calculator = FrankenCalculator(
        jit_model, device=device, forces_mode="torch.autograd", gnn_config=gnn_cfg
    )
    md = init_langevin_md(calculator)
    md.run(2)
    jit_energy = md.atoms.get_total_energy()
    np.testing.assert_allclose(jit_energy, EXPECTED_ENERGIES[gnn_cfg.path_or_id], rtol=1e-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gnn_cfg", GNN_CONFIGS)
@pytest.mark.skipif(
    not compiler_supports_cxx20(),
    reason="torch.compile requires a C++ compiler with C++20 support",
)
def test_calculator_compile(device, gnn_cfg):
    if device == "cpu" and gnn_cfg.family == "pet":
        pytest.skip(reason="torch.compile segfault for the `get_adaptive_cutoffs_grid` function")
    np.random.seed(1)
    torch.manual_seed(1)
    rf_cfg = MultiscaleGaussianRFConfig(
        num_random_features=128,
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6,
    )
    # Define the rng_seed and initialize the model
    model = FrankenPotential(gnn_cfg, rf_cfg).to(device)
    num_lin_models = 1  # only a single weight for MD
    rf_weights = torch.randn(
        (num_lin_models, model.rf.total_random_features)
    ).to(device=device)
    model.rf.weights = rf_weights
    model = torch.compile(model)
    calculator = FrankenCalculator(model, device=device, forces_mode="torch.autograd")
    md = init_langevin_md(calculator)
    md.run(2)
    compiled_energy = md.atoms.get_total_energy()
    np.testing.assert_allclose(compiled_energy, EXPECTED_ENERGIES[gnn_cfg.path_or_id], rtol=1e-1)

@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64, "float32", "float64"])
@pytest.mark.parametrize("gnn_cfg", GNN_CONFIGS)
def test_calculator_precision(dtype, gnn_cfg):
    model = FrankenPotential(
        gnn_cfg,
        MultiscaleGaussianRFConfig(num_random_features=16),
    ).to(dtype=torch.float64)
    # Preserve a mixed-precision checkpoint when no conversion is requested.
    model.rf.weights = torch.randn(
        (1, model.rf.total_random_features), dtype=torch.float32
    )
    calculator = FrankenCalculator(model, dtype=dtype)
    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    expected_dtype = torch.float64 if dtype is None else resolved_dtype
    assert calculator.dtype == expected_dtype
    assert model.rf.weights.dtype == (torch.float32 if dtype is None else resolved_dtype)
    atoms = read(DATASET_REGISTRY.get_path("test", "md", None, False), index=0)
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert isinstance(energy, float)
    assert np.isfinite(energy)
    assert forces.dtype == (np.float64 if expected_dtype == torch.float64 else np.float32)
    assert np.isfinite(forces).all()


@pytest.mark.parametrize("dtype", [torch.float16, "float16", "invalid"])
def test_calculator_invalid_precision(dtype):
    with pytest.raises(ValueError, match="dtype must be"):
        FrankenCalculator("unused.pt", dtype=dtype)


@pytest.fixture
def gp_calculator():
    with mocked_gnn("cpu", torch.float32):
        model = FrankenPotential(
            DEFAULT_GNN_CONFIGS[0],
            GaussianRFConfig(num_random_features=8, length_scale=1.0),
            scale_by_Z=False,
        )
    model._cholesky_factor = pack_upper(torch.eye(8) * 2)
    model._l2_penalty = 1.0
    model.gnn.parameters.return_value = [torch.zeros(1)]
    model.gnn.buffers.return_value = []
    with patch("franken.calculators.ase_calc.FrankenAtomsDataset"):
        calc = FrankenCalculator(model, device="cpu")

    def configuration(index, no_targets=True):
        return Configuration(
            atom_pos=torch.tensor(calc.atoms.positions, dtype=torch.float32),
            atomic_numbers=torch.tensor(calc.atoms.numbers),
            natoms=torch.tensor(len(calc.atoms)),
            cell=torch.tensor(calc.atoms.cell.array, dtype=torch.float32),
        )

    calc.dataset.__getitem__.side_effect = configuration
    return calc


def test_gp_requests_and_cache(gp_calculator):
    calc = gp_calculator
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0.8, 0.2, 0.1]], cell=[4, 4, 4])
    props = [
        "energy_gp_uncertainty",
        "per_atom_energy_gp_uncertainty",
        "forces_gp_uncertainty",
    ]
    with patch.object(calc.franken, "forward", wraps=calc.franken.forward) as forward:
        calc.calculate(atoms, properties=["energy", "forces", *props])
        assert forward.call_count == 1
        data = calc.dataset.__getitem__(0, no_targets=True)
        expected = calc.franken.predict(["energy", "forces", *props], data)
        for prop in ["energy", "forces", *props]:
            np.testing.assert_allclose(
                calc.results[prop], expected[prop].squeeze(0).detach().numpy(),
                rtol=1e-5, atol=1e-6,
            )
        assert isinstance(calc.results[props[0]], float)
        assert calc.results[props[1]].shape == (2,)
        assert calc.results[props[2]].shape == (2, 3)
        calls = forward.call_count
        calc.get_property(props[0], atoms)
        assert forward.call_count == calls
        atoms.positions[1, 0] += 0.1
        calc.get_property(props[0], atoms)
        assert forward.call_count == calls + 1
        assert props[1] not in calc.results
        assert set(calc.results) == {props[0]}
        assert forward.call_args.args[0] == [props[0]]

    calc.reset()
    calc.get_property("energy", atoms)
    assert not any(prop in calc.results for prop in props)
    calc.get_property(props[1], atoms)
    assert props[1] in calc.results
    atoms.positions[1, 0] += 0.1
    calc.calculate(atoms, properties=["energy"])
    assert set(calc.results) == {"energy"}

@pytest.mark.parametrize("missing", ["factor", "penalty"])
def test_unavailable_gp(gp_calculator, missing):
    model = gp_calculator.franken
    if missing == "factor":
        model._cholesky_factor = None
    else:
        model._l2_penalty = None
    with patch("franken.calculators.ase_calc.FrankenAtomsDataset"):
        calc = FrankenCalculator(model, device="cpu")
    assert "energy_gp_uncertainty" not in model.implemented_properties
    assert "energy_gp_uncertainty" not in calc.implemented_properties
    with pytest.raises(PropertyNotImplementedError):
        calc.get_property("energy_gp_uncertainty", ase.Atoms("H"))


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_gp_penalty(gp_calculator, penalty):
    gp_calculator.franken._l2_penalty = penalty
    with pytest.raises(ValueError, match="finite positive l2_penalty"):
        gp_calculator.calculate(
            ase.Atoms("H", cell=[4, 4, 4]),
            properties=["energy_gp_uncertainty"],
        )
