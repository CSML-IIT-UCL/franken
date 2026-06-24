import importlib.util
import os
import subprocess

import ase
import ase.md
import ase.md.npt
import numpy as np
import pytest
import torch
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from franken.backbones.wrappers.common_patches import unpatch_e3nn
from franken.config import MaceBackboneConfig, MultiscaleGaussianRFConfig, PETBackboneConfig
from franken.calculators.ase_calc import FrankenCalculator
from franken.rf.model import FrankenPotential
from franken.datasets.registry import DATASET_REGISTRY

from .conftest import DEVICES


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
    rf_cfg = MultiscaleGaussianRFConfig(num_random_features=128)
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
    rf_cfg = MultiscaleGaussianRFConfig(num_random_features=128)
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
    rf_cfg = MultiscaleGaussianRFConfig(num_random_features=128)
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
    np.random.seed(1)
    torch.manual_seed(1)
    rf_cfg = MultiscaleGaussianRFConfig(num_random_features=128)
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
