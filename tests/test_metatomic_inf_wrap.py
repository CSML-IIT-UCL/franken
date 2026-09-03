"""
Test the model conversion to LAMMPS (essentially testing torch-scriptability, not LAMMPS directly)
"""

import importlib.util
import os
import pytest
import ase
import ase.md.velocitydistribution
import ase.build
import ase.units
import numpy as np
import torch

metatomic_torch = pytest.importorskip("metatomic.torch")
metatomic_ase = pytest.importorskip("metatomic_ase")
pytest.importorskip("metatensor.torch")
pytest.importorskip("metatrain")

load_atomistic_model = metatomic_torch.load_atomistic_model
MetatomicCalculator = metatomic_ase.MetatomicCalculator

HAS_MACE = importlib.util.find_spec("mace") is not None
HAS_PET = importlib.util.find_spec("metatrain.pet") is not None

from franken.backbones.wrappers.common_patches import unpatch_e3nn
from franken.calculators.metatomic_inf_wrap import create_metatomic
from franken.config import BackboneConfig, GaussianRFConfig, MultiscaleGaussianRFConfig
from franken.data import FrankenAtomsDataset
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics
from franken.utils.misc import garbage_collection_cuda
from franken.datasets.registry import DATASET_REGISTRY
from .conftest import DEVICES
from .utils import are_dicts_close, cleanup_dir, create_temp_dir


RF_PARAMETRIZE = [
    GaussianRFConfig(num_random_features=128, length_scale=1.0),
    MultiscaleGaussianRFConfig(num_random_features=128),
]

BACKBONES = [
    pytest.param(
        ("pet", "PET_OMat/xs_1.0"),
        marks=pytest.mark.skipif(not HAS_PET, reason="PET dependencies not installed"),
    ),
    pytest.param(
        ("mace", "mace_mp/small"),
        marks=pytest.mark.skipif(not HAS_MACE, reason="MACE dependencies not installed"),
    ),
]

@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backbone", BACKBONES)
def test_preserves_info(rf_cfg, device, backbone):
    """Test for checking save and load methods of FrankenPotential"""
    gnn_cfg = BackboneConfig.from_ckpt(
        dict(family=backbone[0], path_or_id=backbone[1])
    )

    dtype = torch.float32
    temp_dir = None
    try:
        # Step 1: Create a temporary directory for saving the model
        temp_dir = create_temp_dir()

        data_path = DATASET_REGISTRY.get_path("test", "test", None, False)
        dataset = FrankenAtomsDataset(
            data_path=data_path,
            split="train",
            gnn_config=gnn_cfg,
        )
        model = FrankenPotential(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
            scale_by_Z=True,
            num_species=dataset.num_species,
        ).to(device, dtype=dtype)
        with torch.no_grad():
            gnn_features_stats = Statistics()
            for data, _ in dataset:  # type: ignore
                data = data.to(device=device, dtype=dtype)
                gnn_features = model.gnn.descriptors(data)
                gnn_features_stats.update(
                    gnn_features, atomic_numbers=data.atomic_numbers
                )

            model.input_scaler.set_from_statistics(gnn_features_stats)
            garbage_collection_cuda()

        # Step 2: Save the model to the temporary directory
        model_save_path = os.path.join(temp_dir, "model_checkpoint.pth")
        model.save(model_save_path)

        # Step 3: Run create_metatomic
        unpatch_e3nn()  # MACE needs it before jit script
        comp_model_path = create_metatomic(model_path=model_save_path, rf_weight_id=None, dtype=dtype)

        # Step 4: Load saved model
        mta_model = load_atomistic_model(comp_model_path).to(device)
        mta_model_unwrap1 = mta_model.module
        mta_franken = mta_model_unwrap1.model

        # Step 5: Compare rf.state_dict between the original and loaded models
        assert are_dicts_close(
            model.rf.state_dict(), mta_franken.rf.state_dict(), verbose=True
        ), "The rf.state_dict() of the loaded model does not match the original model."

        assert are_dicts_close(
            model.input_scaler.state_dict(),
            mta_franken.input_scaler.state_dict(),
            verbose=True,
        ), "The input_scaler.state_dict() of the loaded model does not match the original model."

        assert are_dicts_close(
            model.energy_shift.state_dict(),
            mta_franken.energy_shift.state_dict(),
            verbose=True,
        ), "The energy_shift.state_dict() of the loaded model does not match the original model."
    finally:
        if temp_dir is not None:
            cleanup_dir(temp_dir)


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("backbone", BACKBONES)
def test_calc_for_asemd(rf_cfg, device, dtype, backbone):
    gnn_cfg = BackboneConfig.from_ckpt(
        dict(family=backbone[0], path_or_id=backbone[1])
    )
    temp_dir = None
    try:
        # Step 1: Create a temporary directory for saving the model
        temp_dir = create_temp_dir()

        data_path = DATASET_REGISTRY.get_path("test", "test", None, False)
        dataset = FrankenAtomsDataset(
            data_path=data_path,
            split="train",
            gnn_config=gnn_cfg,
        )
        model = FrankenPotential(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
            scale_by_Z=True,
            num_species=dataset.num_species,
        ).to(device, dtype=dtype)
        with torch.no_grad():
            gnn_features_stats = Statistics()
            for data, _ in dataset:  # type: ignore
                data = data.to(device=device, dtype=dtype)
                gnn_features = model.gnn.descriptors(data)
                gnn_features_stats.update(
                    gnn_features, atomic_numbers=data.atomic_numbers
                )

            model.input_scaler.set_from_statistics(gnn_features_stats)
            garbage_collection_cuda()

        # Step 2: Save the model to the temporary directory
        model_save_path = os.path.join(temp_dir, "model_checkpoint.pth")
        model.save(model_save_path)

        # Step 3: Run create_metatomic
        unpatch_e3nn()  # MACE needs it before jit script
        comp_model_path = create_metatomic(model_path=model_save_path, rf_weight_id=None, dtype=dtype)

        # Step 4: Crease ASE structure for MD
        primitive = ase.build.bulk(name="C", crystalstructure="diamond", a=3.567)
        atoms = ase.build.make_supercell(primitive, 3 * np.eye(3))
        ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=300)
        atoms.calc = MetatomicCalculator(comp_model_path, device=device)
        integrator = ase.md.Langevin(
            atoms,
            timestep=1.0 * ase.units.fs,
            temperature_K=300,
            friction=0.1 / ase.units.fs,
        )
        # run short simulation just to check that things don't blow up!
        for _ in range(5):
            integrator.run(1)
            print(atoms.get_total_energy())
    finally:
        if temp_dir is not None:
            cleanup_dir(temp_dir)
