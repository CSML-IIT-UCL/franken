"""
Test the model conversion to LAMMPS (essentially testing torch-scriptability, not LAMMPS directly)
"""

import os

import pytest
import torch

from franken.backbones.wrappers.common_patches import unpatch_e3nn
from franken.backbones.wrappers.mace_wrap import atom_numbers_to_node_attrs
from franken.config import BackboneConfig, GaussianRFConfig, MultiscaleGaussianRFConfig
from franken.data import FrankenAtomsDataset
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics
from franken.utils.misc import garbage_collection_cuda
from franken.datasets.registry import DATASET_REGISTRY
from franken.calculators.mace_inf_wrap import MaceInferenceWrapper

from .conftest import DEVICES
from .utils import are_dicts_close, cleanup_dir, create_temp_dir


RF_PARAMETRIZE = [
    GaussianRFConfig(num_random_features=128, length_scale=1.0),
    MultiscaleGaussianRFConfig(num_random_features=128),
]


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backbone", [("pet", "PET_OMat/xs_1.0"), ("mace", "mace_mp/small")])
def test_wrap_compile(rf_cfg, device, backbone):
    """Test for checking save and load methods of FrankenPotential"""
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
        ).to(device)

        with torch.no_grad():
            gnn_features_stats = Statistics()
            for data, _ in dataset:  # type: ignore
                data = data.to(device=device)
                gnn_features = model.gnn.descriptors(data)
                gnn_features_stats.update(
                    gnn_features, atomic_numbers=data.atomic_numbers
                )

            model.input_scaler.set_from_statistics(gnn_features_stats)
            garbage_collection_cuda()

        # Step 2: Save the model to the temporary directory
        model_save_path = os.path.join(temp_dir, "model_checkpoint.pth")
        model.save(model_save_path)

        # Step 3: Run create_lammps_model
        unpatch_e3nn()  # needed in case some previous test ran the patching code
        comp_model_path = MaceInferenceWrapper.init_wrapper(model_path=model_save_path, rf_weight_id=None)

        # Step 4: Load saved model
        comp_model = torch.jit.load(comp_model_path, map_location="cpu").to(device=device)

        # Step 4: Compare rf.state_dict between the original and loaded models
        with pytest.raises(RuntimeError) as exc:
            assert are_dicts_close(
                model.rf.state_dict(), comp_model.model.rf.state_dict(), verbose=True
            )
        assert "Float did not match Double" in str(exc.value)
        assert are_dicts_close(
            model.rf.double().state_dict(), comp_model.model.rf.state_dict(), verbose=True
        ), "The rf.state_dict() of the loaded model does not match the original model."

        with pytest.raises(RuntimeError) as exc:
            assert are_dicts_close(
                model.input_scaler.state_dict(),
                comp_model.model.input_scaler.state_dict(),
                verbose=True,
            )
        assert "Float did not match Double" in str(exc.value)
        assert are_dicts_close(
            model.input_scaler.double().state_dict(),
            comp_model.model.input_scaler.state_dict(),
            verbose=True,
        ), "The input_scaler.state_dict() of the loaded model does not match the original model."

        with pytest.raises(RuntimeError) as exc:
            assert are_dicts_close(
                model.energy_shift.state_dict(),
                comp_model.model.energy_shift.state_dict(),
                verbose=True,
            )
        assert "Float did not match Double" in str(exc.value)
        assert are_dicts_close(
            model.energy_shift.double().state_dict(),
            comp_model.model.energy_shift.state_dict(),
            verbose=True,
        ), "The energy_shift.state_dict() of the loaded model does not match the original model."
    finally:
        if temp_dir is not None:
            cleanup_dir(temp_dir)


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backbone", [("pet", "PET_OMat/xs_1.0"), ("mace", "mace_mp/small")])
def test_wrap_asemd(rf_cfg, device, backbone):
    unpatch_e3nn()  # needed in case some previous test ran the patching code
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
        ).to(device)
        with torch.no_grad():
            gnn_features_stats = Statistics()
            for data, _ in dataset:  # type: ignore
                data = data.to(device=device)
                gnn_features = model.gnn.descriptors(data)
                gnn_features_stats.update(
                    gnn_features, atomic_numbers=data.atomic_numbers
                )

            model.input_scaler.set_from_statistics(gnn_features_stats)
            garbage_collection_cuda()

        # Step 2: Save the model to the temporary directory
        model_save_path = os.path.join(temp_dir, "model_checkpoint.pth")
        model.save(model_save_path)

        # Step 3: Initialize MACE LAMMPS inference wrapper and re-load it
        unpatch_e3nn()  # needed in case some previous test ran the patching code
        comp_model_path = MaceInferenceWrapper.init_wrapper(model_path=model_save_path, rf_weight_id=None)
        comp_model = torch.jit.load(comp_model_path, map_location="cpu").to(device=device)

        # Step 4: run compiled model for the training dataset.
        #         we can't actually run MD because this only works with a LAMMPS calculator
        #         the comp_data dictionary would be filled in with the LAMMPS-MACE C++ code.
        for dataset_el in dataset:
            assert isinstance(dataset_el, tuple)
            config = dataset_el[0].to(device=device)
            node_attrs = atom_numbers_to_node_attrs(
                frame_nums=config.atomic_numbers, all_nums=model.gnn.supported_atomic_types().to(device), 
                dtype=config.atom_pos.dtype
            )
            assert config.edge_index is not None
            comp_data = {
                "node_attrs": node_attrs,
                "cell": config.cell,
                "edge_index": config.edge_index.transpose(0, 1),
                "positions": config.atom_pos,
                "shifts": config.shifts,
                "unit_shifts": config.unit_shifts,
            }
            out_data = comp_model(comp_data, torch.empty((1,)))
            print(out_data["total_energy_local"])
    finally:
        if temp_dir is not None:
            cleanup_dir(temp_dir)
