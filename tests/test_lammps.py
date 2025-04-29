"""
Test the model conversion to LAMMPS (essentially testing torch-scriptability, not LAMMPS directly)
"""

import os
import pytest
import torch

from franken.config import GaussianRFConfig, MaceBackboneConfig, MultiscaleGaussianRFConfig
from franken.data import BaseAtomsDataset
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics
from franken.utils.misc import garbage_collection_cuda
from franken.datasets.registry import DATASET_REGISTRY
from franken.calculators.lammps_calc import LammpsFrankenCalculator

from .conftest import DEVICES
from .utils import are_dicts_close, cleanup_dir, create_temp_dir


RF_PARAMETRIZE = [
    GaussianRFConfig(num_random_features=128, length_scale=1.0),
    MultiscaleGaussianRFConfig(num_random_features=128),
]


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
def test_lammps_compile(rf_cfg, device):
    """Test for checking save and load methods of FrankenPotential"""
    gnn_cfg = MaceBackboneConfig("MACE-L0")
    temp_dir = None
    try:
        # Step 1: Create a temporary directory for saving the model
        temp_dir = create_temp_dir()

        data_path = DATASET_REGISTRY.get_path("test", "test", None, False)
        dataset = BaseAtomsDataset.from_path(
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
        comp_model_path = LammpsFrankenCalculator.create_lammps_model(model_path=model_save_path, rf_weight_id=None)

        # Step 4: Load saved model
        comp_model = torch.jit.load(comp_model_path, map_location=device)

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

