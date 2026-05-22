from copy import deepcopy
import os

os.environ["OMP_NUM_THREADS"] = "8"
from unittest.mock import DEFAULT, patch

import numpy as np
import pytest
import torch

from franken.config import GaussianRFConfig, MultiscaleGaussianRFConfig
from franken.calculators.ase_calc import FrankenCalculator
from franken.data import FrankenAtomsDataset
from franken.data.base import ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY, Configuration, TargetType
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics
from franken.utils.misc import garbage_collection_cuda
from franken.datasets.registry import DATASET_REGISTRY

from .conftest import DEFAULT_GNN_CONFIGS, DEVICES
from .utils import are_dicts_close, cleanup_dir, create_temp_dir, mocked_gnn

RF_PARAMETRIZE = [
    GaussianRFConfig(num_random_features=128, length_scale=1.0),
    MultiscaleGaussianRFConfig(num_random_features=128),
]
ALL_TARGETS: list[TargetType] = [ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY]


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
def test_deterministic_initialization(rf_cfg, device):
    for gnn_cfg in DEFAULT_GNN_CONFIGS:
        # Instantiate two models with the same rng_seed
        model1 = FrankenPotential(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
        ).to(device)

        model2 = FrankenPotential(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
        ).to(device)

        # Compare their rf.state_dict()
        assert are_dicts_close(
            model1.rf.state_dict(), model2.rf.state_dict()
        ), f"Model initializations are not deterministic for {device=}, {gnn_cfg=}, {rf_cfg=}"


def mocked_torch_save_load():
    return patch.multiple("torch", load=DEFAULT, save=DEFAULT)


def mocked_dataset(num_atoms, dtype, device, num_configs: int = 1):
    data = []
    for _ in range(num_configs):
        data.append(
            Configuration(
                torch.randn(num_atoms, 3, dtype=dtype),
                torch.randint(1, 100, (num_atoms,)),
                torch.tensor(num_atoms),
            ).to(device)
        )
    return data


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale_by_Z", [True, False])
def test_save_load_functionality(rf_cfg, device, scale_by_Z):
    """Test for checking save and load methods of FrankenPotential"""
    for gnn_cfg in DEFAULT_GNN_CONFIGS:
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
                scale_by_Z=scale_by_Z,
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

            # Step 3: Load the model from the saved checkpoint
            loaded_model = FrankenPotential.load(model_save_path, map_location=device)

            # Step 4: Compare rf.state_dict between the original and loaded models
            assert are_dicts_close(
                model.rf.state_dict(), loaded_model.rf.state_dict(), verbose=True
            ), "The rf.state_dict() of the loaded model does not match the original model."
            assert are_dicts_close(
                model.input_scaler.state_dict(),
                loaded_model.input_scaler.state_dict(),
                verbose=True,
            ), "The input_scaler.state_dict() of the loaded model does not match the original model."

            # Step 5: Compare the hyperparameters between the original and loaded models
            assert (
                model.hyperparameters == loaded_model.hyperparameters
            ), "The hyperparameters of the loaded model do not match the original model."
        finally:
            if temp_dir is not None:
                cleanup_dir(temp_dir)


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale_by_Z", [True, False])
def test_multiweight_save_load_functionality(rf_cfg, device, scale_by_Z):
    """Test for checking save and load methods of FrankenPotential"""
    for gnn_cfg in DEFAULT_GNN_CONFIGS:
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
                scale_by_Z=scale_by_Z,
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

            multi_weights = torch.randn(
                17, model.rf.total_random_features, device=device, dtype=torch.float64
            )

            # Step 2: Save the model to the temporary directory
            model_save_path = os.path.join(temp_dir, "model_checkpoint.pth")
            model.save(model_save_path, multi_weights)

            # Step 3: Load the model from the saved checkpoint
            loaded_model = FrankenPotential.load(
                model_save_path,
                map_location=device,
                rf_weight_id=10,
            )

            model.rf.weights.copy_(multi_weights[10].reshape_as(model.rf.weights))

            # Step 4: Compare rf.state_dict between the original and loaded models
            assert are_dicts_close(
                model.rf.state_dict(), loaded_model.rf.state_dict(), verbose=True
            ), "The rf.state_dict() of the loaded model does not match the original model."

            # Step 5: Compare the hyperparameters between the original and loaded models
            assert (
                model.hyperparameters == loaded_model.hyperparameters
            ), "The hyperparameters of the loaded model do not match the original model."
        finally:
            if temp_dir is not None:
                cleanup_dir(temp_dir)


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multiweights", [True, False])
def test_inference_force_mode(rf_cfg, device, multiweights: bool):
    for gnn_cfg in DEFAULT_GNN_CONFIGS:
        data_path = DATASET_REGISTRY.get_path("test", "test", None, False)
        dataset = FrankenAtomsDataset(
            data_path=data_path,
            split="train",
            gnn_config=gnn_cfg,
            num_random_subsamples=1,
        )

        model = FrankenPotential(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
            scale_by_Z=True,
            num_species=dataset.num_species,
        ).to(device)

        if multiweights:
            dummy_multiweights = torch.randn(
                (10, model.rf.weights.shape[-1]),
                dtype=model.rf.weights.dtype,
                device=model.rf.weights.device,
            )
            model.rf.weights = dummy_multiweights
        else:
            model.rf.weights.copy_(torch.randn(model.rf.weights.shape))

        model_func = deepcopy(model)
        model_func.force_func_grad = True
        calc_func = FrankenCalculator(model_func, device=device)
        calc_autograd = FrankenCalculator(model, device=device)
        for atoms in dataset.ase_atoms[:1]:
            calc_func.calculate(atoms)
            calc_autograd.calculate(atoms)
            print(
                np.max(
                    np.abs(
                        calc_func.results["forces"] - calc_autograd.results["forces"]
                    )
                )
            )
            assert np.allclose(
                calc_func.results["forces"],
                calc_autograd.results["forces"],
                rtol=1e-3,
                atol=1e-3,
            )
            assert np.allclose(
                calc_func.results["energy"],
                calc_autograd.results["energy"],
                rtol=1e-3,
                atol=1e-3,
            )


def random_cfg(num_atoms, dtype, device, atomic_numbers=None):
    if atomic_numbers is not None:
        num_atoms = atomic_numbers.shape[0]
    else:
        atomic_numbers = torch.randint(1, 100, (num_atoms,))
    num_edges = num_atoms * 2
    return Configuration(
        torch.randn(num_atoms, 3, dtype=dtype),
        atomic_numbers=atomic_numbers,
        natoms=torch.tensor(num_atoms),
        edge_index=torch.randint(0, num_atoms, (num_edges, 2), dtype=torch.int32),
        unit_shifts=torch.randn(num_edges, 3, dtype=torch.int32),
        cell=torch.randn((3, 3), dtype=dtype)
    ).to(device)


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
def test_fmap_batched(rf_cfg, device):
    num_atoms = 10
    dtype = torch.float32
    with mocked_gnn(device=device, dtype=dtype, feature_dim=32):
        model = FrankenPotential(
            gnn_config="test", # type: ignore
            rf_config=rf_cfg,
        ).to(device)

    cfg1 = random_cfg(num_atoms, dtype, device)
    cfg2 = random_cfg(num_atoms, dtype, device)
    cfg_batched = Configuration.concatenate([cfg1, cfg2])

    out1 = model.grad_feature_map(cfg1, ALL_TARGETS)
    out2 = model.grad_feature_map(cfg2, ALL_TARGETS)
    out_batched = model.grad_feature_map(cfg_batched, ALL_TARGETS)
    out = {tt: torch.cat([out1[tt], out2[tt]], dim=1) for tt in ALL_TARGETS}
    for tt in ALL_TARGETS:
        torch.testing.assert_close(out_batched[tt], out[tt], rtol=1e-3, atol=1e-3, 
                                    msg=f"Batched (autograd) equality failed on target {tt}")
        torch.testing.assert_close(out_batched[tt], out[tt], rtol=1e-3, atol=1e-3, 
                                    msg=f"Batched (func) equality failed on target {tt}")


@pytest.mark.parametrize("num_systems", [1, 3])
def test_fmap_shapes(num_systems):
    num_atoms = 10
    dtype = torch.float32
    with mocked_gnn(device="cpu", dtype=dtype, feature_dim=32):
        model = FrankenPotential(
            gnn_config="test", # type: ignore
            rf_config=RF_PARAMETRIZE[0],
        )
    data = Configuration.concatenate([
        random_cfg(num_atoms, dtype, "cpu") for _ in range(num_systems)
    ])
    out_fmap = model.grad_feature_map(data, ALL_TARGETS)
    for k, v in out_fmap.items():
        if k == ENERGY_TARGET_KEY:
            assert v.shape == (RF_PARAMETRIZE[0].num_random_features, num_systems)
        if k == FORCES_TARGET_KEY:
            assert v.shape == (RF_PARAMETRIZE[0].num_random_features, num_systems * num_atoms * 3)
        if k == STRESS_TARGET_KEY:
            assert v.shape == (RF_PARAMETRIZE[0].num_random_features, num_systems * 3 * 3)


@pytest.mark.parametrize("num_linear_models", [1, 8])
@pytest.mark.parametrize("num_systems", [1, 3])
def test_gradient_shapes(num_linear_models, num_systems):
    num_atoms = 10
    dtype = torch.float32
    with mocked_gnn(device="cpu", dtype=dtype, feature_dim=32):
        model = FrankenPotential(
            gnn_config="test", # type: ignore
            rf_config=RF_PARAMETRIZE[0],
        )
    weights = torch.randn((num_linear_models, model.rf.total_random_features))
    data = Configuration.concatenate([
        random_cfg(num_atoms, dtype, "cpu") for _ in range(num_systems)
    ])
    out_ag = model._predict(weights, data, ALL_TARGETS, mode="torch.autograd")
    for k, v in out_ag.items():
        if k == ENERGY_TARGET_KEY:
            assert v.shape == (num_linear_models, num_systems)
        if k == FORCES_TARGET_KEY:
            assert v.shape == (num_linear_models, num_systems * num_atoms, 3)
        if k == STRESS_TARGET_KEY:
            assert v.shape == (num_linear_models, num_systems, 3, 3)
    out_func = model._predict(weights, data, ALL_TARGETS, mode="torch.func")
    for k, v in out_func.items():
        if k == ENERGY_TARGET_KEY:
            assert v.shape == (num_linear_models, num_systems)
        if k == FORCES_TARGET_KEY:
            assert v.shape == (num_linear_models, num_systems * num_atoms, 3)
        if k == STRESS_TARGET_KEY:
            assert v.shape == (num_linear_models, num_systems, 3, 3)


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multiweights", [True, False])
@pytest.mark.parametrize("targets", [
    [ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY], 
    [ENERGY_TARGET_KEY, FORCES_TARGET_KEY], 
    [ENERGY_TARGET_KEY]
])
class TestModelGradients:
    def test_gradients_mocked(self, rf_cfg, device, multiweights, targets: list[TargetType]):
        num_atoms = 10
        dtype = torch.float32
        with mocked_gnn(device=device, dtype=dtype, feature_dim=32):
            model = FrankenPotential(
                gnn_config="test", # type: ignore
                rf_config=rf_cfg,
            ).to(device)

        num_lin_models = 8 if multiweights else 1
        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )
        data = random_cfg(num_atoms, dtype, device)
        out_ag = model._predict(weights, data, targets, mode="torch.autograd")
        out_func = model._predict(weights, data, targets, mode="torch.func")
        for tt in targets:
            torch.testing.assert_close(out_func[tt], out_ag[tt], rtol=1e-3, atol=1e-3, 
                                       msg=f"Func-Autograd equality failed on target {tt}. AG={out_ag[tt]} FUNC={out_func[tt]}")
    
    def test_batched_gradients_mocked(self, rf_cfg, device, multiweights, targets):
        num_atoms = 10
        num_lin_models = 8 if multiweights else 1
        dtype = torch.float32
        with mocked_gnn(device=device, dtype=dtype, feature_dim=32):
            model = FrankenPotential(
                gnn_config="test", # type: ignore
                rf_config=rf_cfg,
            ).to(device)

        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )
        cfgs = [random_cfg(num_atoms, dtype, device) for _ in range(2)]
        cfg_batched = Configuration.concatenate(cfgs)
        out_ag_indiv = [model._predict(weights, cfg, targets, mode="torch.autograd") for cfg in cfgs]
        out_ag = {tt: torch.cat([out[tt] for out in out_ag_indiv], dim=1) for tt in targets}
        out_ag_batched = model._predict(weights, cfg_batched, targets, mode="torch.autograd")
        out_func_batched = model._predict(weights, cfg_batched, targets, mode="torch.func")
        for tt in targets:
            torch.testing.assert_close(out_ag_batched[tt], out_ag[tt], rtol=1e-3, atol=1e-3, 
                                       msg=f"Batched (autograd) equality failed on target {tt}")
            torch.testing.assert_close(out_func_batched[tt], out_ag[tt], rtol=1e-3, atol=1e-3, 
                                       msg=f"Batched (func) equality failed on target {tt}")
            
    @pytest.mark.parametrize("gnn_cfg", DEFAULT_GNN_CONFIGS)
    def test_gradients_real(self, rf_cfg, device, multiweights: bool, targets, gnn_cfg):
        model = FrankenPotential(gnn_cfg, rf_cfg).to(device)

        num_lin_models = 10 if multiweights else 1
        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )
        data_path = DATASET_REGISTRY.get_path("test", "test", None, False)
        dataset = FrankenAtomsDataset(
            data_path=data_path,
            split="train",
            gnn_config=gnn_cfg,
        )
        data, _ = dataset[0]  # type: ignore
        data = data.to(device)
        out_ag = model._predict(weights, data, targets, mode="torch.autograd")
        out_func = model._predict(weights, data, targets, mode="torch.func")
        for tt in targets:
            torch.testing.assert_close(out_func[tt], out_ag[tt], rtol=1e-3, atol=1e-3, 
                                       msg=f"Func-Autograd equality failed on target {tt}")
            
    
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("targets", [
    [ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY], 
    [ENERGY_TARGET_KEY, FORCES_TARGET_KEY], 
])
@pytest.mark.parametrize("gnn_cfg", DEFAULT_GNN_CONFIGS)
def test_jitted_model(device, targets, gnn_cfg):
    rf_cfg = RF_PARAMETRIZE[0]
    model = FrankenPotential(gnn_cfg, rf_cfg).to(device)
    jit_model = torch.jit.script(model)

    num_lin_models = 1
    weights = torch.randn(
        (num_lin_models, model.rf.total_random_features), device=device
    )
    data_path = DATASET_REGISTRY.get_path("test", "test", None, False)
    dataset = FrankenAtomsDataset(
        data_path=data_path,
        split="train",
        gnn_config=gnn_cfg,
    )
    data, _ = dataset[0]  # type: ignore
    data = data.to(device)
    out_ag_jit = jit_model._predict(weights, data, targets, mode="torch.autograd")
    with pytest.raises(torch.jit.Error):
        jit_model._predict(weights, data, targets, mode="torch.func")
    out_func = model._predict(weights, data, targets, mode="torch.func")
    for tt in targets:
        torch.testing.assert_close(out_func[tt], out_ag_jit[tt], rtol=1e-3, atol=1e-3, 
                                    msg=f"Func-Autograd equality failed on target {tt}")


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multiweights", [True, False])
class TestEnergyShift:
    num_atoms = 10
    atomic_nums = torch.tensor([1, 1, 1, 1, 1, 8, 8, 8, 8, 8])
    atomic_energies = {1: -1.0, 8: 0.5}
    dtype = torch.float32

    def test_simple(self, rf_cfg, device, multiweights):
        with mocked_gnn(device=device, dtype=self.dtype, feature_dim=32):
            model = FrankenPotential(
                gnn_config="test", # type: ignore
                rf_config=rf_cfg,
                atomic_energies=self.atomic_energies,
                num_species=2,
            ).to(device)

        num_lin_models = 8 if multiweights else 1
        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )
        data = random_cfg(self.num_atoms, self.dtype, device, atomic_numbers=self.atomic_nums)

        out_ag = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.autograd", add_energy_shift=False)
        out_ag_shift = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.autograd", add_energy_shift=True)
        torch.testing.assert_close(out_ag[FORCES_TARGET_KEY], out_ag_shift[FORCES_TARGET_KEY])
        torch.testing.assert_close(out_ag[STRESS_TARGET_KEY], out_ag_shift[STRESS_TARGET_KEY])
        torch.testing.assert_close(
            out_ag[ENERGY_TARGET_KEY] + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 5, 
            out_ag_shift[ENERGY_TARGET_KEY]
        )
        out_fn = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.func", add_energy_shift=False)
        out_fn_shift = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.func", add_energy_shift=True)
        torch.testing.assert_close(out_fn[FORCES_TARGET_KEY], out_fn_shift[FORCES_TARGET_KEY])
        torch.testing.assert_close(out_fn[STRESS_TARGET_KEY], out_fn_shift[STRESS_TARGET_KEY])
        torch.testing.assert_close(
            out_fn[ENERGY_TARGET_KEY] + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 5, 
            out_fn_shift[ENERGY_TARGET_KEY]
        )

    def test_w_feature_maps(self, rf_cfg, device, multiweights):
        with mocked_gnn(device=device, dtype=self.dtype, feature_dim=32):
            model = FrankenPotential(
                gnn_config="test", # type: ignore
                rf_config=rf_cfg,
                atomic_energies=self.atomic_energies,
                num_species=2,
            ).to(device)
        num_lin_models = 8 if multiweights else 1
        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )

        data = random_cfg(self.num_atoms, self.dtype, device, atomic_numbers=self.atomic_nums)
        fmaps = model.grad_feature_map(data, ALL_TARGETS)
        # ffmap = ffmap.view(ffmap.shape[0], -1)
        out_from_fmaps = model.predict_from_fmaps(data, fmaps, weights, add_energy_shift=False)
        out_from_fmaps_shift = model.predict_from_fmaps(data, fmaps, weights, add_energy_shift=True)
        torch.testing.assert_close(out_from_fmaps[FORCES_TARGET_KEY], out_from_fmaps_shift[FORCES_TARGET_KEY])
        torch.testing.assert_close(out_from_fmaps[STRESS_TARGET_KEY], out_from_fmaps_shift[STRESS_TARGET_KEY])
        torch.testing.assert_close(
            out_from_fmaps[ENERGY_TARGET_KEY] + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 5, 
            out_from_fmaps_shift[ENERGY_TARGET_KEY]
        )

    def test_unknown_species(self, rf_cfg, device, multiweights):
        """Shift for unknown species should be 0"""
        atomic_nums = torch.tensor([1, 1, 1, 1, 1, 8, 8, 8, 9, 10])
        with mocked_gnn(device=device, dtype=self.dtype, feature_dim=32):
            model = FrankenPotential(
                gnn_config="test", # type: ignore
                rf_config=rf_cfg,
                atomic_energies=self.atomic_energies,
                num_species=2,
            ).to(device)
        num_lin_models = 8 if multiweights else 1
        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )

        data = random_cfg(self.num_atoms, self.dtype, device, atomic_numbers=atomic_nums)

        out_ag = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.autograd", add_energy_shift=False)
        out_ag_shift = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.autograd", add_energy_shift=True)
        torch.testing.assert_close(out_ag[FORCES_TARGET_KEY], out_ag_shift[FORCES_TARGET_KEY])
        torch.testing.assert_close(out_ag[STRESS_TARGET_KEY], out_ag_shift[STRESS_TARGET_KEY])
        torch.testing.assert_close(
            out_ag[ENERGY_TARGET_KEY] + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 3, 
            out_ag_shift[ENERGY_TARGET_KEY]
        )
        out_fn = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.func", add_energy_shift=False)
        out_fn_shift = model.predict(ALL_TARGETS, data, weights, differential_mode="torch.func", add_energy_shift=True)
        torch.testing.assert_close(out_fn[FORCES_TARGET_KEY], out_fn_shift[FORCES_TARGET_KEY])
        torch.testing.assert_close(out_fn[STRESS_TARGET_KEY], out_fn_shift[STRESS_TARGET_KEY])
        torch.testing.assert_close(
            out_fn[ENERGY_TARGET_KEY] + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 3, 
            out_fn_shift[ENERGY_TARGET_KEY]
        )

    def test_batched(self, rf_cfg, device, multiweights):
        with mocked_gnn(device=device, dtype=self.dtype, feature_dim=32):
            model = FrankenPotential(
                gnn_config="test", # type: ignore
                rf_config=rf_cfg,
                atomic_energies=self.atomic_energies,
                num_species=2,
            ).to(device)
        num_lin_models = 8 if multiweights else 1
        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )
        cfg1 = random_cfg(self.num_atoms, self.dtype, device, atomic_numbers=self.atomic_nums)
        cfg2 = random_cfg(self.num_atoms, self.dtype, device, atomic_numbers=self.atomic_nums)
        cfg_batched = Configuration.concatenate([cfg1, cfg2])

        out_ag_1 = model.predict(ALL_TARGETS, cfg1, weights, differential_mode="torch.autograd", add_energy_shift=True)
        out_ag_2 = model.predict(ALL_TARGETS, cfg2, weights, differential_mode="torch.autograd", add_energy_shift=True)
        out_ag = {tt: torch.cat([out_ag_1[tt], out_ag_2[tt]], dim=1) for tt in ALL_TARGETS}
        out_ag_batched = model.predict(ALL_TARGETS, cfg_batched, weights, differential_mode="torch.autograd", add_energy_shift=True)
        out_func_batched = model.predict(ALL_TARGETS, cfg_batched, weights, differential_mode="torch.func", add_energy_shift=True)
        for tt in ALL_TARGETS:
            torch.testing.assert_close(out_ag_batched[tt], out_ag[tt], rtol=1e-3, atol=1e-3, 
                                       msg=f"Batched (autograd) equality failed on target {tt}")
            torch.testing.assert_close(out_func_batched[tt], out_ag[tt], rtol=1e-3, atol=1e-3, 
                                       msg=f"Batched (func) equality failed on target {tt}")


class TestStatistics:
    def test_online_algo(self):
        dim = 128
        atomic_numbers = torch.tensor([1, 5, 1])
        data = torch.randn(100, len(atomic_numbers), dim)
        st = Statistics(input_dim=dim)

        for data_item in data:
            st.update(data_item, atomic_numbers)

        # Compute the expected values
        global_mean = data.view(-1, dim).mean(0)
        global_std = data.view(-1, dim).std(0)

        torch.testing.assert_close(st.statistics[0]["mean"], global_mean.double())
        torch.testing.assert_close(
            st.statistics[0]["std"], global_std.double(), rtol=1e-2, atol=1e-2
        )

    def test_per_atom(self):
        dim = 128
        atomic_numbers = torch.tensor([1, 5, 1])
        data = torch.randn(100, len(atomic_numbers), dim)
        st = Statistics(input_dim=dim)

        for data_item in data:
            st.update(data_item, atomic_numbers)

        # Compute the expected values
        atom1_mean = data[:, [0, 2]].view(-1, dim).mean(0)
        atom5_mean = data[:, [1]].view(-1, dim).mean(0)
        atom1_std = data[:, [0, 2]].view(-1, dim).std(0)
        atom5_std = data[:, [1]].view(-1, dim).std(0)

        torch.testing.assert_close(st.statistics[1]["mean"], atom1_mean.double())
        torch.testing.assert_close(
            st.statistics[1]["std"], atom1_std.double(), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(st.statistics[5]["mean"], atom5_mean.double())
        torch.testing.assert_close(
            st.statistics[5]["std"], atom5_std.double(), rtol=1e-2, atol=1e-2
        )

