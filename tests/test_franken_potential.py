from pathlib import Path
import os



os.environ["OMP_NUM_THREADS"] = "8"
from unittest.mock import DEFAULT, patch

import ase
import numpy as np
import pytest
import torch
from ase import units
from ase.io import read
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from franken.trainers.log_utils import LogCollection
from franken.config import GaussianRFConfig, HPSearchConfig, MultiscaleGaussianRFConfig, SolverConfig
from franken.calculators.ase_calc import FrankenCalculator
from franken.autotune.script import init_loaders, run_autotune
from franken.data import BaseAtomsDataset
from franken.data.base import Configuration
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics
from franken.trainers.rf_cuda_lowmem import RandomFeaturesTrainer
from franken.utils.misc import garbage_collection_cuda
from franken.datasets.registry import DATASET_REGISTRY

from .conftest import DEFAULT_GNN_CONFIGS, DEVICES
from .utils import are_dicts_close, cleanup_dir, create_temp_dir, mocked_gnn

RF_PARAMETRIZE = [
    GaussianRFConfig(num_random_features=128, length_scale=1.0),
    MultiscaleGaussianRFConfig(num_random_features=128),
]


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
            dataset = BaseAtomsDataset.from_path(
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
            dataset = BaseAtomsDataset.from_path(
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
        dataset = BaseAtomsDataset.from_path(
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

        calc_func = FrankenCalculator(model, device=device, forces_mode="torch.func")
        calc_autograd = FrankenCalculator(
            model,
            device=device,
            forces_mode="torch.autograd",
        )
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


@pytest.mark.parametrize("rf_cfg", RF_PARAMETRIZE)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("multiweights", [True, False])
class TestModelGradients:
    def test_gradients_mocked(self, rf_cfg, device, multiweights):
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
        data = Configuration(
            torch.randn(num_atoms, 3, dtype=dtype),
            torch.randint(1, 100, (num_atoms,)),
            torch.tensor(num_atoms),
        ).to(device)
        energy_autograd, forces_autograd = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.autograd"
        )
        energy_func, forces_func = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.func"
        )
        torch.testing.assert_close(energy_func, energy_autograd, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(forces_func, forces_autograd, rtol=1e-3, atol=1e-3)
    
    def test_batched_gradients_mocked(self, rf_cfg, device, multiweights):
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
        atomic_nums = torch.randint(1, 100, (num_atoms,))
        cfg1 = Configuration(
            torch.randn(num_atoms, 3, dtype=dtype), atomic_nums, torch.tensor(num_atoms)
        ).to(device)
        cfg2 = Configuration(
            torch.randn(num_atoms, 3, dtype=dtype), atomic_nums, torch.tensor(num_atoms)
        ).to(device)
        cfg_batched = Configuration.concatenate([cfg1, cfg2])

        energy_autograd_1, forces_autograd_1 = model.energy_and_forces(
            cfg1, weights=weights, forces_mode="torch.autograd"
        )
        energy_autograd_2, forces_autograd_2 = model.energy_and_forces(
            cfg2, weights=weights, forces_mode="torch.autograd"
        )
        assert forces_autograd_1 is not None and forces_autograd_2 is not None
        energy_autograd = torch.cat([energy_autograd_1, energy_autograd_2], dim=1)
        forces_autograd = torch.cat([forces_autograd_1, forces_autograd_2], dim=1)
        energy_autograd_batched, forces_autograd_batched = model.energy_and_forces(
            cfg_batched, weights=weights, forces_mode="torch.autograd"
        )
        torch.testing.assert_close(energy_autograd_batched, energy_autograd, msg=f"Batched energies not equal: actual={energy_autograd_batched}  expected={energy_autograd}")
        torch.testing.assert_close(forces_autograd_batched, forces_autograd, rtol=1e-4, atol=1e-4, msg=f"Batched forces (autograd) not equal: actual={forces_autograd_batched}  expected={forces_autograd}")
        energy_func_batched, forces_func_batched = model.energy_and_forces(
            cfg_batched, weights=weights, forces_mode="torch.func"
        )
        torch.testing.assert_close(energy_func_batched, energy_autograd, msg=f"Batched energies not equal: actual={energy_autograd_batched}  expected={energy_autograd}")
        torch.testing.assert_close(forces_func_batched, forces_autograd, rtol=1e-4, atol=1e-4, msg=f"Batched forces (func) not equal: actual={forces_autograd_batched}  expected={forces_autograd}")

    @pytest.mark.parametrize("gnn_cfg", DEFAULT_GNN_CONFIGS)
    def test_gradients_real(self, rf_cfg, device, multiweights: bool, gnn_cfg):
        model = FrankenPotential(gnn_cfg, rf_cfg).to(device)

        num_lin_models = 10 if multiweights else 1
        weights = torch.randn(
            (num_lin_models, model.rf.total_random_features), device=device
        )
        data_path = DATASET_REGISTRY.get_path("test", "test", None, False)
        dataset = BaseAtomsDataset.from_path(
            data_path=data_path,
            split="train",
            gnn_config=gnn_cfg,
        )
        data, _ = dataset[0]  # type: ignore
        data = data.to(device)
        energy_autograd, forces_autograd = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.autograd"
        )
        energy_func, forces_func = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.func"
        )
        torch.testing.assert_close(energy_func, energy_autograd, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(forces_func, forces_autograd, rtol=1e-3, atol=1e-3)


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
        data = Configuration(
            torch.randn(self.num_atoms, 3, dtype=self.dtype),
            self.atomic_nums,
            torch.tensor(self.num_atoms),
        ).to(device)

        e_ag, f_ag = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.autograd", add_energy_shift=False
        )
        e_ag_shift, f_ag_shift = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.autograd", add_energy_shift=True
        )
        torch.testing.assert_close(f_ag, f_ag_shift)
        torch.testing.assert_close(e_ag + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 5, e_ag_shift) 

        e_fn, f_fn = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.func", add_energy_shift=False
        )
        e_fn_shift, f_fn_shift = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.func", add_energy_shift=True
        )
        torch.testing.assert_close(f_fn, f_fn_shift)
        torch.testing.assert_close(e_fn + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 5, e_fn_shift) 

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
        data = Configuration(
            torch.randn(self.num_atoms, 3, dtype=self.dtype),
            self.atomic_nums,
            torch.tensor(self.num_atoms),
        ).to(device)

        ffmap, efmap = model.grad_feature_map(data)
        ffmap = ffmap.view(ffmap.shape[0], -1)
        e_from_fmaps, f_from_fmaps = model.energy_and_forces_from_fmaps(
            data, energy_fmap=efmap, forces_fmap=ffmap, weights=weights, add_energy_shift=False
        )
        e_from_fmaps_shift, f_from_fmaps_shift = model.energy_and_forces_from_fmaps(
            data, energy_fmap=efmap, forces_fmap=ffmap, weights=weights, add_energy_shift=True
        )
        torch.testing.assert_close(f_from_fmaps, f_from_fmaps_shift)
        torch.testing.assert_close(e_from_fmaps + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 5, e_from_fmaps_shift) 

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
        data = Configuration(
            torch.randn(self.num_atoms, 3, dtype=self.dtype),
            atomic_nums,
            torch.tensor(self.num_atoms),
        ).to(device)

        e_ag, f_ag = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.autograd", add_energy_shift=False
        )
        e_ag_shift, f_ag_shift = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.autograd", add_energy_shift=True
        )
        torch.testing.assert_close(f_ag, f_ag_shift)
        torch.testing.assert_close(e_ag + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 3, e_ag_shift) 

        e_fn, f_fn = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.func", add_energy_shift=False
        )
        e_fn_shift, f_fn_shift = model.energy_and_forces(
            data, weights=weights, forces_mode="torch.func", add_energy_shift=True
        )
        torch.testing.assert_close(f_fn, f_fn_shift)
        torch.testing.assert_close(e_fn + self.atomic_energies[1] * 5 + self.atomic_energies[8] * 3, e_fn_shift) 
    
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
        atomic_nums = torch.randint(1, 100, (self.num_atoms,))
        cfg1 = Configuration(
            torch.randn(self.num_atoms, 3, dtype=self.dtype), atomic_nums, torch.tensor(self.num_atoms)
        ).to(device)
        cfg2 = Configuration(
            torch.randn(self.num_atoms, 3, dtype=self.dtype), atomic_nums, torch.tensor(self.num_atoms)
        ).to(device)
        cfg_batched = Configuration.concatenate([cfg1, cfg2])

        e_ag1, f_ag1 = model.energy_and_forces(
            cfg1, weights=weights, forces_mode="torch.autograd", add_energy_shift=True
        )
        e_ag2, f_ag2 = model.energy_and_forces(
            cfg2, weights=weights, forces_mode="torch.autograd", add_energy_shift=True
        )
        assert f_ag1 is not None and f_ag2 is not None
        e_ag = torch.cat([e_ag1, e_ag2], dim=1)
        f_ag = torch.cat([f_ag1, f_ag2], dim=1)
        e_ag_batch, f_ag_batch = model.energy_and_forces(
            cfg_batched, weights=weights, forces_mode="torch.autograd", add_energy_shift=True
        )
        torch.testing.assert_close(e_ag, e_ag_batch, msg=f"Batched energies not equal: actual={e_ag}  expected={e_ag_batch}")
        torch.testing.assert_close(f_ag, f_ag_batch, rtol=1e-4, atol=1e-4, msg=f"Batched forces not equal: actual={f_ag}  expected={f_ag_batch}")


@pytest.mark.parametrize("gnn_cfg", DEFAULT_GNN_CONFIGS)
@pytest.mark.parametrize("device", DEVICES)
def test_autotune(gnn_cfg, device):
    loaders = init_loaders(
        gnn_cfg,
        DATASET_REGISTRY.get_path("test", "train", None, False),
        test_path=DATASET_REGISTRY.get_path("test", "test", None, False),
        val_path=DATASET_REGISTRY.get_path("test", "val", None, False),
    )
    rf_cfg = GaussianRFConfig(
        num_random_features=128,
        length_scale=HPSearchConfig(values=[0.5, 1.0]),
    )
    solver_cfg = SolverConfig(
        l2_penalty=HPSearchConfig(value=1e-4),
        force_weight=HPSearchConfig(start=0.1, stop=0.9, num=2, scale='linear')
    )
    temp_dir = None
    try:
        # Step 1: Create a temporary directory for saving the model
        temp_dir = Path(create_temp_dir())
        trainer = RandomFeaturesTrainer(
            train_dataloader=loaders["train"],
            random_features_normalization=None,
            log_dir=temp_dir,
            save_every_model=False,
            device=device,
        )
        run_autotune(
            gnn_cfg=gnn_cfg,
            rf_cfg=rf_cfg,
            solver_cfg=solver_cfg,
            loaders=loaders,
            scale_by_species=False,
            jac_chunk_size="auto",
            trainer=trainer
        )
        print(f"{list(temp_dir.glob('*'))}")
        assert (temp_dir / "best.json").is_file()
        assert (temp_dir / "log.json").is_file()
        assert (temp_dir / "best_ckpt.pt").is_file()
        logs = LogCollection.from_json(temp_dir / "log.json")
        assert len(logs) == 4
    finally:
        if temp_dir is not None:
            cleanup_dir(str(temp_dir))


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


class TestFit:
    pass


class TestSerialize:
    pass
