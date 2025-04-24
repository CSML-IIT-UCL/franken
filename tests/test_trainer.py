import json
import logging
import pathlib
from unittest.mock import DEFAULT, MagicMock, Mock, mock_open, patch

import pytest
import torch

from franken.config import GaussianRFConfig, MaceBackboneConfig
from franken.data.base import Configuration, Target
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics
from franken.trainers import RandomFeaturesTrainer
from franken.trainers.log_utils import DataSplit, LogCollection, LogEntry
from tests.utils import mocked_gnn
from .conftest import DEVICES


@pytest.mark.parametrize("device", DEVICES)
class TestTrainer:
    def trainer(self, device, dtype=torch.float32, **kwargs):
        return RandomFeaturesTrainer(
            # it's okay to pass None here as long as we don't test `fit` or `set_statistics`
            train_dataloader=None, # type: ignore
            random_features_normalization=None,
            log_dir=None,
            save_every_model=False,
            device=device,
            dtype=dtype,
            **kwargs
        )

    def gen_efmap_vals(self, num_feat, num_configs, num_atoms, device, dtype):
        return [
            (
                torch.randn(num_feat, num_atoms * 3, device=device, dtype=dtype),
                torch.randn(num_feat, device=device, dtype=dtype),
            )
            for _ in range(num_configs)
        ]

    def gen_data(self, num_configs, num_atoms, dtype, split="train"):
        data = [
            (
                Configuration(torch.randn(num_atoms, 3, dtype=dtype), torch.randint(1, 100, (num_atoms, )), torch.tensor(num_atoms)),
                Target(torch.randn(1) ** 2, torch.randn(num_atoms, 3) ** 2)
            )
            for _ in range(num_configs)
        ]
        def iterator(*args, **kwargs):
            for x in data:
                yield x
        dl_mock = MagicMock()
        dl_mock.__len__ = MagicMock(return_value=num_configs)
        dl_mock.__iter__ = iterator
        dl_mock.dataset.__len__ = MagicMock(return_value=num_configs)
        dl_mock.dataset.split = split
        return dl_mock

    def init_model(self, efmaps, device, dtype):
        # We need the following from the potential model
        # - model.rf.total_random_features
        # - model.grad_feature_map(data)
        with mocked_gnn(device=device, dtype=dtype, feature_dim=32):
            underlying = FrankenPotential(
                    gnn_config="test", # type: ignore
                    rf_config=GaussianRFConfig(num_random_features=64, length_scale=1.0)
                ).to(device)
            model = Mock(wraps=underlying)
            model.rf = Mock(wraps=underlying.rf)
            if efmaps is not None:
                # will iterate through efmap_vals at every call
                model.grad_feature_map = MagicMock(side_effect=efmaps)
                model.rf.total_random_features = efmaps[0][0].shape[0]
            else:
                model.rf.total_random_features = underlying.rf.total_random_features
            return model

    def test_covs_and_coeffs(self, device):
        trainer = self.trainer(device, torch.float32)
        num_configs = 3
        num_atoms = 10
        num_rf = 4
        efmap_vals = self.gen_efmap_vals(num_rf, num_configs, num_atoms, device, trainer.buffer_dt)
        dataloader = self.gen_data(num_configs, num_atoms, trainer.buffer_dt)
        # Expected energy and forces covariance and coeffs
        exp_f_cov = sum([efmap[0] @ efmap[0].T for efmap in efmap_vals])
        exp_f_coef = sum([-efmap[0] @ data[1].forces.to(device).view(-1) / num_atoms for efmap, data in zip(efmap_vals, dataloader)])
        exp_e_cov = sum([torch.outer(efmap[1], efmap[1]) for efmap in efmap_vals])
        exp_e_coef = sum([efmap[1] * data[1].energy.to(device) / num_atoms for efmap, data in zip(efmap_vals, dataloader)])

        model = self.init_model(efmap_vals, device, trainer.buffer_dt)
        trainer._compute_covs_and_coeffs(model, dataloader)
        # Check outputs
        torch.testing.assert_close(
            torch.triu(trainer.covariance, 1), torch.triu(exp_e_cov, 1) # type: ignore
        )
        torch.testing.assert_close(
            trainer.diag_energy, torch.diagonal(exp_e_cov) # type: ignore
        )
        torch.testing.assert_close(
            torch.tril(trainer.covariance, -1), torch.tril(exp_f_cov, -1) # type: ignore
        )
        torch.testing.assert_close(
            trainer.diag_forces, torch.diagonal(exp_f_cov) # type: ignore
        )
        torch.testing.assert_close(
            trainer.coeffs_energy, exp_e_coef
        )
        torch.testing.assert_close(
            trainer.coeffs_forces, exp_f_coef
        )
        # Check behavior
        assert model.grad_feature_map.call_count == num_configs

    def test_solve(self, device):
        # Start by computing covs and coeffs
        trainer = self.trainer(device, torch.float32)
        num_configs = 3
        num_atoms = 10
        num_rf = 32
        loss_lerp_weight = 0.1
        L2_penalty = 1e-6
        efmap_vals = self.gen_efmap_vals(num_rf, num_configs, num_atoms, device, trainer.buffer_dt)
        dataloader = self.gen_data(num_configs, num_atoms, trainer.buffer_dt)
        # Expected energy and forces covariance and coeffs
        exp_f_cov = sum([efmap[0] @ efmap[0].T for efmap in efmap_vals])
        exp_f_coef = sum([-efmap[0] @ data[1].forces.to(device).view(-1) / num_atoms for efmap, data in zip(efmap_vals, dataloader)])
        exp_e_cov = sum([torch.outer(efmap[1], efmap[1]) for efmap in efmap_vals])
        exp_e_coef = sum([efmap[1] * data[1].energy.to(device) / num_atoms for efmap, data in zip(efmap_vals, dataloader)])
        lerped_cov = exp_f_cov * loss_lerp_weight + exp_e_cov * (1 - loss_lerp_weight)
        exp_sol = torch.linalg.solve(
            lerped_cov + torch.eye(num_rf, dtype=trainer.buffer_dt, device=device) * L2_penalty,
            exp_f_coef * loss_lerp_weight + exp_e_coef * (1 - loss_lerp_weight)
        )
        model = self.init_model(efmap_vals, device, trainer.buffer_dt)
        trainer._compute_covs_and_coeffs(model, dataloader)
        solution = trainer.solve(force_weight=loss_lerp_weight, l2_penalty=L2_penalty)
        torch.testing.assert_close(exp_sol, solution)

    def test_compute_train_preds_shortcut(self, device):
        trainer = self.trainer(device, torch.float64, save_fmaps=True)
        num_configs = 5
        num_atoms = 10
        dataloader = self.gen_data(num_configs, num_atoms, trainer.buffer_dt, split="train")
        model = self.init_model(None, device, trainer.buffer_dt)
        trainer._compute_covs_and_coeffs(model, dataloader)
        weights = trainer.solve(force_weight=0.3, l2_penalty=1e-3)
        weights = weights.view(1, -1)  # fit converts to this shape

        logs = LogCollection([LogEntry("", 0, 0, 0, [], [])])
        logs_fast = trainer.evaluate(model, dataloader, logs, weights, ["forces_MAE", "energy_MAE"])
        model.energy_and_forces.assert_not_called()
        model.energy_and_forces_from_fmaps.assert_called()
        model.energy_and_forces_from_fmaps.reset_mock()

        trainer.save_fmaps = False
        logs = LogCollection([LogEntry("", 0, 0, 0, [], [])])
        logs_slow = trainer.evaluate(model, dataloader, logs, weights, ["forces_MAE", "energy_MAE"])
        model.energy_and_forces.assert_called()
        model.energy_and_forces_from_fmaps.assert_not_called()

        metrics_fast = logs_fast[0].metrics
        metrics_slow = logs_slow[0].metrics
        for mfast, mslow in zip(metrics_fast, metrics_slow):
            assert mfast.name == mslow.name
            torch.testing.assert_close(mslow.value, mfast.value, msg=f"Slow and fast computations don't match metric {mfast.name}")


class TestSerializeBestModel:
    @pytest.fixture
    def log_collection(self):
        return LogCollection(logs=[LogEntry.from_dict(log) for log in [{
            "checkpoint": {"hash": "rand_uuid", "rf_weight_id": 0},
            "timings": {"cov_coeffs": 1.0, "solve": 1.0},
            "metrics": {
                "train": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
                "validation": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
                "test": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
            },
            "hyperparameters": {
                "franken": {
                    "gnn_backbone_id": "SchNet-S2EF-OC20-All",
                    "interaction_block": 3,
                    "kernel_type": "gaussian",
                },
                "random_features": {
                    "num_random_features": 1024,
                },
                "input_scaler": {"scale_by_Z": True, "num_species": 2},
                "solver": {
                    "l2_penalty": 1e-6,
                    "force_weight": 0.1,
                    "dtype": "torch.float64",
                },
            },
        },{
            "checkpoint": {"hash": "rand_uuid2", "rf_weight_id": 1},
            "timings": {"cov_coeffs": 1.0, "solve": 1.0},
            "metrics": {
                "train": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
                "validation": {"energy_MAE": 1.0, "forces_MAE": 0.5, "forces_cosim": 1.0},
                "test": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
            },
            "hyperparameters": {
                "franken": {
                    "gnn_backbone_id": "SchNet-S2EF-OC20-All",
                    "interaction_block": 3,
                    "kernel_type": "gaussian",
                },
                "random_features": {
                    "num_random_features": 1024,
                },
                "input_scaler": {"scale_by_Z": True, "num_species": 2},
                "solver": {
                    "l2_penalty": 1e-6,
                    "force_weight": 0.1,
                    "dtype": "torch.float64",
                },
            },
        }]])

    def test_should_save(self, log_collection, caplog):
        with patch.object(pathlib.Path, 'exists') as mock_exists:
            with mocked_gnn("cpu", torch.float32):
                with patch.multiple("franken.trainers.base", LogCollection=DEFAULT) as p2:
                    with patch.multiple("torch", load=DEFAULT, save=DEFAULT) as p3:
                        with patch("builtins.open", mock_open()) as file_open:
                            trainer = RandomFeaturesTrainer(
                                train_dataloader=None, # type: ignore
                                random_features_normalization=None,
                                log_dir=pathlib.Path("test_dir"),
                                save_every_model=False,
                                device="cpu",
                                dtype=torch.float32
                            )
                            model = FrankenPotential(
                                gnn_config=MaceBackboneConfig("test"),
                                rf_config=GaussianRFConfig(num_random_features=1024, length_scale=1.0)
                            ).to("cpu")
                            all_weights = torch.randn(2, 1024, dtype=trainer.buffer_dt)
                            # 1. best_model_file exists but contains same best model as log_collection
                            mock_exists.return_value = True
                            p2["LogCollection"].from_json.return_value = log_collection
                            # log_collection[1] is best model (read will be used to open best_model_file)
                            file_open.return_value.read.return_value = json.dumps(log_collection[1].to_dict())
                            with caplog.at_level(logging.DEBUG):
                                trainer.serialize_best_model(
                                    model=model,
                                    all_weights=all_weights,
                                    split=DataSplit.VALIDATION
                                )
                            p3["save"].assert_not_called()
                            assert caplog.text == ""
                            # 2. best_model_file exists and does not contain same best model as log_collection
                            mock_exists.return_value = True
                            p2["LogCollection"].from_json.return_value = log_collection
                            # log_collection[0] is not best model (read will be used to open best_model_file)
                            file_open.return_value.read.return_value = json.dumps(log_collection[0].to_dict())
                            with caplog.at_level(logging.DEBUG):
                                trainer.serialize_best_model(
                                    model=model,
                                    all_weights=all_weights,
                                    split=DataSplit.VALIDATION
                                )
                            p3["save"].assert_called_once()
                            assert "Identified new best model" in caplog.text
                            assert "Saved best model" in caplog.text
                            assert p3["save"].call_args.args[1] == pathlib.Path("test_dir/best_ckpt.pt")
                            p3["save"].reset_mock()
                            # 3. best_model_file does not exist
                            mock_exists.return_value = False
                            p2["LogCollection"].from_json.return_value = log_collection
                            with caplog.at_level(logging.DEBUG):
                                trainer.serialize_best_model(
                                    model=model,
                                    all_weights=all_weights,
                                    split=DataSplit.VALIDATION
                                )
                            p3["save"].assert_called_once()
                            assert "Identified new best model" in caplog.text
                            assert "Saved best model" in caplog.text
                            assert p3["save"].call_args.args[1] == pathlib.Path("test_dir/best_ckpt.pt")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_correct(self, dtype, log_collection):
        with patch.object(pathlib.Path, 'exists') as mock_exists:
            mock_exists.return_value = False
            trainer = RandomFeaturesTrainer(
                train_dataloader=None, # type: ignore
                random_features_normalization=None,
                log_dir=pathlib.Path("test_dir"),
                save_every_model=False,
                device="cpu",
                dtype=dtype
            )
            all_weights = torch.randn(2, 1024, dtype=trainer.buffer_dt)
            with mocked_gnn("cpu", torch.float32):
                with patch.multiple("franken.trainers.base", LogCollection=DEFAULT) as p2:
                    with patch.multiple("torch", load=DEFAULT, save=DEFAULT) as p3:
                        with patch("builtins.open", mock_open()):
                            p2["LogCollection"].from_json.return_value = log_collection
                            model = FrankenPotential(
                                gnn_config=MaceBackboneConfig("test"),
                                rf_config=GaussianRFConfig(1024, 1.0)
                            ).to("cpu")
                            trainer.serialize_best_model(
                                model=model,
                                all_weights=all_weights,
                                split=DataSplit.VALIDATION
                            )
                            # check model saved with correct parameters.
                            assert p3["save"].call_args.args[1] == pathlib.Path("test_dir/best_ckpt.pt")
                            save_ckpt = p3["save"].call_args.args[0]
                            assert save_ckpt["multi_weights"] is None
                            # check model weights. This checks dtype!
                            torch.testing.assert_close(model.rf.weights, all_weights[1].view(1, -1))


def test_get_statistics():
    dloader = Mock()
    dloader.dataset = Mock()
    with patch.multiple("franken.trainers.base", compute_dataset_statistics=DEFAULT) as p2:
        p2["compute_dataset_statistics"].return_value = Statistics()
        trainer = RandomFeaturesTrainer(
            train_dataloader=dloader, # type: ignore
            random_features_normalization=None,
            log_dir=pathlib.Path("test_dir"),
            save_every_model=False,
            device="cpu",
            dtype=torch.float32
        )
        with mocked_gnn("cpu", torch.float32):
            model = FrankenPotential(
                gnn_config=MaceBackboneConfig("test"),
                rf_config=GaussianRFConfig(num_random_features=64, length_scale=1.0)
            ).to("cpu")
            trainer.get_statistics(model)
            p2["compute_dataset_statistics"].assert_called_once()
            p2["compute_dataset_statistics"].reset_mock()
            trainer.get_statistics(model)
            p2["compute_dataset_statistics"].assert_not_called()
            # Initialize a new GNN (different parameters)
            model = FrankenPotential(
                gnn_config=MaceBackboneConfig("test2"),
                rf_config=GaussianRFConfig(num_random_features=64, length_scale=1.0)
            ).to("cpu")
            trainer.get_statistics(model)
            p2["compute_dataset_statistics"].assert_called_once()
