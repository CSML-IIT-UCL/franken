from unittest.mock import Mock, patch

import pytest
import torch

from franken.config import GaussianRFConfig, MaceBackboneConfig
from franken.rf.model import FrankenPotential
from franken.trainers.log_utils import (
    DataSplit,
    HyperParameterGroup,
    LogCollection,
    LogEntry,
)
from franken.trainers.rf_lowmem import LowMemRandomFeaturesTrainer
from franken.trainers.rf_trainer import RandomFeaturesTrainer
from franken.utils.linalg.tri import pack_upper
from tests.conftest import DEVICES
from tests.utils import mocked_gnn


def make_model(device="cpu"):
    return FrankenPotential(
        MaceBackboneConfig("test"),
        GaussianRFConfig(num_random_features=4, length_scale=1.0),
    ).to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_covs_and_coeffs_cache_roundtrip_with_three_targets(device):
    trainer = RandomFeaturesTrainer(
        train_dataloader=None,
        training_targets=["energy", "forces", "stress"],
        l2_penalty=0.1,
        target_weight={},
        device=device,
    )
    matrices = {}
    for target in trainer.training_targets:
        value = torch.randn(4, 4, device=device)
        matrices[target] = value @ value.T
    coeffs = {target: torch.randn(4, device=device) for target in matrices}

    trainer._offload_covs_and_coeffs(matrices, coeffs)
    with pytest.raises(RuntimeError, match="previous fit"):
        trainer.on_fit_start(None)
    restored_covs, restored_coeffs = trainer._restore_covs_and_coeffs(n_features=4)

    assert set(trainer._covs_cache) == set(matrices)
    assert set(restored_covs) == set(matrices)
    assert set(restored_coeffs) == set(coeffs)
    for target in matrices:
        assert trainer._covs_cache[target].device.type == "cpu"
        assert trainer._coeffs_cache[target].device.type == "cpu"
        torch.testing.assert_close(restored_covs[target], matrices[target])
        torch.testing.assert_close(restored_coeffs[target], coeffs[target])
    trainer._purge_covs_and_coeffs_cache()
    assert trainer._covs_cache is None
    assert trainer._coeffs_cache is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "trainer_cls", [RandomFeaturesTrainer, LowMemRandomFeaturesTrainer]
)
@pytest.mark.parametrize("save_cho_factor", [True, False])
def test_winning_factor(tmp_path, device, dtype, trainer_cls, save_cho_factor):
    trainer = trainer_cls(
        train_dataloader=None,
        training_targets=["energy", "forces"],
        l2_penalty=[0.1, 0.5],
        target_weight={"forces": [1.0, 3.0]},
        random_features_normalization=None,
        log_dir=tmp_path,
        device=device,
        dtype=dtype,
        save_cho_factor=save_cho_factor,
    )
    a = torch.randn(4, 4, device=device, dtype=dtype)
    b = torch.randn(4, 4, device=device, dtype=dtype)
    energy_cov, forces_cov = a @ a.T, b @ b.T
    coeffs = {
        t: torch.randn(4, device=device, dtype=dtype) for t in trainer.training_targets
    }
    if trainer_cls is LowMemRandomFeaturesTrainer:
        shared = energy_cov.triu(1) + forces_cov.tril(-1)
        covs = {
            "energy": (shared, energy_cov.diagonal().clone(), True),
            "forces": (shared, forces_cov.diagonal().clone(), False),
        }
    else:
        covs = {"energy": energy_cov, "forces": forces_cov}

    with mocked_gnn(device, torch.float32):
        model = make_model(device)
        with (
            patch.object(trainer, "on_fit_start"),
            patch.object(trainer, "patch_e3nn"),
            patch.object(trainer, "_covs_and_coeffs", return_value=(covs, coeffs)),
            patch.object(trainer, "solve", wraps=trainer.solve) as solve,
        ):
            logs, weights = trainer.fit(model)
            assert solve.call_count == 4
            assert model._cho_factor is None
            for i, log in enumerate(logs):
                log.add_metric("energy_MAE", 0.0 if i == 1 else 1.0, DataSplit.TRAIN)
            selected = logs[1]
            hp = selected.to_dict()["hyperparameters"]["solver"]
            expected = (
                energy_cov * hp["energy_weight"] + forces_cov * hp["forces_weight"]
            ) / (hp["energy_weight"] + hp["forces_weight"])
            expected = (
                expected + torch.eye(4, device=device, dtype=dtype) * hp["l2_penalty"]
            )

            trainer.serialize_logs(model, logs, weights, ["energy_MAE"])
            assert solve.call_count == 4 + save_cho_factor
            assert trainer._covs_cache is None
            assert trainer._coeffs_cache is None
            if save_cho_factor:
                assert model._cho_factor.shape == (10,)
                factor = model.get_cho_factor()
                torch.testing.assert_close(factor, factor.triu())
                torch.testing.assert_close(
                    factor.T @ factor, expected / hp["l2_penalty"]
                )
            else:
                assert model.get_cho_factor() is None
            torch.testing.assert_close(model.rf.weights, weights[1].reshape(1, -1))
            loaded = FrankenPotential.load(
                tmp_path / "best_ckpt.pt", map_location=device
            )
            if save_cho_factor:
                torch.testing.assert_close(loaded._cho_factor, model._cho_factor)
                loaded.double().cpu()
                assert loaded._cho_factor.dtype == torch.float64
                assert loaded._cho_factor.device.type == "cpu"
            else:
                assert loaded.get_cho_factor() is None

            # Repeating serialization for the same winner does not solve again.
            trainer.serialize_best_model(model, weights, ["energy_MAE"])
            assert solve.call_count == 4 + save_cho_factor

            # A later losing RF trial leaves the previous best checkpoint intact.
            previous_checkpoint = (tmp_path / "best_ckpt.pt").read_bytes()
            other_model = make_model(device)
            other_model.rf_config = GaussianRFConfig(
                num_random_features=4, length_scale=2.0
            )
            other_logs, other_weights = trainer.fit(other_model)
            for log in other_logs:
                log.add_metric("energy_MAE", 2.0, DataSplit.TRAIN)
            trainer.serialize_logs(
                other_model, other_logs, other_weights, ["energy_MAE"]
            )
            assert solve.call_count == 8 + save_cho_factor
            assert trainer._covs_cache is None
            assert trainer._coeffs_cache is None
            assert other_model._cho_factor is None
            assert (tmp_path / "best_ckpt.pt").read_bytes() == previous_checkpoint


@pytest.mark.parametrize("device", DEVICES)
def test_factor_checkpoint_compatibility(tmp_path, device):
    with mocked_gnn(device, torch.float32):
        model = make_model(device)
        assert model.get_cho_factor() is None
        model._cho_factor = pack_upper(torch.eye(4, device=device, dtype=torch.float64))
        model.save(tmp_path / "new.pt")
        checkpoint = torch.load(tmp_path / "new.pt", weights_only=False)
        assert checkpoint["_cho_factor"].shape == (10,)
        assert "_cho_factor" not in model.state_dict()
        checkpoint.pop("_cho_factor")
        torch.save(checkpoint, tmp_path / "old.pt")
        old_model = FrankenPotential.load(tmp_path / "old.pt", map_location=device)
        assert old_model._cho_factor is None
        assert old_model.get_cho_factor() is None

        model.save(tmp_path / "multi.pt", multi_weights=torch.ones(2, 4, device=device))
        multi = FrankenPotential.load(
            tmp_path / "multi.pt", rf_weight_id=1, map_location=device
        )
        assert multi._cho_factor is None

        checkpoint["_cho_factor"] = torch.zeros(9)
        torch.save(checkpoint, tmp_path / "invalid.pt")
        with pytest.raises(ValueError, match="invalid size"):
            FrankenPotential.load(tmp_path / "invalid.pt")


def test_failed_factor_does_not_publish_best(tmp_path):
    trainer = RandomFeaturesTrainer(
        train_dataloader=None,
        training_targets=["energy"],
        l2_penalty=0.1,
        target_weight={},
        log_dir=tmp_path,
        device="cpu",
    )
    with mocked_gnn("cpu", torch.float32):
        model = make_model()
        log = LogEntry(
            "trial",
            0,
            0.0,
            0.0,
            hyperparameters=[
                HyperParameterGroup.from_dict(
                    "solver", {"l2_penalty": 0.1, "energy_weight": 1.0}
                )
            ],
        )
        log.add_metric("energy_MAE", 1.0, DataSplit.TRAIN)
        LogCollection([log]).save_json(tmp_path / "log.json")
        trainer._offload_covs_and_coeffs(
            {"energy": torch.eye(4)},
            {"energy": torch.ones(4)},
        )
        with patch.object(
            trainer, "solve", side_effect=torch.linalg.LinAlgError("failed")
        ):
            with pytest.raises(torch.linalg.LinAlgError):
                trainer.serialize_best_model(model, torch.ones(1, 4), ["energy_MAE"])
        assert trainer._covs_cache is None
        assert trainer._coeffs_cache is None
        assert not (tmp_path / "best.json").exists()
        assert not (tmp_path / "best_ckpt.pt").exists()


@pytest.mark.parametrize("failure_stage", ["logs", "checkpoint"])
def test_early_serialization_failure_purges_cache(tmp_path, failure_stage):
    trainer = RandomFeaturesTrainer(
        train_dataloader=None,
        training_targets=["energy"],
        l2_penalty=0.1,
        target_weight={},
        log_dir=tmp_path,
        device="cpu",
    )
    trainer._offload_covs_and_coeffs(
        {"energy": torch.eye(4)}, {"energy": torch.ones(4)}
    )
    model = Mock()
    logs = LogCollection([LogEntry("trial", 0, 0.0, 0.0, hyperparameters=[])])
    failing_object, method = (
        (logs, "save_json") if failure_stage == "logs" else (model, "save")
    )
    with patch.object(failing_object, method, side_effect=OSError("failed")):
        with pytest.raises(OSError, match="failed"):
            trainer.serialize_logs(model, logs, torch.ones(1, 4), ["energy_MAE"])
    assert trainer._covs_cache is None
    assert trainer._coeffs_cache is None


@pytest.mark.parametrize("penalty", [0.0, -1.0, float("inf"), float("nan")])
def test_prior_normalized_factor_requires_positive_ridge(penalty):
    trainer = RandomFeaturesTrainer(
        train_dataloader=None,
        training_targets=["energy"],
        l2_penalty=penalty,
        target_weight={},
        device="cpu",
    )
    trainer._offload_covs_and_coeffs(
        {"energy": torch.eye(4)}, {"energy": torch.ones(4)}
    )
    winner = LogEntry(
        "trial",
        0,
        0.0,
        0.0,
        hyperparameters=[
            HyperParameterGroup.from_dict("solver", {"l2_penalty": penalty})
        ],
    )
    with patch.object(trainer, "solve") as solve:
        with pytest.raises(ValueError, match="finite positive l2_penalty"):
            trainer._best_cho_factor(Mock(), winner)
        solve.assert_not_called()
