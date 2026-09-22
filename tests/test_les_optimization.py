"""Small optimizer tests, independent of downloaded backbone checkpoints."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from franken.autotune.cli import parse_cli
from franken.config import LESTrainingConfig
from franken.data.base import Configuration, Target
from franken.trainers.log_utils import DataSplit, LogEntry
from franken.trainers.rf_ewalds import RandomFeaturesEwaldsTrainer


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.les = torch.nn.Linear(1, 1, bias=False).double()
        self.gnn = torch.nn.Linear(1, 1, bias=False).double()
        self.rf = torch.nn.Module()
        self.rf.weights = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
        with torch.no_grad():
            self.les.weight.zero_()
            self.gnn.weight.fill_(1)
        self.requested = []

    def predict(self, targets, data, **kwargs):
        self.requested.append(targets)
        energy = self.les(self.gnn(data.atom_pos[:, :1])).sum() + self.rf.weights.sum()
        result = {"energy": energy.reshape(1, 1)}
        if "forces" in targets:
            result["forces"] = self.les.weight.expand(1, 3)
        return result


def make_trainer(config, n=5):
    dataset = [
        (
            Configuration(
                torch.tensor([[float(i), 0, 0]], dtype=torch.float64),
                torch.tensor([1]),
                torch.tensor([1]),
            ),
            Target(torch.tensor([2.0 * i], dtype=torch.float64), torch.ones(1, 3)),
        )
        for i in range(1, n + 1)
    ]
    trainer = RandomFeaturesEwaldsTrainer(
        SimpleNamespace(dataset=dataset),
        ["energy", "forces"],
        1e-6,
        {"energy": 1.0, "forces": 0.0},
        device="cpu",
        dtype=torch.float64,
        training_config=config,
        best_model_selection=["energy_MAE"],
    )
    trainer._print_eval = Mock()
    return trainer


HPS = {"energy_weight": 1.0, "forces_weight": 0.0, "l2_penalty": 1e-6}


def test_accumulated_gradient_matches_full_mean_and_partial_batch():
    trainer = make_trainer(LESTrainingConfig())
    model = ToyModel()
    for indices in ([0, 1, 2, 3, 4], [4]):
        model.zero_grad()
        loss = trainer._backward_batch(model, indices, {"energy": 1.0})
        x = torch.tensor([i + 1.0 for i in indices], dtype=torch.float64)
        torch.testing.assert_close(loss, (2 * x).square().mean())
        torch.testing.assert_close(
            model.les.weight.grad.squeeze(), -4 * x.square().mean()
        )


def test_accumulated_gradient_with_energy_and_force_losses():
    trainer = make_trainer(LESTrainingConfig())
    model = ToyModel()
    loss = trainer._backward_batch(
        model, list(range(5)), {"energy": 0.25, "forces": 0.75}
    )
    x = torch.arange(1, 6, dtype=torch.float64)
    torch.testing.assert_close(loss, 0.25 * (2 * x).square().mean() + 0.75)
    torch.testing.assert_close(
        model.les.weight.grad.squeeze(), -x.square().mean() - 1.5
    )


@pytest.mark.parametrize("batch_size,steps", [(None, 2), (2, 6)])
def test_adam_epochs_and_persistent_state(batch_size, steps):
    trainer = make_trainer(
        LESTrainingConfig(
            epochs_per_cycle=2, batch_size=batch_size, learning_rate=0.05, lr_decay=0.5
        )
    )
    model = ToyModel()
    with patch.object(
        trainer, "_backward_batch", wraps=trainer._backward_batch
    ) as batch:
        trainer._fit_les(model, 0, HPS)
        seen = [call.args[1] for call in batch.call_args_list]
    per_epoch = steps // 2
    for start in (0, per_epoch):
        assert sorted(
            i for group in seen[start : start + per_epoch] for i in group
        ) == list(range(5))
    if batch_size == 2:
        assert [len(group) for group in seen] == [2, 2, 1, 2, 2, 1]
    optim = trainer._les_optimizer
    assert optim.state[model.les.weight]["step"].item() == steps
    trainer._fit_les(model, 1, HPS)
    assert trainer._les_optimizer is optim
    assert optim.state[model.les.weight]["step"].item() == 2 * steps
    assert optim.param_groups[0]["lr"] == 0.025
    assert all(targets == ["energy"] for targets in model.requested)
    assert model.gnn.weight.item() == 1.0
    assert model.rf.weights.item() == 0.0
    assert model.gnn.weight.requires_grad  # Original flags restored.


def test_lbfgs_reduces_loss_uses_fixed_dataset_and_resets_history():
    trainer = make_trainer(LESTrainingConfig(optimizer="lbfgs", lbfgs_max_iter=10))
    model = ToyModel()
    real_optimizer = torch.optim.LBFGS
    with patch("torch.optim.LBFGS", wraps=real_optimizer) as construct:
        with patch.object(
            trainer, "_backward_batch", wraps=trainer._backward_batch
        ) as batch:
            trainer._fit_les(model, 0, HPS)
            trainer._fit_les(model, 1, HPS)
        assert construct.call_count == 2
    assert all(call.args[1] == list(range(5)) for call in batch.call_args_list)
    assert model.les.weight.item() == pytest.approx(2.0, abs=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimizer": "bad"},
        {"batch_size": 0},
        {"batch_size": -1},
        {"optimizer": "lbfgs", "batch_size": 2},
        {"num_cycles": 0},
        {"epochs_per_cycle": 0},
        {"learning_rate": float("nan")},
        {"learning_rate": -1},
        {"lbfgs_history_size": 0},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        LESTrainingConfig(**kwargs)


@pytest.mark.parametrize("restore_best", [True, False])
def test_fit_restores_both_components_from_best_rff_stage(tmp_path, restore_best):
    trainer = make_trainer(LESTrainingConfig(num_cycles=2, restore_best=restore_best))
    trainer.log_dir = tmp_path
    model = ToyModel()
    trainer.on_fit_start = Mock()
    trainer.covariances = Mock(return_value=({}, {}, None))
    trainer._fit_rff = Mock(
        side_effect=[
            torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([[3.0]], dtype=torch.float64),
        ]
    )
    trainer.create_log_entry = Mock(side_effect=lambda *a: LogEntry("toy", 0, 0, 0))
    trainer.val_dataloader = object()
    scores = iter([3.0, 4.0, 1.0, 5.0])

    def record(hps, model, weights, epoch, step):
        log = LogEntry("toy", 0, 0, 0)
        log.add_metric("energy_MAE", next(scores), DataSplit.VAL)
        trainer._remember_best(model, weights, log, epoch + 1, step)

    def update_les(model, epoch, rf_hps):
        with torch.no_grad():
            model.les.weight.fill_(10.0 * (epoch + 1))
        record(rf_hps, model, None, epoch, "les")

    trainer._print_eval = record
    trainer._fit_les = update_les
    _, weights = trainer.fit(model)
    assert model.rf.weights.item() == 3.0
    assert weights.item() == 3.0
    assert model.les.weight.item() == (10.0 if restore_best else 20.0)
    assert trainer.best_stage["cycle"] == 2
    assert trainer.best_stage["step"] == "rff"
    assert (tmp_path / "best_stage.json").exists()


def test_best_selection_without_validation_and_nonfinite_metrics():
    trainer = make_trainer(LESTrainingConfig())
    model = ToyModel()
    for value in (2.0, float("nan"), 3.0):
        log = LogEntry("toy", 0, 0, 0)
        log.add_metric("energy_MAE", value, DataSplit.TRAIN)
        trainer._remember_best(model, None, log, 1, "les")
    assert trainer.best_stage["score"] == 2.0
    assert trainer.best_stage["split"] == "train"


def test_cli_training_options():
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
        "8",
        "--les",
    ]
    config = parse_cli(
        args
        + [
            "--les-optimizer",
            "adam",
            "--les-batch-size",
            "3",
            "--les-num-cycles",
            "4",
            "--les-epochs-per-cycle",
            "2",
            "--les-no-restore-best",
        ]
    )
    assert config.les_training.batch_size == 3
    assert config.les_training.num_cycles == 4
    assert config.les_training.epochs_per_cycle == 2
    assert config.les_training.restore_best is False
    defaults = parse_cli(args).les_training
    assert defaults.batch_size is None
    assert defaults.restore_best is True
