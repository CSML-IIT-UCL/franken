"""Small optimizer tests, independent of downloaded backbone checkpoints."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from franken.autotune.cli import parse_cli
from franken.config import LESTrainingConfig
from franken.data.base import Configuration, Target
from franken.les.les_head import LESHead
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


@pytest.mark.parametrize("save_fmaps", [False, True])
def test_projected_force_loss_preserves_linear_solver_precision(save_fmaps):
    """The force must use A_F @ w, not a rounded backward pass through RFF."""

    class CancellationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rf = torch.nn.Module()
            self.rf.weights = torch.nn.Parameter(
                torch.tensor([[1e8, -1e8 + 1]], dtype=torch.float64)
            )
            self.les = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.les.weight)

        def grad_feature_map(self, data, targets):
            return {"forces": -torch.ones(2, 3)}

        def predict_les(self, data, targets, is_training=False):
            data.atom_pos.requires_grad_(True)
            energy = self.les.weight.sum() * data.atom_pos.sum()
            force = -torch.autograd.grad(
                energy, data.atom_pos, create_graph=is_training
            )[0]
            return {"forces": force.unsqueeze(0)}

        def predict(self, targets, data, is_training=False, **kwargs):
            data.atom_pos.requires_grad_(True)
            features = torch.stack([data.atom_pos.sum(), data.atom_pos.sum()])
            energy = (self.rf.weights @ features.double()).sum()
            energy = energy + self.les.weight.sum() * data.atom_pos.sum()
            force = -torch.autograd.grad(
                energy, data.atom_pos, create_graph=is_training
            )[0]
            return {"forces": force.unsqueeze(0)}

    data = Configuration(torch.ones(1, 3), torch.tensor([1]), torch.tensor([1]))
    dataset = [(data, Target(energy=torch.zeros(1), forces=torch.zeros(1, 3)))]
    trainer = RandomFeaturesEwaldsTrainer(
        SimpleNamespace(dataset=dataset),
        ["forces"],
        1e-6,
        {"forces": 1.0},
        device="cpu",
        dtype=torch.float64,
        save_fmaps=save_fmaps,
        training_config=LESTrainingConfig(
            optimizer="lbfgs", mode="variable_projection"
        ),
    )
    model = CancellationModel()
    # The double coefficients differ by one, but their float32 adjoints cancel
    # to zero before reaching the positions in the combined-energy path.
    torch.testing.assert_close(
        model.predict(["forces"], data)["forces"], torch.zeros(1, 1, 3)
    )
    if save_fmaps:
        trainer.fmaps = {"forces": [-torch.ones(2, 3, dtype=torch.float64)]}
    loss = trainer._backward_batch(model, [0], {"forces": 1.0})
    torch.testing.assert_close(loss, torch.tensor(3.0, dtype=torch.float64))
    torch.testing.assert_close(model.les.weight.grad, torch.tensor([[6.0]]))
    assert model.rf.weights.grad is None


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_les_force_and_parameter_derivatives(periodic, activation):
    head = LESHead(
        input_dim=3, hidden_dim=(4, 2), les_output_scale=1.0, activation=activation
    ).double()
    positions = torch.tensor([[0.2, 0.3, 0.1], [1.4, 0.8, 0.5]], dtype=torch.float64)
    data = Configuration(
        positions,
        torch.tensor([1, 1]),
        torch.tensor([2]),
        cell=torch.eye(3, dtype=torch.float64) * (6 if periodic else 0),
    )

    def energy(pos):
        # Position-dependent charges exercise both the explicit Coulomb force
        # and the derivative propagated through atomic descriptors.
        return head(pos.square(), pos, data).sum()

    pos = positions.clone().requires_grad_()
    force = -torch.autograd.grad(energy(pos), pos, create_graph=True)[0]
    direction = torch.randn_like(pos)
    direction /= direction.norm()
    eps = 1e-6
    finite_difference = (
        energy(pos + eps * direction) - energy(pos - eps * direction)
    ) / (2 * eps)
    torch.testing.assert_close(
        finite_difference, -(force * direction).sum(), atol=1e-8, rtol=1e-5
    )

    loss = energy(pos).square() + 0.2 * (force - 0.1).square().sum()
    parameter = head.linear_nn.weight
    gradient = torch.autograd.grad(loss, parameter)[0]
    direction = torch.randn_like(parameter)
    direction /= direction.norm()
    initial = parameter.detach().clone()
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            parameter.copy_(initial + sign * eps * direction)
        e = energy(pos)
        f = -torch.autograd.grad(e, pos)[0]
        values.append(e.detach().square() + 0.2 * (f - 0.1).square().sum())
    with torch.no_grad():
        parameter.copy_(initial)
    torch.testing.assert_close(
        (values[0] - values[1]) / (2 * eps),
        (gradient * direction).sum(),
        atol=1e-8,
        rtol=1e-5,
    )


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
    torch.testing.assert_close(loss, 0.25 * (2 * x).square().mean() + 3 * 0.75)
    torch.testing.assert_close(
        model.les.weight.grad.squeeze(), -x.square().mean() - 4.5
    )


class PolynomialModel(torch.nn.Module):
    """Conservative linear energy with interchangeable short/long-range heads."""

    def __init__(self):
        super().__init__()
        self.rf = torch.nn.Module()
        self.rf.total_random_features = 3
        self.rf.weights = torch.nn.Parameter(torch.zeros(1, 3, dtype=torch.float64))
        self.les = torch.nn.Linear(3, 1, bias=False).double()

    def grad_feature_map(self, data, targets):
        x = data.atom_pos[:, 0]
        energy = torch.stack((torch.ones_like(x), x.square() / 2, x**3 / 3))
        forces = torch.zeros(3, x.numel(), 3, dtype=x.dtype)
        forces[1, :, 0] = -x
        forces[2, :, 0] = -x.square()
        return {
            "energy": energy.mean(1, keepdim=True),
            "forces": forces.flatten(1) / data.natoms,
        }

    def _predict_weights(self, data, weights):
        fmaps = self.grad_feature_map(data, ["energy", "forces"])
        return {
            "energy": data.natoms * (weights @ fmaps["energy"]),
            "forces": (data.natoms * (weights @ fmaps["forces"])).reshape(1, -1, 3),
        }

    def predict(self, targets, data, **kwargs):
        return self._predict_weights(data, self.rf.weights + self.les.weight)

    def predict_les(self, data, targets, **kwargs):
        return self._predict_weights(data, self.les.weight)


@pytest.mark.parametrize("normalization", [None, "leading_eig"])
def test_les_gradient_matches_rff_normal_equations(normalization):
    # Different atom counts catch per-structure normalization errors; multiple
    # force components catch a sum-versus-mean mismatch hidden by energy-only tests.
    dataset = []
    for n in (2, 4, 7):
        positions = torch.zeros(n, 3, dtype=torch.float64)
        positions[:, 0] = torch.linspace(0.2, 1.7, n)
        data = Configuration(
            positions, torch.ones(n, dtype=torch.long), torch.tensor([n])
        )
        target = Target(
            torch.tensor([0.7 * n], dtype=torch.float64),
            torch.randn(n, 3, dtype=torch.float64),
        )
        dataset.append((data, target))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, collate_fn=lambda x: x
    )
    trainer = RandomFeaturesEwaldsTrainer(
        loader,
        ["energy", "forces"],
        0.03,
        {"energy": 1.0, "forces": 0.2},
        random_features_normalization=normalization,
        device="cpu",
        dtype=torch.float64,
    )
    model = PolynomialModel()
    covs, _, norms = trainer.covariances(model, loader)
    trainer._loss_normalization = norms
    coeffs = trainer.residual_coeffs(model, loader, norms)
    weights = {"energy": 1 / 1.2, "forces": 0.2 / 1.2}
    with torch.no_grad():
        model.rf.weights.normal_()
    trainer._backward_batch(model, list(range(len(dataset))), weights)
    expected = sum(
        weights[t] * (covs[t] @ model.rf.weights.detach().flatten() - coeffs[t])
        for t in weights
    ) * (2 / len(dataset))
    torch.testing.assert_close(model.les.weight.grad.flatten(), expected)

    # The actual closed-form residual solve must be a stationary point of the
    # same data loss plus its ridge penalty. The toy heads use identical linear
    # features, so the LES gradient also gives the data gradient w.r.t. RFF.
    solution = trainer.solve(
        covs, coeffs, l2_penalty=0.03, energy_weight=1.0, forces_weight=0.2
    )
    with torch.no_grad():
        model.rf.weights.copy_(solution.reshape(1, -1))
    model.zero_grad()
    trainer._backward_batch(model, list(range(len(dataset))), weights)
    torch.testing.assert_close(
        model.les.weight.grad.flatten() + 2 * 0.03 * solution / len(dataset),
        torch.zeros_like(solution),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("normalization", [None, "leading_eig"])
@pytest.mark.parametrize("save_fmaps", [False, True])
def test_variable_projection_gradient_and_accepted_state(normalization, save_fmaps):
    dataset = []
    for n in (2, 3, 5):
        data = Configuration(
            torch.randn(n, 3, dtype=torch.float64),
            torch.ones(n, dtype=torch.long),
            torch.tensor([n]),
        )
        dataset.append(
            (
                data,
                Target(
                    torch.randn(1, dtype=torch.float64),
                    torch.randn(n, 3, dtype=torch.float64),
                ),
            )
        )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, collate_fn=lambda x: x
    )
    trainer = RandomFeaturesEwaldsTrainer(
        loader,
        ["energy", "forces"],
        0.1,
        {"energy": 1.0, "forces": 0.2},
        device="cpu",
        dtype=torch.float64,
        random_features_normalization=normalization,
        save_fmaps=save_fmaps,
        training_config=LESTrainingConfig(
            optimizer="lbfgs", mode="variable_projection"
        ),
    )
    model = PolynomialModel()
    trainer._print_eval = Mock()
    hps = {"energy_weight": 1.0, "forces_weight": 0.2, "l2_penalty": 0.1}
    weights = {"energy": 1 / 1.2, "forces": 0.2 / 1.2}
    covs, _, norms = trainer.covariances(model, loader)
    trainer._loss_normalization = norms
    trainer._prepare_projection(covs, hps)

    # Independent differentiable elimination of the linear block. Compare the
    # envelope gradient AND the line-search value, including the changing ridge.
    xs, ys = [], []
    for data, target in dataset:
        for key, fmap in model.grad_feature_map(data, list(weights)).items():
            scale = weights[key] ** 0.5
            if norms is not None:
                scale = scale / norms[key].sqrt()
            xs.append(scale * fmap.T)
            ys.append(scale * target[key].flatten() / data.natoms)
    x, y = torch.cat(xs), torch.cat(ys)
    theta = model.les.weight.flatten()
    w = torch.linalg.solve(
        x.T @ x + 0.1 * torch.eye(3, dtype=x.dtype), x.T @ (y - x @ theta)
    )
    expected_loss = (
        (x @ (w + theta) - y).square().sum() + 0.1 * w.square().sum()
    ) / len(dataset)
    expected_gradient = torch.autograd.grad(expected_loss, model.les.weight)[0]
    accepted_theta = model.les.weight.detach().clone()

    def step(closure):
        torch.testing.assert_close(closure(), expected_loss.detach())
        torch.testing.assert_close(model.les.weight.grad, expected_gradient)
        # Simulate a final rejected trial followed by the optimizer restoring
        # its accepted LES parameters. RFF must also be restored to that point.
        with torch.no_grad():
            model.les.weight.add_(0.5)
        closure()
        with torch.no_grad():
            model.les.weight.copy_(accepted_theta)

    with patch("torch.optim.LBFGS") as optimizer:
        optimizer.return_value.step.side_effect = step
        optimizer.return_value.zero_grad.side_effect = model.zero_grad
        trainer._fit_les(model, 0, hps)
    torch.testing.assert_close(model.rf.weights.flatten(), w.detach())

    trainer.on_fit_start = Mock()
    trainer.create_log_entry = Mock(side_effect=lambda *a: LogEntry("toy", 0, 0, 0))
    trainer.training_config.num_cycles = 2
    trainer.training_config.restore_best = False
    objectives = []

    def record(*args, **kwargs):
        combined = (model.rf.weights + model.les.weight).flatten()
        objectives.append(
            (
                (x @ combined - y).square().sum()
                + 0.1 * model.rf.weights.square().sum()
            ).item()
        )

    trainer._print_eval = record
    with patch.object(
        trainer, "_prepare_projection", wraps=trainer._prepare_projection
    ) as prepare:
        trainer.fit(model)
    assert prepare.call_count == 1
    assert len(objectives) == 4
    assert all(b <= a + 1e-10 for a, b in zip(objectives, objectives[1:]))
    assert objectives[-1] < objectives[0]
    assert trainer._projection_factor is None


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
        {"mode": "bad"},
        {"mode": "variable_projection", "optimizer": "adam"},
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
    # Even cycle zero must fit y - LES(initial), rather than the full target.
    assert all(
        "direct_coeffs" not in call.kwargs for call in trainer._fit_rff.call_args_list
    )
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
    projected = parse_cli(
        args
        + [
            "--les-optimizer",
            "lbfgs",
            "--les-mode",
            "variable_projection",
            "--les-activation",
            "silu",
        ]
    )
    assert projected.les_training.mode == "variable_projection"
    assert projected.les.activation == "silu"
