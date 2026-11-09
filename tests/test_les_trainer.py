"""Small optimizer tests, independent of downloaded backbone checkpoints."""

import json
import math
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from franken.data.dataset import FrankenAtomsDataset
from franken.datasets.registry import DATASET_REGISTRY
from tests.test_trainer import make_dloader_from_dset

from .conftest import DEFAULT_GNN_CONFIGS, do_random_seed
from franken.config import GaussianRFConfig, LESConfig, MaceBackboneConfig
from franken.data.base import ENERGY_TARGET_KEY, FORCES_TARGET_KEY, Configuration, Target
from franken.rf.les_model import LESFrankenPotential
from franken.trainers.log_utils import DataSplit, LogEntry
from franken.trainers.rf_les import RandomFeaturesEwaldsTrainer
from tests.utils import mocked_gnn


HPS = {"energy_weight": 1.0, "forces_weight": 0.0, "l2_penalty": 1e-6}


def init_mock_model(lescfg, efmaps, device, dtype, feature_dim=32):
    # We need the following from the potential model
    # - model.rf.total_random_features
    # - model.grad_feature_map(data)
    with mocked_gnn(device=device, dtype=dtype, feature_dim=feature_dim):
        underlying = LESFrankenPotential(
            les_config=lescfg, 
            gnn_config=MaceBackboneConfig("invalid"),  # pyright: ignore[reportArgumentType]
            rf_config=GaussianRFConfig(num_random_features=64, length_scale=1.0),
            scale_by_Z=False
        ).to(device=device, dtype=dtype)
        model = Mock(wraps=underlying)
        model.rf = MagicMock(spec=underlying.rf)
        model.rf.weights.shape = underlying.rf.weights.shape
        model.hyperparameters = underlying.hyperparameters
        model.training = underlying.training
        if efmaps is not None:
            # will iterate through efmap_vals at every call
            model.grad_feature_map = MagicMock(side_effect=efmaps)
            model.rf.total_random_features = efmaps[0][FORCES_TARGET_KEY].shape[0]
        else:
            model.rf.total_random_features = underlying.rf.total_random_features
        return model

    
def make_trainer(
    cfg: LESConfig,
    device="cpu",
    targets=[ENERGY_TARGET_KEY, FORCES_TARGET_KEY],
    dtype=torch.float32,
    save_fmaps=True,
    dataset=None,
    n=5,
    log_dir=None,
):
    if dataset is None:
        dataset = [(
                Configuration(
                    torch.tensor([[float(i), float(i) + math.sin(float(i)), math.cos(i)]], dtype=torch.float64),
                    torch.tensor([1]),
                    torch.tensor([1]),
                ),
                Target(torch.tensor([2.0 * i], dtype=torch.float64), torch.ones(1, 3) * i),
            )
            for i in range(1, n + 1)
        ]
    dloader = make_dloader_from_dset(dataset)
    trainer = RandomFeaturesEwaldsTrainer(
        train_dataloader=dloader,
        training_targets=targets,
        log_dir=log_dir,
        l2_penalty=1e-5,
        target_weight={ENERGY_TARGET_KEY: 1.0, FORCES_TARGET_KEY: 1.0},
        optimizer=cfg.optimizer,
        num_cycles=cfg.num_cycles, # type: ignore
        adam_epochs_per_cycle=cfg.adam_epochs_per_cycle, # type: ignore
        adam_batch_size=cfg.adam_batch_size,
        adam_learning_rate=cfg.adam_learning_rate, # type: ignore
        lr_decay=cfg.lr_decay, # type: ignore
        lbfgs_max_iter=cfg.lbfgs_max_iter, # type: ignore
        lbfgs_max_eval=cfg.lbfgs_max_eval, # type: ignore
        lbfgs_learning_rate=cfg.lbfgs_learning_rate, # type: ignore
        lbfgs_history_size=cfg.lbfgs_history_size, # type: ignore
        lbfgs_tolerance_grad=cfg.lbfgs_tolerance_grad, # type: ignore
        lbfgs_tolerance_change=cfg.lbfgs_tolerance_change, # type: ignore
        mode=cfg.mode,
        random_features_normalization=None,#"leading_eig",
        save_every_model=False,
        save_fmaps=save_fmaps,
        dtype=dtype,
        device=device,
    )
    return trainer
    # return Mock(wraps=trainer)


@pytest.mark.parametrize("gnn_cfg", DEFAULT_GNN_CONFIGS)
def test_varproj_maxiter(gnn_cfg):
    rf_cfg = GaussianRFConfig(num_random_features=32, length_scale=1.0)
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = FrankenAtomsDataset(
        data_path=data_path,
        split="train",
        gnn_config=gnn_cfg,
    )
    do_random_seed(1)
    tr1 = make_trainer(LESConfig(
        mode="variable_projection", num_cycles=1, lbfgs_max_iter=2, lbfgs_max_eval=50), 
        dataset=dataset
    )
    m1 = LESFrankenPotential(
        gnn_config=gnn_cfg,
        rf_config=rf_cfg,
        les_config=LESConfig(),
        num_species=dataset.num_species, # type: ignore
    )
    lc1, w1 = tr1.fit(m1)
    do_random_seed(1)
    print("\n")
    tr2 = make_trainer(LESConfig(
        mode="variable_projection", num_cycles=2, lbfgs_max_iter=1, lbfgs_max_eval=50), 
        dataset=dataset
    )
    m2 = LESFrankenPotential(
        gnn_config=gnn_cfg,
        rf_config=rf_cfg,
        les_config=LESConfig(),
        num_species=dataset.num_species, # type: ignore
    )
    lc2, w2 = tr2.fit(m2)
    do_random_seed(1)
    tr3 = make_trainer(LESConfig(mode="variable_projection", num_cycles=1, lbfgs_max_iter=1), dataset=dataset)
    m3 = LESFrankenPotential(
        gnn_config=gnn_cfg,
        rf_config=rf_cfg,
        les_config=LESConfig(),
        num_species=dataset.num_species, # type: ignore
    )
    lc3, w3 = tr3.fit(m3)
    torch.testing.assert_close(w1, w2)
    assert not torch.all(torch.isclose(w1, w3))


def test_no_save_fmaps():
    num_sys = 2
    lescfg = LESConfig(optimizer="lbfgs", mode="variable_projection", num_cycles=3, lbfgs_max_eval=1)
    with pytest.warns(UserWarning, match="save_fmaps is False. This requires many extra passes"):
        trainer = make_trainer(lescfg, save_fmaps=False, n=num_sys)
    model = init_mock_model(lescfg, efmaps=None, device="cpu", dtype=torch.float64)
    call_count = 0
    def wait(out, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return out
    with (
        patch("franken.trainers.rf_les.torch.cholesky_solve", new=wait),
        patch("franken.trainers.rf_les.torch.linalg.cholesky", new=lambda x, *args, **kwargs: x),
    ):
        trainer.fit(model)
    # one at start, one in each les solve
    expected_call_count = 1 + lescfg.num_cycles
    assert call_count == expected_call_count
    # the first one + for each cycle one in the closure, one outside
    assert model.grad_feature_map.call_count == num_sys + num_sys * lescfg.num_cycles * 2


@pytest.mark.parametrize("num_cycles", [1, 2])
def test_time_logs(num_cycles):
    num_sys = 2
    lescfg = LESConfig(optimizer="lbfgs", mode="variable_projection", num_cycles=num_cycles, lbfgs_max_eval=1)
    trainer = make_trainer(
        lescfg, save_fmaps=True, n=num_sys
    )
    model = init_mock_model(lescfg, efmaps=None, device="cpu", dtype=torch.float64)
    call_count = 0
    def wait(out, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.5)
        return out
    with (
        patch("franken.trainers.rf_les.torch.cholesky_solve", new=wait),
        patch("franken.trainers.rf_les.torch.linalg.cholesky", new=lambda x, *args, **kwargs: x),
    ):
        logc, weights = trainer.fit(model)
    expected_call_count = 1 + num_cycles  # one at start, one in each les solve
    assert call_count == expected_call_count
    times = logc.logs[0].timings
    for t in times:
        if t.name == "rff_solve":
            assert t.value > expected_call_count * 0.5
    # only called at the beginning.
    assert model.grad_feature_map.call_count == num_sys


class TestLESTrainingHistory():
    def make_trainer(self, log_dir):
        trainer = make_trainer(LESConfig(), log_dir=log_dir, save_fmaps=True)
        trainer.training_history = []
        trainer.train_dataloader = Mock(name="train_loader")
        trainer.val_dataloader = Mock(name="validation_loader")
        trainer.create_log_entry = Mock(
            side_effect=lambda *args: LogEntry("model123", 0)
        )
        trainer.eval_summary = Mock(return_value="evaluation")

        def evaluate(model, loader, log_collection, all_weights):
            split = (
                DataSplit.TRAIN
                if loader is trainer.train_dataloader
                else DataSplit.VALIDATION
            )
            for name, value in (("energy_RMSE", 1.5), ("forces_RMSE", 20.0)):
                log_collection[0].add_metric(name, value, split)
            return log_collection

        trainer.evaluate = Mock(side_effect=evaluate)
        return trainer

    def test_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = init_mock_model(LESConfig(), efmaps=None, device="cpu", dtype=torch.float64)
            trainer = self.make_trainer(Path(tmp) / "run")
            path = Path(tmp) / "run" / "training_history.json"
            trainer._print_eval(model, 0, step="rff")
            first = json.loads(path.read_text())
            assert first == trainer.training_history
            assert len(trainer.training_history) == 1

            trainer._print_eval(model, 0, step="les")
            trainer._print_eval(model, 1, step="rff")
            saved = json.loads(path.read_text())
            assert saved == trainer.training_history
            assert (
                [(row["cycle"], row["step"]) for row in saved]
                == [(1, "rff"), (1, "les"), (2, "rff")]
            )
            for row in saved:
                for split in ("train", "validation"):
                    assert row["metrics"][split]["energy_RMSE"] == 1.5
                    assert row["metrics"][split]["forces_RMSE"] == 20.0
            assert not path.with_suffix(".json.tmp").exists()

    def test_no_logdir(self):
        trainer = self.make_trainer(None)
        trainer.val_dataloader = None
        trainer._print_eval(Mock(), 0, step="joint")
        assert trainer.evaluate.call_count == 1
        assert trainer.training_history[0]["step"] == "joint"
        assert set(trainer.training_history[0]["metrics"]) == {"train"}

    @patch("franken.trainers.rf_les.dist_utils.get_rank", return_value=1)
    def test_other_ranks(self, rank):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer._print_eval(Mock(), 0, step="les")
            assert trainer.evaluate.call_count == 2
            assert trainer.training_history == []
            assert list(Path(tmp).iterdir()) == []


def test_serialize_best_model(log_collection):
    with (
        mocked_gnn("cpu", torch.float32),
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        temp_dir = Path(temp_dir)
        with open(temp_dir / "training_history.json", "w") as fh:
            json.dump("test", fh)

        trainer = RandomFeaturesEwaldsTrainer(
            train_dataloader=None, # type: ignore
            training_targets=[ENERGY_TARGET_KEY, FORCES_TARGET_KEY],
            l2_penalty=1e-6,
            target_weight={},
            random_features_normalization=None,
            log_dir=temp_dir,
            save_every_model=False,
            device="cpu",
            dtype=torch.float32,
            optimizer="adam",
            mode="alternating",
            num_cycles=1,
            adam_epochs_per_cycle=1,
            adam_batch_size=1,
            adam_learning_rate=1,
            lr_decay=1,
            lbfgs_history_size=1,
            lbfgs_learning_rate=1,
            lbfgs_max_iter=1,
            lbfgs_max_eval=1,
            lbfgs_tolerance_change=1,
            lbfgs_tolerance_grad=1
        )
        model = LESFrankenPotential(
            gnn_config=MaceBackboneConfig("test"),
            rf_config=GaussianRFConfig(num_random_features=1024, length_scale=1.0),
            les_config=LESConfig(),
        ).to("cpu")
        all_weights = torch.randn(2, 1024, dtype=trainer.buffer_dt)
        # 1. don't need to save
        with patch.object(trainer, "has_new_best_model") as trainer_patch:
            trainer_patch.return_value = (False, log_collection[1])
            should_save = trainer.serialize_best_model(
                model=model,
                all_weights=all_weights,
                split=DataSplit.VALIDATION,
                best_model_selection=["energy_MAE", "forces_MAE"],
            )
            assert not should_save
            assert (temp_dir / "training_history.json").exists()
            assert not (temp_dir / "best_training_history.json").exists()
        # 2. need to save
        with patch.object(trainer, "has_new_best_model") as trainer_patch:
            trainer_patch.return_value = (True, log_collection[1])
            should_save = trainer.serialize_best_model(
                model=model,
                all_weights=all_weights,
                split=DataSplit.VALIDATION,
                best_model_selection=["energy_MAE", "forces_MAE"],
            )
            assert should_save
            assert (temp_dir / "training_history.json").exists()
            assert (temp_dir / "best_training_history.json").exists()
            with open(temp_dir / "best_training_history.json", "r") as fh:
                assert fh.readline() == '"test"'


@pytest.mark.parametrize("save_fmaps", [False, True])
def test_preserve_precision(save_fmaps):
    """The force must use A_F @ w, not a rounded backward pass through RFF."""
    dt = torch.float64  # TODO: Test that it doesn't work in f32

    class CancellationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rf = torch.nn.Module()
            self.rf.weights = torch.nn.Parameter(
                torch.tensor([[1e8, -1e8 + 1]], dtype=dt)
            )
            self.les = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.les.weight)

        def grad_feature_map(self, data, targets):
            return {"forces": -torch.ones(2, 3)}

        def predict_les(self, data, targets):
            data.atom_pos.requires_grad_(True)
            energy = self.les.weight.sum() * data.atom_pos.sum()
            force = -torch.autograd.grad(
                energy, data.atom_pos, create_graph=True
            )[0]
            return {"forces": force.unsqueeze(0)}

        def predict(self, targets, data, is_training=False, **kwargs):
            data.atom_pos.requires_grad_(True)
            features = torch.stack([data.atom_pos.sum(), data.atom_pos.sum()])
            energy = (self.rf.weights @ features.to(dtype=dt)).sum()
            energy = energy + self.les.weight.sum() * data.atom_pos.sum()
            force = -torch.autograd.grad(
                energy, data.atom_pos, create_graph=is_training
            )[0]
            return {"forces": force.unsqueeze(0)}

    data = Configuration(torch.ones(1, 3), torch.tensor([1]), torch.tensor([1]))
    dataset = [(data, Target(energy=torch.zeros(1), forces=torch.zeros(1, 3)))]
    trainer = make_trainer(
        LESConfig(optimizer="lbfgs", mode="variable_projection"), 
        device="cpu",
        targets=[FORCES_TARGET_KEY],
        dtype=dt,
        dataset=dataset,
        save_fmaps=save_fmaps
    )
    model = CancellationModel()
    # If dtype is propagated correctly we should see a difference 
    # between f32 and f64. With f32, the product of [1e8, -1e8 + 1] (RFs)
    # and the data ([[-1, -1, -1], [-1, -1, -1]]) rounds to zero (not
    # enough precision to resolve the +1). With f64 it does not.
    torch.testing.assert_close(
        model.predict(["forces"], data)["forces"], torch.zeros(1, 1, 3)
    )
    if save_fmaps:
        trainer.fmaps = {"forces": [-torch.ones(2, 3, dtype=dt)]}
    loss = trainer._full_gradient(model, None)
    torch.testing.assert_close(loss, torch.tensor(3.0, dtype=dt))
    torch.testing.assert_close(model.les.weight.grad, torch.tensor([[6.0]]))
    assert model.rf.weights.grad is None


@pytest.mark.parametrize("periodic", [False, True])
def test_les_force_and_parameter_derivatives(periodic):
    Les = pytest.importorskip("les").Les
    head = Les(
        les_arguments={
            "use_atomwise": True,
            "n_hidden": [4, 2],
            "output_scaling_factor": 1.0,
        }
    )
    head.atomwise(torch.zeros(1, 3), torch.zeros(1, dtype=torch.long))
    head = head.double()
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
        return head(positions=pos, desc=pos.square(), cell=data.cell.unsqueeze(0))[
            "E_lr"
        ].sum()

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
    parameter = head.atomwise.linear_nn.linear.weight
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
