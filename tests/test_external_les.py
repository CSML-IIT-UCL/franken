"""External LES integration, using a backbone that needs no checkpoint."""

import sys

import pytest
import torch

from franken.config import (
    GaussianRFConfig,
    LESConfig,
    MaceBackboneConfig,
)
from franken.data.base import Configuration, Target
from franken.rf.les_model import LESFrankenPotential
from franken.rf.model import FrankenPotential
from franken.trainers.rf_ewalds import RandomFeaturesEwaldsTrainer


class ToyBackbone(torch.nn.Module):
    def feature_dim(self):
        return 4

    def descriptors(self, data):
        return torch.cat((data.atom_pos, data.atom_pos.sum(1, keepdim=True)), dim=1)

    def franken_train(self):
        pass

    def franken_val(self):
        pass


def model_kwargs():
    return dict(
        gnn_config=MaceBackboneConfig("mace_mp/small"),
        rf_config=GaussianRFConfig(num_random_features=8, length_scale=1.0),
    )


@pytest.fixture
def model(monkeypatch, request):
    pytest.importorskip("les")
    monkeypatch.setattr("franken.rf.model.load_checkpoint", lambda cfg: ToyBackbone())
    result = LESFrankenPotential(
        **model_kwargs(),
        les_config=LESConfig(
            hidden_dim=(4, 2), is_periodic=getattr(request, "param", None), N_max=6
        ),
    ).double()
    # Constant, nonzero charges isolate the electrostatic geometry derivatives.
    with torch.no_grad():
        for parameter in result.les.parameters():
            parameter.zero_()
        result.les.atomwise.outnet[-1].linear.bias.fill_(0.5)
    return result


def configuration(periodic=True):
    return Configuration(
        atom_pos=torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.4, 0.5]], dtype=torch.float64),
        atomic_numbers=torch.tensor([1, 1]),
        natoms=torch.tensor([2]),
        cell=torch.eye(3, dtype=torch.float64) * 8 if periodic else None,
        edge_index=torch.empty((0, 2), dtype=torch.long),
        unit_shifts=torch.empty((0, 3), dtype=torch.long),
    )


def test_optional_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "les", None)
    with pytest.raises(ImportError, match=r"franken\[les\]"):
        LESFrankenPotential(**model_kwargs(), les_config=LESConfig())


@pytest.mark.parametrize("model", [None, True, False], indirect=True)
def test_upstream_parameters_materialized_and_roundtrip(model, tmp_path):
    from les import Les

    assert isinstance(model.les, Les)
    assert model.les.atomwise.n_in == 4
    assert model.les.N_max == 6
    assert list(model.les.parameters())
    # Checkpoint reconstruction must work when the model's dtype was changed.
    path = tmp_path / "les.pt"
    model.save(path)
    loaded = FrankenPotential.load(path, map_location="cpu").double()
    data = configuration(periodic=model.les.is_periodic is not False)
    torch.testing.assert_close(
        model.predict_les(data, ["energy", "forces"])["energy"],
        loaded.predict_les(data, ["energy", "forces"])["energy"],
    )


@pytest.mark.parametrize("model", [None, True, False], indirect=True)
@pytest.mark.parametrize("constant_charges", [True, False])
def test_force_finite_difference_and_training_gradients(model, constant_charges):
    if not constant_charges:
        for module in model.les.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
    data = configuration(periodic=model.les.is_periodic is not False)
    result = model.predict_les(data, ["energy", "forces"], is_training=True)
    assert result["energy"].shape == (1, 1)
    assert result["forces"].shape == (1, 2, 3)
    loss = result["energy"].square().sum() + result["forces"].square().sum()
    loss.backward()
    gradient = model.les.atomwise.outnet[-1].linear.bias.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    eps = 1e-5
    plus, minus = data.atom_pos.detach().clone(), data.atom_pos.detach().clone()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    data.atom_pos = plus
    eplus = model.predict_les(data, ["energy"])["energy"]
    data.atom_pos = minus
    eminus = model.predict_les(data, ["energy"])["energy"]
    torch.testing.assert_close(
        result["forces"][0, 0, 0],
        (-(eplus - eminus) / (2 * eps)).squeeze(),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("model", [None, True], indirect=True)
def test_periodic_stress_finite_difference(model):
    data = configuration()
    stress = model.predict_les(data, ["stress"])["stress"][0, 0, 0, 0]
    eps = 1e-5
    energies = []
    for sign in (1, -1):
        strained = configuration()
        strained.atom_pos[:, 0] *= 1 + sign * eps
        strained.cell[:, 0] *= 1 + sign * eps
        energies.append(model.predict_les(strained, ["energy"])["energy"])
    expected = ((energies[0] - energies[1]) / (2 * eps * data.cell.det())).squeeze()
    torch.testing.assert_close(stress, expected, rtol=1e-5, atol=1e-7)


def test_batch_matches_individual_predictions(model):
    first, second = configuration(), configuration()
    second.atom_pos *= 1.2
    batch = Configuration(
        atom_pos=torch.cat((first.atom_pos, second.atom_pos)),
        atomic_numbers=torch.tensor([1, 1, 1, 1]),
        natoms=torch.tensor([2, 2]),
        cell=torch.stack((first.cell, second.cell)),
        batch_ids=torch.tensor([0, 0, 1, 1]),
    )
    for predict in (
        model.predict_les,
        lambda data, targets: model.predict(targets, data),
    ):
        predictions = predict(batch, ["energy", "forces"])
        individual = [predict(d, ["energy", "forces"]) for d in (first, second)]
        for target in ("energy", "forces"):
            torch.testing.assert_close(
                predictions[target], torch.cat([p[target] for p in individual], dim=1)
            )


def test_legacy_checkpoint_has_clear_error(model, tmp_path):
    path = tmp_path / "les.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["les"].pop("backend")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="former custom LES head"):
        FrankenPotential.load(path)


@pytest.mark.parametrize("mode", ["alternating", "variable_projection"])
def test_external_head_with_existing_lbfgs_trainer(model, mode, monkeypatch):
    data = configuration()

    class Dataset(list):
        atomic_energies = {1: 0.0}

    dataset = Dataset([(data, Target(torch.tensor([0.1]), torch.zeros(2, 3)))])
    trainer = RandomFeaturesEwaldsTrainer(
        train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=None),
        training_targets=["energy", "forces"],
        l2_penalty=1e-3,
        target_weight={"energy": 1.0, "forces": 1.0},
        random_features_normalization=None,
        device="cpu",
        dtype=torch.float64,
        les_config=LESConfig(
            optimizer="lbfgs",
            mode=mode,
            num_cycles=1,
            lbfgs_max_iter=2,
            restore_best=False,
        ),
    )
    # This smoke test exercises the fit, leaving metric reporting to its own tests.
    monkeypatch.setattr(trainer, "_print_eval", lambda *args, **kwargs: None)
    logs, weights = trainer.fit(model)
    assert len(logs) == 1
    assert torch.isfinite(weights).all()
    assert torch.isfinite(model.predict_les(data, ["energy"])["energy"]).all()
