"""External LES integration, using a backbone that needs no checkpoint."""

import sys

import pytest
import torch

from franken.config import (
    GaussianRFConfig,
    LESConfig,
    MaceBackboneConfig,
)
from franken.data.base import ENERGY_TARGET_KEY, FORCES_TARGET_KEY, Configuration
from franken.rf.les_model import LESFrankenPotential
from franken.rf.model import FrankenPotential
from tests.conftest import DEVICES


class ToyBackbone(torch.nn.Module):
    def feature_dim(self):
        return 4

    def descriptors(self, data):
        return torch.cat((data.atom_pos, data.atom_pos.sum(1, keepdim=True)), dim=1)

    def franken_train(self):
        pass

    def franken_val(self):
        pass


MODEL_KWARGS = dict(
    gnn_config=MaceBackboneConfig("mace_mp/small"),
    rf_config=GaussianRFConfig(num_random_features=8, length_scale=1.0),
)


@pytest.fixture
def model(monkeypatch, request):
    def make_model(is_periodic=None):
        pytest.importorskip("les")
        monkeypatch.setattr("franken.rf.model.load_checkpoint", lambda cfg: ToyBackbone())
        result = LESFrankenPotential(
            **MODEL_KWARGS,
            is_periodic=is_periodic,
            les_config=LESConfig(),
        ).double()
        with torch.no_grad():
            for parameter in result.les.parameters():
                parameter.zero_()
            result.les.atomwise.outnet[-1].linear.bias.fill_(0.5)
        return result
    return make_model


def configuration(periodic=True):
    return Configuration(
        atom_pos=torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.4, 0.5]], dtype=torch.float64),
        atomic_numbers=torch.tensor([1, 1]),
        natoms=torch.tensor([2]),
        cell=torch.eye(3, dtype=torch.float64) * 8 if periodic else None,
        edge_index=torch.empty((0, 2), dtype=torch.long),
        unit_shifts=torch.empty((0, 3), dtype=torch.long),
    )


def test_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "les", None)
    with pytest.raises(ImportError, match=r"franken\[les\]"):
        LESFrankenPotential(**MODEL_KWARGS, les_config=LESConfig())


class TestPeriodic:
    @pytest.mark.parametrize('periodic', [False, None])
    def test_no_cell(self, model, periodic):
        les_pot = model(is_periodic=periodic)
        # Should not raise errors both if the cell is None or if it is not
        data = configuration()
        data.cell = None
        les_pot.predict_les(data, [ENERGY_TARGET_KEY, FORCES_TARGET_KEY])
        data = configuration()
        les_pot.predict_les(data, [ENERGY_TARGET_KEY, FORCES_TARGET_KEY])

    def test_incorrect_data(self, model):
        # This error triggers only with the 'new' code-path which does not detect
        # periodic / non-periodic data.
        les_pot = model(is_periodic=True)
        data = configuration()
        # Cannot invert empty cell
        data.cell = None
        with pytest.raises(torch._C._LinAlgError):
            les_pot.predict_les(data, ["energy", "forces"])


@pytest.mark.parametrize("device", DEVICES)
def test_save_load(model, device, tmp_path):
    from les import Les

    les_pot = model().to(device=device, dtype=torch.float64)
    assert isinstance(les_pot.les, Les)
    assert les_pot.les.atomwise.n_in == 4
    assert les_pot.les.N_max == 10
    assert list(les_pot.les.parameters())
    # Checkpoint reconstruction must work when the model's dtype was changed.
    path = tmp_path / "les.pt"
    les_pot.save(path)
    loaded_pot = FrankenPotential.load(path, map_location="cpu").double()
    assert isinstance(loaded_pot, LESFrankenPotential)
    # Test parameters
    torch.testing.assert_close(les_pot.state_dict(), loaded_pot.state_dict())
    # Test hyperparameters
    assert (
        les_pot.hyperparameters == loaded_pot.hyperparameters
    ), "The hyperparameters of the loaded model do not match the original model."

    # Test predictions
    data = configuration(periodic=True)
    keys = [ENERGY_TARGET_KEY, FORCES_TARGET_KEY]
    orig_preds = les_pot.predict_les(data, keys)
    load_preds = loaded_pot.predict_les(data, keys)
    for k in keys:
        torch.testing.assert_close(orig_preds[k], load_preds[k])


@pytest.mark.parametrize("model", [None, True, False], indirect=True)
@pytest.mark.parametrize("constant_charges", [True, False])
def test_force_finite_difference_and_training_gradients(model, constant_charges):
    les_pot = model().to(dtype=torch.float64)
    if not constant_charges:
        for module in les_pot.les.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
    data = configuration(periodic=les_pot.les.is_periodic is not False)
    result = les_pot.predict_les(data, ["energy", "forces"])
    assert result["energy"].shape == (1, 1)
    assert result["forces"].shape == (1, 2, 3)
    loss = result["energy"].square().sum() + result["forces"].square().sum()
    loss.backward()
    gradient = les_pot.les.atomwise.outnet[-1].linear.bias.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    eps = 1e-5
    plus, minus = data.atom_pos.detach().clone(), data.atom_pos.detach().clone()
    plus[0, 0] += eps
    minus[0, 0] -= eps
    data.atom_pos = plus
    eplus = les_pot.predict_les(data, ["energy"])["energy"]
    data.atom_pos = minus
    eminus = les_pot.predict_les(data, ["energy"])["energy"]
    torch.testing.assert_close(
        result["forces"][0, 0, 0],
        (-(eplus - eminus) / (2 * eps)).squeeze(),
        rtol=1e-5,
        atol=1e-7,
    )


def test_batch_matches_individual_predictions(model):
    les_pot = model().to(dtype=torch.float64)
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
        les_pot.predict_les,
        lambda data, targets: les_pot.predict(targets, data),
    ):
        predictions = predict(batch, ["energy", "forces"])
        individual = [predict(d, ["energy", "forces"]) for d in (first, second)]
        for target in ("energy", "forces"):
            torch.testing.assert_close(
                predictions[target], torch.cat([p[target] for p in individual], dim=1)
            )
