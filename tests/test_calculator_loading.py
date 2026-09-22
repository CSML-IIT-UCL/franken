"""Checkpoint dispatch tests without downloaded backbone models."""

from unittest.mock import patch

import ase
import numpy as np
import pytest
import torch

from franken.calculators import FrankenCalculator
from franken.config import GaussianRFConfig, LESConfig, MaceBackboneConfig
from franken.data.base import Configuration
from franken.rf.les_model import LESFrankenPotential
from franken.rf.model import FrankenPotential


class ToyBackbone(torch.nn.Module):
    def feature_dim(self):
        return 4

    def descriptors(self, data: Configuration):
        pos = data.atom_pos
        return torch.cat((pos, pos.sum(dim=1, keepdim=True)), dim=1)

    @torch.jit.export
    def franken_val(self):
        pass


class ToyDataset:
    def __init__(self, **kwargs):
        self.atoms = []

    def add_configuration(self, atoms):
        self.atoms.append(atoms)
        return len(self.atoms) - 1

    def __getitem__(self, index, no_targets=False):
        atoms = self.atoms[index]
        return Configuration(
            atom_pos=torch.tensor(atoms.positions, dtype=torch.float32),
            atomic_numbers=torch.tensor(atoms.numbers),
            natoms=torch.tensor([len(atoms)]),
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
        )


class ScriptedPotential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = ToyBackbone()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, targets: list[str], data: Configuration):
        energy = self.scale * data.atom_pos.square().sum()
        return {"energy": energy.reshape(1, 1)}


@pytest.fixture
def toy_backbone(monkeypatch):
    monkeypatch.setattr("franken.rf.model.load_checkpoint", lambda cfg: ToyBackbone())
    monkeypatch.setattr("franken.calculators.ase_calc.FrankenAtomsDataset", ToyDataset)


@pytest.fixture
def atoms():
    return ase.Atoms("H2", positions=[[0.1, 0.2, 0.3], [1.2, 0.4, 0.5]], cell=[8] * 3)


def make_model(model_cls):
    kwargs = dict(
        gnn_config=MaceBackboneConfig("mace_mp/small"),
        rf_config=GaussianRFConfig(num_random_features=8, length_scale=1.0),
        atomic_energies={1: -1.0},
    )
    if model_cls is LESFrankenPotential:
        kwargs["les_config"] = LESConfig(hidden_dim=(4, 2), les_output_scale=1.0)
    model = model_cls(**kwargs)
    with torch.no_grad():
        model.rf.weights.fill_(0.1)
        if model_cls is LESFrankenPotential:
            # Make the LES contribution nonzero, so dropping it fails comparisons.
            for parameter in model.les.parameters():
                parameter.zero_()
            model.les.outnet[-1].bias.fill_(0.5)
    return model


@pytest.mark.parametrize("model_cls", [FrankenPotential, LESFrankenPotential])
def test_auto_load_and_calculator_roundtrip(toy_backbone, atoms, tmp_path, model_cls):
    model = make_model(model_cls)
    path = tmp_path / "model.pt"
    model.save(path)
    with patch("torch.load", wraps=torch.load) as read:
        loaded = FrankenPotential.load(path, map_location="cpu")
    assert type(loaded) is model_cls
    assert read.call_count == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    atoms.calc = FrankenCalculator(model, device="cpu")
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()
    # The second positional argument must remain the device.
    with patch("torch.load", wraps=torch.load) as read:
        atoms.calc = FrankenCalculator(path, "cpu")
    assert type(atoms.calc.franken) is model_cls
    assert read.call_count == 1
    np.testing.assert_allclose(atoms.get_potential_energy(), expected_energy)
    np.testing.assert_allclose(atoms.get_forces(), expected_forces)


def test_explicit_les_loader_still_works(toy_backbone, tmp_path):
    model = make_model(LESFrankenPotential)
    path = tmp_path / "les.pt"
    model.save(path)
    loaded = LESFrankenPotential.load(path, map_location="cpu")
    assert isinstance(loaded, LESFrankenPotential)
    torch.testing.assert_close(loaded.les.outnet[-1].bias, model.les.outnet[-1].bias)


@pytest.mark.parametrize("has_les", [False, True])
def test_dispatch_preserves_load_options(tmp_path, has_les):
    checkpoint = {"les": {}} if has_les else {}
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    model_cls = LESFrankenPotential if has_les else FrankenPotential
    with patch.object(model_cls, "_from_checkpoint", return_value="loaded") as build:
        result = FrankenPotential.load(
            path, map_location="cpu", rf_weight_id=2, backbone_path_or_id="custom.pt"
        )
    assert result == "loaded"
    build.assert_called_once_with(
        checkpoint,
        map_location="cpu",
        rf_weight_id=2,
        backbone_path_or_id="custom.pt",
    )


@pytest.mark.parametrize("from_path", [False, True])
def test_torchscript_loading_is_preserved(toy_backbone, atoms, tmp_path, from_path):
    model = torch.jit.script(ScriptedPotential())
    source = model
    if from_path:
        source = tmp_path / "scripted.pt"
        torch.jit.save(model, source)
    with patch.object(FrankenPotential, "load") as load:
        atoms.calc = FrankenCalculator(
            source, "cpu", gnn_config=MaceBackboneConfig("mace_mp/small")
        )
        load.assert_not_called()
    np.testing.assert_allclose(
        atoms.get_potential_energy(), 2 * np.square(atoms.positions).sum(), rtol=1e-6
    )
