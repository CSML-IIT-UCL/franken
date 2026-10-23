import numpy as np
import pytest
from ase import Atoms
from franken.calculators.ase_calc import FrankenCalculator
from franken.data.base import ENERGY_TARGET_KEY
import torch

from franken.rf.extrapolation import ExtrapolationGrade


def test_build_active_set_uses_requested_rows_per_species():
    atomic_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    atomic_numbers = torch.tensor([1, 1, 1, 8, 8])

    active_set = ExtrapolationGrade.build_active_set_from_features(
        atomic_features,
        atomic_numbers,
        max_rows=2,
        regularization=0.0,
    )

    assert active_set.active_matrices[1].shape == (2, 3)
    assert active_set.active_matrices[8].shape == (2, 3)
    assert active_set.inverse_matrices[1].shape == (3, 2)
    assert active_set.inverse_matrices[8].shape == (3, 2)


def test_maxvol_selection_bounds_training_gamma():
    rows = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.2],
            [0.2, 1.0, 1.0],
            [1.0, 0.1, 1.0],
        ],
        dtype=torch.float64,
    )

    selected = ExtrapolationGrade.select_active_rows(
        rows,
        method="maxvol",
        qr_tolerance=1e-12,
        maxvol_tolerance=1.01,
        maxvol_iters=100,
    )
    active_inverse = torch.linalg.pinv(rows[selected])
    gamma = torch.abs(rows @ active_inverse).max()

    assert selected.shape == (rows.shape[1],)
    assert gamma <= 1.01


def test_build_active_set_can_use_maxvol():
    atomic_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.2, 1.5],
        ],
        dtype=torch.float64,
    )
    atomic_numbers = torch.tensor([6, 6, 6, 6])

    active_set = ExtrapolationGrade.build_active_set_from_features(
        atomic_features,
        atomic_numbers,
        regularization=0.0,
        selection_method="maxvol",
    )
    gamma = active_set.grade_configuration(
        atomic_features,
        atomic_numbers,
    )

    assert active_set.selection_method == "maxvol"
    assert active_set.active_matrices[6].shape == (2, 2)
    assert gamma <= 1.01


def test_build_active_set_can_pool_all_species():
    atomic_features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 1.5],
        ],
        dtype=torch.float64,
    )
    atomic_numbers = torch.tensor([1, 1, 8, 8])

    active_set = ExtrapolationGrade.build_active_set_from_features(
        atomic_features,
        atomic_numbers,
        regularization=0.0,
        selection_method="maxvol",
        per_species=False,
    )

    assert active_set.per_species is False
    assert list(active_set.active_matrices) == [0]
    assert active_set.active_matrices[0].shape == (2, 2)
    assert active_set.inverse_matrices[0].shape == (2, 2)


def test_grade_configuration_matches_manual_gamma():
    active_set = ExtrapolationGrade.from_matrices(
        {
            1: torch.eye(2),
            8: torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        },
        regularization=0.0,
    )
    scorer = active_set
    atomic_features = torch.tensor([[0.5, -1.0], [1.0, 4.0]])
    atomic_numbers = torch.tensor([1, 8])

    gamma_config, gamma_atoms = scorer.grade_configuration(
        atomic_features,
        atomic_numbers,
        return_atomic=True,
    )

    torch.testing.assert_close(
        gamma_atoms, torch.tensor([1.0, 2.0], dtype=gamma_atoms.dtype)
    )
    torch.testing.assert_close(
        gamma_config, torch.tensor(2.0, dtype=gamma_config.dtype)
    )


def test_active_set_with_too_few_rows_is_finite():
    atomic_features = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    atomic_numbers = torch.tensor([6, 6])

    active_set = ExtrapolationGrade.build_active_set_from_features(
        atomic_features,
        atomic_numbers,
    )
    scorer = active_set
    gamma = scorer.grade_configuration(atomic_features, atomic_numbers)

    assert active_set.active_matrices[6].shape == (2, 3)
    assert active_set.inverse_matrices[6].shape == (3, 2)
    assert torch.isfinite(gamma)
    assert gamma <= 1.0 + 1e-12


def test_active_set_roundtrip(tmp_path):
    active_set = ExtrapolationGrade.from_matrices({1: torch.eye(2)}, regularization=0.0)
    path = tmp_path / "active_set.pt"

    active_set.save(path)
    loaded = ExtrapolationGrade.load(path)

    torch.testing.assert_close(loaded.active_matrices[1], active_set.active_matrices[1])
    torch.testing.assert_close(
        loaded.inverse_matrices[1], active_set.inverse_matrices[1]
    )
    assert loaded.per_species is True


class _DummyGNN:
    def franken_val(self):
        pass


class _DummyFranken(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = _DummyGNN()
        self.gnn_config = None

    def atomic_feature_map(self, data):
        return torch.eye(
            data.atom_pos.shape[0],
            dtype=data.atom_pos.dtype,
            device=data.atom_pos.device,
        )

    def forward(self, targets, data):
        return {ENERGY_TARGET_KEY: torch.zeros(1, dtype=data.atom_pos.dtype)}


@pytest.mark.parametrize("from_file", [False, True])
def test_calculator_exposes_extrapolation_grade(tmp_path, from_file):
    active_set = ExtrapolationGrade.from_matrices({1: torch.eye(2)}, regularization=0.0)
    if from_file:
        path = tmp_path / "extrapolation.pt"
        active_set.save(path)
        active_set = path
    calculator = FrankenCalculator(
        _DummyFranken(),
        device="cpu",
        extrapolation_grade=active_set,
    )
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])

    calculator.calculate(atoms, properties=["energy", "extrapolation_grade"])

    assert calculator.results["atomic_extrapolation"].shape == (2,)
    assert "extrapolation_grade" in calculator.results
    np.testing.assert_allclose(calculator.results["extrapolation_grade"], 1.0)


def test_global_grades_and_device_conversion():
    scorer = ExtrapolationGrade.from_matrices(
        {0: torch.eye(2)}, regularization=0, per_species=False
    ).to(device="cpu", dtype=torch.float64)
    grade, atomic = scorer.grade_configuration(
        torch.tensor([[0.5, 0.0], [0.0, 2.0]]), torch.tensor([1, 8]), return_atomic=True
    )
    torch.testing.assert_close(atomic, torch.tensor([0.5, 2.0], dtype=torch.float64))
    assert grade == 2
    assert scorer.per_species is False


def test_calculator_without_scorer():
    calculator = FrankenCalculator(_DummyFranken(), device="cpu")
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.75, 0, 0]])
    calculator.calculate(atoms, properties=["energy"])
    assert "extrapolation_grade" not in calculator.results
    with pytest.raises(RuntimeError, match="required"):
        calculator.calculate(atoms, properties=["extrapolation_grade"])


@pytest.mark.parametrize("per_species", [False, True])
def test_build_from_model_matches_precomputed_features(per_species):
    from franken.config import GaussianRFConfig
    from franken.data import Configuration
    from franken.rf.model import FrankenPotential
    from .conftest import DEFAULT_GNN_CONFIGS
    from .utils import mocked_gnn

    with mocked_gnn("cpu", torch.float32):
        model = FrankenPotential(
            DEFAULT_GNN_CONFIGS[0],
            GaussianRFConfig(num_random_features=4, length_scale=1.0),
            scale_by_Z=False,
        )
    config = Configuration(
        torch.randn(8, 3),
        atomic_numbers=torch.tensor([1, 8] * 4),
        natoms=torch.tensor(8),
    )
    descriptors = model.input_scaler(
        model.gnn.descriptors(config), atomic_numbers=config.atomic_numbers
    )
    raw = model.atomic_feature_map(config)
    torch.testing.assert_close(raw, model.rf.atomic_feature_map(descriptors))
    assert not raw.requires_grad
    expected = ExtrapolationGrade.build_active_set_from_features(
        raw, config.atomic_numbers, per_species=per_species
    )
    # Also exercise dataloader-style tuples.
    scorer = ExtrapolationGrade.build_active_set(
        model, [(config, {})], per_species=per_species
    )
    for species in expected.species:
        torch.testing.assert_close(
            scorer.active_matrices[species], expected.active_matrices[species]
        )
    torch.testing.assert_close(
        scorer.grade_model_configuration(model, config),
        expected.grade_configuration(raw, config.atomic_numbers),
    )


@pytest.mark.parametrize("method", ["pivoted-qr", "maxvol"])
def test_selection_progress_without_changing_selection(method, capsys, monkeypatch):
    import franken.rf.extrapolation as module

    postfixes = []
    original_tqdm = module.tqdm

    class RecordingProgress(original_tqdm):
        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            postfixes.append(kwargs)
            return super().set_postfix(ordered_dict, refresh=refresh, **kwargs)

    monkeypatch.setattr(module, "tqdm", RecordingProgress)
    rows = torch.tensor([[2., 0.], [0., 3.], [1., 1.]], dtype=torch.float64)
    numbers = torch.ones(3, dtype=torch.long)
    visible = ExtrapolationGrade.build_active_set_from_features(
        rows, numbers, selection_method=method, show_progress=True,
    )
    output = capsys.readouterr().err
    assert "Selecting active rows (QR)" in output
    assert "2/2" in output
    maxvol_lines = [line for line in output.split("\r") if "MaxVol:" in line]
    if method == "maxvol":
        assert maxvol_lines
        assert all("%" not in line and "/300" not in line for line in maxvol_lines)
        assert "1it" in maxvol_lines[-1]
    else:
        assert not maxvol_lines
    assert "log_volume" not in output
    assert postfixes == []
    hidden = ExtrapolationGrade.build_active_set_from_features(
        rows, numbers, selection_method=method, show_progress=False,
    )
    assert capsys.readouterr().err == ""
    torch.testing.assert_close(visible.active_matrices[1], hidden.active_matrices[1])
