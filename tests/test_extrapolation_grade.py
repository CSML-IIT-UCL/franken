import torch

from franken.rf.extrapolation import ActiveSet, ExtrapolationGrade


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
    gamma = ExtrapolationGrade(active_set).grade_configuration(
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
    active_set = ActiveSet.from_matrices(
        {
            1: torch.eye(2),
            8: torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        },
        regularization=0.0,
    )
    scorer = ExtrapolationGrade(active_set)
    atomic_features = torch.tensor([[0.5, -1.0], [1.0, 4.0]])
    atomic_numbers = torch.tensor([1, 8])

    gamma_config, gamma_atoms = scorer.grade_configuration(
        atomic_features,
        atomic_numbers,
        return_atomic=True,
    )

    torch.testing.assert_close(gamma_atoms, torch.tensor([1.0, 2.0], dtype=gamma_atoms.dtype))
    torch.testing.assert_close(gamma_config, torch.tensor(2.0, dtype=gamma_config.dtype))


def test_active_set_with_too_few_rows_is_finite():
    atomic_features = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    atomic_numbers = torch.tensor([6, 6])

    active_set = ExtrapolationGrade.build_active_set_from_features(
        atomic_features,
        atomic_numbers,
    )
    scorer = ExtrapolationGrade(active_set)
    gamma = scorer.grade_configuration(atomic_features, atomic_numbers)

    assert active_set.active_matrices[6].shape == (2, 3)
    assert active_set.inverse_matrices[6].shape == (3, 2)
    assert torch.isfinite(gamma)
    assert gamma <= 1.0 + 1e-12


def test_active_set_roundtrip(tmp_path):
    active_set = ActiveSet.from_matrices({1: torch.eye(2)}, regularization=0.0)
    path = tmp_path / "active_set.pt"

    active_set.save(path)
    loaded = ActiveSet.load(path)

    torch.testing.assert_close(loaded.active_matrices[1], active_set.active_matrices[1])
    torch.testing.assert_close(
        loaded.inverse_matrices[1], active_set.inverse_matrices[1]
    )
    assert loaded.per_species is True
