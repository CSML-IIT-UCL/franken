import pytest
import torch


def test_registry_init():
    from franken.metrics.registry import registry

    assert hasattr(registry._instance, "_metrics")


def test_available_metrics():
    import franken.metrics as fm

    for name in ["energy_MAE", "forces_MAE", "forces_cosim"]:
        assert name in fm.available_metrics()


def test_register():
    import franken.metrics as fm
    from franken.metrics.base import BaseMetric

    class MockMetric(BaseMetric):
        pass

    assert "mock_metric" not in fm.available_metrics()
    fm.register("mock_metric", MockMetric)
    assert "mock_metric" in fm.available_metrics()


def test_init_metric():
    import franken.metrics as fm
    from franken.metrics.base import BaseMetric

    class MockMetric(BaseMetric):
        def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
            super().__init__("mock_metric", device, dtype)

    fm.register("mock_metric", MockMetric)
    metric = fm.init_metric("mock_metric", torch.device("cpu"))
    assert isinstance(metric, MockMetric)


@pytest.mark.parametrize(
    (
        "metric_name",
        "targets_energy",
        "targets_forces",
        "pred_energy",
        "pred_forces",
        "atomic_numbers",
        "expected_scalar",
        "expected_entries",
        "expected_shape",
    ),
    [
        (
            "energy_MAE",
            torch.tensor(3.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor(1.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            None,
            1000.0,
            None,
            None,
        ),
        (
            "energy_RMSE",
            torch.tensor(3.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor(1.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            None,
            1000.0,
            None,
            None,
        ),
        (
            "forces_MAE",
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            torch.tensor(0.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            None,
            500.0,
            None,
            None,
        ),
        (
            "forces_RMSE",
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            torch.tensor(0.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            None,
            ((5.0 / 6.0) ** 0.5) * 1000.0,
            None,
            None,
        ),
        (
            "forces_RMSE2",
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            torch.tensor(0.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            None,
            ((5.0 / 6.0) ** 0.5) * 1000.0,
            None,
            None,
        ),
        (
            "forces_cosim",
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            None,
            0.0,
            None,
            None,
        ),
        (
            "forces_MAE_species",
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            torch.tensor(0.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([1, 1, 8], dtype=torch.long),
            None,
            [
                (0, 0, 500.0),
                (0, 1, (1.0 / 3.0) * 1000.0),
                (0, 8, (2.0 / 3.0) * 1000.0),
            ],
            (1, 91),
        ),
        (
            "forces_RMSE_species",
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            torch.tensor([0.0, 0.0]),
            torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                ]
            ),
            torch.tensor([1, 1, 8], dtype=torch.long),
            None,
            [
                (0, 1, ((1.0 / 3.0) ** 0.5) * 1000.0),
                (0, 8, ((4.0 / 3.0) ** 0.5) * 1000.0),
                (0, 0, (((1.0 / 3.0) ** 0.5 + (4.0 / 3.0) ** 0.5) / 2.0) * 1000.0),
                (1, 1, 0.0),
                (1, 8, 0.0),
                (1, 0, 0.0),
            ],
            (2, 91),
        ),
    ],
)
def test_metric_values_parametrized(
    metric_name,
    targets_energy,
    targets_forces,
    pred_energy,
    pred_forces,
    atomic_numbers,
    expected_scalar,
    expected_entries,
    expected_shape,
):
    import franken.metrics as fm
    from franken.data.base import Target

    metric = fm.init_metric(metric_name, torch.device("cpu"))
    targets = Target(energy=targets_energy, forces=targets_forces)
    predictions = Target(energy=pred_energy, forces=pred_forces)

    if atomic_numbers is None:
        metric.update(predictions, targets)
    else:
        assert metric.requires_species
        metric.update(predictions, targets, atomic_numbers=atomic_numbers)

    values = metric.compute()

    if expected_scalar is not None:
        assert values.item() == pytest.approx(expected_scalar)
    else:
        assert expected_shape is not None
        assert values.shape == expected_shape
        assert expected_entries is not None
        for model_idx, species_idx, expected in expected_entries:
            assert values[model_idx, species_idx].item() == pytest.approx(expected)


@pytest.mark.parametrize("metric_name", ["energy_MAE", "energy_RMSE"])
def test_energy_metrics_require_forces(metric_name):
    import franken.metrics as fm
    from franken.data.base import Target

    metric = fm.init_metric(metric_name, torch.device("cpu"))
    targets = Target(energy=torch.tensor(3.0), forces=None)
    predictions = Target(energy=torch.tensor(1.0), forces=None)
    with pytest.raises(NotImplementedError):
        metric.update(predictions, targets)
