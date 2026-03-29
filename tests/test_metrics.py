import pytest
import torch

from franken.metrics.registry import registry
import franken.metrics as fm
from franken.data.base import Target, Configuration

ALL_METRICS = [
    "forces_MAE_species", "forces_RMSE_species", 
    "energy_MAE", "energy_RMSE", 
    "forces_MAE", "forces_RMSE", "forces_RMSE2",
    "forces_cosim"
]


def test_registry_init():
    assert hasattr(registry._instance, "_metrics")


def test_available_metrics():
    for name in ["energy_MAE", "forces_MAE", "forces_cosim"]:
        assert name in fm.available_metrics()


def test_register():
    class MockMetric(fm.BaseMetric):
        pass

    assert "mock_metric" not in fm.available_metrics()
    fm.register("mock_metric", MockMetric)
    assert "mock_metric" in fm.available_metrics()


def test_init_metric():
    class MockMetric(fm.BaseMetric):
        def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
            super().__init__("mock_metric", device, dtype)

    fm.register("mock_metric", MockMetric)
    metric = fm.init_metric("mock_metric", torch.device("cpu"))
    assert isinstance(metric, MockMetric)


@pytest.mark.parametrize("metric_name", ["forces_MAE_species", "forces_RMSE_species", "energy_MAE", "forces_MAE"])
class TestShapes:
    def test_simple(self, metric_name):
        metric = fm.init_metric(metric_name, torch.device("cpu"))
        n_atoms = 4
        data = Configuration(
            atom_pos=torch.randn(n_atoms, 3),
            atomic_numbers=torch.randint(0, 40, (n_atoms, )),
            natoms=torch.tensor(n_atoms)
        )
        targets_energy = torch.tensor([1.0])
        targets_forces = torch.randn(n_atoms, 3)
        pred_energy = torch.randn(1, 1)
        pred_forces = torch.randn(1, n_atoms, 3)
        targets = Target(energy=targets_energy, forces=targets_forces)
        predictions = Target(energy=pred_energy, forces=pred_forces)
        metric.update(predictions, targets, data)
        values = metric.compute()
        assert values.shape[0] == 1

    def test_missing_model_dim(self, metric_name):
        metric = fm.init_metric(metric_name, torch.device("cpu"))
        n_atoms = 4
        data = Configuration(
            atom_pos=torch.randn(n_atoms, 3),
            atomic_numbers=torch.randint(0, 40, (n_atoms, )),
            natoms=torch.tensor(n_atoms)
        )
        targets_energy = torch.tensor([1.0])
        targets_forces = torch.randn(n_atoms, 3)
        pred_energy = torch.randn(1)
        pred_forces = torch.randn(n_atoms, 3)
        targets = Target(energy=targets_energy, forces=targets_forces)
        predictions = Target(energy=pred_energy, forces=pred_forces)
        metric.update(predictions, targets, data)
        values = metric.compute()
        assert values.shape[0] == 1  # should default to 1 model

    def test_multimodel(self, metric_name):
        metric = fm.init_metric(metric_name, torch.device("cpu"))
        n_atoms = 4
        n_models = 10
        data = Configuration(
            atom_pos=torch.randn(n_atoms, 3),
            atomic_numbers=torch.randint(0, 40, (n_atoms, )),
            natoms=torch.tensor(n_atoms)
        )
        targets_energy = torch.tensor([1.0])
        targets_forces = torch.randn(n_atoms, 3)
        pred_energy = torch.randn(n_models, 1)
        pred_forces = torch.randn(n_models, n_atoms, 3)
        targets = Target(energy=targets_energy, forces=targets_forces)
        predictions = Target(energy=pred_energy, forces=pred_forces)
        metric.update(predictions, targets, data)
        values = metric.compute()
        assert values.shape[0] == n_models


@pytest.mark.parametrize("metric_name", ALL_METRICS)
@pytest.mark.parametrize("n_models", [1, 10])
def test_batched_configs(metric_name, n_models):
    metric = fm.init_metric(metric_name, torch.device("cpu"))
    n_atoms = 4
    
    atomic_nums = torch.randint(1, 100, (n_atoms,))
    cfg1 = Configuration(
        torch.randn(n_atoms, 3), atomic_nums, torch.tensor(n_atoms)
    )
    cfg2 = Configuration(
        torch.randn(n_atoms, 3), atomic_nums, torch.tensor(n_atoms)
    )
    cfg_batched = Configuration.concatenate([cfg1, cfg2])
    tgt1 = Target(
        torch.tensor([1.0]), torch.randn(n_atoms, 3)
    )
    tgt2 = Target(
        torch.tensor([2.0]), torch.randn(n_atoms, 3)
    )
    tgt_batched = Target(
        torch.cat([tgt1.energy, tgt2.energy], dim=0), torch.cat([tgt1.forces, tgt2.forces], dim=0)
    )
    prd1 = Target(
        torch.randn(n_models, 1), torch.randn(n_models, n_atoms, 3)
    )
    prd2 = Target(
        torch.randn(n_models, 1), torch.randn(n_models, n_atoms, 3)
    )
    prd_batched = Target(
        torch.cat([prd1.energy, prd2.energy], dim=1), torch.cat([prd1.forces, prd2.forces], dim=1)
    )
    metric.update(prd1, tgt1, cfg1)
    metric.update(prd2, tgt2, cfg2)
    vals_indiv = metric.compute(reset=True)
    metric.update(prd_batched, tgt_batched, cfg_batched)
    vals_batched = metric.compute()
    torch.testing.assert_close(vals_indiv, vals_batched)


@pytest.mark.parametrize(
    (
        "metric_name",
        "targets_energy",
        "targets_forces",
        "pred_energy",
        "pred_forces",
        "expected_scalar",
        "expected_entries",
        "expected_shape",
    ),
    [
        (
            "energy_MAE",
            torch.tensor(3.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            1000.0,
            None,
            None,
        ),
        (
            "energy_RMSE",
            torch.tensor(3.0),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
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
    expected_scalar,
    expected_entries,
    expected_shape,
):
    metric = fm.init_metric(metric_name, torch.device("cpu"))
    n_atoms = targets_forces.shape[0]
    if "species" in metric_name:
        atomic_numbers = torch.tensor([1, 1, 8])
    else:
        atomic_numbers = torch.randint(0, 40, (n_atoms, ))
    data = Configuration(
        atom_pos=torch.randn(n_atoms, 3),
        atomic_numbers=atomic_numbers,
        natoms=torch.tensor(n_atoms)
    )
    targets = Target(energy=targets_energy, forces=targets_forces)
    predictions = Target(energy=pred_energy, forces=pred_forces)
    metric.update(predictions, targets, data)

    values = metric.compute()
    print(f"{values=}")

    if expected_scalar is not None:
        assert values.item() == pytest.approx(expected_scalar)
    else:
        assert expected_shape is not None
        assert values.shape == expected_shape
        assert expected_entries is not None
        for model_idx, species_idx, expected in expected_entries:
            assert values[model_idx, species_idx].item() == pytest.approx(expected)
