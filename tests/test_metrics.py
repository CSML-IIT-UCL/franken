import pytest
import torch

from franken.metrics.registry import metric_registry
import franken.metrics as fm
from franken.data.base import Target, Configuration


def test_registry_init():
    assert hasattr(metric_registry, "_metrics")


def test_available_metrics():
    for name in ["energy_MAE", "forces_MAE", "forces_cosim"]:
        assert name in fm.available_metrics()


def test_register():
    class MockMetric(fm.BaseMetric):
        name = "mock_metric"

    assert "mock_metric" not in fm.available_metrics()
    metric_registry.register()(MockMetric)
    assert "mock_metric" in fm.available_metrics()
    metric_registry._metrics.pop("mock_metric")  # reset state


def test_init_metric():
    class MockMetric(fm.BaseMetric):
        name = "mock_metric"
        def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32):
            super().__init__(device, dtype, units={})
    metric_registry.register()(MockMetric)
    metric = fm.init_metric("mock_metric", torch.device("cpu"))
    assert isinstance(metric, MockMetric)
    metric_registry._metrics.pop("mock_metric")  # reset state


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
        for metric_val in values:
            # NOTE: species metric will have a different name (hence use startswith)
            assert metric_val[0].startswith(metric_name)
            assert metric_val[1].shape[0] == 1

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
        for metric_val in values:
            # NOTE: species metric will have a different name (hence use startswith)
            assert metric_val[0].startswith(metric_name)
            assert metric_val[1].shape[0] == 1

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
        for metric_val in values:
            # NOTE: species metric will have a different name (hence use startswith)
            assert metric_val[0].startswith(metric_name)
            assert metric_val[1].shape[0] == n_models

def random_cfg(num_atoms, atomic_numbers=None):
    if atomic_numbers is not None:
        num_atoms = atomic_numbers.shape[0]
    else:
        atomic_numbers = torch.randint(1, 90, (num_atoms,))
    num_edges = num_atoms * 2
    return Configuration(
        torch.randn(num_atoms, 3),
        atomic_numbers=atomic_numbers,
        natoms=torch.tensor(num_atoms),
        edge_index=torch.randint(0, num_atoms, (num_edges, 2)),
        unit_shifts=torch.randint(1, 5, (num_edges, 3), dtype=torch.int32),
        cell=torch.randn((3, 3))
    )

def random_target(cfg: Configuration, batch_size: int | None) -> Target:
    if batch_size is None:
        return Target(
            torch.randn(1)**2, torch.randn(cfg.atom_pos.shape[0], 3), torch.randn(3, 3)
        )
    else:
        return Target(
            torch.randn(batch_size, 1)**2, torch.randn(batch_size, cfg.atom_pos.shape[0], 3), torch.randn(batch_size, 3, 3)
        )

@pytest.mark.parametrize("metric_name", metric_registry.available_metrics)
@pytest.mark.parametrize("n_models", [1, 10])
def test_batched_configs(metric_name, n_models):
    metric = fm.init_metric(metric_name, torch.device("cpu"))
    n_atoms = 4
    
    cfgs = [random_cfg(n_atoms) for _ in range(5)]
    tgts = [random_target(cfg, None) for cfg in cfgs]
    prds = [random_target(cfg, n_models) for cfg in cfgs]
    cfg_b1 = Configuration.concatenate([cfgs[0], cfgs[1], cfgs[2]])
    tgt_b1 = Target.concatenate([tgts[0], tgts[1], tgts[2]])
    prd_b1 = Target.concatenate([prds[0], prds[1], prds[2]])
    cfg_b2 = Configuration.concatenate([cfgs[3], cfgs[4]])
    tgt_b2 = Target.concatenate([tgts[3], tgts[4]])
    prd_b2 = Target.concatenate([prds[3], prds[4]])
    # batched metric
    metric.update(prd_b1, tgt_b1, cfg_b1)
    metric.update(prd_b2, tgt_b2, cfg_b2)
    vals_batched = metric.compute()
    # single metric
    for i in range(len(cfgs)):
        metric.update(prds[i], tgts[i], cfgs[i])
    vals_indiv = metric.compute()
    assert len(vals_indiv) == len(vals_batched)
    for metric_indiv, metric_batched in zip(vals_indiv, vals_batched):
        assert metric_indiv[0] == metric_batched[0]  # metric name
        torch.testing.assert_close(metric_indiv[1], metric_batched[1], msg=f"Expected {metric_indiv[1]}, found {metric_batched[1]}")


class TestMetricValues:
    natoms = 2

    @pytest.mark.parametrize(
        ("name", "targets", "predictions", "expected"),
        [
            ("energy_MAE", torch.tensor(3.0), torch.tensor([[1.0]]), torch.tensor([1000.0])),
            ("energy_RMSE", torch.tensor(3.0), torch.tensor([[1.0]]), torch.tensor([1000.0])),
        ]
    )
    def test_energy_metrics(self, name, targets, predictions, expected):
        metric = fm.init_metric(name, torch.device("cpu"))
        data = Configuration(
            atom_pos=torch.randn(self.natoms, 3),
            atomic_numbers=torch.randint(0, 40, (self.natoms, )),
            natoms=torch.tensor(self.natoms)
        )
        targets = Target(energy=targets, forces=None)
        predictions = Target(energy=predictions, forces=None)
        metric.update(predictions, targets, data)
        values = metric.compute()
        assert len(values) == 1
        assert values[0][0] == name
        torch.testing.assert_close(values[0][1], expected)

    @pytest.mark.parametrize(
        ("name", "targets", "predictions", "expected"),
        [
            ("forces_MAE", torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), torch.tensor([500.0])),
            ("forces_RMSE",torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), torch.tensor([((5.0 / 6.0) ** 0.5) * 1000.0])),
            ("forces_cosim", torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), torch.tensor([0.0])),
        ]
    )
    def test_forces_metrics(self, name, targets, predictions, expected):
        metric = fm.init_metric(name, torch.device("cpu"))
        data = Configuration(
            atom_pos=torch.randn(self.natoms, 3),
            atomic_numbers=torch.randint(0, 40, (self.natoms, )),
            natoms=torch.tensor(self.natoms)
        )
        targets = Target(energy=None, forces=targets)
        predictions = Target(energy=None, forces=predictions)
        metric.update(predictions, targets, data)
        values = metric.compute()
        assert len(values) == 1
        assert values[0][0] == name
        torch.testing.assert_close(values[0][1], expected)

    @pytest.mark.parametrize(
        ("name", "targets", "predictions", "expected"),
        [(
            "forces_MAE_species",
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            [
                ("forces_MAE_species_1", torch.tensor([(1.0 / 3.0) * 1000.0])),
                ("forces_MAE_species_8", torch.tensor([(2.0 / 3.0) * 1000.0])),
            ]
        ),
        (
            "forces_RMSE_species",
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            torch.tensor([
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            ]),
            [
                ("forces_RMSE_species_1", torch.tensor([((1.0 / 3.0) ** 0.5) * 1000.0, 0.0])),
                ("forces_RMSE_species_8", torch.tensor([((4.0 / 3.0) ** 0.5) * 1000.0, 0.0])),
            ]
        )]
    )
    def test_perspecies_forces_metrics(self, name, targets, predictions, expected):
        metric = fm.init_metric(name, torch.device("cpu"))
        data = Configuration(
            atom_pos=torch.randn(3, 3),
            atomic_numbers=torch.tensor([1, 1, 8]),
            natoms=torch.tensor(3)
        )
        targets = Target(energy=None, forces=targets)
        predictions = Target(energy=None, forces=predictions)
        metric.update(predictions, targets, data)
        values = metric.compute()
        assert len(values) == len(expected)
        for act_m, exp_m in zip(values, expected):
            assert act_m[0] == exp_m[0], "name incorrect"
            torch.testing.assert_close(act_m[1], exp_m[1])
