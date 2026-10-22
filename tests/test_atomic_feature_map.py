import pytest
import torch

from franken.config import GaussianRFConfig
from franken.data import Configuration
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics

from .conftest import DEFAULT_GNN_CONFIGS, DEVICES
from .utils import mocked_gnn


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale_by_Z", [False, True])
def test_model_atomic_feature_map_uses_scaled_descriptors(device, scale_by_Z):
    with mocked_gnn(device, torch.float32, feature_dim=4):
        model = FrankenPotential(
            DEFAULT_GNN_CONFIGS[0],
            GaussianRFConfig(num_random_features=8, length_scale=1.0),
            scale_by_Z=scale_by_Z,
            num_species=2 if scale_by_Z else 1,
        ).to(device)
    numbers = torch.tensor([1, 8, 1, 8], device=device)
    config = Configuration(
        torch.randn(4, 3, device=device, requires_grad=True),
        atomic_numbers=numbers,
        natoms=torch.tensor(4, device=device),
    )
    statistics = Statistics(input_dim=4)
    statistics.update(model.gnn.descriptors(config).detach(), numbers)
    model.input_scaler.set_from_statistics(statistics)
    descriptors = model.input_scaler(
        model.gnn.descriptors(config), atomic_numbers=numbers
    )
    expected = model.rf.atomic_feature_map(descriptors)
    actual = model.atomic_feature_map(config)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (4, 8)
    assert actual.device == config.atom_pos.device
    assert actual.requires_grad
    expected_grad = torch.autograd.grad(expected.sum(), config.atom_pos)[0]
    actual_grad = torch.autograd.grad(actual.sum(), config.atom_pos)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
    assert actual_grad.abs().sum() > 0
