import pytest
import torch

from franken.rf.heads import (
    BiasedOrthogonalRFF,
    Linear,
    MultiScaleOrthogonalRFF,
    OrthogonalRFF,
    TensorSketch,
)

RF_PARAMETRIZE = [
    "poly",
    "gaussian",
    "linear",
    "biased-gaussian",
    "multiscale-gaussian",
]


def init_rf(rf_type: str, *args, **kwargs):
    if rf_type == "poly":
        return TensorSketch(*args, **kwargs)
    elif rf_type == "gaussian":
        return OrthogonalRFF(*args, **kwargs)
    elif rf_type == "linear":
        return Linear(*args, **kwargs)
    elif rf_type == "biased-gaussian":
        return BiasedOrthogonalRFF(*args, **kwargs)
    elif rf_type == "multiscale-gaussian":
        return MultiScaleOrthogonalRFF(*args, **kwargs)
    else:
        raise ValueError(rf_type)


@pytest.mark.parametrize("rf_type", RF_PARAMETRIZE)
class TestDtype:
    @pytest.mark.parametrize("dt", [torch.float32, torch.float64])
    def test_dtype_match(self, dt, rf_type):
        rf = init_rf(
            rf_type,
            input_dim=64,
        )
        data = torch.randn(10, 64, dtype=dt)
        atomic_nums = torch.randint(1, 100, (10,))
        fmap = rf.feature_map(data, atomic_numbers=atomic_nums)
        assert fmap.dtype == dt
        for buf_name, buf in rf.named_buffers():
            if buf_name == "weights":
                # weights not touched by this test so they'll be f32
                assert (
                    buf.dtype == torch.get_default_dtype()
                ), f"weights has unexpected type {buf.dtype}"
            elif buf.numel() > 1 and buf.dtype.is_floating_point:
                assert buf.dtype == dt, f"Buffer {buf_name} has incorrect dtype."


class TestFeatureSizes:
    def test_orff_offset(self):
        rf_offset = init_rf(
            "gaussian", input_dim=32, use_offset=True, num_random_features=128
        )
        rf_no_offset = init_rf(
            "gaussian", input_dim=32, use_offset=False, num_random_features=128
        )
        assert rf_offset.num_random_features == 128
        assert rf_no_offset.num_random_features == 128
        assert rf_offset.total_random_features == 128
        assert rf_no_offset.total_random_features == 256
        assert rf_offset.rff_matrix.shape == (128, 32)
        assert rf_no_offset.rff_matrix.shape == (128, 32)
        assert rf_offset.random_offset.shape == (128,)

    @pytest.mark.parametrize("rf_type", ["poly", "gaussian"])
    def test_per_species_kernel_nonlin1(self, rf_type):
        rf = init_rf(
            rf_type,
            input_dim=32,
            num_random_features=128,
            num_species=4,
            chemically_informed_ratio=None,
        )
        assert rf.num_random_features == 128
        assert rf.total_random_features == 128 * 4

    @pytest.mark.parametrize("rf_type", ["poly", "gaussian"])
    def test_per_species_kernel_nonlin2(self, rf_type):
        rf = init_rf(
            rf_type,
            input_dim=32,
            num_random_features=128,
            num_species=4,
            chemically_informed_ratio=0.4,
        )
        assert rf.num_random_features == 128
        assert rf.total_random_features == 128 * (4 + 1)

    def test_per_species_kernel_lin1(self):
        rf = init_rf(
            "linear", input_dim=32, num_species=4, chemically_informed_ratio=None
        )
        assert rf.num_random_features == 33
        assert rf.total_random_features == 33 * 4

    def test_per_species_kernel_lin2(self):
        rf = init_rf(
            "linear", input_dim=32, num_species=4, chemically_informed_ratio=0.4
        )
        assert rf.num_random_features == 33
        assert rf.total_random_features == (33) * (4 + 1)


class TestEdgeCaseInputs:
    @pytest.mark.parametrize("rf_type", RF_PARAMETRIZE)
    def test_zero_lengthscale(self, rf_type):
        with pytest.raises(ValueError):
            if rf_type != "multiscale-gaussian":
                init_rf(rf_type, input_dim=32, length_scale=0)
            else:
                init_rf(rf_type, input_dim=32, length_scale_low=0)

    @pytest.mark.parametrize("rf_type", RF_PARAMETRIZE)
    def test_negative_lengthscale(self, rf_type):
        with pytest.raises(ValueError):
            if rf_type != "multiscale-gaussian":
                init_rf(rf_type, input_dim=32, length_scale=-1.1)
            else:
                init_rf(rf_type, input_dim=32, length_scale_low=-1.1)


class TestScatterSum:
    @pytest.mark.parametrize("rf_type", ["poly", "gaussian"])
    def test_batched_simple(self, rf_type):
        torch.manual_seed(1)
        rf = init_rf(
            rf_type,
            input_dim=32,
            num_random_features=128,
            num_species=None,
            chemically_informed_ratio=None,
        )
        dt = torch.float32
        data = torch.randn(10, 32, dtype=dt)
        atomic_nums = torch.tensor([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]).long()

        batch_ids = torch.cat([torch.zeros(5), torch.ones(5)]).long()
        fmap_batched = rf.feature_map(data, atomic_numbers=atomic_nums, batch_ids=batch_ids)
        fmap_single = torch.cat([
            rf.feature_map(data[:5], atomic_numbers=atomic_nums[:5]),
            rf.feature_map(data[5:], atomic_numbers=atomic_nums[5:])
        ])
        torch.testing.assert_close(fmap_batched, fmap_single)

    @pytest.mark.parametrize("rf_type", ["poly", "gaussian"])
    def test_batched_perspecies(self, rf_type):
        torch.manual_seed(1)
        rf = init_rf(
            rf_type,
            input_dim=8,
            num_random_features=4,
            num_species=4,
            chemically_informed_ratio=None,
        )
        dt = torch.float32
        data = torch.randn(10, 8, dtype=dt)
        atomic_nums = torch.tensor([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]).long()

        batch_ids = torch.cat([torch.zeros(5), torch.ones(5)]).long()
        fmap_batched = rf.feature_map(data, atomic_numbers=atomic_nums, batch_ids=batch_ids)
        fmap_single = torch.cat([
            rf.feature_map(data[:5], atomic_numbers=atomic_nums[:5]),
            rf.feature_map(data[5:], atomic_numbers=atomic_nums[5:])
        ])
        torch.testing.assert_close(fmap_batched, fmap_single)

    @pytest.mark.parametrize("rf_type", ["poly", "gaussian"])
    def test_batched_chemratio(self, rf_type):
        torch.manual_seed(1)
        rf = init_rf(
            rf_type,
            input_dim=32,
            num_random_features=128,
            num_species=4,
            chemically_informed_ratio=0.2,
        )
        dt = torch.float32
        data = torch.randn(10, 32, dtype=dt)
        atomic_nums = torch.tensor([0, 1, 2, 3, 0, 0, 1, 2, 3, 0]).long()

        batch_ids = torch.cat([torch.zeros(5), torch.ones(5)]).long()
        fmap_batched = rf.feature_map(data, atomic_numbers=atomic_nums, batch_ids=batch_ids)
        fmap_single = torch.cat([
            rf.feature_map(data[:5], atomic_numbers=atomic_nums[:5]),
            rf.feature_map(data[5:], atomic_numbers=atomic_nums[5:])
        ])
        torch.testing.assert_close(fmap_batched, fmap_single)


@pytest.mark.parametrize("rf_type", RF_PARAMETRIZE)
@pytest.mark.parametrize("num_species,kappa", [(None, None), (2, None), (2, 0.3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_atomic_features_reproduce_pooled_map(rf_type, num_species, kappa, dtype):
    rf = init_rf(
        rf_type,
        input_dim=4,
        **({"num_random_features": 8} if rf_type != "linear" else {}),
        num_species=num_species,
        chemically_informed_ratio=kappa,
    )
    descriptors = torch.randn(7, 4, dtype=dtype)
    atomic_numbers = torch.tensor([1, 1, 8, 1, 8, 8, 8])
    batch_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    raw = rf.atomic_feature_map(descriptors)
    assert raw.shape == (7, rf.num_random_features)
    atomic = rf.feature_map(descriptors, atomic_numbers, batch_ids, per_atom=True)
    pooled = rf.feature_map(descriptors, atomic_numbers, batch_ids)
    assert atomic.shape == (7, rf.total_random_features)
    assert atomic.dtype == dtype
    for i in range(2):
        mask = batch_ids == i
        torch.testing.assert_close(atomic[mask].mean(0), pooled[i])
        separate = rf.feature_map(
            descriptors[mask], atomic_numbers[mask], per_atom=True
        )
        torch.testing.assert_close(atomic[mask], separate)


@pytest.mark.parametrize("rf_type", ["gaussian", "multiscale-gaussian"])
def test_atomic_features_without_offset(rf_type):
    rf = init_rf(
        rf_type, input_dim=4, num_random_features=8, use_offset=False, num_species=2
    )
    h = torch.randn(5, 4)
    numbers = torch.tensor([1, 8, 1, 8, 8])
    raw = rf.atomic_feature_map(h)
    pooled = rf.feature_map(h, numbers)
    atomic = rf.feature_map(h, numbers, per_atom=True)
    assert raw.shape == (5, 16)
    assert pooled.shape == (1, 32)
    torch.testing.assert_close(atomic.mean(0), pooled[0])


class _ScriptFeatureMap(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, descriptors: torch.Tensor, numbers: torch.Tensor):
        return self.head.feature_map(descriptors, atomic_numbers=numbers)


@pytest.mark.parametrize("rf_type", RF_PARAMETRIZE)
def test_feature_map_remains_scriptable(rf_type):
    head = init_rf(rf_type, input_dim=4, num_species=2)
    wrapper = _ScriptFeatureMap(head)
    scripted = torch.jit.script(wrapper)
    h = torch.randn(5, 4)
    numbers = torch.tensor([1, 8, 1, 8, 8])
    torch.testing.assert_close(scripted(h, numbers), wrapper(h, numbers))


@pytest.mark.parametrize("kappa", [None, 0.3, 0.0, 1.0])
def test_biased_gaussian_constant_is_pooled_per_species(kappa):
    rf = BiasedOrthogonalRFF(
        input_dim=4,
        num_random_features=8,
        num_species=2,
        chemically_informed_ratio=kappa,
        bias=4.0,
    )
    h = torch.randn(3, 4)
    numbers = torch.tensor([1, 1, 8])
    raw = rf.atomic_feature_map(h)
    torch.testing.assert_close(raw[:, -1], torch.full((3,), 2.0))
    pooled = rf.feature_map(h, numbers).reshape(-1, 8)
    if kappa is None:
        expected = torch.tensor([1.0, 1.0])
    else:
        expected = torch.tensor([2 * (1 - kappa), kappa, kappa])
    torch.testing.assert_close(pooled[:, -1], expected)


@pytest.mark.parametrize("rf_type", ["gaussian", "biased-gaussian"])
def test_atomic_contributions_reproduce_total_energy_and_gradients(rf_type):
    rf = init_rf(rf_type, input_dim=4, num_species=2, chemically_informed_ratio=0.3)
    h = torch.randn(3, 4, requires_grad=True)
    numbers = torch.tensor([1, 1, 8])
    weights = torch.randn(rf.total_random_features)
    atomic = rf.feature_map(h, numbers, per_atom=True)
    pooled = rf.feature_map(h, numbers)
    energy_atomic = (atomic @ weights).sum()
    energy_pooled = len(h) * (pooled @ weights).sum()
    torch.testing.assert_close(energy_atomic, energy_pooled)
    torch.testing.assert_close(
        torch.autograd.grad(energy_atomic, h)[0],
        torch.autograd.grad(energy_pooled, h)[0],
    )


@pytest.mark.parametrize("kappa", [None, 0.3])
def test_atomic_contributions_allow_missing_species_within_batch(kappa):
    rf = Linear(input_dim=2, num_species=2, chemically_informed_ratio=kappa)
    h = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    numbers = torch.tensor([1, 1, 8])
    batch_ids = torch.tensor([0, 0, 1])
    contributions = rf.feature_map(h, numbers, batch_ids, per_atom=True)
    pooled = rf.feature_map(h, numbers, batch_ids)
    torch.testing.assert_close(contributions[:2].mean(0), pooled[0])
    torch.testing.assert_close(contributions[2], pooled[1])
    species_blocks = contributions.reshape(3, -1, 3)
    offset = 0 if kappa is None else 1
    torch.testing.assert_close(species_blocks[:2, offset + 1], torch.zeros(2, 3))
    torch.testing.assert_close(species_blocks[2, offset], torch.zeros(3))
