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


if __name__ == "__main__":
    pytest.main()
