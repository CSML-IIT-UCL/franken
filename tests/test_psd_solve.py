import pytest
import torch

from franken.trainers.rf_lowmem import LowMemRandomFeaturesTrainer
from franken.utils.linalg.tri import pack_upper, unpack_upper
from tests.conftest import DEVICES, SKIP_NO_CUDA


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [0, 1, 5])
def test_packed_upper(device, n):
    matrix = torch.randn(n, n, device=device, dtype=torch.float64)
    packed = pack_upper(matrix)
    assert packed.shape == (n * (n + 1) // 2,)
    assert packed.untyped_storage().nbytes() == packed.numel() * packed.element_size()
    indices = torch.triu_indices(n, n, device=device)
    torch.testing.assert_close(packed, matrix[indices[0], indices[1]])
    torch.testing.assert_close(unpack_upper(packed, n), torch.triu(matrix))


def test_invalid_packed_upper():
    with pytest.raises(ValueError):
        pack_upper(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        unpack_upper(torch.zeros(4), 2)


@SKIP_NO_CUDA
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("column_contiguous", [False, True])
def test_lowmem_factor_orientation(n, column_contiguous):
    pytest.importorskip("cupy")
    a = torch.randn(n, n, device="cuda", dtype=torch.float64)
    cov = a @ a.T
    expected = cov + torch.eye(n, device="cuda", dtype=torch.float64) * 0.1
    cov = cov.T.contiguous().T if column_contiguous else cov.contiguous()
    trainer = LowMemRandomFeaturesTrainer(
        train_dataloader=None,
        l2_penalty=0.1,
        training_targets=["energy", "forces"],
        target_weight={},
        device="cuda",
    )
    solution, packed = trainer.psd_solve(
        cov,
        torch.ones(n, device="cuda", dtype=torch.float64),
        0.1,
        return_cho_factor=True,
    )
    factor = unpack_upper(packed, n)
    torch.testing.assert_close(
        solution,
        torch.linalg.solve(expected, torch.ones(n, device="cuda", dtype=torch.float64)),
    )
    torch.testing.assert_close(factor.T @ factor, expected)
