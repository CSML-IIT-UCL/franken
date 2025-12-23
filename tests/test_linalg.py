from functools import partial
from unittest.mock import DEFAULT, patch

import numpy as np
import pytest
import torch
try:
    import cupy # pyright: ignore[reportMissingImports]  # noqa: F401
    cupy_available = True
except ImportError:
    cupy_available = False

from franken.utils.linalg.cov import (
    _lowmemcov_rank1_update,
    _lowmemcov_rankk_update,
    _rank1_update,
    _rankk_update,
    rank1_update,
    rankk_update
)
from franken.utils.linalg.psdsolve import (
    _lowmem_psd_ridge,
    _naive_psd_ridge,
    psd_ridge
)
from franken.utils.linalg.tri import (
    _trilerp_cpu,
    _trilerp_triton,
    inplace_triangular_divide,
    triangular_lerp
)
from .conftest import (
    DEVICES,
    SKIP_NO_CUDA,
)


FAIL_NO_CUPY_MARK = pytest.mark.xfail(not cupy_available, reason="No CuPy")
DEVICES_FAIL_NOCUPY = [
    "cpu",
    pytest.param("cuda:0", marks=[SKIP_NO_CUDA, FAIL_NO_CUPY_MARK]),  # type: ignore
]


class TestCovUpdates:
    mat_size = 10

    def A(self, device, dtype):
        gen = torch.Generator(device=device)
        gen.manual_seed(37)
        mat = torch.randn(self.mat_size, self.mat_size, device=device, dtype=dtype, generator=gen)
        return (mat @ mat.T) / self.mat_size

    def d(self, device, dtype):
        gen = torch.Generator(device=device)
        gen.manual_seed(38)
        diag = torch.randn(self.mat_size, device=device, dtype=dtype, generator=gen)
        return diag

    def vec(self, device, dtype):
        gen = torch.Generator(device=device)
        gen.manual_seed(39)
        vec = torch.randn(self.mat_size, device=device, dtype=dtype, generator=gen)
        return vec

    def veck(self, device, dtype):
        gen = torch.Generator(device=device)
        gen.manual_seed(40)
        veck = torch.randn((self.mat_size, 3), device=device, dtype=dtype, generator=gen)
        return veck

    @pytest.mark.parametrize("device", DEVICES_FAIL_NOCUPY)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("upper", [True, False])
    def test_lowmem_rank1(self, dtype, device, upper):
        A = self.A(device, dtype)
        Acp = torch.clone(A)
        d = self.d(device, dtype)
        vec = self.vec(device, dtype)

        tri_func = partial(torch.triu, diagonal=1) if upper else partial(torch.tril, diagonal=-1)
        other_tri_func = partial(torch.tril, diagonal=-1) if upper else partial(torch.triu, diagonal=1)
        exp_cov = A + tri_func(torch.outer(vec, vec))
        exp_diag = d + vec ** 2

        # TEST FUNCTION HERE
        out_cov, out_diag = _lowmemcov_rank1_update(A, d, vec, upper=upper)

        # must be in-place
        assert out_cov.data_ptr() == A.data_ptr()
        assert out_diag.data_ptr() == d.data_ptr()
        # correctness
        torch.testing.assert_close(tri_func(out_cov), tri_func(exp_cov))
        torch.testing.assert_close(out_diag, exp_diag)
        # preserving the other triangular matrix
        torch.testing.assert_close(other_tri_func(out_cov), other_tri_func(Acp))

    @pytest.mark.parametrize("device", DEVICES_FAIL_NOCUPY)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("upper", [True, False])
    def test_lowmem_rankk(self, device, dtype, upper):
        A = self.A(device, dtype)
        Acp = torch.clone(A)
        d = self.d(device, dtype)
        veck = self.veck(device, dtype)

        tri_func = partial(torch.triu, diagonal=1) if upper else partial(torch.tril, diagonal=-1)
        other_tri_func = partial(torch.tril, diagonal=-1) if upper else partial(torch.triu, diagonal=1)
        exp_cov = A + tri_func(veck @ veck.T)
        exp_diag = d + (veck ** 2).sum(1)

        # TEST FUNCTION HERE
        out_cov, out_diag = _lowmemcov_rankk_update(A, d, veck, upper=upper)

        # must be in-place
        assert out_cov.data_ptr() == A.data_ptr()
        assert out_diag.data_ptr() == d.data_ptr()
        # correctness
        torch.testing.assert_close(tri_func(out_cov), tri_func(exp_cov))
        torch.testing.assert_close(out_diag, exp_diag)
        # preserving the other triangular matrix
        torch.testing.assert_close(other_tri_func(out_cov), other_tri_func(Acp))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("upper", [True, False])
    def test_nocupy_rank1(self, dtype, device, upper):
        A = self.A(device, dtype)
        Acp = torch.clone(A)
        d = self.d(device, dtype)
        vec = self.vec(device, dtype)

        tri_func = partial(torch.triu, diagonal=1) if upper else partial(torch.tril, diagonal=-1)
        other_tri_func = partial(torch.tril, diagonal=-1) if upper else partial(torch.triu, diagonal=1)
        exp_cov = A + tri_func(torch.outer(vec, vec))
        exp_diag = d + vec ** 2

        with patch.multiple(
            "franken.utils.linalg.cov", cupy=None
        ) as _:
            # TEST FUNCTION HERE
            out_cov, out_diag = _rank1_update(A, d, vec, upper=upper)
        # must be in-place
        assert out_cov.data_ptr() == A.data_ptr()
        assert out_diag.data_ptr() == d.data_ptr()
        # correctness
        torch.testing.assert_close(tri_func(out_cov), tri_func(exp_cov))
        torch.testing.assert_close(out_diag, exp_diag)
        # preserving the other triangular matrix
        torch.testing.assert_close(other_tri_func(out_cov), other_tri_func(Acp))

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("upper", [True, False])
    def test_nocupy_rankk(self, device, dtype, upper):
        A = self.A(device, dtype)
        Acp = torch.clone(A)
        d = self.d(device, dtype)
        veck = self.veck(device, dtype)

        tri_func = partial(torch.triu, diagonal=1) if upper else partial(torch.tril, diagonal=-1)
        other_tri_func = partial(torch.tril, diagonal=-1) if upper else partial(torch.triu, diagonal=1)
        exp_cov = A + tri_func(veck @ veck.T)
        exp_diag = d + (veck ** 2).sum(1)

        with patch.multiple(
            "franken.utils.linalg.cov", cupy=None
        ) as _:
            # TEST FUNCTION HERE
            out_cov, out_diag = _rankk_update(A, d, veck, upper=upper)
        # must be in-place
        assert out_cov.data_ptr() == A.data_ptr()
        assert out_diag.data_ptr() == d.data_ptr()
        # correctness
        torch.testing.assert_close(tri_func(out_cov), tri_func(exp_cov))
        torch.testing.assert_close(out_diag, exp_diag)
        # preserving the other triangular matrix
        torch.testing.assert_close(other_tri_func(out_cov), other_tri_func(Acp))

    @SKIP_NO_CUDA
    def test_lowmem_cuda_nocupy(self):
        upper = True
        dtype = torch.float32
        A = self.A("cuda", dtype)
        d = self.d("cuda", dtype)
        vec = self.vec("cuda", dtype)
        veck = self.veck("cuda", dtype)

        with patch.multiple(
            "franken.utils.linalg.cov", cupy=None
        ) as _:
            with pytest.raises(AssertionError):
                _lowmemcov_rank1_update(A, d, vec, upper=upper)
            with pytest.raises(AssertionError):
                _lowmemcov_rankk_update(A, d, veck, upper=upper)

    @SKIP_NO_CUDA
    @FAIL_NO_CUPY_MARK
    def test_rank1_dispatcher_cuda(self):
        upper = True
        dtype = torch.float32
        A = self.A("cuda", dtype)
        d = self.d("cuda", dtype)
        vec = self.vec("cuda", dtype)
        with patch.multiple("franken.utils.linalg.cov", _lowmemcov_rank1_update=DEFAULT, _rank1_update=DEFAULT) as mocks:
            with patch.multiple("franken.utils.linalg.cov", cupy=None):
                with pytest.warns(UserWarning, match="`cupy` is not available"):
                    rank1_update(A, d, vec, upper=upper)
            mocks['_rank1_update'].assert_called_once()
            rank1_update(A, d, vec, upper=upper)
            mocks['_lowmemcov_rank1_update'].assert_called_once()

    @SKIP_NO_CUDA
    @FAIL_NO_CUPY_MARK
    def test_rankk_dispatcher_cuda(self):
        upper = True
        dtype = torch.float32
        A = self.A("cuda", dtype)
        d = self.d("cuda", dtype)
        veck = self.veck("cuda", dtype)
        with patch.multiple("franken.utils.linalg.cov", _lowmemcov_rankk_update=DEFAULT, _rankk_update=DEFAULT) as mocks:
            with patch.multiple("franken.utils.linalg.cov", cupy=None):
                with pytest.warns(UserWarning, match="`cupy` is not available"):
                    rankk_update(A, d, veck, upper=upper)
            mocks['_rankk_update'].assert_called_once()
            rankk_update(A, d, veck, upper=upper)
            mocks['_lowmemcov_rankk_update'].assert_called_once()


class TestPSDSolvers:
    mat_size = 5  # needs to be small otherwise too much numerical error

    def A(self, device, dtype):
        gen = torch.Generator(device=device)
        gen.manual_seed(37)
        mat = torch.randn(self.mat_size, self.mat_size, device=device, dtype=dtype, generator=gen)
        return (mat @ mat.T) / self.mat_size

    def B(self, device, dtype):
        gen = torch.Generator(device=device)
        gen.manual_seed(33)
        return torch.randn(self.mat_size, 2, device=device, dtype=dtype, generator=gen) / self.mat_size

    def expected(self, A, B, penalty):
        A = A + torch.eye(A.shape[0], dtype=A.dtype, device=A.device) * penalty
        return torch.linalg.solve(A.double(), B.double()).to(dtype=A.dtype)

    @SKIP_NO_CUDA
    @FAIL_NO_CUPY_MARK
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_lowmem_cuda(self, dtype):
        A = self.A("cuda", dtype)
        B = self.B("cuda", dtype)
        penalty = 1e-5
        expected = self.expected(A, B, penalty)

        # To make sure only upper part of A is accessed set the lower to zero
        A = torch.triu(A)
        Acopy = torch.clone(A)
        result = _lowmem_psd_ridge(A, B, penalty)

        # A should have been overwritten (by cholesky)
        assert not torch.isclose(A, Acopy).all()
        # rhs MAY be overwritten by result
        # (only true if B is F-contiguous which is not the case here.)
        #assert result.data_ptr() == B.data_ptr()
        # correctness
        torch.testing.assert_close(result, expected)

    def test_lowmem_cpu(self):
        A = self.A("cpu", torch.float32)
        B = self.B("cpu", torch.float32)
        penalty = 1e-5
        with pytest.raises(AssertionError):
            _lowmem_psd_ridge(A, B, penalty)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_naive(self, device, dtype):
        A = self.A(device, dtype)
        B = self.B(device, dtype)
        penalty = 1e-5
        expected = self.expected(A, B, penalty)

        # To make sure only upper part of A is accessed set the lower to zero
        A = torch.triu(A)
        Acopy = torch.clone(A)
        Bcopy = torch.clone(B)
        result = _naive_psd_ridge(A, B, penalty)

        # A should have been been overwritten (only diagonal)
        assert not torch.isclose(A, Acopy, rtol=penalty / 10, atol=penalty / 10).all()
        # rhs should be NOT overwritten by result
        torch.testing.assert_close(B, Bcopy)
        # correctness
        torch.testing.assert_close(result, expected)

    @SKIP_NO_CUDA
    @FAIL_NO_CUPY_MARK
    def test_dispatcher_cuda(self):
        with patch.multiple(
            "franken.utils.linalg.psdsolve", _naive_psd_ridge=DEFAULT, _lowmem_psd_ridge=DEFAULT
        ) as mocks:
            A = self.A("cuda", torch.float32)
            B = self.B("cuda", torch.float32)
            psd_ridge(A, B, 1e-5)
            mocks['_lowmem_psd_ridge'].assert_called_once()

    @SKIP_NO_CUDA
    def test_dispatcher_cuda_nocupy(self):
        with patch.multiple(
            "franken.utils.linalg.psdsolve", _naive_psd_ridge=DEFAULT, _lowmem_psd_ridge=DEFAULT, cupy=None
        ) as mocks:
            A = self.A("cuda", torch.float32)
            B = self.B("cuda", torch.float32)
            with pytest.warns(UserWarning, match="`cupy` is not available"):
                psd_ridge(A, B, 1e-5)
            mocks['_naive_psd_ridge'].assert_called_once()

    def test_dispatcher_cpu(self):
        with patch.multiple(
            "franken.utils.linalg.psdsolve", _naive_psd_ridge=DEFAULT, _lowmem_psd_ridge=DEFAULT
        ) as mocks:
            A = self.A("cpu", torch.float32)
            B = self.B("cpu", torch.float32)
            psd_ridge(A, B, 1e-5)
            mocks['_naive_psd_ridge'].assert_called_once()


@pytest.mark.parametrize("device", DEVICES)
class TestTridivide:
    def test_tridivide_up(self, device):
        mat = np.random.randn(5, 5)
        val = 12.0
        triu_ids = np.triu_indices(mat.shape[0], 1)
        expected = np.copy(mat)
        expected[triu_ids] /= val

        expected_torch = torch.from_numpy(expected).to(device=device)
        mat_torch = torch.from_numpy(mat).to(device=device)
        actual = inplace_triangular_divide(mat_torch, val, upper=True)
        torch.testing.assert_close(actual, expected_torch)

    def test_tridivide_lo(self, device):
        mat = np.random.randn(5, 5)
        val = 12.0
        tril_ids = np.tril_indices(mat.shape[0], -1)
        expected = np.copy(mat)
        expected[tril_ids] /= val

        expected_torch = torch.from_numpy(expected).to(device=device)
        mat_torch = torch.from_numpy(mat).to(device=device)
        actual = inplace_triangular_divide(mat_torch, val, upper=False)
        torch.testing.assert_close(actual, expected_torch)


class TestTrilerp:
    @pytest.mark.parametrize("weight", [0.0, 0.2, 1.0])
    def test_cpu_noinplace(self, weight):
        mat = torch.randn(5, 5)
        triu = torch.triu(mat, diagonal=1)
        tril = torch.tril(mat, diagonal=-1)
        expected = triu + weight * (tril.T - triu)

        actual = _trilerp_cpu(mat, weight, None)
        torch.testing.assert_close(torch.triu(actual, diagonal=1), expected)
        assert mat.data_ptr() != actual.data_ptr(), "This should not be inplace"

    @pytest.mark.parametrize("weight", [0.0, 0.2, 1.0])
    def test_cpu_inplace(self, weight):
        mat = torch.randn(5, 5)
        triu = torch.triu(mat, diagonal=1)
        tril = torch.tril(mat, diagonal=-1)
        expected = triu + weight * (tril.T - triu)

        actual = _trilerp_cpu(mat, weight, out=mat)
        assert mat.data_ptr() == actual.data_ptr(), "This should be inplace"
        torch.testing.assert_close(torch.triu(actual, diagonal=1), expected)

    @SKIP_NO_CUDA
    @pytest.mark.parametrize("weight", [0.0, 0.2, 1.0])
    def test_cuda_noinplace(self, weight):
        mat = torch.randn(5, 5, device="cuda")
        triu = torch.triu(mat, diagonal=1)
        tril = torch.tril(mat, diagonal=-1)
        expected = triu + weight * (tril.T - triu)

        actual = _trilerp_triton(mat, weight, None)
        torch.testing.assert_close(torch.triu(actual, diagonal=1), expected)
        assert mat.data_ptr() != actual.data_ptr(), "This should not be inplace"

    @SKIP_NO_CUDA
    @pytest.mark.parametrize("weight", [0.0, 0.2, 1.0])
    def test_cuda_inplace(self, weight):
        mat = torch.randn(5, 5, device="cuda")
        weight = 0.2
        triu = torch.triu(mat, diagonal=1)
        tril = torch.tril(mat, diagonal=-1)
        expected = triu + weight * (tril.T - triu)

        actual = _trilerp_triton(mat, weight, out=mat)
        assert mat.data_ptr() == actual.data_ptr(), "This should be inplace"
        torch.testing.assert_close(torch.triu(actual, diagonal=1), expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_wrapper(self, device):
        mat = torch.randn(5, 5, device=device)
        weight = 0.2
        triu = torch.triu(mat, diagonal=1)
        tril = torch.tril(mat, diagonal=-1)
        expected_mat = triu + weight * (tril.T - triu)
        diag_up = torch.randn(5, device=device)
        diag_lo = torch.randn(5, device=device)
        expected_diag = diag_up + weight * (diag_lo - diag_up)

        actual_mat, actual_diag = triangular_lerp(mat, diag_up, diag_lo, weight, inplace=True)
        assert mat.data_ptr() == actual_mat.data_ptr(), "This should be inplace"
        assert actual_diag.data_ptr() == diag_up.data_ptr(), "This should be inplace"
        torch.testing.assert_close(torch.triu(actual_mat, diagonal=1), expected_mat)
        torch.testing.assert_close(actual_diag, expected_diag)
