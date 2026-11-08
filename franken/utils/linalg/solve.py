import warnings

import torch

try:
    import cupy.cuda
    from cupy_backends.cuda.libs import cublas, cusolver
except ImportError:
    cupy = None
    cusolver = None
    cublas = None

from franken.utils.linalg.tri import pack_upper


def calc_cholesky_factor(cov: torch.Tensor, penalty: float) -> torch.Tensor:
    # Add diagonal without copies
    cov.diagonal().add_(penalty)
    # Solve with cholesky on GPU
    L = torch.linalg.cholesky(cov, upper=True)
    return L


def psd_solve(
    cov: torch.Tensor,
    rhs: torch.Tensor,
    penalty: float,
    return_cholesky_factor: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Solve ridge regression via Cholesky factorization, overwriting :attr:`cov` and :attr:`rhs`.

    Multiple right-hand sides are supported.
    Instead of providing the data matrix (commonly :math:`X` in ridge-regression notation),
    and labels (commonly :math:`y`), we are given directly :math:`\text{cov} = X^{\top} X`
    and :math:`\text{rhs} = X^{\top} y`.
    Since :attr:`cov` is symmetric only its **upper triangle** will be accessed.

    To limit memory usage, the :attr:`cov` matrix **may be overwritten**, and :math:`rhs`
    may also be overwritten (depending on its memory layout).

    Args:
        cov (Tensor): covariance of the linear system
        rhs (Tensor): right hand side (one or more) of the linear system
        penalty (float): Tikhonov l2 penalty
        return_cholesky_factor (bool): Also return the packed upper Cholesky factor.

    Returns:
        The ridge regression coefficients, or (coefficients, packed factor)
        when return_cholesky_factor is True.
    """
    L = calc_cholesky_factor(cov, penalty)
    rhs_shape = rhs.shape
    solution = torch.cholesky_solve(rhs.view(cov.shape[0], -1), L, upper=True)
    solution = solution.view(rhs_shape)
    if return_cholesky_factor:
        return solution, pack_upper(L).detach()
    return solution


def lowmem_psd_solve(
    cov: torch.Tensor,
    rhs: torch.Tensor,
    penalty: float,
    return_cholesky_factor: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Solve ridge regression with the low-memory CUDA solver when available.

    Multiple right-hand sides are supported.
    Instead of providing the data matrix (commonly :math:`X` in ridge-regression notation),
    and labels (commonly :math:`y`), we are given directly :math:`\text{cov} = X^{\top} X`
    and :math:`\text{rhs} = X^{\top} y`.
    Since :attr:`cov` is symmetric only its **upper triangle** will be accessed.

    To limit memory usage, the :attr:`cov` matrix **may be overwritten**, and :math:`rhs`
    may also be overwritten (depending on its memory layout).

    Args:
        cov (Tensor): covariance of the linear system
        rhs (Tensor): right hand side (one or more) of the linear system
        penalty (float): Tikhonov l2 penalty
        return_cholesky_factor (bool): Also return the packed upper Cholesky factor.

    Returns:
        The ridge regression coefficients, or (coefficients, packed factor)
        when return_cholesky_factor is True.
    """
    if cupy is None or cov.device.type != "cuda":
        if cov.device.type == "cuda":
            warnings.warn(
                "low-memory solver cannot be used because `cupy` is not available. "
                "Install `cupy` if you encounter memory problems."
            )
        return psd_solve(cov, rhs, penalty, return_cholesky_factor)

    assert cusolver is not None and cublas is not None and cupy is not None
    assert cov.device.type == "cuda"
    dtype = cov.dtype
    n = cov.shape[0]

    # Add diagonal without copies
    cov.diagonal().add_(penalty)

    if dtype == torch.float32:
        potrf = cusolver.spotrf
        potrf_bufferSize = cusolver.spotrf_bufferSize
        potrs = cusolver.spotrs
    elif dtype == torch.float64:
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
        potrs = cusolver.dpotrs
    else:
        raise ValueError(dtype)

    # cov must be f-contiguous (column-contiguous, stride is (1, n))
    assert cov.dim() == 2
    assert cov.shape[0] == cov.shape[1]
    transpose = False
    if n != 1:
        if cov.stride(0) != 1:
            cov = cov.T
            transpose = True
    assert cov.stride(0) == 1
    cov_cp = cupy.asarray(cov)

    # save rhs shape to restore it later on.
    rhs_shape = rhs.shape
    rhs = rhs.reshape(n, -1)
    n_rhs = rhs.shape[1]
    if rhs.stride(0) != 1:  # force rhs to be f-contiguous
        # `contiguous` causes a copy
        rhs = rhs.T.contiguous().T
    assert rhs.stride(0) == 1
    rhs_cp = cupy.asarray(rhs)

    handle = cupy.cuda.device.get_cusolver_handle()
    uplo = cublas.CUBLAS_FILL_MODE_LOWER if transpose else cublas.CUBLAS_FILL_MODE_UPPER
    dev_info = torch.empty(
        1, dtype=torch.int32
    )  # don't allocate with cupy as it uses a separate mem pool
    dev_info_cp = cupy.asarray(dev_info)

    worksize = potrf_bufferSize(handle, uplo, n, cov_cp.data.ptr, n)
    workspace = torch.empty(worksize, dtype=dtype)
    workspace_cp = cupy.asarray(workspace)

    # Cholesky factorization
    potrf(
        handle,
        uplo,
        n,
        cov_cp.data.ptr,
        n,
        workspace_cp.data.ptr,
        worksize,
        dev_info_cp.data.ptr,
    )
    if (dev_info_cp != 0).any():
        raise torch.linalg.LinAlgError(  # type: ignore
            f"Error reported by {potrf.__name__} in cuSOLVER. devInfo = {dev_info_cp}."
        )

    # Solve: A * X = B
    potrs(
        handle,
        uplo,
        n,
        n_rhs,
        cov_cp.data.ptr,
        n,
        rhs_cp.data.ptr,
        n,
        dev_info_cp.data.ptr,
    )
    if (dev_info_cp != 0).any():
        raise torch.linalg.LinAlgError(  # type: ignore
            f"Error reported by {potrf.__name__} in cuSOLVER. devInfo = {dev_info_cp}."
        )

    solution = torch.as_tensor(rhs).reshape(rhs_shape)
    if return_cholesky_factor:
        # Restore the input orientation: its upper triangle now holds the factor.
        return solution, pack_upper(cov.T if transpose else cov).detach()
    return solution
