import warnings
from typing import Optional, Tuple, TypeVar

import scipy.linalg.blas
import torch

try:
    import cupy
    import cupy.cublas
except ImportError:
    cupy = None

from franken.utils.linalg.tri import inplace_triangular_divide

OptTensorType = TypeVar("OptTensorType", bound=torch.Tensor | None, covariant=True)


def rank1_update(
    cov: torch.Tensor, diag: OptTensorType, vec: torch.Tensor, upper: bool = True
) -> Tuple[torch.Tensor, OptTensorType]:
    r"""in-place rank-1 update to a triangular matrix

    If cupy is available, a low-memory-footprint algorithm is used.

    Returns the updated covariance and diagonal vector:

    .. math::
        \text{out}_1 = \text{cov} + \text{vec} \text{vec}^T

    .. math::
        \text{out}_2 = \text{diag} + \text{vec}^2

    Args:
        cov (Tensor): the square matrix which is updated in-place
        diag (Tensor): the diagonal of the triangular matrix
        vec (Tensor): the vector comprising the symmetric rank-1 update
        upper (bool): whether to update the lower or upper triangle of :attr:`cov`.
    """
    if cupy is not None or cov.device.type == "cpu":
        return _lowmemcov_rank1_update(cov, diag, vec, upper)
    warnings.warn(
        "low-memory covariance updates cannot be used "
        "because `cupy` is not available. Install `cupy` "
        "if you encounter memory problems."
    )
    return _rank1_update(cov, diag, vec, upper)


def _rank1_update(
    cov: torch.Tensor, diag: OptTensorType, vec: torch.Tensor, upper: bool = True
) -> Tuple[torch.Tensor, OptTensorType]:
    tri_fn = torch.triu if upper else torch.tril
    cov.add_(tri_fn(torch.outer(vec, vec)))
    if diag is not None:
        diag.add_(vec**2)
    return cov, diag


def _lowmemcov_rank1_update(
    cov: torch.Tensor, diag: OptTensorType, vec: torch.Tensor, upper: bool = True
) -> Tuple[torch.Tensor, OptTensorType]:
    # Use the DSYR function in blas
    # A := alpha*x*x**T + A
    assert vec.device == cov.device
    assert vec.dtype == cov.dtype
    if cov.device.type == "cpu":
        # cov needs to be F-contiguous for the overwriting to work
        transposed = False
        if cov.shape[0] > 1 and cov.stride(1) == 1:  # then it's C-contiguous
            cov = cov.T
            upper = not upper
            transposed = True
        # Note that cov_np and cov share the same memory, so changes in cov_np are reflected in cov.
        cov_np = cov.numpy()
        energy_fmap_np = vec.numpy()
        if cov.dtype == torch.float64:
            scipy.linalg.blas.dsyr(
                alpha=1.0, x=energy_fmap_np, a=cov_np, lower=not upper, overwrite_a=True
            )
        elif cov.dtype == torch.float32:
            scipy.linalg.blas.ssyr(
                alpha=1.0, x=energy_fmap_np, a=cov_np, lower=not upper, overwrite_a=True
            )
        else:
            raise ValueError(cov.dtype)
        if transposed:
            cov = cov.T
    else:
        assert cupy is not None
        energy_fmap_cp = cupy.asarray(vec.view(-1, 1))
        cov_cp = cupy.asarray(cov)
        # It's a bit dumb but cupy doesn't have great coverage of the cuBLAS level 2 API, so
        # we use 'syrk' again instead of 'syr'.
        cupy.cublas.syrk(
            trans="N",
            a=energy_fmap_cp,
            out=cov_cp,
            alpha=1.0,
            beta=1.0,
            lower=not upper,
        )
    if diag is not None:
        diag.add_(vec**2)
    return cov, diag


def rankk_update(
    cov: torch.Tensor, diag: OptTensorType, update: torch.Tensor, upper: bool = True
) -> Tuple[torch.Tensor, OptTensorType]:
    r"""in-place rank-k update to a triangular matrix.

    If cupy is available, a low-memory-footprint algorithm is used.

    Returns the updated covariance and diagonal vector:

    .. math::
        \text{out}^{(1)} = \text{cov} + \text{update} \text{update}^T

    .. math::
        \text{out}^{(2)}_i = \text{diag}_i + \sum_{j} \text{update}_{i,j}^2

    Args:
        cov (Tensor): the square matrix which is updated in-place
        diag (Tensor): the diagonal of the triangular matrix
        update (Tensor): the rank-k update matrix
        upper (bool): whether to update the lower or upper triangle of :attr:`cov`.
    """
    if cupy is not None or cov.device.type == "cpu":
        return _lowmemcov_rankk_update(cov, diag, update, upper)
    warnings.warn(
        "low-memory covariance updates cannot be used "
        "because `cupy` is not available. Install `cupy` "
        "if you encounter memory problems."
    )
    return _rankk_update(cov, diag, update, upper)


def _rankk_update(
    cov: torch.Tensor, diag: OptTensorType, update: torch.Tensor, upper: bool = True
) -> tuple[torch.Tensor, OptTensorType]:
    tri_fn = torch.triu if upper else torch.tril
    cov.add_(tri_fn(update @ update.T))
    if diag is not None:
        diag.add_((update**2).sum(1))
    return cov, diag


def _lowmemcov_rankk_update(
    cov: torch.Tensor, diag: OptTensorType, update: torch.Tensor, upper: bool = True
) -> tuple[torch.Tensor, OptTensorType]:
    # DSYRK
    # C := alpha*A*A**T + beta*C,
    assert update.device == cov.device
    assert update.dtype == cov.dtype

    if cov.device.type == "cpu":
        # cov needs to be F-contiguous for the overwriting to work
        transposed = False
        if cov.shape[0] > 1 and cov.stride(1) == 1:  # then it's C-contiguous
            cov = cov.T
            upper = not upper
            transposed = True
        # Note that cov_np and cov share the same memory, so changes in cov_np are reflected in cov.
        cov_np = cov.numpy()
        forces_fmap_np = update.numpy()
        if cov.dtype == torch.float64:
            scipy.linalg.blas.dsyrk(
                alpha=1.0,
                a=forces_fmap_np,
                beta=1.0,
                c=cov_np,
                lower=not upper,
                overwrite_c=True,
            )
        elif cov.dtype == torch.float32:
            scipy.linalg.blas.ssyrk(
                alpha=1.0,
                a=forces_fmap_np,
                beta=1.0,
                c=cov_np,
                lower=not upper,
                overwrite_c=True,
            )
        else:
            raise ValueError(cov.dtype)
        if transposed:
            cov = cov.T
    else:
        assert cupy is not None
        forces_fmap_cp = cupy.asarray(update)
        cov_cp = cupy.asarray(cov)
        cupy.cublas.syrk(
            trans="N",
            a=forces_fmap_cp,
            out=cov_cp,
            alpha=1.0,
            beta=1.0,
            lower=not upper,
        )
    if diag is not None:
        diag.add_((update**2).sum(1))
    return cov, diag


def triu_to_cov(triu: torch.Tensor, cov: Optional[torch.Tensor] = None) -> torch.Tensor:
    tril_indices = torch.tril_indices(triu.shape[0], triu.shape[1])
    if cov is None:
        cov = triu.clone()
    else:
        assert cov.shape == triu.shape
        cov.copy_(triu)
    cov[tril_indices.unbind()] = cov.T[tril_indices.unbind()]
    return cov


def tril_to_cov(tril: torch.Tensor, cov: Optional[torch.Tensor] = None) -> torch.Tensor:
    tril_indices = torch.tril_indices(tril.shape[0], tril.shape[1])
    if cov is None:
        cov = tril.clone()
    else:
        assert cov.shape == tril.shape
        cov.copy_(tril)
    cov.T[tril_indices.unbind()] = cov[tril_indices.unbind()]
    return cov


def normalize_leading_eig(cov: torch.Tensor, coeffs: torch.Tensor) -> None:
    r"""Normalize covariance matrix dividing by the largest eigenvalue in-place"""
    norm, _ = torch.lobpcg(cov, k=1, largest=True)
    cov.div_(norm)
    coeffs.div_(norm)


def lowmem_normalize_leading_eig(
    tri_cov: torch.Tensor, diag: torch.Tensor, coeffs: torch.Tensor, upper: bool
) -> None:
    r"""Normalize low-memory covariance matrix dividing by the largest eigenvalue.

    The low-memory covariance matrix is stored only in one triangle of :attr:`tri_cov`.

    We need to compute the eigenvalues from the triangular parts of the matrix.
    Since this is not really possible with LOBPCG, we convert to full matrices
    and compute the eigenvalues of those.
    TODO: This requires a new copy of `cov` which wastes a bunch of memory.
    """
    tri_cov.diagonal().copy_(diag)
    if upper:
        cov = triu_to_cov(tri_cov)
    else:
        cov = tril_to_cov(tri_cov)
    norm, _ = torch.lobpcg(cov, k=1, largest=True)
    diag.div_(norm[0])
    coeffs.div_(norm[0])
    inplace_triangular_divide(tri_cov, norm[0].item(), upper=upper)
