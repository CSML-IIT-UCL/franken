"""Utilities for dealing with triangular matrices in-place."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    import triton.language as tl
    from triton import jit
except ImportError:

    def jit(*a, **kw):
        pass

    class mock_tl:
        constexpr = None

    tl = mock_tl()


def pack_upper(matrix: torch.Tensor) -> torch.Tensor:
    """Pack a square matrix's upper triangle row by row, including the diagonal."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, found {matrix.shape}")
    n = matrix.shape[0]
    mask = torch.ones(n, n, dtype=torch.bool, device=matrix.device).triu_()
    return matrix.masked_select(mask)


def unpack_upper(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Reconstruct an upper-triangular matrix from row-wise packed values."""
    if n < 0 or packed.ndim != 1 or packed.numel() != n * (n + 1) // 2:
        raise ValueError("Packed upper triangle has an invalid size for n={n}. Found {packed.shape}")
    mask = torch.ones(n, n, dtype=torch.bool, device=packed.device).triu_()
    return packed.new_zeros((n, n)).masked_scatter_(mask, packed)


def naive_lerp(
    start: torch.Tensor,
    end: torch.Tensor,
    weight: Union[float, torch.Tensor],
    inplace: bool = False,
) -> torch.Tensor:
    r"""Linear interpolation wrapper.
    Computes the linear interpolation between :attr:`start` and :attr:`end` based on
    the specified :attr:`weight`. If :attr:`inplace` is True, the result is stored overwriting
    the :attr:`start` tensor.

    .. math::
        \text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)

    Args:
        input (Tensor): the tensor with the starting points
        end (Tensor): the tensor with the ending points
        weight (float or tensor): the weight for the interpolation formula
        inplace (bool): whether the interpolation should be computed in place.
    """
    if not inplace:
        return torch.lerp(start, end, weight)
    return start.lerp_(end, weight)


def triangular_lerp(
    matrix: torch.Tensor,
    diag_upper: torch.Tensor,
    diag_lower: torch.Tensor,
    weight: float,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Linearly interpolate the upper and lower triangular parts of a matrix.
    The diagonals of the upper and lower triangles must be specified separately using
    the :attr:`diag_upper` and :attr:`diag_lower` arguments. If :attr:`inplace` is True,
    the result is stored overwriting :attr:`diag_upper` and the upper triangle
    of :attr:`matrix`.

    The first output is the linear interpolation of the triangular matrix

    .. math:
        \text{out}_1 = \text{upper}(\text{matrix}) + \text{weight} \times (\text{lower}(\text{matrix}) - \text{upper}(\text{matrix}))

    the second output is the linear interpolation of the diagonals

    .. math:
        \text{out}_2 = \text{diag_upper} + \text{weight} \times (\text{diag_lower} - \text{diag_upper})

    Args:
        matrix (Tensor): the square matrix whose triangular parts are interpolated
        diag_upper (Tensor): a vector containing the start point of interpolation
        diag_lower (Tensor): a vector containing the end point of interpolation
        weight (float or tensor): the weight for the interpolation formula
        inplace (bool): whether the interpolation should be computed in place.
    """
    diag_lerped = naive_lerp(diag_upper, diag_lower, weight, inplace)
    if matrix.device.type == "cpu":
        triang_lerped = _trilerp_cpu(
            matrix, weight=weight, out=matrix if inplace else None
        )
    else:
        triang_lerped = _trilerp_triton(
            matrix, weight=weight, out=matrix if inplace else None
        )
    return triang_lerped, diag_lerped


def _trilerp_cpu(
    matrix: torch.Tensor, weight: float, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    up_ids = np.triu_indices(matrix.shape[0], k=1)

    if out is None:
        out = torch.empty_like(matrix)
    out_np = out.numpy()
    matrix_np = matrix.numpy()
    lerped = matrix_np[up_ids] + weight * (matrix_np.T[up_ids] - matrix_np[up_ids])
    out_np[up_ids] = lerped
    return torch.from_numpy(out_np)


@jit
def trilerp_kernel(
    a_ptr,
    N,
    stride_a1,
    stride_a2,
    out_ptr,
    stride_o1,
    stride_o2,
    weight,
    BLOCK_SIZE_N: tl.constexpr,
):
    # TODO: Check if this could be replaced by cublas GEAM (a cuda specific extension!). This is in cupy!
    # Compute the block index in the upper triangle that we're
    # going to compute
    pid = tl.program_id(axis=0)
    # triangular program ID to block ID
    row = ((tl.sqrt_rn((8 * pid + 1).to(tl.float32)) - 1.0) / 2.0).to(tl.int32)
    pid_a1 = pid - row * (row + 1) // 2
    pid_a2 = row

    offs_a1_up = pid_a1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_a2_up = pid_a2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_up_ptrs = a_ptr + (
        offs_a1_up[:, None] * stride_a1 + offs_a2_up[None, :] * stride_a2
    )
    offs_a1_lo = offs_a2_up
    offs_a2_lo = offs_a1_up
    a_lo_ptrs = a_ptr + (
        offs_a1_lo[:, None] * stride_a1 + offs_a2_lo[None, :] * stride_a2
    )
    out_ptrs = out_ptr + (
        offs_a1_up[:, None] * stride_o1 + offs_a2_up[None, :] * stride_o2
    )

    mask = (
        (offs_a1_up[:, None] < offs_a2_up[None, :])
        & (offs_a1_up[:, None] < N)
        & (offs_a2_up[None, :] < N)
    )
    up = tl.load(a_up_ptrs, mask=mask, other=0.0)
    lo = tl.load(a_lo_ptrs, mask=mask.T, other=0.0)
    interp = up * (1 - weight) + lo.T * weight
    tl.store(out_ptrs, interp, mask=mask)


def _trilerp_triton(
    matrix: torch.Tensor, weight: float, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    assert matrix.shape[0] == matrix.shape[1], "Incompatible dimensions"
    N = matrix.shape[0]
    s1 = matrix.stride(0)
    s2 = matrix.stride(1)
    if out is None:
        out = torch.empty_like(matrix)
    o1 = out.stride(0)
    o2 = out.stride(1)
    block_size = 64  # fixed, this is not a hot-loop function.
    grid = (math.ceil(N / block_size) * (math.ceil(N / block_size) + 1) // 2,)
    trilerp_kernel[grid](matrix, N, s1, s2, out, o1, o2, weight, block_size)  # type: ignore
    return out


@jit
def inplace_triangular_divide_kernel(
    a_ptr, N, stride_a1, stride_a2, val, upper, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # triangular program ID to block ID
    row = ((tl.sqrt_rn((8 * pid + 1).to(tl.float32)) - 1.0) / 2.0).to(tl.int32)
    pid_a1 = pid - row * (row + 1) // 2
    pid_a2 = row
    if upper:
        offs_a1_up = pid_a1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_a2_up = pid_a2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (
            (offs_a1_up[:, None] < offs_a2_up[None, :])
            & (offs_a1_up[:, None] < N)
            & (offs_a2_up[None, :] < N)
        )
        a_ptrs = a_ptr + (
            offs_a1_up[:, None] * stride_a1 + offs_a2_up[None, :] * stride_a2
        )
    else:
        offs_a1_lo = pid_a2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_a2_lo = pid_a1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (
            (offs_a1_lo[:, None] > offs_a2_lo[None, :])
            & (offs_a1_lo[:, None] < N)
            & (offs_a2_lo[None, :] < N)
        )
        a_ptrs = a_ptr + (
            offs_a1_lo[:, None] * stride_a1 + offs_a2_lo[None, :] * stride_a2
        )
    tri_mat = tl.load(a_ptrs, mask=mask, other=0.0)
    scaled_tri_mat = tri_mat / val
    tl.store(a_ptrs, scaled_tri_mat, mask=mask)


def _cpu_triangular_divide(
    matrix: torch.Tensor, value: float, upper: bool
) -> torch.Tensor:
    if upper:
        tri_ids = np.triu_indices(matrix.shape[0], k=1)
    else:
        tri_ids = np.tril_indices(matrix.shape[0], k=-1)
    matrix_np = matrix.numpy()
    matrix_np[tri_ids] /= value
    return torch.from_numpy(matrix_np)


def inplace_triangular_divide(
    matrix: torch.Tensor, value: float, upper: bool
) -> torch.Tensor:
    """Divide a triangular matrix by a scalar inplace.

    The diagonal of the matrix is not touched by this function.

    Args:
        matrix (Tensor): a square matrix
        value (float): the divisor
        upper (bool): whether to divide the upper or lower triangle of :attr:`matrix`.

    """
    assert matrix.shape[0] == matrix.shape[1], "Incompatible dimensions"
    if matrix.device.type == "cpu":
        matrix = _cpu_triangular_divide(matrix, value, upper)
    else:
        N = matrix.shape[0]
        s1 = matrix.stride(0)
        s2 = matrix.stride(1)
        block_size = 64
        grid = (math.ceil(N / block_size) * (math.ceil(N / block_size) + 1) // 2,)
        inplace_triangular_divide_kernel[grid](
            matrix, N, s1, s2, value, upper, block_size
        )  # type: ignore
    return matrix
