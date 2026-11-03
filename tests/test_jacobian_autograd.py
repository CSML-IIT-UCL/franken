"""Autograd-only Jacobians for kernels without forward-mode AD support."""

import pytest
import torch

from franken.utils.jac import jacobian_autograd


class ReverseOnlyCube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**3

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return 3 * x**2 * gradient


@pytest.mark.parametrize("features", [1, 16])
@pytest.mark.parametrize("has_aux", [False, True])
def test_reverse_only_kernel_matches_analytic_jacobian(features, has_aux):
    x = torch.tensor([0.3, -0.8, 1.1], dtype=torch.float64)
    matrix = torch.arange(1, features * 3 + 1, dtype=x.dtype).reshape(features, 3)

    def func(value):
        output = matrix @ ReverseOnlyCube.apply(value)
        return (output, output[None]) if has_aux else output

    with torch.no_grad():
        actual = jacobian_autograd(func, has_aux=has_aux)(x)
    if has_aux:
        actual, aux = actual
        torch.testing.assert_close(aux, (matrix @ x**3)[None])
        assert not aux.requires_grad
    torch.testing.assert_close(actual, matrix * 3 * x.square())
    assert not actual.requires_grad
    assert not x.requires_grad


@pytest.mark.parametrize("features", [1, 16])
def test_multiple_inputs_and_unused_coordinate(features):
    x = torch.tensor([0.2, 0.4], dtype=torch.float64)
    strain = torch.tensor([0.3], dtype=torch.float64)
    ignored = torch.zeros(2, dtype=torch.float64)
    scales = torch.arange(1, features + 1, dtype=x.dtype)

    def func(pos, metadata, displacement, unused):
        assert metadata == "constant"
        return scales * ((pos**3).sum() + displacement.square().sum())

    actual = jacobian_autograd(func, argnums=(0, 2, 3))(x, "constant", strain, ignored)
    expected = (
        scales[:, None] * 3 * x.square(),
        scales[:, None] * 2 * strain,
        torch.zeros(features, 2, dtype=x.dtype),
    )
    for block, ref in zip(actual, expected):
        torch.testing.assert_close(block, ref)


def test_constant_output_has_zero_jacobian():
    x = torch.zeros(3)
    result = jacobian_autograd(lambda value: torch.ones(5))(x)
    torch.testing.assert_close(result, torch.zeros(5, 3))


class DetachedBackwardCube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**3

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return 3 * x**2 * gradient.detach()


def test_detached_backward_raises_instead_of_returning_zero_derivatives():
    def func(value):
        return DetachedBackwardCube.apply(value).repeat(8)

    with pytest.raises(RuntimeError, match="detached its backward inputs"):
        jacobian_autograd(func)(torch.ones(3))
