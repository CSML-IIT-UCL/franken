import logging
from typing import Any, Callable, List, Literal, Optional

import torch

import franken.data.base
from franken.data.base import Configuration
from franken.utils.jac import jacfwd, tune_jacfwd_chunksize

logger = logging.getLogger("franken")
__all__ = (
    "forces_bwdad",
    "forces_stress_bwdad",
    "forces_fwdad",
    "forces_stress_fwdad",
)


"""Backward auto-diff (torch.autograd)
useful for when there is a single (or few) outputs and many inputs
"""


def forces_bwdad(
    data: Configuration,
    fn: Callable,
    **extra_args: torch.Tensor,
):
    data.atom_pos.requires_grad_(True)
    _, energy = fn(data.atom_pos, displacement=None, data=data, **extra_args)
    with torch.enable_grad():
        forces = _forces_bwdad_helper(energies=energy, data=data)
    return {
        franken.data.base.FORCES_TARGET_KEY: forces.detach(),
        franken.data.base.ENERGY_TARGET_KEY: energy.detach(),
    }


def _forces_bwdad_helper(energies: torch.Tensor, data: Configuration):
    n_sols = energies.shape[0]
    n_atoms = data.atom_pos.shape[0]
    force_lst = []
    for i in range(n_sols):  # each model (M) independently
        cur_energy: torch.Tensor = energies[i]
        # NOTE: complex type annotation required by jit.script
        grad_out: List[Optional[torch.Tensor]] = [torch.ones_like(cur_energy)]
        grads = torch.autograd.grad(
            outputs=[cur_energy],
            inputs=[data.atom_pos],
            grad_outputs=grad_out,  # type: ignore
            retain_graph=i < n_sols - 1,
        )
        grad = grads[0]
        assert grad is not None
        force_lst.append(grad)
    forces = -torch.stack(force_lst, 0).view(n_sols, n_atoms, 3)
    return forces


def forces_stress_bwdad(
    data: Configuration,
    fn: Callable,
    **extra_args: torch.Tensor,
):
    n_systems = data.natoms.numel()
    displacement = torch.zeros(
        (n_systems, 3, 3),
        dtype=data.atom_pos.dtype,
        device=data.atom_pos.device,
    ).requires_grad_(True)
    data.atom_pos.requires_grad_(True)
    with torch.enable_grad():
        _, energy = fn(
            data.atom_pos, displacement=displacement, data=data, **extra_args
        )
    forces, stress = _forces_stress_bwdad_helper(
        energies=energy, displacement=displacement, data=data
    )
    return {
        franken.data.base.FORCES_TARGET_KEY: forces.detach(),
        franken.data.base.STRESS_TARGET_KEY: stress.detach(),
        franken.data.base.ENERGY_TARGET_KEY: energy.detach(),
    }


def _forces_stress_bwdad_helper(
    energies: torch.Tensor, displacement: torch.Tensor, data: Configuration
):
    """
    energies: [n_sols, n_systems]
    displacement: [n_systems, 3, 3]
    """
    n_atoms = data.atom_pos.shape[0]
    n_sols = energies.shape[0]
    n_systems = data.natoms.numel()

    force_lst, virial_lst = [], []
    for i in range(n_sols):  # each model (M) independently
        cur_energy: torch.Tensor = energies[i]
        # NOTE: complex type annotation required by jit.script
        grad_out: List[Optional[torch.Tensor]] = [torch.ones_like(cur_energy)]
        grads = torch.autograd.grad(
            outputs=[cur_energy],
            inputs=[data.atom_pos, displacement],
            grad_outputs=grad_out,  # type: ignore
            retain_graph=i < n_sols - 1,
        )
        g0 = grads[0]
        assert g0 is not None
        force_lst.append(g0)
        g1 = grads[1]
        assert g1 is not None
        virial_lst.append(g1)
    forces = -torch.stack(force_lst, 0).view(n_sols, n_atoms, 3)
    virial = -torch.stack(virial_lst, 0).view(n_sols, n_systems, 3, 3)
    stress = _virial_to_stress(virial, data)
    return forces, stress


"""Forward auto-diff (torch.func)
useful for when there are several outputs and inputs
"""


@torch.jit.unused
@torch.no_grad
def forces_stress_fwdad(
    data: Configuration,
    fn: Callable,
    cache_key: str,
    cache: dict[str, Any],
    chunk_size: int | Literal["auto"],
    **extra_args: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int | Literal["auto"]]:
    n_systems = data.natoms.numel()
    displacement = torch.zeros(
        (n_systems, 3, 3),
        dtype=data.atom_pos.dtype,
        device=data.atom_pos.device,
    )
    args = [data.atom_pos, displacement, data, *extra_args.values()]
    if (jacfn := cache.get(cache_key)) is None:
        chunk_size = _get_jacobian_chunk_size(
            fn, args, argnums=0, chunk_size=chunk_size
        )
        jacfn = jacfwd(fn, argnums=(0, 1), has_aux=True, chunk_size=chunk_size)
        cache[cache_key] = jacfn
    (force_fm, virial_fm), energy_fm = jacfn(*args)
    stress_fm = _virial_to_stress(-virial_fm, data)
    return {
        franken.data.base.FORCES_TARGET_KEY: -force_fm,
        franken.data.base.STRESS_TARGET_KEY: stress_fm,
        franken.data.base.ENERGY_TARGET_KEY: energy_fm,
    }, chunk_size


@torch.jit.unused
@torch.no_grad
def forces_fwdad(
    data: Configuration,
    fn: Callable,
    cache_key: str,
    cache: dict[str, Any],
    chunk_size: int | Literal["auto"],
    **extra_args: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int | Literal["auto"]]:
    args = [data.atom_pos, None, data, *extra_args.values()]
    if (jacfn := cache.get(cache_key)) is None:
        chunk_size = _get_jacobian_chunk_size(
            fn, args, argnums=0, chunk_size=chunk_size
        )
        jacfn = jacfwd(fn, argnums=0, has_aux=True, chunk_size=chunk_size)
        cache[cache_key] = jacfn
    force_fm, energy_fm = jacfn(*args)
    return {
        franken.data.base.FORCES_TARGET_KEY: -force_fm,
        franken.data.base.ENERGY_TARGET_KEY: energy_fm,
    }, chunk_size


def _get_jacobian_chunk_size(
    fn,
    inputs,
    argnums: int | tuple[int, int],
    chunk_size: int | Literal["auto"],
) -> int:
    if isinstance(chunk_size, int):
        return chunk_size
    elif chunk_size == "auto":
        # TODO: We can probably cache the value for different functions to avoid multiple tuner runs.
        jac_chunk_size = tune_jacfwd_chunksize(
            test_sample=inputs,
            func=fn,
            argnums=argnums,
            has_aux=True,
        )
        logger.info(f"jacobian chunk size automatically set to {jac_chunk_size}")
        return jac_chunk_size
    else:
        raise RuntimeError(f"Unexpected chunk_size {chunk_size}")


def _virial_to_stress(virial: torch.Tensor, data: Configuration) -> torch.Tensor:
    cell = data.cell
    assert cell is not None
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    return -stress
