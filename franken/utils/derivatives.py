from typing import Callable, List, Optional

import torch

import franken.data.base
from franken.data.base import Configuration


def full_forces_autograd(
    data: Configuration,
    fn: Callable,
    **extra_args: torch.Tensor,
):
    data.atom_pos.requires_grad_(True)
    _, energy = fn(data.atom_pos, displacement=None, data=data, **extra_args)
    forces = forces_autograd(energies=energy, data=data)
    return {
        franken.data.base.FORCES_TARGET_KEY: forces.detach(),
        franken.data.base.ENERGY_TARGET_KEY: energy.detach(),
    }


def forces_autograd(energies: torch.Tensor, data: Configuration):
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


def full_forces_stress_autograd(
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
    _, energy = fn(data.atom_pos, displacement=displacement, data=data, **extra_args)
    forces, stress = forces_stress_autograd(
        energies=energy, displacement=displacement, data=data
    )
    return {
        franken.data.base.FORCES_TARGET_KEY: forces.detach(),
        franken.data.base.STRESS_TARGET_KEY: stress.detach(),
        franken.data.base.ENERGY_TARGET_KEY: energy.detach(),
    }


def forces_stress_autograd(
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
    stress = virial_to_stress(virial, data)
    return forces, stress


def virial_to_stress(virial: torch.Tensor, data: Configuration) -> torch.Tensor:
    cell = data.cell
    assert cell is not None
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    return -stress
