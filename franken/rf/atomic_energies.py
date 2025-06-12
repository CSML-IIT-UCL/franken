from typing import Mapping

import torch


class AtomicEnergiesShift(torch.nn.Module):
    atomic_energies: torch.Tensor
    Z_keys: list[int]

    def __init__(
        self,
        num_species: int,
        atomic_energies: Mapping[int, torch.Tensor | float] | None = None,
    ):
        """
        Initialize the AtomicEnergiesShift module.

        Args:
            num_species:
            atomic_energies: A dictionary mapping atomic numbers to atomic energies.
        """
        super().__init__()

        self.num_species = num_species
        self.register_buffer("atomic_energies", torch.zeros(num_species))
        self.register_buffer(
            "z_keys", torch.zeros((self.num_species,), dtype=torch.long)
        )  # placeholder
        self.is_initialized = False

        if atomic_energies is not None:
            self.set_from_atomic_energies(atomic_energies)

    def set_from_atomic_energies(
        self, atomic_energies: Mapping[int, torch.Tensor | float]
    ):
        assert (
            len(atomic_energies) == self.num_species
        ), f"{len(atomic_energies)=} != {self.num_species=}"
        device = self.atomic_energies.device
        self.atomic_energies = torch.stack(
            [
                v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
                for v in atomic_energies.values()
            ]
        ).to(device)
        self.z_keys = torch.tensor(
            list(atomic_energies.keys()),
            dtype=torch.long,
            device=self.atomic_energies.device,
        )
        self.is_initialized = True

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Calculate the energy shift for a given set of atomic numbers.

        Args:
            atomic_numbers: A tensor containing atomic numbers for which to calculate the energy shift.

        Returns:
            A tensor representing the total energy shift for the provided atomic numbers.
        """

        shift = torch.tensor(
            0.0, dtype=self.atomic_energies.dtype, device=self.atomic_energies.device
        )

        for z, atom_ene in zip(self.z_keys, self.atomic_energies):
            mask = atomic_numbers == int(z.item())
            shift += torch.sum(atom_ene * mask)

        return shift

    def __repr__(self):
        formatted_energies = " , ".join(
            [
                f"{z.item()}: {atom_ene}"
                for z, atom_ene in zip(self.z_keys, self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}({formatted_energies})"
