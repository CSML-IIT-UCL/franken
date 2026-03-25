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

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        batch_ids: torch.Tensor | None = None,
        num_systems: int | None = None,
    ) -> torch.Tensor:
        """
        Calculate the energy shift for a given set of atomic numbers.

        Args:
            atomic_numbers: Atomic numbers for all atoms.
            batch_ids: Optional system index per atom. If provided, returns one
                shift per system.
            num_systems: Optional number of systems. If omitted, inferred from
                ``batch_ids``.

        Returns:
            Scalar shift for single-system inputs, or a tensor of shape
            ``[n_systems]`` when ``batch_ids`` is provided.
        """
        if batch_ids is None:
            shift = torch.tensor(
                0.0,
                dtype=self.atomic_energies.dtype,
                device=self.atomic_energies.device,
            )
            for z, atom_ene in zip(self.z_keys, self.atomic_energies):
                mask = atomic_numbers == int(z.item())
                shift += torch.sum(atom_ene * mask)
            return shift

        batch_ids = batch_ids.to(dtype=torch.long, device=atomic_numbers.device)
        if num_systems is None:
            num_systems = (
                int(batch_ids.max().item()) + 1 if batch_ids.numel() > 0 else 0
            )

        shift = torch.zeros(
            (num_systems,),
            dtype=self.atomic_energies.dtype,
            device=self.atomic_energies.device,
        )
        for z, atom_ene in zip(self.z_keys, self.atomic_energies):
            contrib = atom_ene * (atomic_numbers == z).to(
                dtype=self.atomic_energies.dtype
            )
            shift = shift.index_add(0, batch_ids, contrib)
        return shift

    def __repr__(self):
        formatted_energies = " , ".join(
            [
                f"{z.item()}: {atom_ene}"
                for z, atom_ene in zip(self.z_keys, self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}({formatted_energies})"
