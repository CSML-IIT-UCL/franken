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
        self.register_buffer(
            "atomic_energies", torch.zeros(num_species, dtype=torch.float64)
        )
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
                (
                    v.detach().to(device=device, dtype=torch.float64).clone()
                    if isinstance(v, torch.Tensor)
                    else torch.tensor(v, device=device, dtype=torch.float64)
                )
                for v in atomic_energies.values()
            ]
        ).to(device)
        self.z_keys = torch.tensor(
            list(atomic_energies.keys()),
            dtype=torch.long,
            device=self.atomic_energies.device,
        )
        self.is_initialized = True

    def _apply(self, fn, recurse=True):
        # Model-wide dtype conversions must not round the energy reference.
        atomic_energies = self.atomic_energies
        super()._apply(fn, recurse=recurse)
        self.atomic_energies = atomic_energies.to(
            device=self.atomic_energies.device, dtype=torch.float64
        )
        return self

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
        # Also enforce double accumulation for scripted modules, whose generic
        # conversion methods do not call the Python _apply override.
        atomic_energies = self.atomic_energies.to(dtype=torch.float64)
        if batch_ids is None:
            shift = torch.tensor(
                0.0,
                dtype=torch.float64,
                device=self.atomic_energies.device,
            )
            for z, atom_ene in zip(self.z_keys, atomic_energies):
                mask = atomic_numbers == z
                shift += torch.sum(atom_ene * mask)
            return shift

        batch_ids = batch_ids.to(dtype=torch.long, device=atomic_numbers.device)
        if num_systems is None:
            num_systems = (
                int(batch_ids.max().item()) + 1 if batch_ids.numel() > 0 else 0
            )

        shift = torch.zeros(
            (num_systems,),
            dtype=torch.float64,
            device=self.atomic_energies.device,
        )
        for z, atom_ene in zip(self.z_keys, atomic_energies):
            contrib = atom_ene * (atomic_numbers == z).to(dtype=torch.float64)
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
