import abc
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch.utils.data
import torch.distributed
from torch import Tensor
import ase.io
import ase.data

from franken.config import BackboneConfig
from franken.data.distributed_sampler import SimpleUnevenDistributedSampler
from franken.rf.atomic_energies import AtomicEnergiesShift


logger = logging.getLogger("franken")


@torch.jit.script
class Configuration:
    """Container for a single configuration (molecule or crystal).

    The set of attributes which are non empty depends on the GNN backbone which the
    :class:`Configuration` object will be passed to.
    """

    def __init__(
        self,
        atom_pos: Tensor,
        atomic_numbers: Tensor,
        natoms: Tensor,
        node_attrs: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        shifts: Optional[Tensor] = None,
        unit_shifts: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
        batch_ids: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ):
        self.atom_pos = atom_pos
        self.atomic_numbers = atomic_numbers
        self.natoms = natoms
        self.node_attrs = node_attrs
        self.edge_index = edge_index
        self.shifts = shifts
        self.unit_shifts = unit_shifts
        self.cell = cell
        self.batch_ids = batch_ids
        self.pbc = pbc

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> "Configuration":
        # optional-type refinement must be on local variables (torch.jit.script)
        node_attrs = self.node_attrs
        if node_attrs is not None:
            node_attrs = node_attrs.to(device=device, dtype=dtype)
        edge_index = self.edge_index
        if edge_index is not None:
            edge_index = edge_index.to(device=device, dtype=dtype)
        shifts = self.shifts
        if shifts is not None:
            shifts = shifts.to(device=device, dtype=dtype)
        unit_shifts = self.unit_shifts
        if unit_shifts is not None:
            unit_shifts = unit_shifts.to(device=device, dtype=dtype)
        cell = self.cell
        if cell is not None:
            cell = cell.to(device=device, dtype=dtype)
        batch_ids = self.batch_ids
        if batch_ids is not None:
            batch_ids = batch_ids.to(device=device, dtype=dtype)
        pbc = self.pbc
        if pbc is not None:
            pbc = pbc.to(device=device, dtype=dtype)
        return Configuration(
            atom_pos=self.atom_pos.to(device=device, dtype=dtype),
            atomic_numbers=self.atomic_numbers.to(device=device, dtype=dtype),
            natoms=self.natoms.to(device=device, dtype=dtype),
            node_attrs=node_attrs,
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell,
            batch_ids=batch_ids,
            pbc=pbc,
        )


@dataclasses.dataclass
class Target:
    """Container class for the target variables of a single configuration."""

    energy: Tensor
    forces: Optional[Tensor]

    def to(self, device=None, dtype=None) -> "Target":
        return Target(
            energy=self.energy.to(device=device, dtype=dtype),
            forces=(
                self.forces.to(device=device, dtype=dtype)
                if self.forces is not None
                else None
            ),
        )


class BaseAtomsDataset(torch.utils.data.Dataset, abc.ABC):
    """Base class for atom datasets.

    The data is loaded entirely into memory from the provided :attr:`dataset_dir` and the
    file-path specified in the :attr:`DATASET_REGISTRY`. Each dataset must be stored in a
    single `.extxyz` file which is read using `ase <https://wiki.fysik.dtu.dk/ase/>`_.

    Args:
        data_path (str or None): path to the '.extxyz' file to be loaded with `ase`.
            This can be None in which case no data will be loaded (useful for running MD).
        split (str): the split ('train', 'test', 'val', 'md') which will be used.  The 'md'
            split can be used to initialize an empty dataset to help running MD simulations.
        dataset_dir (str or Path): points to the base folder containing datasets,
            if set to `None` the root path of the repository is used.
        num_configurations: (optional int): maximum number of configurations to
            load. The default is to load all configurations in the file.
    """

    def __init__(
        self,
        data_path: str | Path | None,
        split: str,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
    ):
        self.split = split
        self.data_path = data_path

        ase_atoms: list[ase.Atoms] = []
        if self.data_path is not None:
            read_ase_atoms = ase.io.read(self.data_path, index=":")
            if isinstance(read_ase_atoms, ase.Atoms):
                # workaround edge case of a single configuration
                ase_atoms = [read_ase_atoms]
            else:
                ase_atoms = read_ase_atoms

        if subsample_rng is not None:
            rng = np.random.default_rng(subsample_rng)
            rng.shuffle(ase_atoms)  # type: ignore

        if (num_random_subsamples is not None) and (
            1 <= num_random_subsamples <= len(ase_atoms)
        ):
            ase_atoms = ase_atoms[:num_random_subsamples]

        self.ase_atoms = ase_atoms
        self.atomic_energies_ = None
        self.energy_shifts_ = None
        self.mean_absolute_deviations_ = None

    def add_configuration(self, atoms: ase.Atoms) -> int:
        self.ase_atoms.append(atoms)
        return len(self.ase_atoms) - 1

    def __len__(self):
        return len(self.ase_atoms)

    def compute_average_atomic_energies(self) -> Dict[int, torch.Tensor]:
        """
        Function to compute the average interaction energy of each chemical element
        returns dictionary of E0s
        """
        len_train = len(self)
        zs = self.species
        len_zs = len(zs)
        A = torch.zeros((len_train, len_zs))
        B = torch.zeros(len_train)
        for i in range(len_train):
            atoms = self.ase_atoms[i]
            B[i] = atoms.get_potential_energy(apply_constraint=False)
            for j, z in enumerate(zs):
                A[i, j] = torch.count_nonzero(
                    torch.tensor(atoms.get_atomic_numbers() == z)
                )
        try:
            E0s = torch.linalg.lstsq(A, B, rcond=None)[0]
            atomic_energies_dict = {}
            for i, z in enumerate(zs):
                atomic_energies_dict[z] = E0s[i]
        except torch.linalg.LinAlgError:
            logging.warning(
                "Failed to compute atomic energies using least squares regression, using the same for all atoms"
            )
            atomic_energies_dict = {}
            for i, z in enumerate(zs):
                atomic_energies_dict[z] = B.sum() / A.sum()

        return atomic_energies_dict

    @property
    def atomic_energies(self):
        # average atomic energies
        if self.atomic_energies_ is None:
            if self.split != "train":
                raise RuntimeError(
                    f"Atomic energies should only be computed on the training set. Found: {self.split}."
                )
            self.atomic_energies_ = self.compute_average_atomic_energies()
        return self.atomic_energies_

    @property
    def energy_shifts(self):
        # precompute energy shifts based on atomic energies
        if self.energy_shifts_ is None:
            if self.split != "train":
                raise RuntimeError(
                    f"Energy shifts should only be computed for the training set. Found: {self.split}."
                )
            shifter = AtomicEnergiesShift(
                num_species=self.num_species, atomic_energies=self.atomic_energies
            )
            self.energy_shifts_ = [
                shifter(torch.from_numpy(atoms.get_atomic_numbers()))
                for atoms in self.ase_atoms
            ]

        return self.energy_shifts_

    @property
    def mean_absolute_deviations(self):
        """Returns the mean absolute deviations of the potential energy and forces in the dataset."""
        if self.mean_absolute_deviations_ is None:
            energies = np.array(
                [a.get_potential_energy(apply_constraint=False) for a in self.ase_atoms]
            )
            forces = np.concatenate([a.get_forces() for a in self.ase_atoms], axis=0)
            mad_energy = np.mean(np.absolute(energies - np.mean(energies)))
            mad_forces = np.mean(
                np.absolute(forces - np.mean(forces, axis=0, keepdims=True))
            )
            self.mean_absolute_deviations_ = {
                "energy": float(mad_energy),
                "forces": float(mad_forces),
            }
        return self.mean_absolute_deviations_

    @property
    def species(self):
        _species = set()
        for a in self.ase_atoms:
            _species.update(a.get_atomic_numbers().tolist())
        return sorted(list(_species))

    @property
    def num_species(self):
        return len(self.species)

    @abc.abstractmethod
    def __getitem__(
        self, idx, no_targets: bool = False
    ) -> Union[Configuration, Tuple[Configuration, Target]]:
        pass

    @staticmethod
    def from_path(
        data_path: str | Path | None,
        split: str,
        gnn_config: BackboneConfig | None,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
    ) -> "BaseAtomsDataset":
        """Factory method to initialize an atoms dataset with a specific GNN type.

        Datasets are tightly coupled to the GNN they will be used with: different
        kinds of GNNs require slightly different variations on the dataset class,
        for example see :class:`franken.datasets.atoms_dataset.MACEAtomsDataset` and
        :class:`franken.datasets.atoms_dataset.FairchemAtomsDataset`.
        This method will take care of instantiating the correct dataset class for
        the given GNN.

        Args:
            data_path (str or None): path to the '.extxyz' file to be loaded with `ase`.
                This can be None in which case no data will be loaded (useful for running MD).
            split (str): the split ('train', 'test', 'val', 'md') which will be used.  The 'md'
                split can be used to initialize an empty dataset to help running MD simulations.
            gnn_backbone_id (str): name of the GNN to be used with the dataset. If `None` is passed,
                we will initialize a simple dataset without any inter-atomic graph information.
            num_random_subsamples: (optional int): maximum number of configurations to
                load. The default is to load all configurations in the file.
            subsample_rng: (optional int): random number generator seed used while subsampling data.

        """
        if gnn_config is not None:
            backbone_family = gnn_config.family
            backbone_id = gnn_config.path_or_id
        else:
            backbone_family = None
            backbone_id = None
        if backbone_family == "mace":
            try:
                from franken.data.mace import MACEAtomsDataset

                return MACEAtomsDataset(
                    data_path,
                    split,
                    num_random_subsamples,
                    subsample_rng,
                    backbone_id,
                    precompute=True,
                )
            except ImportError as import_err:
                print(
                    f"franken wasn't able to load {backbone_id}. Is {backbone_family} installed?",
                    file=sys.stderr,
                )
                raise import_err
        elif backbone_family == "fairchem":
            try:
                from franken.data.fairchem import FairchemAtomsDataset

                return FairchemAtomsDataset(
                    data_path,
                    split,
                    num_random_subsamples,
                    subsample_rng,
                    backbone_id,
                    precompute=True,
                )
            except ImportError as import_err:
                print(
                    f"franken wasn't able to load {backbone_id}. Is {backbone_family} installed?",
                    file=sys.stderr,
                )
                raise import_err
        elif backbone_family == "sevenn":
            try:
                from franken.data.sevenn import SevennAtomsDataset
            except ImportError as e:
                logger.error(
                    f"franken wasn't able to load {backbone_id}. Is {backbone_family} installed?",
                    exc_info=e,
                )
                raise
            return SevennAtomsDataset(
                data_path,
                split,
                num_random_subsamples,
                subsample_rng,
                backbone_id,
                precompute=True,
            )
        elif backbone_family is None:
            return SimpleAtomsDataset(
                data_path, split, num_random_subsamples, subsample_rng
            )
        else:
            raise ValueError(backbone_family)

    def get_dataloader(self, distributed: bool) -> torch.utils.data.DataLoader:
        """Get a dataloader corresponding to a :class:`~franken.datasets.atoms_dataset.BaseAtomsDataset`.

        The dataloader creation is specific to the problem: batch size is fixed to 1 and no extra workers are
        used since fetching the data is fast.
        If :attr:`distributed` is True the dataloader will use a :class:`torch.utils.data.DistributedSampler`
        to distribute the dataset's samples among available processes.

        Args:
            dset (BaseAtomsDataset): the dataset used by the dataloader
            split (str): the data-split ('train', 'test', 'val')
            distributed (bool): whether the dataloader should be distributed among available processes
        """

        def empty_collate_fn(batch):
            assert len(batch) == 1
            return batch[0]

        sampler = None
        if distributed and torch.distributed.is_initialized():
            sampler = SimpleUnevenDistributedSampler(self)
        elif distributed and not torch.distributed.is_initialized():
            logger.warning(
                "The distributed flag was set to True, but torch.distributed is not initialized. "
            )
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            num_workers=0,
            collate_fn=empty_collate_fn,
        )


class SimpleAtomsDataset(BaseAtomsDataset):
    def __init__(
        self,
        data_path: str | Path | None,
        split: str,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
    ):
        super().__init__(data_path, split, num_random_subsamples, subsample_rng)

    def __getitem__(self, idx, no_targets: bool = False):
        """Returns an array of (inputs, outputs) with inputs being a configuration
        and outputs being the target (energy and forces).
        Note: ONLY for the 'train' split, the energy_shift is removed from the target.
        """
        atoms = self.ase_atoms[idx]
        positions = atoms.get_positions()
        atomic_numbers = np.array(
            [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
        )
        cell = np.array(atoms.get_cell())

        data = Configuration(
            atom_pos=torch.from_numpy(positions).to(dtype=torch.float32),
            atomic_numbers=torch.from_numpy(atomic_numbers).to(dtype=torch.int64),
            natoms=torch.tensor(len(atomic_numbers)).view(1),
            cell=torch.from_numpy(cell).to(dtype=torch.float32),
        )
        if no_targets:
            return data
        energy = torch.tensor(
            self.ase_atoms[idx].get_potential_energy(apply_constraint=False)
        )
        if self.split == "train":
            energy = energy - self.energy_shifts[idx]
        target = Target(
            energy=energy,
            forces=torch.Tensor(self.ase_atoms[idx].get_forces(apply_constraint=False)),
        )
        return data, target
