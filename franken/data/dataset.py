from typing import Dict, Union, Tuple
from pathlib import Path
import logging
import warnings

import torch
import numpy as np
import ase
import ase.io
from tqdm import tqdm

import franken.utils.distributed as dist_utils
from franken.config import BackboneConfig
from franken.data.base import Configuration, Target
from franken.data.distributed_sampler import SimpleUnevenDistributedSampler
from franken.rf.atomic_energies import AtomicEnergiesShift


logger = logging.getLogger("franken")


class FrankenAtomsDataset(torch.utils.data.Dataset):
    """Base class for atomic datasets.

    The data is loaded entirely into memory from the provided :attr:`dataset_dir` and the
    file-path specified in the :attr:`DATASET_REGISTRY`. Each dataset must be stored in a
    single `.extxyz` file which is read using `ase <https://wiki.fysik.dtu.dk/ase/>`_.

    Args:
        data_path (str or None): path to the '.extxyz' file to be loaded with `ase`.
            This can be None in which case no data will be loaded (useful for running MD).
        split (str): the split ('train', 'test', 'val', 'md') which will be used.  The 'md'
            split can be used to initialize an empty dataset to help running MD simulations.
        gnn_config (BackboneConfig or None): Loads data to be used with a specific model.
            Datasets are tightly coupled to the GNN they will be used with, mostly in relation
            to the way in which neighbor lists are calculated. For this reason the dataset class
            will hold a reference to the model. If None, no neighbor calculations will be performed
            and the dataset will only return atomic positions.
        num_random_subsamples (optional int): maximum number of configurations to
            load. The default is to load all configurations in the file.
        subsample_rng (optional int): random number generator seed used while subsampling data.
        precompute (bool): Whether to precompute all structures in the dataset and drop the
            reference to the GNN at the start.
    """

    def __init__(
        self,
        data_path: str | Path | None,
        split: str,
        gnn_config: BackboneConfig | None,
        num_random_subsamples: int | None = None,
        subsample_rng: int | None = None,
        precompute: bool = True,
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

        self.gnn = None
        if gnn_config is not None:
            from franken.backbones.utils import load_checkpoint

            self.gnn = load_checkpoint(gnn_config)

        if precompute and len(self.ase_atoms) > 0:
            self.graphs = self.convert_all(self.ase_atoms)
            del self.gnn

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
        except torch.linalg.LinAlgError:  # pyright: ignore[reportPrivateImportUsage]
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

    def get_dataloader(self, distributed: bool) -> torch.utils.data.DataLoader:
        """Get a dataloader corresponding to a :class:`~franken.data.FrankenAtomsDataset`.

        The dataloader creation is specific to the problem: batch size is fixed to 1 and no extra workers are
        used since fetching the data is fast.
        If :attr:`distributed` is True the dataloader will use a :class:`torch.utils.data.DistributedSampler`
        to distribute the dataset's samples among available processes.

        Args:
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

    def convert_all(self, atoms_list):
        graphs = []
        atoms_iter = atoms_list
        process_rank = dist_utils.get_rank()
        if process_rank == 0:
            desc = f"ASE -> Franken ({self.split})"
            atoms_iter = tqdm(atoms_list, desc=desc)
        for atoms in atoms_iter:
            graphs.append(self.convert(atoms))
        return graphs

    def convert(self, frame: ase.Atoms):
        # Build partial configuration (no edges)
        dtype = torch.get_default_dtype()

        pos = frame.get_positions()
        pbc = frame.get_pbc()
        cell = frame.get_cell()
        atomic_numbers = frame.get_atomic_numbers()

        pos_pt = torch.tensor(pos, dtype=dtype)
        pbc_pt = torch.tensor(pbc, dtype=torch.bool)
        atomic_numbers_pt = torch.tensor(atomic_numbers, dtype=torch.int32)

        cell_vectors_are_not_zero = np.any(cell != 0, axis=1)
        if not np.all(cell_vectors_are_not_zero == pbc):
            warnings.warn(
                "A conversion was requested for an `ase.Atoms` object "
                "with one or more non-zero cell vectors but where the corresponding "
                "boundary conditions are set to `False`. "
                "The corresponding cell vectors will be set to zero.",
            )
        cell_pt = torch.zeros((3, 3), dtype=dtype)
        cell_pt[pbc_pt] = torch.tensor(cell[pbc], dtype=dtype)  # type: ignore

        partial_config = Configuration(
            atom_pos=pos_pt,
            atomic_numbers=atomic_numbers_pt,
            natoms=torch.tensor(len(atomic_numbers_pt)).view(1),
            cell=cell_pt,
            pbc=pbc_pt,
        )
        if self.gnn is not None:
            # Add edges
            config = self.gnn.get_neighbors(partial_config)
        else:
            config = partial_config
        return config

    def __getitem__(
        self, idx, no_targets: bool = False
    ) -> Union[Configuration, Tuple[Configuration, Target]]:
        """Returns an array of (inputs, outputs) with inputs being a configuration
        and outputs being the target (energy and forces).
        Note: ONLY for the 'train' split, the energy_shift is removed from the target.
        """
        if self.graphs is None:
            config = self.convert(self.ase_atoms[idx])
        else:
            config = self.graphs[idx]

        if no_targets:
            return config
        energy = torch.tensor(
            self.ase_atoms[idx].get_potential_energy(apply_constraint=False)
        )
        if self.split == "train":
            energy = energy - self.energy_shifts[idx]
        target = Target(
            energy=energy,
            forces=torch.Tensor(self.ase_atoms[idx].get_forces(apply_constraint=False)),
        )
        return config, target
