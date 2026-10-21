"""KF electrolyte configurations grouped by geometry and ion-pair count."""

from collections import defaultdict
from pathlib import Path
import random
import shutil
import zipfile

import ase.io
import numpy as np

from franken.datasets.registry import DATASET_REGISTRY, BaseRegisteredDataset
from franken.utils.file_utils import download_file


ELECTROLYTE_GROUPS = {
    f"electrolyte_{geometry}_{pairs}": (geometry, pairs)
    for geometry, counts in (("bulk", range(6)), ("interface", (1, 2, 10)))
    for pairs in counts
}
ELECTROLYTE_DATASETS = [
    "electrolyte",
    "electrolyte_bulk",
    "electrolyte_interface",
    *ELECTROLYTE_GROUPS,
]


@DATASET_REGISTRY.register(ELECTROLYTE_DATASETS)
class ElectrolyteRegisteredDataset(BaseRegisteredDataset):
    relative_paths = {
        name: {
            "train": f"electrolyte/{name}/train.xyz",
            "val": f"electrolyte/{name}/validation.xyz",
        }
        for name in ELECTROLYTE_DATASETS
    }
    source_url = (
        "https://raw.githubusercontent.com/BingqingCheng/cace-lr-fit/"
        "main/fit-electrolyte/electrolyte.xyz.zip"
    )
    source_md5 = "f0b1d1a53c2896050768634194445db8"

    @classmethod
    def get_path(
        cls, name: str, split: str, base_path: Path | None, download: bool = True
    ) -> Path:
        if base_path is None:
            raise KeyError(None)
        path = base_path / cls.relative_paths[name][split]
        if not path.is_file() and download:
            cls.download(base_path)
        if path.is_file():
            return path
        raise ValueError(f"Dataset not found at '{path.resolve()}'")

    @staticmethod
    def group_key(atoms) -> tuple[str, int]:
        """Count matched K/F pairs and distinguish cubic bulk from slabs."""
        n_pairs = int(np.count_nonzero(atoms.numbers == 19))
        if n_pairs != np.count_nonzero(atoms.numbers == 9):
            raise ValueError("Electrolyte configuration has unequal K and F counts")
        lengths = atoms.cell.lengths()
        cubic = np.allclose(lengths, lengths[0]) and np.allclose(
            atoms.cell.angles(), 90.0
        )
        return ("bulk" if cubic else "interface", n_pairs)

    @classmethod
    def download(cls, base_path: Path):
        """Extract the source and split each group 80/20 with seed 42.

        The combined dataset concatenates the group splits. Energies, forces,
        periodic cells, and per-atom charge arrays are preserved by ASE.
        """
        folder = base_path / "electrolyte"
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / "electrolyte.xyz.zip"
        source = folder / "electrolyte.xyz"
        download_file(
            url=cls.source_url,
            filename=archive,
            expected_md5=cls.source_md5,
            desc="Downloading electrolyte dataset",
        )
        with zipfile.ZipFile(archive) as zf:
            with zf.open("electrolyte.xyz") as src, source.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        groups = defaultdict(list)
        expected = set(ELECTROLYTE_GROUPS.values())
        for atoms in ase.io.iread(source, format="extxyz"):
            key = cls.group_key(atoms)
            if key not in expected:
                raise ValueError(f"Unexpected electrolyte group: {key}")
            groups[key].append(atoms)
        for key in expected:
            if len(groups[key]) < 2:
                raise ValueError(f"Electrolyte group {key} needs at least two frames")

        all_train, all_validation = [], []
        geometry_splits = {geometry: ([], []) for geometry in ("bulk", "interface")}
        for name, key in ELECTROLYTE_GROUPS.items():
            frames = groups[key]
            random.Random(42).shuffle(frames)
            n_val = min(max(1, round(0.2 * len(frames))), len(frames) - 1)
            validation, train = frames[:n_val], frames[n_val:]
            cls.write_splits(base_path, name, train, validation)
            geometry_train, geometry_validation = geometry_splits[key[0]]
            geometry_train.extend(train)
            geometry_validation.extend(validation)
            all_train.extend(train)
            all_validation.extend(validation)
        cls.write_splits(base_path, "electrolyte", all_train, all_validation)
        for geometry, (train, validation) in geometry_splits.items():
            cls.write_splits(base_path, f"electrolyte_{geometry}", train, validation)
        archive.unlink()

    @classmethod
    def write_splits(cls, base_path: Path, name: str, train, validation):
        for split, frames in (("train", train), ("val", validation)):
            path = base_path / cls.relative_paths[name][split]
            path.parent.mkdir(parents=True, exist_ok=True)
            ase.io.write(path, frames, format="extxyz")
