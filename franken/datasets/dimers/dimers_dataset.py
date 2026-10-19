"""Biomolecular dimer scans, split by separation in source-file order."""

from pathlib import Path

import ase.io

from franken.datasets.registry import DATASET_REGISTRY, BaseRegisteredDataset
from franken.utils.file_utils import download_file


DIMER_LABELS = ("AA", "CA", "CC", "CP", "PA", "PP")
DIMER_DATASETS = {
    name: (label, first_only)
    for label in DIMER_LABELS
    for name, first_only in ((f"dimers_{label}", False), (f"dimers_{label}_0", True))
}


@DATASET_REGISTRY.register(list(DIMER_DATASETS))
class DimersRegisteredDataset(BaseRegisteredDataset):
    relative_paths = {
        name: {
            "train": f"dimers/{name}/train.xyz",
            "val": f"dimers/{name}/validation.xyz",
        }
        for name in DIMER_DATASETS
    }
    source_files = {
        "bio_dimers.xyz": "cdf5f3e139bd56a2c31014c8e50e8b93",
        "bio_dimers_monomers.xyz": "3d4b87ec5e1a79b75c41c48a78f78552",
    }

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

    @classmethod
    def download(cls, base_path: Path):
        """Cache both sources and prepare 10/3 splits for every scan.

        Monomers are retained for reference, but are not included in either split.
        Energies, forces, periodic cells, and scan metadata are preserved by ASE.
        """
        folder = base_path / "dimers"
        folder.mkdir(exist_ok=True, parents=True)
        for filename, checksum in cls.source_files.items():
            download_file(
                url=(
                    "https://archive.materialscloud.org/records/405an-d8183/files/"
                    f"{filename}?download=1"
                ),
                filename=folder / filename,
                expected_md5=checksum,
                desc=f"Downloading {filename}",
            )

        scans = {label: {} for label in DIMER_LABELS}
        for atoms in ase.io.iread(folder / "bio_dimers.xyz", format="extxyz"):
            label = atoms.info["label"]
            scans[label].setdefault(int(atoms.info["dimer_id"]), []).append(atoms)
        for label, label_scans in scans.items():
            if not label_scans:
                raise ValueError(f"No {label} dimer scans found")
            for dimer_id, frames in label_scans.items():
                if len(frames) != 13:
                    raise ValueError(
                        f"{label} dimer {dimer_id} has {len(frames)} frames; expected 13"
                    )

        for name, (label, first_only) in DIMER_DATASETS.items():
            label_scans = scans[label]
            if first_only:
                selected = [next(iter(label_scans.values()))]
            else:
                selected = label_scans.values()
            train, validation = [], []
            for frames in selected:
                train.extend(frames[:10])
                validation.extend(frames[10:13])
            for split, frames in (("train", train), ("val", validation)):
                path = base_path / cls.relative_paths[name][split]
                path.parent.mkdir(exist_ok=True, parents=True)
                ase.io.write(path, frames, format="extxyz")
