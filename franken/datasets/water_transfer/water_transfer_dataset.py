from pathlib import Path
import shutil

import ase.io

from franken.datasets.registry import DATASET_REGISTRY, BaseRegisteredDataset
from franken.datasets.water.water_dataset import WaterRegisteredDataset


@DATASET_REGISTRY.register("water-transfer")
class WaterTransferRegisteredDataset(BaseRegisteredDataset):
    relative_paths = {
        "water-transfer": {
            "train": "water-transfer/train.xyz",
            "val": "water-transfer/val.xyz",
        },
    }

    @classmethod
    def get_path(
        cls, name: str, split: str, base_path: Path | None, download: bool = True
    ):
        if base_path is None:
            raise KeyError(None)
        path = base_path / cls.relative_paths[name][split]
        if not path.is_file():
            source_split = {"train": "val", "val": "train"}[split]
            source_path = WaterRegisteredDataset.get_path(
                "water", source_split, base_path, download=download
            )
            path.parent.mkdir(exist_ok=True, parents=True)
            if split == "train":
                shutil.copyfile(source_path, path)
            else:
                frames = ase.io.read(source_path, index=":450", format="extxyz")
                ase.io.write(path, frames, format="extxyz")
        return path

    @classmethod
    def download(cls, base_path: Path):
        for split in cls.relative_paths["water-transfer"]:
            cls.get_path("water-transfer", split, base_path)


if __name__ == "__main__":
    WaterTransferRegisteredDataset.download(Path(__file__).parent.parent)
