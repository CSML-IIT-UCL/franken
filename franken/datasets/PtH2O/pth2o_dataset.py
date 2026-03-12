from pathlib import Path
import shutil
import zipfile

import ase
import ase.io
import numpy as np

from franken.datasets.registry import DATASET_REGISTRY, BaseRegisteredDataset
from franken.utils.file_utils import download_file


@DATASET_REGISTRY.register("PtH2O")
class PtH2ORegisteredDataset(BaseRegisteredDataset):
    """
    FigShare path: https://figshare.com/articles/dataset/Dataset_and_training_files_for_Is_the_water_Pt_111_interface_ordered_at_room_temperature_/14791755
    Paper URL: https://pubs.aip.org/aip/jcp/article-abstract/155/22/224701/1030633/Is-the-water-Pt-111-interface-ordered-at-room
    Citation: A. E. G Mikkelsen, J. Schiøtz, T. Vegge and K. W. Jacobsen, "Is the water/Pt(111) interface ordered at room temperature?",
              The Journal of Chemical Physics vol. 155 (22), 2021.
    """

    relative_paths = {
        "PtH2O": {
            "train": "PtH2O/train.extxyz",
            "val": "PtH2O/valid.extxyz",
            "test": "PtH2O/test.extxyz",
        },
    }

    @classmethod
    def get_path(
        cls, name: str, split: str, base_path: Path | None, download: bool = True
    ):
        if base_path is None:
            raise KeyError(None)
        relative_path = cls.relative_paths[name][split]
        path = base_path / relative_path
        if not path.is_file() and download:
            cls.download(base_path)
        if path.is_file():
            return path
        else:
            raise ValueError(f"Dataset not found at '{path.resolve()}'")

    @classmethod
    def download(cls, base_path: Path):
        pth2o_base_path = base_path / "PtH2O"
        pth2o_base_path.mkdir(exist_ok=True, parents=True)
        # Download
        download_file(
            url="https://api.figshare.com/v2/file/download/29141586",
            filename=pth2o_base_path / "data.zip",
            desc="Downloading PtH2O dataset",
            expected_md5="acd748f7f32c66961c90cb15457f7bae",
        )
        # Extract
        with zipfile.ZipFile(pth2o_base_path / "data.zip", "r") as zf:
            zf.extractall(pth2o_base_path)
        # Read full dataset
        full_traj = ase.io.read(
            pth2o_base_path / "Dataset_and_training_files" / "dataset.traj", index=":"
        )
        assert isinstance(full_traj, list)
        # Split into train/val/test
        np.random.seed(42)
        np.random.shuffle(full_traj)
        train_traj = full_traj[:30_000]
        valid_traj = full_traj[30_000:31_000]
        test_traj = full_traj[31_000:]
        # Saved shuffled to disk
        ase.io.write(pth2o_base_path / "train.extxyz", train_traj)
        ase.io.write(pth2o_base_path / "valid.extxyz", valid_traj)
        ase.io.write(pth2o_base_path / "test.extxyz", test_traj)
        # Cleanup
        (pth2o_base_path / "data.zip").unlink()
        shutil.rmtree(pth2o_base_path / "Dataset_and_training_files")


if __name__ == "__main__":
    PtH2ORegisteredDataset.download(Path(__file__).parent.parent)
