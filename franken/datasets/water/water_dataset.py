from pathlib import Path
import re
import zipfile

import ase
import ase.io

from franken.datasets.registry import DATASET_REGISTRY, BaseRegisteredDataset
from franken.utils.file_utils import download_file


@DATASET_REGISTRY.register("water")
class WaterRegisteredDataset(BaseRegisteredDataset):
    relative_paths = {
        "water": {
            "train": "water/ML_AB_dataset_1.xyz",
            "val": "water/ML_AB_dataset_2-val.xyz",
        },
    }
    zip_file_names = ["ML_AB_dataset_1", "ML_AB_dataset_2", "ML_AB_128h2o_validation"]

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
        water_base_path = base_path / "water"
        water_base_path.mkdir(exist_ok=True, parents=True)

        # NOTE: cannot check MD5 here since it changes at every download. As a dumb fallback we check the file-size.
        download_file(
            url="https://zenodo.org/api/records/10723405/files-archive",
            filename=water_base_path / "data.zip",
            desc="Downloading water dataset",
            expected_size=35866571,
        )
        # Extract from zip and convert VASP -> XYZ format
        with zipfile.ZipFile(water_base_path / "data.zip", mode="r") as zf:
            for file_name in cls.zip_file_names:
                with zf.open(file_name, "r") as fh:
                    vasp_data = fh.read().decode("utf-8")
                    xyz_data = vasp_mlff_to_xyz(vasp_data)
                with open(water_base_path / f"{file_name}.xyz", "w") as fh:
                    fh.write(xyz_data)
                # Sanity check
                traj = ase.io.read(water_base_path / f"{file_name}.xyz", index=":")
                assert isinstance(traj, list)
                for i, atoms in enumerate(traj):
                    atoms.get_potential_energy()
                    atoms.get_forces()
        # Split a validation set from dataset-2
        dataset = ase.io.read(
            water_base_path / "ML_AB_dataset_2.xyz", index=":", format="extxyz"
        )
        assert isinstance(dataset, list)
        dataset_no_overlap = dataset[473:]
        ase.io.write(water_base_path / "ML_AB_dataset_2-val.xyz", dataset_no_overlap)
        # Cleanup
        (water_base_path / "data.zip").unlink()


def vasp_mlff_to_xyz_oneconfig(data):
    # Parse sections using regular expressions
    num_atoms = int(re.search(r"The number of atoms\s*[-=]+\s*(\d+)", data).group(1))
    energy = float(
        re.search(r"Total energy \(eV\)\s*[-=]+\s*([-+]?\d*\.\d+|\d+)", data).group(1)
    )

    # Extract lattice vectors
    lattice_match = re.search(
        r"Primitive lattice vectors \(ang.\)\s*[-=]+\s*([\d\s.-]+)", data
    )
    lattice_lines = lattice_match.group(1).strip().split("\n")
    lattice = [line.split() for line in lattice_lines]

    # Flatten and format lattice as a string for XYZ format
    lattice_flat = " ".join([" ".join(line) for line in lattice])

    # Extract atomic positions
    positions_match = re.search(
        r"Atomic positions \(ang.\)\s*[-=]+\s*([\d\s.-]+)", data
    )
    positions_lines = positions_match.group(1).strip().split("\n")
    positions = [line.split() for line in positions_lines]

    # Extract forces
    forces_match = re.search(r"Forces \(eV ang.\^-1\)\s*[-=]+\s*([\d\s.-]+)", data)
    forces_lines = forces_match.group(1).strip().split("\n")
    forces = [line.split() for line in forces_lines]

    # Extract stress tensor (two lines) without separators
    stress_match_1 = re.search(
        r"Stress \(kbar\)\s*[-=]+\s*XX YY ZZ\s*[-=]+\s*([\d\s.-]+)", data
    )
    stress_match_2 = re.search(r"XY YZ ZX\s*[-=]+\s*([\d\s.-]+)", data)

    # Ensure we only capture numerical values and not separator lines
    stress_values_1 = (
        stress_match_1.group(1).strip().split()[:3]
    )  # Take first three values for XX YY ZZ
    stress_values_2 = (
        stress_match_2.group(1).strip().split()[:3]
    )  # Take first three values for XY YZ ZX
    stress_values_kbar = [float(x) for x in stress_values_1 + stress_values_2]

    # Convert stress from kbar (VASP output) to eV/Å^3, which is the internal
    # stress unit expected by ASE/franken for extxyz stress tensors.
    kbar_to_ev_per_ang3 = 1.0 / 1602.176634
    stress_values = [v * kbar_to_ev_per_ang3 for v in stress_values_kbar]
    xx, yy, zz, xy, yz, zx = stress_values

    stress_tensor = (
        f"{xx:.8f} {xy:.8f} {zx:.8f} "
        f"{xy:.8f} {yy:.8f} {yz:.8f} "
        f"{zx:.8f} {yz:.8f} {zz:.8f}"
    )

    # Create the extended XYZ content for this configuration
    xyz_content = []
    xyz_content.append(f"{num_atoms}")
    xyz_content.append(
        f'Lattice="{lattice_flat}" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} stress="{stress_tensor}"'
    )

    # Atom types (order them according to the positions provided)
    atom_type_lines = (
        re.search(r"Atom types and atom numbers\s*[-=]+\s*([\w\s\d]+)", data)
        .group(1)
        .strip()
        .split("\n")
    )
    atom_types = []
    for line in atom_type_lines:
        element, count = line.split()
        atom_types.extend([element] * int(count))

    # Add each atom's data line by line
    for idx, (position, force) in enumerate(zip(positions, forces)):
        element = atom_types[idx]
        px, py, pz = position
        fx, fy, fz = force
        xyz_content.append(f"{element} {px} {py} {pz} {fx} {fy} {fz}")

    return "\n".join(xyz_content)


def vasp_mlff_to_xyz(data):
    # Split the data by configurations using "Configuration num." as the delimiter
    configurations = re.split(r"Configuration num\.\s*\d+", data)
    xyz_all = []

    # Process each configuration if it is not empty
    for config in configurations:
        config = config.strip()
        if config:  # Only parse if the configuration is not empty
            try:
                xyz_all.append(vasp_mlff_to_xyz_oneconfig(config))
            except AttributeError:
                pass  # some errors are expected.

    # Join all configurations with a newline
    return "\n".join(xyz_all)


if __name__ == "__main__":
    WaterRegisteredDataset.download(Path(__file__).parent.parent)
