

from collections import defaultdict
from dataclasses import dataclass
import datetime
import json
import os
import pathlib
import re
import tempfile
from typing import Literal

import ase
from ase.io import read, write
from matplotlib import pyplot as plt
import numpy as np

import franken
import franken.autotune
from franken.config import AutotuneConfig, BackboneConfig, DatasetConfig, LESConfig, MaceBackboneConfig, MultiscaleGaussianRFConfig, RFConfig, SolverConfig
import franken.data
import franken.data.base
from franken.data.dataset import FrankenAtomsDataset
from franken.rf.model import FrankenPotential


# Download the data from here:
# https://archive.materialscloud.org/records/405an-d8183
monomers = read("../../../franken/datasets/dimers/bio_dimers_monomers.xyz", index=":")
dimers = read("../../../franken/datasets/dimers/bio_dimers.xyz", index=":")


@dataclass
class DimersDataset:
    label: Literal["CC"] | Literal["CP"] | Literal["PP"]
    id: int
    monomerA: ase.Atoms
    monomerB: ase.Atoms
    energyA: float
    energyB: float
    data: list[ase.Atoms]
    distances: list[float]

    def get_train_set(self) -> list[ase.Atoms]:
        return self.data[:10]

    def get_test_set(self) -> list[ase.Atoms]:
        return self.data[10:]
    

def get_single_dataset(allowed_states: set[Literal["CC"] | Literal["CP"] | Literal["PP"]], dimers: list[ase.Atoms]):
    c_data = []
    c_id = -1
    for dimer in dimers:
        label = dimer.info.get("label")
        if label not in allowed_states:
            continue

        dimer_id = int(dimer.info.get("dimer_id"))
        if dimer_id != c_id and c_id != -1:
            dataset = DimersDataset(
                label=label,
                id=c_id,
                monomerA=monoA,
                monomerB=monoB,
                energyA=energyA,
                energyB=energyB,
                data=c_data,
                distances=[d.info.get("distance") for d in c_data]
            )
            yield dataset
            c_data = []
            c_id = dimer_id
        if c_id == -1:
            c_id = dimer_id

        monoA = dimer[:dimer.info.get("indexB")]
        monoB = dimer[dimer.info.get("indexB"):]
        energyA = dimer.info.get("energyA")
        energyB = dimer.info.get("energyB")
        c_data.append(dimer)
        # print(f"Parsing dimer #{len(c_data):2} with ID={dimer_id:4} ({label}). "
        #       f"{monoA.get_chemical_formula()}({dimer.info.get('chargeA')}) "
        #       f"{monoB.get_chemical_formula()}({dimer.info.get('chargeB')})")


def autotune_franken(
    dset: DimersDataset, 
    solver: SolverConfig,
    bbone: BackboneConfig,
    rfs: RFConfig,
):
    # Write dset to a temporary folder
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dset = dset.get_train_set()
        test_dset = dset.get_test_set()
        train_path = os.path.join(tmpdir, "train.xyz")
        test_path = os.path.join(tmpdir, "test.xyz")
        write(train_path, train_dset)
        write(test_path, test_dset)

        dset_cfg = DatasetConfig(
            name=f"dimer_{dset.id}", train_path=train_path, val_path=test_path
        )
        cfg = AutotuneConfig(
            dataset=dset_cfg,
            solver=solver,
            backbone=bbone,
            rfs=rfs,
            les=None,
            eval_splits=["val"],
            run_dir=f"./franken_outputs/dimer_{dset.id}/",
        )
        run_dir = franken.autotune.autotune(cfg)
        assert isinstance(run_dir, pathlib.Path)
        return run_dir / "best_ckpt.pt"
        # model = FrankenPotential.load(path=run_dir / "best_ckpt.pt")
        # train_dset = FrankenAtomsDataset(
        #     data_path=train_path,
        #     split="train",
        #     gnn_config=bbone,
        # )
        # test_dset = FrankenAtomsDataset(
        #     data_path=test_path,
        #     split="test",
        #     gnn_config=bbone,
        # )
        # for data, tgt in test_dset:
        #     pred = model.predict(
        #         targets=[franken.data.base.ENERGY_TARGET_KEY, franken.data.base.FORCES_TARGET_KEY],
        #         data=data,
        #     )
        
def train_franken_les(
    dset: DimersDataset, 
    solver: SolverConfig,
    bbone: BackboneConfig,
    rfs: RFConfig,
):
    les_cfg = LESConfig(
        hidden_dim=(64, 32),
        les_output_scale=0.1
    )
    # Write dset to a temporary folder
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dset = dset.get_train_set()
        test_dset = dset.get_test_set()
        train_path = os.path.join(tmpdir, "train.xyz")
        test_path = os.path.join(tmpdir, "test.xyz")
        write(train_path, train_dset)
        write(test_path, test_dset)

        dset_cfg = DatasetConfig(
            name=f"dimer_{dset.id}", train_path=train_path, val_path=test_path
        )
        cfg = AutotuneConfig(
            dataset=dset_cfg,
            solver=solver,
            backbone=bbone,
            rfs=rfs,
            les=les_cfg,
            eval_splits=["val"],
            run_dir=f"./les_outputs/dimer_{dset.id}/",
        )
        run_dir = franken.autotune.autotune(cfg)
        assert isinstance(run_dir, pathlib.Path)
        return run_dir / "best_ckpt.pt"


def test_franken(
    dset: DimersDataset,
    model_path: pathlib.Path,
    model_cls=FrankenPotential,
):
    # Write dset to a temporary folder
    all_preds = []
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.xyz")
        write(train_path, dset.data)
        dset_cfg = DatasetConfig(
            name=f"dimer_{dset.id}", train_path=train_path
        )
        model = model_cls.load(model_path)
        model.eval()
        frk_dset = FrankenAtomsDataset(
            data_path=train_path,
            split="train",
            gnn_config=model.gnn_config,
        )
        for i, (data, tgt) in enumerate(frk_dset):
            preds = model.predict(
                targets=[franken.data.base.ENERGY_TARGET_KEY, franken.data.base.FORCES_TARGET_KEY],
                data=data,
            )
            all_preds.append(preds)
    return all_preds


def get_newest_rundir(dir: pathlib.Path) -> pathlib.Path:
    """
    Find the newest run directory based on timestamp in the folder name.
    
    Folder format: run_DDMMYY_HHMMSS_randomstring
    Example: run_260723_214650_abc20294
    Args:
        dir: Path object pointing to the directory containing run folders
    Returns:
        Path to the newest run directory, or None if no matching folders found
    """
    if not dir.exists() or not dir.is_dir():
        raise ValueError(dir)
    
    pattern = re.compile(r'^run_(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_[a-zA-Z0-9]+$')
    valid_dirs = []
    for item in dir.iterdir():
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        try:
            day, month, year, hour, minute, second = match.groups()
            timestamp = datetime.datetime.strptime(
                f"{day}{month}{year}_{hour}{minute}{second}",
                "%d%m%y_%H%M%S"
            )
            valid_dirs.append((timestamp, item))
        except ValueError:
            continue
    if not valid_dirs:
        return None
    # Sort by timestamp descending and return the newest
    valid_dirs.sort(key=lambda x: x[0], reverse=True)
    return valid_dirs[0][1]


if __name__ == "__main__":
    datasets = list(get_single_dataset({"CC"}, dimers))
    print(f"Loaded {len(datasets)} datasets")
    bb_cfg = MaceBackboneConfig("mace_mp/small")
    rf_cfg = MultiscaleGaussianRFConfig(
        num_random_features=2048, 
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6
    )

    if False:  # train standard franken (autotune)
        slv_cfg = SolverConfig(
            l2_penalty=np.logspace(-11, -6, 6).tolist(),
            force_weight=np.logspace(-1, 4, 6).tolist(),
        )
        for dataset in datasets:
            autotune_franken(
                dset=dataset,
                solver=slv_cfg,
                bbone=bb_cfg,
                rfs=rf_cfg,
            )
    if False:  # examine trained models
        energy_rmse, forces_rmse = [], []
        binding_energies = {}
        for d, dataset in enumerate(datasets):
            franken_dir = get_newest_rundir(
                pathlib.Path(f"franken_outputs/dimer_{dataset.id}")
            )
            with open(franken_dir / "best.json", "r") as fh:
                best_log = json.load(fh)
            energy_rmse.append(best_log["metrics"]["validation"]["energy_RMSE"])
            forces_rmse.append(best_log["metrics"]["validation"]["forces_RMSE"])
            # test model on all the data for distance plot
            preds = test_franken(dataset, franken_dir / "best_ckpt.pt")
            # Calculate binding energy
            binding_energies[dataset.id] = defaultdict(list)
            for i in range(len(dataset.data)):
                true_binding_energy = dataset.data[i].get_potential_energy() - dataset.energyA - dataset.energyB
                pred_binding_energy = preds[i][franken.data.base.ENERGY_TARGET_KEY].item() - dataset.energyA - dataset.energyB
                binding_energies[dataset.id]["true"].append(true_binding_energy)
                binding_energies[dataset.id]["pred"].append(pred_binding_energy)
                binding_energies[dataset.id]["dist"].append(dataset.distances[i])
            if d >= 0:
                break
        mean_energy_rmse = np.mean(energy_rmse)
        mean_forces_rmse = np.mean(forces_rmse)
        print(f"Standard Franken.")
        print(f"\tAverage Energy RMSE: {mean_energy_rmse:.1f}")
        print(f"\tAverage Forces RMSE: {mean_forces_rmse:.1f}")
        fig, ax = plt.subplots()
        for d, b_energy in enumerate(binding_energies.values()):
            ax.scatter(b_energy["dist"][:10], b_energy["true"][:10], 
                    s=100, edgecolors='b', marker='o', c="none", label="True train" if d == 0 else None)
            ax.scatter(b_energy["dist"][:10], b_energy["pred"][:10], 
                    s=100, c='b', marker='x', label="Franken train" if d == 0 else None)
            ax.scatter(b_energy["dist"][10:], b_energy["true"][10:], 
                    s=100, edgecolors='r', marker='o', c="none", label="True test" if d == 0 else None)
            ax.scatter(b_energy["dist"][10:], b_energy["pred"][10:], 
                    s=100, c='r', marker='x', label="Franken test" if d == 0 else None)
        ax.set_xlabel("Distance (A)")
        ax.set_ylabel("Energy (eV)")
        ax.legend(loc="best")
        plt.show()
    if True:  # train LES franken
        for dataset in datasets:
            franken_dir = get_newest_rundir(
                pathlib.Path(f"franken_outputs/dimer_{dataset.id}")
            )
            with open(franken_dir / "best.json", "r") as fh:
                best_log = json.load(fh)
            slv_cfg = SolverConfig(
                l2_penalty=1e-11,#best_log["hyperparameters"]["solver"]["l2_penalty"],
                energy_weight=1,
                force_weight=1#best_log["hyperparameters"]["solver"]["forces_weight"],
            )
            train_franken_les(
                dset=dataset,
                solver=slv_cfg,
                bbone=bb_cfg,
                rfs=rf_cfg,
            )
            # Only do the first dimer to start!
            break



# Lattice="30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0" 
# Properties=
# species:S:1:pos:R:3:forces:R:3 
# dimer_id=0 
# label=CC 
# energy=-13964.1459944979 
# chargeA=1 
# energyA=-7741.80993019298 
# chargeB=-1 
# energyB=-6221.39955938734 
# indexB=16 
# distance=7.096956324842526 
# distance_initial=6.630174209406314 
# pbc="T T T"
