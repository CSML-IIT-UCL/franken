"""Fit and evaluate short- and long-range Franken models on CC dimers.

This is the full-batch Adam version of ``main-mgb.py``.  LES optimization uses
the public ``LESTrainingConfig`` API: each Adam epoch accumulates the gradient
over every training configuration before performing one parameter update.
"""

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
import json
from pathlib import Path
import re
from typing import Literal

import ase
from ase.io import read, write
from matplotlib import pyplot as plt
import numpy as np

import franken.autotune
import franken.data.base
from franken.config import (
    AutotuneConfig,
    BackboneConfig,
    DatasetConfig,
    LESConfig,
    LESTrainingConfig,
    MaceBackboneConfig,
    MultiscaleGaussianRFConfig,
    RFConfig,
    SolverConfig,
)
from franken.data.dataset import FrankenAtomsDataset
from franken.rf.model import FrankenPotential
from plot_training_history import plot_training_history


State = Literal["CC", "CP", "PP"]
ENERGY = franken.data.base.ENERGY_TARGET_KEY
FORCES = franken.data.base.FORCES_TARGET_KEY
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR.parent.parent / "franken/datasets/dimers/bio_dimers.xyz"


@dataclass
class DimersDataset:
    label: State
    id: int
    energy_a: float
    energy_b: float
    data: list[ase.Atoms]
    distances: list[float]

    def split(self) -> tuple[list[ase.Atoms], list[ase.Atoms]]:
        """Use short-distance structures for training and distant ones for validation."""
        return self.data[:10], self.data[10:]


def get_datasets(allowed_states: set[State], dimers: list[ase.Atoms]):
    """Yield one dataset for each contiguous dimer-id group in the source file."""
    selected = (dimer for dimer in dimers if dimer.info["label"] in allowed_states)
    for dimer_id, group in groupby(
        selected, key=lambda dimer: int(dimer.info["dimer_id"])
    ):
        data = list(group)
        first = data[0]
        yield DimersDataset(
            label=first.info["label"],
            id=dimer_id,
            energy_a=first.info["energyA"],
            energy_b=first.info["energyB"],
            data=data,
            distances=[dimer.info["distance"] for dimer in data],
        )


def dataset_config(dset: DimersDataset, name: str) -> DatasetConfig:
    folder = SCRIPT_DIR / f"dimer_{dset.id}_dataset"
    folder.mkdir(exist_ok=True)
    train, validation = dset.split()
    train_path, validation_path = folder / "train.xyz", folder / "test.xyz"
    write(train_path, train)
    write(validation_path, validation)
    return DatasetConfig(
        name=name,
        train_path=str(train_path),
        val_path=str(validation_path),
    )


def train_model(
    dset: DimersDataset,
    solver: SolverConfig,
    backbone: BackboneConfig,
    rfs: RFConfig,
    *,
    use_les: bool,
    les_training: LESTrainingConfig | None = None,
) -> Path:
    """Run the short-range fit or the alternating RFF + LES fit."""
    kind = "les" if use_les else "franken"
    config = AutotuneConfig(
        dataset=dataset_config(dset, f"dimer_{kind}_{dset.id}"),
        solver=solver,
        backbone=backbone,
        rfs=rfs,
        les=(
            LESConfig(hidden_dim=(64, 32), les_output_scale=1.0, dl=3)
            if use_les
            else None
        ),
        les_training=les_training or LESTrainingConfig(),
        best_model_selection=["energy_MAE", "forces_MAE"],
        eval_splits=["val"],
        run_dir=str(SCRIPT_DIR / f"{kind}_outputs/dimer_{dset.id}"),
    )
    run_dir = franken.autotune.autotune(config)
    assert isinstance(run_dir, Path)
    if use_les:
        plot_training_history(run_dir / "training_history.json")
    return run_dir / "best_ckpt.pt"


def newest_run(folder: Path) -> Path:
    """Return the most recent autotune run directory in *folder*."""
    pattern = re.compile(r"run_(\d{6}_\d{6})_[A-Za-z0-9]+$")
    runs = []
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            runs.append((datetime.strptime(match[1], "%y%m%d_%H%M%S"), path))
    if not runs:
        raise FileNotFoundError(f"No run directories found in {folder}")
    return max(runs, key=lambda run: run[0])[1]


def predict(dset: DimersDataset, checkpoint: Path):
    """Predict energy and forces for every separation in a dimer scan."""
    eval_path = SCRIPT_DIR / f"dimer_{dset.id}_dataset/all.xyz"
    write(eval_path, dset.data)
    # The new loader detects regular and LES checkpoints automatically.
    model = FrankenPotential.load(checkpoint)
    model.eval()
    dataset = FrankenAtomsDataset(
        data_path=eval_path,
        split="train",
        gnn_config=model.gnn_config,
    )
    return [model.predict(targets=[ENERGY, FORCES], data=data) for data, _ in dataset]


def evaluate_and_plot(dset: DimersDataset, *, use_les: bool) -> None:
    kind, title = ("les", "LES Franken") if use_les else ("franken", "Standard Franken")
    run = newest_run(SCRIPT_DIR / f"{kind}_outputs/dimer_{dset.id}")
    with (run / "best.json").open() as file:
        metrics = json.load(file)["metrics"]["validation"]
    print(f"{title}.")
    print(f"\tEnergy RMSE: {metrics['energy_RMSE']:.1f}")
    print(f"\tForces RMSE: {metrics['forces_RMSE']:.1f}")

    prediction = predict(dset, run / "best_ckpt.pt")
    monomer_energy = dset.energy_a + dset.energy_b
    reference = [atoms.get_potential_energy() - monomer_energy for atoms in dset.data]
    predicted = [result[ENERGY].item() - monomer_energy for result in prediction]

    fig, ax = plt.subplots()
    for values, color, label, marker in (
        (reference, "b", "True", "o"),
        (predicted, "b", title, "x"),
    ):
        train_style = (
            {"edgecolors": color, "c": "none"} if marker == "o" else {"c": color}
        )
        ax.scatter(
            dset.distances[:10],
            values[:10],
            s=100,
            marker=marker,
            label=f"{label} train",
            **train_style,
        )
        test_style = {"edgecolors": "r", "c": "none"} if marker == "o" else {"c": "r"}
        ax.scatter(
            dset.distances[10:],
            values[10:],
            s=100,
            marker=marker,
            label=f"{label} test",
            **test_style,
        )
    ax.set(xlabel="Distance (A)", ylabel="Energy (eV)")
    ax.legend(loc="best")
    fig.savefig(SCRIPT_DIR / f"{'LR' if use_les else 'SR'}_binding_energy.png")
    plt.close(fig)


if __name__ == "__main__":
    # Download from https://archive.materialscloud.org/records/405an-d8183
    datasets = list(get_datasets({"CC"}, read(DATA_PATH, index=":")))
    print(f"Loaded {len(datasets)} datasets")

    backbone = MaceBackboneConfig("mace_mp/small")
    rfs = MultiscaleGaussianRFConfig(
        num_random_features=2048,
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6,
    )
    full_batch_adam = LESTrainingConfig(
        optimizer="adam",
        num_cycles=50,
        epochs_per_cycle=50,
        batch_size=None,
        learning_rate=1e-4,
        restore_best=True,
    )

    # Retain [:1] for the current dimer-0 experiment; remove it for all CC dimers.
    for dset in datasets[:1]:
        train_model(
            dset,
            SolverConfig(
                l2_penalty=np.logspace(-11, -6, 6).tolist(),
                force_weight=np.logspace(-2, 3, 6).tolist(),
            ),
            backbone,
            rfs,
            use_les=False,
        )
        evaluate_and_plot(dset, use_les=False)

        sr_run = newest_run(SCRIPT_DIR / f"franken_outputs/dimer_{dset.id}")
        with (sr_run / "best.json").open() as file:
            best_solver = json.load(file)["hyperparameters"]["solver"]
        train_model(
            dset,
            SolverConfig(
                l2_penalty=best_solver["l2_penalty"],
                energy_weight=best_solver["energy_weight"],
                force_weight=best_solver["forces_weight"],
            ),
            backbone,
            rfs,
            use_les=True,
            les_training=full_batch_adam,
        )
        evaluate_and_plot(dset, use_les=True)
