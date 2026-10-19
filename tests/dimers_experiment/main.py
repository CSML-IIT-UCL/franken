"""Fit and evaluate short- and long-range Franken models on CC dimers.

LES optimization uses a smooth charge head and full-batch L-BFGS with variable
projection: the short-range RFF solution is updated at every LES trial point.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Literal

import ase
from ase.io import read, write
from matplotlib import pyplot as plt

import franken.autotune
import franken.data.base
from franken.config import (
    AutotuneConfig,
    BackboneConfig,
    DatasetConfig,
    LESConfig,
    MaceBackboneConfig,
    MultiscaleGaussianRFConfig,
    RFConfig,
    SolverConfig,
)
from franken.backbones.utils import CacheDir
from franken.datasets import DATASET_REGISTRY
from franken.data.dataset import FrankenAtomsDataset
from franken.rf.model import FrankenPotential
from plot_training_history import plot_training_history
from plot_HPGridSearch import plot_HPGridSearch


State = Literal["CC", "CP", "PP"]
ENERGY = franken.data.base.ENERGY_TARGET_KEY
FORCES = franken.data.base.FORCES_TARGET_KEY
SCRIPT_DIR = Path(__file__).parent
DATASET_NAME = "dimers_CC_0"


@dataclass
class DimersDataset:
    label: State
    id: int
    energy_a: float
    energy_b: float
    data: list[ase.Atoms]
    distances: list[float]


def load_dataset() -> DimersDataset:
    """Load the registered train/validation splits for the first CC scan."""
    data = []
    for split in ("train", "val"):
        path = DATASET_REGISTRY.get_path(DATASET_NAME, split, CacheDir.get())
        data.extend(read(path, index=":"))
    first = data[0]
    return DimersDataset(
        label=first.info["label"],
        id=int(first.info["dimer_id"]),
        energy_a=first.info["energyA"],
        energy_b=first.info["energyB"],
        data=data,
        distances=[atoms.info["distance"] for atoms in data],
    )


def train_model(
    dset: DimersDataset,
    solver: SolverConfig,
    backbone: BackboneConfig,
    rfs: RFConfig,
    *,
    use_les: bool,
    les_config: LESConfig | None = None,
) -> Path:
    """Run the short-range fit or the alternating RFF + LES fit."""
    kind = "les" if use_les else "franken"
    config = AutotuneConfig(
        dataset=DatasetConfig(name=DATASET_NAME),
        solver=solver,
        backbone=backbone,
        rfs=rfs,
        les=les_config if use_les else None,
        best_model_selection=["energy_MAE", "forces_MAE"],
        eval_splits=["val"],
        run_dir=str(SCRIPT_DIR / f"{kind}_outputs/dimer_{dset.id}"),
        # Match the explicit weights used by train_les_fresh.py. Spectral
        # normalization changes the relative energy/force weighting.
        rf_normalization=None,
        save_fmaps=use_les,
    )
    run_dir = franken.autotune.autotune(config)
    assert isinstance(run_dir, Path)
    if use_les:
        plot_training_history(run_dir)
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
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    write(eval_path, dset.data)
    # The new loader detects regular and LES checkpoints automatically.
    model = FrankenPotential.load(checkpoint)
    model.eval()
    dataset = FrankenAtomsDataset(
        data_path=eval_path,
        split="test",
        gnn_config=model.gnn_config,
    )
    return [model.predict(targets=[ENERGY, FORCES], data=data) for data, _ in dataset]


def evaluate_and_plot(dset: DimersDataset, *, use_les: bool) -> None:
    kind, title = ("les", "LES Franken") if use_les else ("franken", "Standard Franken")
    run = newest_run(SCRIPT_DIR / f"{kind}_outputs/dimer_{dset.id}")
    with (run / "best.json").open() as file:
        metrics = json.load(file)["metrics"]["validation"]
    print(f"{title}.")
    for m_name, m_val in metrics.items():
        print(f"\t{m_name}: {m_val:.2f}")

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
    datasets = [load_dataset()]
    print(f"Loaded {len(datasets)} datasets")

    backbone = MaceBackboneConfig("mace_mp/small")
    rfs = MultiscaleGaussianRFConfig(
        num_random_features=2048,
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6,
    )
    # full_batch_adam = LESConfig(
    #     optimizer="adam",
    #     num_cycles=50,
    #     epochs_per_cycle=20,
    #     batch_size=None,
    #     learning_rate=1e-4,
    #     restore_best=True,
    # )

    les_config = LESConfig(
        hidden_dim=(64, 32),
        les_output_scale=1.0,
        # dl=3,
        N_max=20,
        optimizer="lbfgs",
        mode="variable_projection",
        num_cycles=15,
        batch_size=None,
        lbfgs_learning_rate=1.0,
        lbfgs_max_iter=20,  # [10, 20, 40, 80],
        lbfgs_history_size=20,  # [20, 80]
    )

    for dset in datasets:
        # train_model(
        #     dset,
        #     SolverConfig(
        #         l2_penalty=[1e-8, 1e-6],
        #         force_weight=[1.0, 0.1],
        #     ),
        #     backbone,
        #     rfs,
        #     use_les=False,
        # )
        # evaluate_and_plot(dset, use_les=False)

        # sr_run = newest_run(SCRIPT_DIR / f"franken_outputs/dimer_{dset.id}")
        # with (sr_run / "best.json").open() as file:
        #     best_solver = json.load(file)["hyperparameters"]["solver"]
        train_model(
            dset,
            SolverConfig(
                l2_penalty=[1e-8, 1e-6],
                energy_weight=1.0,
                force_weight=[1.0, 0.1],
            ),
            backbone,
            rfs,
            use_les=True,
            les_config=les_config,
        )
        evaluate_and_plot(dset, use_les=True)

        kind = "les"
        run = newest_run(SCRIPT_DIR / f"{kind}_outputs/dimer_{dset.id}")
        hyperparameter_1 = "forces_relative_weight"
        hyperparameter_2 = "l2_penalty"
        plot_HPGridSearch(run, hyperparameter_1, hyperparameter_2, kind)
