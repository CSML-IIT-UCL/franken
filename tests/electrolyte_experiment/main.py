"""Fit and compare standard and LES Franken models on bulk KF with one ion pair.

The registry downloads electrolyte_bulk_1 and supplies its seeded 161/40
train/validation split. Run with: python tests/electrolyte_experiment/main.py
"""

import json
from pathlib import Path

import franken.autotune
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
from plot_HPGridSearch import plot_HPGridSearch
from plot_training_history import plot_training_history


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAME = "electrolyte_bulk_1"


def train_model(
    solver: SolverConfig,
    backbone: BackboneConfig,
    rfs: RFConfig,
    *,
    use_les: bool,
    les_config: LESConfig | None = None,
) -> Path:
    """Train on the registered splits and return this run's best checkpoint."""
    if use_les and les_config is None:
        raise ValueError("LES training requires les_config")
    kind = "les" if use_les else "franken"
    config = AutotuneConfig(
        dataset=DatasetConfig(name=DATASET_NAME, max_train_samples = 10),
        solver=solver,
        backbone=backbone,
        rfs=rfs,
        les=les_config if use_les else None,
        best_model_selection=["energy_RMSE", "forces_RMSE"],
        eval_splits=["val"],
        run_dir=str(SCRIPT_DIR / f"{kind}_outputs" / DATASET_NAME),
        rf_normalization=None,
        save_fmaps=use_les,
        jac_chunk_size=2,
    )
    run_dir = franken.autotune.autotune(config)
    assert isinstance(run_dir, Path)
    if use_les:
        plot_training_history(run_dir)
    plot_HPGridSearch(run_dir, "forces_weight", "l2_penalty", str(run_dir / kind))
    return run_dir / "best_ckpt.pt"


def evaluate(checkpoint: Path, *, use_les: bool) -> None:
    """Report validation errors for the best model from the completed run."""
    title = "LES Franken" if use_les else "Standard Franken"
    with checkpoint.with_name("best.json").open() as file:
        metrics = json.load(file)["metrics"]["validation"]
    print(f"{title} on {DATASET_NAME}:")
    print(f"\tEnergy RMSE: {metrics['energy_RMSE']:.1f} meV/atom")
    print(f"\tForces RMSE: {metrics['forces_RMSE']:.1f} meV/Å")
    print(f"\tCheckpoint: {checkpoint}")


def main() -> None:
    backbone = MaceBackboneConfig("mace_mp/small")
    rfs = MultiscaleGaussianRFConfig(
        num_random_features=512,
        length_scale_low=4,
        length_scale_high=20,
        length_scale_num=6,
    )
    les_config = LESConfig(
        hidden_dim=(64, 32),
        les_output_scale=1.0,
        N_max=20,
        optimizer="lbfgs",
        mode="variable_projection",
        num_cycles=20,
        batch_size=None,
        lbfgs_learning_rate=1.0,
        lbfgs_max_iter=20,
        lbfgs_history_size=20,
    )

    checkpoint = train_model(
        SolverConfig(l2_penalty=[1e-8, 1e-6], force_weight=[1.0, 0.1]),
        backbone,
        rfs,
        use_les=False,
    )
    evaluate(checkpoint, use_les=False)

    checkpoint = train_model(
        SolverConfig(l2_penalty=1e-11, force_weight=1e-2, energy_weight=1.0),
        backbone,
        rfs,
        use_les=True,
        les_config=les_config,
    )
    evaluate(checkpoint, use_les=True)


if __name__ == "__main__":
    main()
