import json

from franken.autotune import autotune
from franken.config import BackboneConfig, MaceBackboneConfig, DatasetConfig, MultiscaleGaussianRFConfig, SolverConfig, HPSearchConfig, AutotuneConfig

from franken.datasets.registry import DATASET_REGISTRY
from franken.backbones.utils import CacheDir


def run_test_for_bbone(
    gnn_config: BackboneConfig,
    n_train_samples: int,  # 128
    n_rf: int,  # 8192
    seed: int,
):
    train_path = DATASET_REGISTRY.get_path("water", "train", base_path=CacheDir.get())
    val_path = DATASET_REGISTRY.get_path("water", "val", base_path=CacheDir.get())

    dataset_cfg = DatasetConfig(train_path=str(train_path),
                                max_train_samples=n_train_samples,
                                val_path=str(val_path)
    )
    print("Dataset configuration")
    print(dataset_cfg)

    rf_config = MultiscaleGaussianRFConfig(
        num_random_features=n_rf,
        length_scale_low=8,
        length_scale_high=32,
        length_scale_num=4,
        rng_seed=seed,
    )
    solver_cfg = SolverConfig(
        l2_penalty=HPSearchConfig(start=-10, stop=-5, num=5, scale='log'),  # equivalent of numpy.logspace
        force_weight=HPSearchConfig(start=0.01, stop=0.99, num=5, scale='linear'),  # equivalent of numpy.linspace
    )
    autotune_cfg = AutotuneConfig(
        dataset=dataset_cfg,
        solver=solver_cfg,
        backbone=gnn_config,
        rfs=rf_config,
        metrics=["energy_MAE", "forces_MAE"],
        seed=seed,
        jac_chunk_size='auto',
        run_dir="./results",
    )

    run_path = autotune(autotune_cfg)
    assert run_path is not None

    with open(run_path / "best.json", "r") as fh:
        best_log = json.load(fh)
    best_ls = best_log["hyperparameters"]["random_features"]["length_scale"]
    best_l2 = best_log["hyperparameters"]["solver"]["l2_penalty"]
    best_fw = best_log["hyperparameters"]["solver"]["force_weight"]
    print("Best model hyperparameters: ")
    print(f"\tLength-scale: {best_ls:.1f}")
    print(f"\tL2 penalty: {best_l2:.2e}")
    print(f"\tForce-weight: {best_fw:.3f}")

    print("Best model accuracy:")
    print(f"\tforces MAE: {best_log['metrics']['validation']['forces_MAE']}")
    print(f"\tenergy MAE: {best_log['metrics']['validation']['energy_MAE']}")

    print("Best model timings:")
    print(f"\tCov/Coeff time (s): {best_log['timings']['cov_coeffs']}")


if __name__ == "__main__":
    gnn_config = MaceBackboneConfig(
        path_or_id="mace_mp/small",
        interaction_block=2,
    )
    run_test_for_bbone(
        gnn_config=gnn_config,
        n_train_samples=128,
        n_rf=8192,
        seed=1,
    )
