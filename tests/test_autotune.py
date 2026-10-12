from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import pytest

from franken.trainers.log_utils import LogCollection
from franken.config import AutotuneConfig, DatasetConfig, GaussianRFConfig, HPSearchConfig, MaceBackboneConfig, MultiscaleGaussianRFConfig, SolverConfig
from franken.autotune.cli import parse_cli
from franken.autotune.script import init_loaders, run_autotune
from franken.data.base import ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY, TargetType
from franken.trainers.rf_trainer import RandomFeaturesTrainer
from franken.datasets.registry import DATASET_REGISTRY

from .conftest import DEFAULT_GNN_CONFIGS, DEVICES
from .utils import cleanup_dir, create_temp_dir

RF_PARAMETRIZE = [
    GaussianRFConfig(num_random_features=128, length_scale=1.0),
    MultiscaleGaussianRFConfig(num_random_features=128),
]
ALL_TARGETS: list[TargetType] = [ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY]



@pytest.mark.parametrize("gnn_cfg", DEFAULT_GNN_CONFIGS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("atomic_energies", [None, {7: 1.0, 26: 10.0}])
def test_integration(gnn_cfg, device, atomic_energies):
    loaders = init_loaders(
        gnn_cfg,
        DATASET_REGISTRY.get_path("test", "train", None, False),
        test_path=DATASET_REGISTRY.get_path("test", "test", None, False),
        val_path=DATASET_REGISTRY.get_path("test", "val", None, False),
    )
    rf_cfg = GaussianRFConfig(
        num_random_features=128,
        length_scale=HPSearchConfig(values=[0.5, 1.0]),
    )
    cfg = AutotuneConfig(
        dataset=DatasetConfig(),
        solver=SolverConfig(l2_penalty=[1e-4], force_weight=np.linspace(0.1, 0.9, 2).tolist()),
        backbone=gnn_cfg,
        rfs=rf_cfg,
        les=None,
        train_targets=[ENERGY_TARGET_KEY, FORCES_TARGET_KEY],
        atomic_energies=atomic_energies,
        save_every_model=False,
        rf_normalization=None
    )
    temp_dir = None
    try:
        # Step 1: Create a temporary directory for saving the model
        temp_dir = Path(create_temp_dir())
        run_autotune(
            auto_cfg=cfg,
            loaders=loaders,
            device=device,
            log_dir=temp_dir,
        )
        print(f"{list(temp_dir.glob('*'))}")
        assert (temp_dir / "best.json").is_file()
        assert (temp_dir / "log.json").is_file()
        assert (temp_dir / "best_ckpt.pt").is_file()
        logs = LogCollection.from_json(temp_dir / "log.json")
        assert len(logs) == 4
    finally:
        if temp_dir is not None:
            cleanup_dir(str(temp_dir))


def test_parse_cli():
    cfg = parse_cli([
        "--train-path",
        "/tmp/train.xyz",
        "--val-path",
        "/tmp/val.xyz",
        "--backbone",
        "mace",
        "--mace.path-or-id",
        "mace_mp/small",
        "--rf",
        "gaussian",
        "--gaussian.num-rf",
        "128",
        "--gaussian.length-scale",
        "[1.,2.]",
        "--train-targets",
        "energy",
        "forces",
        "stress",
        "--stress-weight",
        "(0.1,0.9,2,linear)",
        "--metrics",
        "energy_MAE",
        "forces_MAE",
        "stress_MAE",
        "--run-dir",
        ".",
    ])

    assert cfg.train_targets == [ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY]
    assert cfg.metrics == ["energy_MAE", "forces_MAE", "stress_MAE"]
    assert cfg.solver.stress_weight == HPSearchConfig(start=0.1, stop=0.9, num=2, scale="linear")
