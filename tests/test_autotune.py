from copy import deepcopy
from pathlib import Path
import os

from tests.test_franken_calculator import GNN_CONFIGS

os.environ["OMP_NUM_THREADS"] = "8"
from unittest.mock import DEFAULT, patch

import numpy as np
import pytest
import torch

from franken.trainers.log_utils import LogCollection
from franken.config import GaussianRFConfig, HPSearchConfig, MultiscaleGaussianRFConfig
from franken.calculators.ase_calc import FrankenCalculator
from franken.autotune.cli import parse_cli
from franken.autotune.script import init_loaders, run_autotune
from franken.data import FrankenAtomsDataset
from franken.data.base import ENERGY_TARGET_KEY, FORCES_TARGET_KEY, STRESS_TARGET_KEY, Configuration, TargetType
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics
from franken.trainers.rf_trainer import RandomFeaturesTrainer
from franken.utils.misc import garbage_collection_cuda
from franken.datasets.registry import DATASET_REGISTRY

from .conftest import DEFAULT_GNN_CONFIGS, DEVICES
from .utils import are_dicts_close, cleanup_dir, create_temp_dir, mocked_gnn

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
    temp_dir = None
    try:
        # Step 1: Create a temporary directory for saving the model
        temp_dir = Path(create_temp_dir())
        trainer = RandomFeaturesTrainer(
            train_dataloader=loaders["train"],
            random_features_normalization=None,
            log_dir=temp_dir,
            save_every_model=False,
            l2_penalty=[1e-4],
            target_weight={FORCES_TARGET_KEY: np.linspace(0.1, 0.9, 2).tolist()},
            training_targets=[ENERGY_TARGET_KEY, FORCES_TARGET_KEY],
            device=device,
        )
        run_autotune(
            gnn_cfg=gnn_cfg,
            rf_cfg=rf_cfg,
            loaders=loaders,
            scale_by_species=False,
            jac_chunk_size="auto",
            trainer=trainer,
            atomic_energies=atomic_energies,

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


def test_parse_cli_supports_stress_weight_and_metrics():
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
