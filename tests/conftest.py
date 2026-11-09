import os
import random
from pathlib import Path

import numpy
import pytest
import torch

from franken import FRANKEN_DIR
from franken.backbones.utils import CacheDir, download_checkpoint
from franken.config import MaceBackboneConfig
from franken.trainers.log_utils import LogCollection, LogEntry

__all__ = [
    "ROOT_PATH",
    "DEFAULT_GNN_CONFIGS",
    "SKIP_NO_CUDA",
    "DEVICES",
    "DEV_CPU_FAIL",
]

ROOT_PATH = FRANKEN_DIR

DEFAULT_GNN_CONFIGS = [
    MaceBackboneConfig("mace_mp/small")
]   # List of gnn_ids to download

SKIP_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=SKIP_NO_CUDA),  # type: ignore
]
DEV_CPU_FAIL = [
    pytest.param(
        dev, marks=pytest.mark.xfail(run=False, reason="Not implemented on CPU")
    )
    if dev == "cpu"
    else dev
    for dev in DEVICES
]


def prepare_gnn_checkpoints():
    # Ensure each gnn_id backbone is downloaded
    for gnn_cfg in DEFAULT_GNN_CONFIGS:
        download_checkpoint(gnn_cfg.path_or_id)
    return CacheDir.get() / "gnn_checkpoints"


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """

    # Cache-dir is either FRANKEN_CACHE_DIR if specified or in the repository folder.
    CacheDir.initialize(
        os.environ.get("FRANKEN_CACHE_DIR", Path(__file__).parent / ".franken")
    )

    prepare_gnn_checkpoints()


def do_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def random_seed():
    """This fixture is called before each test and sets random seeds"""
    do_random_seed(14)


@pytest.fixture
def log_collection():
    return LogCollection(logs=[LogEntry.from_dict(log) for log in [{
        "checkpoint": {"hash": "rand_uuid", "rf_weight_id": 0},
        "timings": {"cov_coeffs": 1.0, "solve": 1.0},
        "metrics": {
            "train": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
            "validation": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
            "test": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
        },
        "hyperparameters": {
            "franken": {
                "gnn_backbone_id": "mace_mp/small",
                "interaction_block": 3,
                "kernel_type": "gaussian",
            },
            "random_features": {
                "num_random_features": 1024,
            },
            "input_scaler": {"scale_by_Z": True, "num_species": 2},
            "solver": {
                "l2_penalty": 1e-6,
                "forces_weight": 0.1,
                "dtype": "torch.float64",
            },
        },
    },{
        "checkpoint": {"hash": "rand_uuid2", "rf_weight_id": 1},
        "timings": {"cov_coeffs": 1.0, "solve": 1.0},
        "metrics": {
            "train": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
            "validation": {"energy_MAE": 1.0, "forces_MAE": 0.5, "forces_cosim": 1.0},
            "test": {"energy_MAE": 1.0, "forces_MAE": 1.0, "forces_cosim": 1.0},
        },
        "hyperparameters": {
            "franken": {
                "gnn_backbone_id": "mace_mp/small",
                "interaction_block": 3,
                "kernel_type": "gaussian",
            },
            "random_features": {
                "num_random_features": 1024,
            },
            "input_scaler": {"scale_by_Z": True, "num_species": 2},
            "solver": {
                "l2_penalty": 1e-6,
                "forces_weight": 0.1,
                "dtype": "torch.float64",
            },
        },
    }]])

