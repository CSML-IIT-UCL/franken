import os
import random
from pathlib import Path

import numpy
import pytest
import torch

from franken import FRANKEN_DIR
from franken.backbones.utils import CacheDir, download_checkpoint
from franken.config import MaceBackboneConfig

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
]  # , "SchNet-S2EF-OC20-All"]  # List of gnn_ids to download

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


@pytest.fixture(autouse=True)
def random_seed():
    """This fixture is called before each test and sets random seeds"""
    random.seed(14)
    numpy.random.seed(14)
    torch.manual_seed(14)
