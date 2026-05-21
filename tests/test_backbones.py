import os
from packaging.version import Version

import pytest
import torch
import importlib.util
import e3nn

from franken.backbones import REGISTRY
from franken.backbones.utils import get_checkpoint_path, load_checkpoint
from franken.config import BackboneConfig, GaussianRFConfig
from franken.data import FrankenAtomsDataset
from franken.data.base import ENERGY_TARGET_KEY, FORCES_TARGET_KEY
from franken.datasets.registry import DATASET_REGISTRY
from franken.rf.model import FrankenPotential

from franken.utils.misc import no_jit

# Check availability of backbones
HAS_MACE = importlib.util.find_spec("mace") is not None
HAS_SEVENN = importlib.util.find_spec("sevenn") is not None
HAS_UPET = True

# Build parametrized model list with skip marks when deps are missing
models = []
for name in REGISTRY.keys():
    kind = REGISTRY[name]["kind"]
    marks = []
    if name in {"PET_OMat/xl_1.0", "PET_OMat/l_1.0", "mace_off/large", "mace_omol/0_4M", "mace_mp/large-0b2", "mace_mp/large"}:
        marks.append(pytest.mark.skip(reason=f"{name} requires too large a model"))
        continue
    if (kind == "mace" and not HAS_MACE) or (kind == "sevenn" and not HAS_SEVENN):
        marks.append(pytest.mark.skip(reason=f"{kind} not installed"))
    elif kind == "mace":
        marks.append(pytest.mark.xfail(Version(e3nn.__version__) >= Version("0.5.5"), reason="Known incompatibility", strict=True))
    elif kind == "sevenn":
        marks.append(pytest.mark.xfail(Version(e3nn.__version__) < Version("0.5.0"), reason="Known incompatibility", strict=True))
    models.append(pytest.param(name, marks=marks))


@pytest.mark.parametrize("model_name", models)
def test_backbone_loading(model_name):
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt(
        {
            "family": registry_entry["kind"],
            "path_or_id": model_name,
        }
    )
    ckpt_path = get_checkpoint_path(gnn_config.path_or_id)
    load_checkpoint(gnn_config)
    assert os.path.isfile(ckpt_path), f"No file found at {ckpt_path}"


@pytest.mark.parametrize("model_name", models)
def test_data_loading(model_name):
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt(
        {
            "family": registry_entry["kind"],
            "path_or_id": model_name,
        }
    )
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = FrankenAtomsDataset(
        data_path=data_path,
        split="train",
        gnn_config=gnn_config,
    )
    config, target = dataset[0] # pyright: ignore[reportGeneralTypeIssues]
    assert config.atom_pos.shape == (2, 3)


@pytest.mark.parametrize("model_name", models)
def test_descriptors(model_name):
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt(
        {
            "family": registry_entry["kind"],
            "path_or_id": model_name,
        }
    )
    bbone = load_checkpoint(gnn_config)
    # Get a random data sample
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = FrankenAtomsDataset(
        data_path=data_path,
        split="train",
        gnn_config=gnn_config,
    )
    data, _ = dataset[0]  # type: ignore
    features = bbone.descriptors(data)
    expected_fdim = bbone.feature_dim()
    assert features.shape[1] == expected_fdim


@pytest.mark.parametrize("model_name", models)
def test_force_maps(model_name):
    from franken.backbones.wrappers.common_patches import patch_e3nn

    patch_e3nn()
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt(
        {
            "family": registry_entry["kind"],
            "path_or_id": model_name,
        }
    )
    # Get a random data sample
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = FrankenAtomsDataset(
        data_path=data_path,
        split="train",
        gnn_config=gnn_config,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # initialize model
    model = FrankenPotential(
        gnn_config=gnn_config,
        rf_config=GaussianRFConfig(num_random_features=128, length_scale=1.0),
    )
    model = model.to(device)
    dataset_el = dataset[0]
    assert isinstance(dataset_el, tuple)
    data = dataset_el[0].to(device)
    with torch.no_grad(), no_jit():
        # Need to call this multiple times to make sure test passes!
        fmaps = model.grad_feature_map(data, targets=[ENERGY_TARGET_KEY, FORCES_TARGET_KEY])
        fmaps = model.grad_feature_map(data, targets=[ENERGY_TARGET_KEY, FORCES_TARGET_KEY])
        fmaps = model.grad_feature_map(data, targets=[ENERGY_TARGET_KEY, FORCES_TARGET_KEY])
