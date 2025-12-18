import pytest
import torch
import importlib.util

from franken.backbones import REGISTRY
from franken.backbones.utils import load_checkpoint
from franken.config import BackboneConfig, GaussianRFConfig
from franken.data import BaseAtomsDataset
from franken.datasets.registry import DATASET_REGISTRY
from franken.rf.model import FrankenPotential

# Check availability of backbones
HAS_MACE = importlib.util.find_spec("mace") is not None
HAS_SEVENN = importlib.util.find_spec("sevenn") is not None
HAS_FAIRCHEM = importlib.util.find_spec("fairchem") is not None

# Build parametrized model list with skip marks when deps are missing
models = []
for name in REGISTRY.keys():
    kind = REGISTRY[name]["kind"]    
    marks = []
    if (kind == "mace" and not HAS_MACE) or (kind == "sevenn" and not HAS_SEVENN) or (kind == "fairchem" and not HAS_FAIRCHEM):
        marks.append(pytest.mark.skip(reason=f"{kind} not installed"))
    models.append(pytest.param(name, marks=marks))

@pytest.mark.parametrize("model_name", models)
def test_backbone_loading(model_name):
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt(
        {
            "family": registry_entry["kind"],
            "path_or_id": model_name,
            "interaction_block": 2,
        }
    )
    load_checkpoint(gnn_config)


@pytest.mark.parametrize("model_name", models)
def test_descriptors(model_name):
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt(
        {
            "family": registry_entry["kind"],
            "path_or_id": model_name,
            "interaction_block": 2,
        }
    )
    bbone = load_checkpoint(gnn_config)
    # Get a random data sample
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = BaseAtomsDataset.from_path(
        data_path=data_path,
        split="train",
        gnn_config=gnn_config,
    )
    data, _ = dataset[0]  # type: ignore
    expected_fdim = bbone.feature_dim()
    features = bbone.descriptors(data)
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
            "interaction_block": 2,
        }
    )
    # Get a random data sample
    data_path = DATASET_REGISTRY.get_path("test", "train", None, False)
    dataset = BaseAtomsDataset.from_path(
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
    data, _ = dataset[0]  # type: ignore
    data = data.to(device)
    emap, fmap = model.grad_feature_map(data)
