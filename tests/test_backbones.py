import pytest
import torch
import importlib.util
import e3nn

from franken.backbones import REGISTRY
from franken.backbones.utils import load_checkpoint
from franken.config import BackboneConfig, GaussianRFConfig
from franken.data import BaseAtomsDataset
from franken.datasets.registry import DATASET_REGISTRY
from franken.rf.model import FrankenPotential

from packaging.version import Version

from franken.utils.misc import no_jit

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
    elif "SchNet" in name:
        marks.append(pytest.mark.xfail(reason="Fails in CI due to unknown reasons", strict=False))
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
    with torch.no_grad(), no_jit():
        # Need to call this multiple times to make sure test passes!
        emap, fmap = model.grad_feature_map(data)
        emap, fmap = model.grad_feature_map(data)
        emap, fmap = model.grad_feature_map(data)
