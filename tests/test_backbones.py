import pytest

from franken.config import BackboneConfig
from franken.data import BaseAtomsDataset
from franken.datasets.registry import DATASET_REGISTRY
from franken.backbones import REGISTRY
from franken.backbones.utils import load_checkpoint


@pytest.mark.parametrize("model_name", ["Egret-1", "MACE-L1", "MACE-OFF-small", "SevenNet0", "SchNet-S2EF-OC20-200k"])
def test_backbone_loading(model_name):
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt({
        "family": registry_entry["kind"],
        "path_or_id": model_name,
        "interaction_block": 2,
    })
    load_checkpoint(gnn_config)


@pytest.mark.parametrize("model_name", ["Egret-1", "MACE-L1", "MACE-OFF-small", "SevenNet0", "SchNet-S2EF-OC20-200k"])  #, "MACE-L1"])
def test_descriptors(model_name):
    registry_entry = REGISTRY[model_name]
    gnn_config = BackboneConfig.from_ckpt({
        "family": registry_entry["kind"],
        "path_or_id": model_name,
        "interaction_block": 2,
    })
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
