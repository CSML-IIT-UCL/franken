import shutil
import tempfile
from unittest.mock import MagicMock, patch

import torch


# Utility function to create a temporary directory
def create_temp_dir() -> str:
    return tempfile.mkdtemp()


# Utility function to clean up a directory
def cleanup_dir(temp_dir: str):
    shutil.rmtree(temp_dir)


def are_dicts_close(dict1, dict2, rtol=1e-4, atol=1e-6, verbose=False):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return False

    if set(dict1.keys()) != set(dict2.keys()):
        if verbose:
            print(f"Dictionaries have different keys: {set(dict1.keys())}, {set(dict2.keys())}")
        return False

    for key in dict1.keys():
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not are_dicts_close(dict1[key], dict2[key], rtol, atol, verbose=verbose):
                return False
        elif isinstance(dict1[key], torch.Tensor) and isinstance(
            dict2[key], torch.Tensor
        ):
            if not torch.allclose(dict1[key], dict2[key], rtol=rtol, atol=atol):
                if verbose:
                    print(f"{key} not equal:\n(1) {dict1[key]}\n(2) {dict2[key]}")
                return False
        else:
            if verbose:
                print("The dictionaries have differnt topology")
            return False
    return True


def mocked_gnn(device, dtype, feature_dim: int = 32, backbone_id: str = "test"):
    # A bunch of code to initialize a mock for the GNN
    gnn = MagicMock()
    gnn.feature_dim = MagicMock(return_value=feature_dim)
    gnn.cutoff_radius = MagicMock(return_value=6.0)
    gnn.num_interaction_layers = MagicMock(return_value=2)
    gnn.supported_atomic_types = MagicMock(
        return_value=torch.tensor([1, 6, 8], dtype=torch.int64)
    )
    gnn.franken_train = MagicMock(return_value=None)
    gnn.franken_val = MagicMock(return_value=None)

    fake_gnn_weight = torch.randn(3, feature_dim, device=device, dtype=dtype)

    def mock_descriptors(data):
        return torch.sin(data.atom_pos) @ fake_gnn_weight

    gnn.descriptors = mock_descriptors

    def load_checkpoint_patch(*args, **kwargs):
        gnn.init_args = MagicMock(return_value=dict(kwargs))
        return gnn

    return patch.multiple("franken.rf.model", load_checkpoint=load_checkpoint_patch)
