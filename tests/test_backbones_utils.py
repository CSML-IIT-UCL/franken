import os
from pathlib import Path
from unittest.mock import patch

import pytest

import franken.backbones.utils


@pytest.fixture
def mock_registry():
    return {
        "UNIMPLEMENTED_MODEL": {
            "kind": "mock",
            "implemented": False,
            "local": "unimplemented.ckpt",
            "remote": "https://example.com",
        }
    }


@pytest.fixture
def mock_cache_folder():
    return Path("/tmp/cache")


def test_model_registry():
    registry = franken.backbones.utils.load_model_registry()
    for model in registry.values():
        for key in ["remote", "local", "kind", "implemented"]:
            assert key in model.keys()


def test_cache_dir_default(mock_cache_folder):
    """Test that the function returns the default path when FRANKEN_CACHE_DIR is not set."""
    # Ensure no environment variable is set
    with patch.dict(os.environ, {}, clear=True):
        # Mock the home path and the Path.exists method
        with patch("pathlib.Path.home", return_value=mock_cache_folder):
            with patch("pathlib.Path.exists", return_value=True) as mock_exists:
                # Call the function
                franken.backbones.utils.CacheDir.initialize()
                result = franken.backbones.utils.CacheDir.get()

                # Check the default path is returned
                assert result == mock_cache_folder / ".franken"
                # Ensure that the path exists
                mock_exists.assert_called_once()
                # Reset cache dir. Avoids leaving stale state around
                franken.backbones.utils.CacheDir.directory = None


def test_cache_dir_with_env_var(mock_cache_folder):
    """Test that the function returns the correct path when FRANKEN_CACHE_DIR is set."""
    # Mock the environment variable
    with patch.dict(os.environ, {"FRANKEN_CACHE_DIR": str(mock_cache_folder)}):
        # Mock the Path.exists method
        with patch("pathlib.Path.exists", return_value=True) as mock_exists:
            # Call the function
            franken.backbones.utils.CacheDir.initialize()
            result = franken.backbones.utils.CacheDir.get()

            # Check the environment variable path is returned
            assert str(result) == str(mock_cache_folder)
            # Ensure that the path exists
            mock_exists.assert_called_once()
            # Reset cache dir. Avoids leaving stale state around
            franken.backbones.utils.CacheDir.directory = None


def test_download_checkpoint_name_error():
    """Test that a NameError is raised for unknown gnn_backbone_id."""
    # Mock the model registry to return an empty registry
    with patch("franken.backbones.utils.load_model_registry", return_value={}):
        # Expect a NameError when the gnn_backbone_id is not in the registry
        with pytest.raises(NameError) as exc_info:
            franken.backbones.utils.download_checkpoint("UNKNOWN_MODEL")
        assert "Unknown UNKNOWN_MODEL GNN backbone" in str(exc_info.value)


def test_download_checkpoint_not_implemented(mock_registry):
    """Test that a NotImplementedError is raised when the model is not implemented."""
    # Mock the model registry to return a registry with a model that is not implemented
    with patch(
        "franken.backbones.utils.load_model_registry", return_value=mock_registry
    ):
        # Expect a NotImplementedError when the gnn_backbone_id is not implemented
        with pytest.raises(NotImplementedError) as exc_info:
            franken.backbones.utils.download_checkpoint("UNIMPLEMENTED_MODEL")
        assert "The model UNIMPLEMENTED_MODEL is not implemented" in str(exc_info.value)


@pytest.mark.skip(reason="Actually downloads the model")
def test_download_checkpoint_successful_download(tmp_path):
    gnn_id = "MACE-L0"
    """Test that the model is downloaded correctly when it is implemented."""
    registry = franken.backbones.utils.load_model_registry()
    with patch.dict(os.environ, {"FRANKEN_CACHE_DIR": str(tmp_path)}):
        franken.backbones.utils.download_checkpoint(gnn_id)
        ckpt = tmp_path / "gnn_checkpoints" / registry[gnn_id]["local"]
        assert ckpt.exists()
        assert ckpt.is_file()


def test_get_checkpoint_path_valid_backbone(mock_registry, mock_cache_folder):
    with patch(
        "franken.backbones.utils.load_model_registry", return_value=mock_registry
    ), patch(
        "franken.backbones.utils.CacheDir.get", return_value=mock_cache_folder
    ), patch("pathlib.Path.exists", return_value=True):
        result = franken.backbones.utils.get_checkpoint_path("UNIMPLEMENTED_MODEL")
        expected_path = mock_cache_folder / "gnn_checkpoints" / "unimplemented.ckpt"
        assert result == expected_path


def test_get_checkpoint_path_invalid_backbone(mock_registry):
    with patch(
        "franken.backbones.utils.load_model_registry", return_value=mock_registry
    ), patch(
        "franken.backbones.utils.make_summary", return_value="available backbones"
    ):
        with pytest.raises(FileNotFoundError) as exc_info:
            franken.backbones.utils.get_checkpoint_path("invalid_backbone")

        assert "GNN Backbone path 'invalid_backbone' does not exist." in str(exc_info.value)
        assert "available backbones" in str(exc_info.value)


def test_get_checkpoint_path_download_required(mock_registry, mock_cache_folder):
    with patch(
        "franken.backbones.utils.load_model_registry", return_value=mock_registry
    ), patch(
        "franken.backbones.utils.CacheDir.get", return_value=mock_cache_folder
    ), patch("pathlib.Path.exists", return_value=False), patch(
        "franken.backbones.utils.download_checkpoint"
    ) as mock_download:
        result = franken.backbones.utils.get_checkpoint_path("UNIMPLEMENTED_MODEL")
        expected_path = mock_cache_folder / "gnn_checkpoints" / "unimplemented.ckpt"
        assert result == expected_path
        mock_download.assert_called_once_with("UNIMPLEMENTED_MODEL")
