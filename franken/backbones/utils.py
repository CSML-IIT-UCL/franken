import importlib.resources
import json
import logging
import os
from pathlib import Path

import requests
import torch

from franken.config import BackboneConfig, asdict_with_classvar
from franken.utils import distributed
from franken.utils.file_utils import download_file


logger = logging.getLogger("franken")


def load_model_registry():
    model_registry_text = (
        importlib.resources.files("franken.backbones")
        .joinpath("registry.json")
        .read_text()
    )
    model_registry = json.loads(model_registry_text)
    return model_registry


class CacheDir:
    directory: Path | None = None

    @staticmethod
    def initialize(cache_dir: Path | str | None = None):
        if CacheDir.is_initialized():
            logger.warning(
                f"Cache directory already initialized at {CacheDir.directory}. Reinitializing."
            )
        # Default cache location: ~/.franken
        default_cache = Path.home() / ".franken"
        if cache_dir is None:
            env_cache_dir = os.environ.get("FRANKEN_CACHE_DIR", None)
            if env_cache_dir is None:
                logger.info(f"Initializing default cache directory at {default_cache}")
                cache_dir = default_cache
            else:
                logger.info(
                    f"Initializing cache directory from $FRANKEN_CACHE_DIR {env_cache_dir}"
                )
                cache_dir = env_cache_dir
        else:
            logger.info(f"Initializing custom cache directory {cache_dir}")
        CacheDir.directory = Path(cache_dir)

        # Ensure the directory exists
        if not CacheDir.directory.exists():
            CacheDir.directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory at: {CacheDir.directory}")

    @staticmethod
    def get() -> Path:
        if not CacheDir.is_initialized():
            CacheDir.initialize()
        assert CacheDir.directory is not None
        return CacheDir.directory

    @staticmethod
    def is_initialized() -> bool:
        return CacheDir.directory is not None


def make_summary(cache_dir: str | None = None):
    """Function to print available models, first those present locally."""
    if cache_dir is not None:
        CacheDir.initialize(cache_dir=cache_dir)
    registry = load_model_registry()
    ckpt_dir = CacheDir.get() / "gnn_checkpoints"

    local_models = []
    remote_models = []
    _summary = ""
    for model, info in registry.items():
        local_path = ckpt_dir / info["local"]
        kind = info["kind"]
        implemented = info.get("implemented", False)
        if implemented:
            if local_path.is_file():
                local_models.append((model, kind))
            else:
                remote_models.append((model, kind))
    if len(local_models) > 0:
        _summary += f"{'DOWNLOADED MODELS':^80}\n"
        _summary += f"{'(' + str(ckpt_dir) + ')':-^80}\n"
        for model, kind in local_models:
            _str = f"{model} ({kind})"
            _summary += f"{_str:<40}\n"

    _summary += f"{'AVAILABLE MODELS':-^80}\n"
    for model, kind in remote_models:
        _str = f"{model} ({kind})"
        _summary += f"{_str:<80}\n"
    _summary += "-" * 80
    return _summary


def get_checkpoint_path(backbone_path_or_id: str) -> Path:
    """Fetches the path of a given backbone. If the backbone is not present, it will be downloaded.

    The backbone can be either specified directly via its file-system path,
    then this function is a thin wrapper -- or it can be specified via its
    ID in the model registry. Then this function takes care of finding the
    correct model path and potentially downloading the backbone from the internet.

    Args:
        backbone_path_or_id (str): file-system path to the backbone
            or the backbone's ID as per the model registry.

    Returns:
        Path: Path to the model on disk

    See Also:
        You can use the command :code:`franken.backbones list` from the command-line
        to find out which backbone IDs are supported out-of-the-box.
    """
    registry = load_model_registry()
    gnn_checkpoints_dir = CacheDir.get() / "gnn_checkpoints"

    if backbone_path_or_id not in registry.keys():
        if not os.path.isfile(backbone_path_or_id):
            raise FileNotFoundError(
                f"GNN Backbone path '{backbone_path_or_id}' does not exist. "
                f"You should either provide an existing backbone path or a backbone ID "
                f"from the registry of available backbones: \n{make_summary()}"
            )
        return Path(backbone_path_or_id)
    else:
        backbone_info = registry[backbone_path_or_id]
        ckpt_path = gnn_checkpoints_dir / backbone_info["local"]
        # Download checkpoint being aware of multiprocessing
        if distributed.get_rank() != 0:
            distributed.barrier()
        else:
            if not ckpt_path.exists():
                download_checkpoint(backbone_path_or_id)
            distributed.barrier()
    return ckpt_path


def download_checkpoint(gnn_backbone_id: str, cache_dir: str | None = None) -> None:
    """Download the model if it's not already present locally."""
    registry = load_model_registry()
    if cache_dir is not None:
        CacheDir.initialize(cache_dir=cache_dir)
    ckpt_dir = CacheDir.get() / "gnn_checkpoints"

    if gnn_backbone_id not in registry.keys():
        raise NameError(
            f"Unknown {gnn_backbone_id} GNN backbone, the current available backbones are\n{make_summary()}"
        )

    if not registry[gnn_backbone_id]["implemented"]:
        raise NotImplementedError(
            f"The model {gnn_backbone_id} is not implemented in franken yet."
        )

    local_path = ckpt_dir / registry[gnn_backbone_id]["local"]
    remote_path = registry[gnn_backbone_id]["remote"]

    if local_path.is_file():
        logger.info(
            f"Model already exists locally at {local_path}. No download needed."
        )
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model from {remote_path} to {local_path}")
    try:
        download_file(url=remote_path, filename=local_path, desc="Downloading model")
    except requests.RequestException as e:
        logger.error(f"Download failed. {e}")
        raise e


def load_checkpoint(gnn_config: BackboneConfig) -> torch.nn.Module:
    gnn_config_dict = asdict_with_classvar(gnn_config)
    gnn_backbone_id = gnn_config_dict.pop("path_or_id")
    backbone_family = gnn_config_dict.pop("family")
    ckpt_path = get_checkpoint_path(gnn_backbone_id)
    err_msg = f"franken wasn't able to load {gnn_backbone_id}. Is {backbone_family} installed?"
    if backbone_family == "fairchem":
        try:
            from franken.backbones.wrappers.fairchem_schnet import FrankenSchNetWrap
        except ImportError as import_err:
            logger.error(err_msg, exc_info=import_err)
            raise
        return FrankenSchNetWrap.load_from_checkpoint(
            str(ckpt_path), gnn_backbone_id=gnn_backbone_id, **gnn_config_dict
        )
    elif backbone_family == "mace":
        try:
            from franken.backbones.wrappers.mace_wrap import FrankenMACE
        except ImportError as import_err:
            logger.error(err_msg, exc_info=import_err)
            raise
        return FrankenMACE.load_from_checkpoint(
            str(ckpt_path), gnn_backbone_id=gnn_backbone_id, **gnn_config_dict
        )
    elif backbone_family == "sevenn":
        try:
            from franken.backbones.wrappers.sevenn import FrankenSevenn
        except ImportError as import_err:
            logger.error(err_msg, exc_info=import_err)
            raise
        return FrankenSevenn.load_from_checkpoint(
            ckpt_path, gnn_backbone_id=gnn_backbone_id, **gnn_config_dict
        )
    else:
        raise ValueError(f"Unknown backbone family {backbone_family}")
