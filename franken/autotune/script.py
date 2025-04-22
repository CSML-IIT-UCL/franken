import dataclasses
import datetime
import json
import logging
import warnings
from pathlib import Path
from typing import Generator, Literal, NamedTuple
from uuid import uuid4

import tyro
import torch.distributed
import torch.utils.data

from franken.config import (
    AutotuneConfig,
    BackboneConfig,
    HPSearchConfig,
    RFConfig,
    SolverConfig,
    asdict_with_classvar,
)
from franken.datasets.registry import DATASET_REGISTRY
from franken.trainers.rf_cuda_lowmem import RandomFeaturesTrainer
import franken.utils.distributed as dist_utils
from franken.backbones.utils import CacheDir
from franken.data import BaseAtomsDataset
from franken.rf.model import FrankenPotential
from franken.trainers import BaseTrainer
from franken.trainers.log_utils import DataSplit, LogEntry
from franken.utils.misc import (
    garbage_collection_cuda,
    get_device_name,
    params_grid,
    pprint_config,
    setup_logger,
)


class BestTrial(NamedTuple):
    trial_id: int
    log: LogEntry


warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp.autocast\(args...\)` is deprecated",
    category=FutureWarning,
)
logger = logging.getLogger("franken")


def init_loaders(
    gnn_cfg: BackboneConfig,
    train_path: Path | None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    num_train_subsamples: int | None = None,
    subsample_rng: int | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    datasets: dict[str, BaseAtomsDataset] = {}
    for split, data_path in zip(
        ["train", "val", "test"], [train_path, val_path, test_path]
    ):
        if data_path is not None:
            dset = BaseAtomsDataset.from_path(
                data_path=data_path,
                split=split,
                gnn_config=gnn_cfg,
                num_random_subsamples=(
                    num_train_subsamples if split == "train" else None
                ),
                subsample_rng=subsample_rng,
            )
            datasets[split] = dset

    dataloaders = {
        split: dset.get_dataloader(distributed=torch.distributed.is_initialized())
        for split, dset in datasets.items()
    }
    return dataloaders


def hp_summary_str(trial_id: int, current_best: BestTrial, rf_params: RFConfig) -> str:
    hp_summary = f"Trial {trial_id + 1:>3} |"
    for k, v in rf_params.to_ckpt().items():
        fmt_val = format(v, ".3f" if isinstance(v, float) else "")
        hp_summary += f" {k:^7}: {fmt_val:^7} |"
    try:
        energy_error = current_best.log.get_metric("energy_MAE", DataSplit.VALIDATION)
        forces_error = current_best.log.get_metric("forces_MAE", DataSplit.VALIDATION)
    except KeyError:
        energy_error = current_best.log.get_metric("energy_MAE", DataSplit.TRAIN)
        forces_error = current_best.log.get_metric("forces_MAE", DataSplit.TRAIN)

    hp_summary += (
        f" Best trial {current_best.trial_id} (energy {energy_error:.2f} meV/atom - "
        f"forces {forces_error:.1f} meV/Ang)"
    )
    return hp_summary


def hps_from_config(cfg):
    hp_iterators = {}
    for field in dataclasses.fields(cfg):
        hp_def = getattr(cfg, field.name)
        if isinstance(hp_def, HPSearchConfig):
            hp_iterators[field.name] = hp_def.get_vals()
        else:

            hp_iterators[field.name] = [hp_def]
    return hp_iterators


def create_rf_hpsearch_grid(
    cfg: RFConfig,
) -> Generator[tuple[int, RFConfig], None, None]:
    """Convert a random-features configuration object to a sequence of hyper-parameters.

    This function is a thin wrapper over :func:`franken.utils.misc.params_grid` which handles unrolling compact
    hyperparameter specs, such as generating linearly or logarithmically spaced HP values.

    Args:
        cfg: The random features configuration object

    Yields:
        A sequence of simple dictionaries going from hyperparameter name to value.
    """
    hp_iterators = hps_from_config(cfg)
    # Convert grid as dictionary to grid as RFConfig classes
    for exp_id, grid_item in params_grid(hp_iterators):
        yield (exp_id, type(cfg)(**grid_item))


def create_solver_hpsearch_grid(cfg: SolverConfig):
    return hps_from_config(cfg)


def run_autotune(
    gnn_cfg: BackboneConfig,
    rf_cfg: RFConfig,
    solver_cfg: SolverConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
    scale_by_species: bool,
    jac_chunk_size: int | Literal["auto"],
    trainer: BaseTrainer,
):
    current_best = BestTrial(None, None)
    rf_param_grid = create_rf_hpsearch_grid(rf_cfg)
    solver_param_grid = create_solver_hpsearch_grid(solver_cfg)
    for trial_id, rf_params in rf_param_grid:
        logger.debug(f"Autotune iteration with RF parameters {rf_params}")

        assert isinstance(loaders["train"].dataset, BaseAtomsDataset)  # for typing
        model = FrankenPotential(
            gnn_config=gnn_cfg,
            rf_config=rf_params,
            scale_by_Z=scale_by_species,
            num_species=loaders["train"].dataset.num_species,
            atomic_energies=None,
            jac_chunk_size=jac_chunk_size,
        )

        logs, weights = trainer.fit(model, solver_param_grid)
        for split_name, loader in loaders.items():
            logs = trainer.evaluate(
                model,
                loader,
                logs,
                weights,
                metrics=[
                    "energy_MAE",
                    "forces_MAE",
                    "energy_RMSE",
                    "forces_RMSE",
                    "forces_cosim",
                ],
            )
        split_for_best_model = (
            DataSplit.VALIDATION if "val" in loaders else DataSplit.TRAIN
        )
        if dist_utils.get_rank() == 0:
            if trainer.log_dir is not None:
                trainer.serialize_logs(model, logs, weights, split_for_best_model)
        dist_utils.barrier()

        # current best model update
        if dist_utils.get_rank() == 0:
            if trainer.log_dir is not None:
                with open(trainer.log_dir / "best.json", "r") as f:
                    try:
                        best_log = LogEntry.from_dict(json.load(f))
                        if best_log != current_best.log:
                            current_best = BestTrial(
                                trial_id=trial_id + 1,
                                log=best_log,
                            )
                    except KeyError:
                        pass

                logger.info(hp_summary_str(trial_id, current_best, rf_params))
        garbage_collection_cuda()


def create_run_folder(base_run_dir: Path) -> Path:
    # Use time + a shortened UUID to ensure uniqueness of
    # the experiment directory. The names will look like
    # 'run_240926_113513_1d93b3ed'
    now_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    exp_dir_name = f"run_{now_str}_{uuid4().hex[:8]}"
    exp_dir = base_run_dir / exp_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_info(cfg: AutotuneConfig, run_dir: Path):
    train_hardware = {
        "num_gpus": dist_utils.get_world_size(),
        "gpu_model": get_device_name("cuda:0"),
        "cpu_model": get_device_name("cpu"),
    }
    cfg_dict = asdict_with_classvar(cfg)
    assert isinstance(cfg_dict, dict)
    with open(run_dir / "configs.json", "w") as f:
        json.dump(cfg_dict | train_hardware, f, indent=4)


def get_dataset_paths(
    train_path: str | None,
    val_path: str | None,
    test_path: str | None,
    dataset_name: str | None,
) -> tuple[Path, Path | None, Path | None]:
    try:
        if dataset_name is None:
            raise KeyError
        out_train_path = DATASET_REGISTRY.get_path(
            dataset_name, "train", CacheDir.get()
        )
        out_val_path = None
        if DATASET_REGISTRY.is_valid_split(dataset_name, "val"):
            out_val_path = DATASET_REGISTRY.get_path(
                dataset_name, "val", CacheDir.get()
            )
        out_test_path = None
        if DATASET_REGISTRY.is_valid_split(dataset_name, "test"):
            out_test_path = DATASET_REGISTRY.get_path(
                dataset_name, "test", CacheDir.get()
            )
        if out_val_path is not None and out_test_path is not None:
            out_test_path = (
                None  # TODO: This is not very good, check with our datasets!
            )
    except KeyError:
        logger.info(f"Dataset with name '{dataset_name}' not found in registry")
        if train_path is None:
            raise ValueError(
                "Either a valid 'dataset_name' or 'train_path' must be "
                "specified in order to load a training dataset."
            )
        out_train_path = Path(train_path)
        out_val_path = Path(val_path) if val_path is not None else None
        out_test_path = Path(test_path) if test_path is not None else None
    return out_train_path, out_val_path, out_test_path


def autotune(cfg: AutotuneConfig):
    run_dir = Path(cfg.run_dir)

    if torch.cuda.is_available():
        rank = dist_utils.init(distributed=torch.cuda.device_count() > 1)
        device = torch.device(torch.cuda.current_device())
    else:
        rank = 0
        device = torch.device("cpu")

    if rank != 0:  # first rank goes forward
        run_dir = None
        dist_utils.barrier()
    else:
        run_dir = create_run_folder(run_dir)
        dist_utils.barrier()  # other ranks follow
    logging_level = cfg.console_logging_level.upper()
    setup_logger(
        level=logging_level, directory=dist_utils.broadcast_obj(run_dir), rank=rank
    )
    pprint_config(asdict_with_classvar(cfg))

    # Global try-catch after setup_logger, to log any exceptions.
    try:
        CacheDir.initialize()

        if rank != 0:  # first rank goes forward
            dist_utils.barrier()
        else:
            assert run_dir is not None
            save_experiment_info(cfg, run_dir)
            logger.info(f"Run folder: {run_dir}")
            dist_utils.barrier()

        train_path, val_path, test_path = get_dataset_paths(
            cfg.dataset.train_path,
            cfg.dataset.val_path,
            cfg.dataset.test_path,
            cfg.dataset.name,
        )

        loaders = init_loaders(
            cfg.backbone,
            train_path,
            val_path,
            test_path,
            cfg.dataset.max_train_samples,
            cfg.seed,
        )

        trainer = RandomFeaturesTrainer(
            train_dataloader=loaders["train"],
            random_features_normalization=cfg.rf_normalization,
            save_every_model=cfg.save_every_model,
            dtype=cfg.dtype,
            save_fmaps=cfg.save_fmaps,
            log_dir=run_dir,
            device=device,
        )

        run_autotune(
            gnn_cfg=cfg.backbone,
            rf_cfg=cfg.rfs,
            solver_cfg=cfg.solver,
            loaders=loaders,
            scale_by_species=cfg.scale_by_species,
            jac_chunk_size=cfg.jac_chunk_size,
            trainer=trainer,
        )
    except Exception as e:
        logger.error("Error encountered in autotune. Exiting.", exc_info=e)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return run_dir


def cli_entry_point():
    args = tyro.cli(
        AutotuneConfig,
        config=(
            tyro.conf.SuppressFixed,
            tyro.conf.FlagConversionOff,
        ),
    )
    autotune(args)


if __name__ == "__main__":
    cli_entry_point()


# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(  # noqa: E731
    AutotuneConfig,
    config=(
        tyro.conf.SuppressFixed,
        tyro.conf.FlagConversionOff,
    ),
)
