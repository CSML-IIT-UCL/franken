import dataclasses
import datetime
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing_extensions import TypeVarTuple, Unpack
from typing import Any, Iterator, NamedTuple, cast
from uuid import uuid4

import torch.distributed
import torch.utils.data

from franken.autotune.cli import build_parser, parse_cli
from franken.config import (
    AutotuneConfig,
    BackboneConfig,
    CheckpointableDataclass,
    DataclassInstance,
    HPSearchConfig,
    SolverConfig,
    asdict_with_classvar,
)
from franken.data.base import (
    ENERGY_TARGET_KEY,
    FORCES_TARGET_KEY,
    STRESS_TARGET_KEY,
    TargetType,
)
from franken.datasets.registry import DATASET_REGISTRY
from franken.rf.les_model import LESFrankenPotential
from franken.trainers import (
    LowMemRandomFeaturesTrainer,
    RandomFeaturesTrainer,
    RandomFeaturesEwaldsTrainer,
)
from franken.trainers.log_utils import DataSplit, LogEntry
import franken.utils.distributed as dist_utils
from franken.backbones.utils import CacheDir
from franken.data import FrankenAtomsDataset
from franken.rf.model import FrankenPotential
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


def set_dataset_atomic_energies(
    loaders: dict[str, torch.utils.data.DataLoader],
    atomic_energies: dict[int, float] | None,
) -> None:
    if atomic_energies is None:
        return

    train_dataset = loaders["train"].dataset
    assert isinstance(train_dataset, FrankenAtomsDataset)

    missing_species = set(train_dataset.species) - set(atomic_energies)
    if missing_species:
        raise ValueError(
            "Missing reference atomic energies for species: "
            f"{sorted(missing_species)}"
        )

    extra_species = set(atomic_energies) - set(train_dataset.species)
    if extra_species:
        logger.warning(
            "Ignoring reference atomic energies for species not present in the "
            "training set: %s",
            sorted(extra_species),
        )

    train_dataset.atomic_energies_ = {  # type: ignore
        z: atomic_energies[z] for z in train_dataset.species
    }
    train_dataset.energy_shifts_ = None


def init_loaders(
    gnn_cfg: BackboneConfig,
    train_path: Path | None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    num_train_subsamples: int | None = None,
    subsample_rng: int | None = None,
) -> dict[str, torch.utils.data.DataLoader]:
    datasets: dict[str, FrankenAtomsDataset] = {}
    for split, data_path in zip(
        ["train", "val", "test"], [train_path, val_path, test_path]
    ):
        if data_path is not None:
            datasets[split] = FrankenAtomsDataset(
                data_path=data_path,
                split=split,
                gnn_config=gnn_cfg,
                num_random_subsamples=(
                    num_train_subsamples if split == "train" else None
                ),
                subsample_rng=subsample_rng,
            )

    dataloaders = {
        split: dset.get_dataloader(distributed=torch.distributed.is_initialized())
        for split, dset in datasets.items()
    }
    return dataloaders


def hp_summary_str(
    trial_id: int, current_best: BestTrial, *cfgs: CheckpointableDataclass | None
) -> str:
    hp_summary = f"Trial {trial_id + 1:>3} |"
    for cfg in cfgs:
        if cfg is None:
            continue
        for k, v in cfg.to_ckpt().items():
            fmt_val = format(v, ".2e" if isinstance(v, float) else "")
            hp_summary += f" {k:^7}: {fmt_val:^7} |"

    def _get_first_available_metric(
        candidates: list[str],
        splits: list[DataSplit],
    ) -> float | None:
        for split in splits:
            for name in candidates:
                try:
                    return current_best.log.get_metric(name, split)
                except KeyError:
                    pass
        return None

    energy_error = _get_first_available_metric(
        ["energy_MAE", "energy_RMSE"],
        [DataSplit.VALIDATION, DataSplit.TRAIN],
    )
    forces_error = _get_first_available_metric(
        ["forces_MAE", "forces_RMSE"],
        [DataSplit.VALIDATION, DataSplit.TRAIN],
    )
    stress_error = _get_first_available_metric(
        ["stress_MAE", "stress_RMSE"],
        [DataSplit.VALIDATION, DataSplit.TRAIN],
    )

    hp_summary += f" Best trial {current_best.trial_id}"
    if energy_error is None:
        energy_error = float("nan")
    hp_summary += f" (energy {energy_error:.2f} meV/atom)"
    if forces_error is None:
        forces_error = float("nan")
    hp_summary += f" (forces {forces_error:.2f} meV/Ang)"
    if stress_error is not None:
        hp_summary += f" (stress {stress_error:.2f} meV/Ang^3)"
    return hp_summary


def hps_from_config(cfg: DataclassInstance):
    hp_iterators = {}
    for field in dataclasses.fields(cfg):
        hp_def = getattr(cfg, field.name)
        if isinstance(hp_def, HPSearchConfig):
            hp_iterators[field.name] = hp_def.get_vals()
        elif isinstance(hp_def, (list, tuple)):
            hp_iterators[field.name] = hp_def
        else:
            hp_iterators[field.name] = [hp_def]
    return hp_iterators


Ts = TypeVarTuple("Ts")


def create_outer_hpsearch_grid(
    cfg: tuple[Unpack[Ts]],
) -> Iterator[tuple[int, tuple[Unpack[Ts]]]]:
    """Expand one or more dataclass configs into a grid of concrete configs.

    Thin wrapper over :func:`franken.utils.misc.params_grid` which handles unrolling compact
    hyperparameter specs, such as generating linearly or logarithmically spaced HP values.

    Args:
        cfg: A configuration or a tuple of configuration data-classes. Field names must be
            unique across all configs in the tuple.

    Yields:
        ``(exp_id, new_cfg)`` pairs, where ``new_cfg`` has the same structure as
        ``cfg`` (a single dataclass or a tuple of dataclasses).
    """
    cfgs: tuple[Any, ...] = cfg  # element types are opaque inside the body

    hp_iterators: dict[str, Any] = {}
    for c in cfgs:
        hps = hps_from_config(c)
        if overlap := hp_iterators.keys() & hps.keys():
            raise ValueError(
                f"Hyperparameter names shared across configs: {sorted(overlap)}"
            )
        hp_iterators |= hps

    for exp_id, grid_item in params_grid(hp_iterators):
        new_cfgs = tuple(
            dataclasses.replace(
                c,
                **{
                    f.name: grid_item[f.name]
                    for f in dataclasses.fields(c)
                    if f.init and f.name in grid_item
                },
            )
            for c in cfgs
        )
        yield exp_id, cast("tuple[Unpack[Ts]]", new_cfgs)


def create_solver_hpsearch_grid(
    cfg: SolverConfig,
) -> tuple[list[float], dict[TargetType, list[float]]]:
    solver_hps = hps_from_config(cfg)
    weight_dict: dict[TargetType, list[float]] = {
        ENERGY_TARGET_KEY: solver_hps["energy_weight"],
        FORCES_TARGET_KEY: solver_hps["forces_weight"],
        STRESS_TARGET_KEY: solver_hps["stress_weight"],
    }
    return solver_hps["l2_penalty"], weight_dict


def run_autotune(
    auto_cfg: AutotuneConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    log_dir: Path | None,
):
    eval_splits = auto_cfg.eval_splits
    is_les = auto_cfg.les is not None
    best_model_selection = auto_cfg.best_model_selection

    if best_model_selection is None or len(best_model_selection) == 0:
        best_model_selection = [f"{tgt}_MAE" for tgt in auto_cfg.train_targets]
    current_best = BestTrial(None, None)  # type: ignore

    if is_les is not None:
        param_grid = list(
            create_outer_hpsearch_grid((auto_cfg.rfs, auto_cfg.les, auto_cfg.solver))
        )
    else:
        param_grid = list(create_outer_hpsearch_grid((auto_cfg.rfs,)))
    print(
        f"Autotuning {'LES ' if is_les else ''} Franken with {len(param_grid)} parameters."
    )
    for trial_id, trial_params in param_grid:
        logger.debug(f"Autotune iteration {trial_id} with parameters {trial_params}")
        if is_les:
            assert len(trial_params) == 3
            new_cfg = dataclasses.replace(
                auto_cfg,
                rfs=trial_params[0],
                les=trial_params[1],
                solver=trial_params[2],
            )
            trainer = init_les_trainer(new_cfg, loaders, device, log_dir)
            model = init_les_model(new_cfg, loaders)
            logs, weights = trainer.fit(model)
        else:
            new_cfg = dataclasses.replace(auto_cfg, rfs=trial_params[0])
            trainer = init_rf_trainer(new_cfg, loaders, device, log_dir)
            model = init_les_model(new_cfg, loaders)
            logs, weights = trainer.fit(model)

        for split_name, loader in loaders.items():
            if eval_splits is not None and split_name not in eval_splits:
                continue
            logs = trainer.evaluate(model, loader, logs, weights)
        split_for_best_model = (
            DataSplit.VALIDATION if "val" in loaders else DataSplit.TRAIN
        )
        if dist_utils.get_rank() == 0:
            if trainer.log_dir is not None:
                trainer.serialize_logs(
                    model,
                    logs,
                    weights,
                    best_model_selection=best_model_selection,
                    best_model_split=split_for_best_model,
                )
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
                logger.info(hp_summary_str(trial_id, current_best, *trial_params))
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
        if train_path is None:
            raise ValueError(
                "Either a valid 'dataset_name' or 'train_path' must be "
                "specified in order to load a training dataset."
            )
        out_train_path = Path(train_path)
        out_val_path = Path(val_path) if val_path is not None else None
        out_test_path = Path(test_path) if test_path is not None else None
    return out_train_path, out_val_path, out_test_path


def init_rf_trainer(
    cfg: AutotuneConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    log_dir: Path | None,
):
    solver_l2, solver_weights = create_solver_hpsearch_grid(cfg.solver)
    trainer_cls = RandomFeaturesTrainer
    if len(cfg.train_targets) == 2:
        trainer_cls = LowMemRandomFeaturesTrainer
    trainer = trainer_cls(
        train_dataloader=loaders["train"],
        l2_penalty=solver_l2,
        training_targets=cfg.train_targets,
        target_weight=solver_weights,
        random_features_normalization=cfg.rf_normalization,
        save_every_model=cfg.save_every_model,
        dtype=cfg.dtype,
        save_fmaps=cfg.save_fmaps,
        log_dir=log_dir,
        device=device,
        metrics=cfg.metrics,
    )
    trainer.val_dataloader = loaders.get("val")
    return trainer


def init_les_trainer(
    cfg: AutotuneConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    log_dir: Path | None,
):
    les_cfg = cfg.les
    # type checking
    assert les_cfg is not None
    l2_penalty = cfg.solver.l2_penalty
    if not isinstance(l2_penalty, float):
        raise ValueError(
            f"l2 penalty must be a single float for LES trainer. Found {l2_penalty}"
        )
    tgt_weights = cfg.solver.get_weights()
    for wname, wval in tgt_weights.items():
        if not isinstance(wval, float):
            raise ValueError(
                f"{wname} must be a single float for LES trainer. Found {wval}"
            )
    tgt_weights = cast(dict[TargetType, float], tgt_weights)
    if not isinstance(les_cfg.num_cycles, int):
        raise ValueError(
            f"num_cycles must be a single int for LES trainer. Found {les_cfg.num_cycles}"
        )
    if not isinstance(les_cfg.lbfgs_max_iter, int):
        raise ValueError(
            f"lbfgs_max_iter must be a single int for LES trainer. Found {les_cfg.lbfgs_max_iter}"
        )
    if not isinstance(les_cfg.lbfgs_lr, (int, float)):
        raise ValueError(
            f"lbfgs_lr must be a single float for LES trainer. Found {les_cfg.lbfgs_lr}"
        )
    if not isinstance(les_cfg.lbfgs_lr_decay, (int, float)):
        raise ValueError(
            f"lbfgs_lr_decay must be a single float for LES trainer. Found {les_cfg.lbfgs_lr_decay}"
        )
    if not isinstance(les_cfg.lbfgs_history_size, int):
        raise ValueError(
            f"lbfgs_history_size must be a single int for LES trainer. Found {les_cfg.lbfgs_history_size}"
        )
    if not isinstance(les_cfg.lbfgs_tolerance_grad, (int, float)):
        raise ValueError(
            f"lbfgs_tolerance_grad must be a single float for LES trainer. Found {les_cfg.lbfgs_tolerance_grad}"
        )
    if not isinstance(les_cfg.lbfgs_tolerance_change, (int, float)):
        raise ValueError(
            f"lbfgs_tolerance_change must be a single float for LES trainer. Found {les_cfg.lbfgs_tolerance_change}"
        )

    trainer = RandomFeaturesEwaldsTrainer(
        train_dataloader=loaders["train"],
        l2_penalty=l2_penalty,
        training_targets=cfg.train_targets,
        target_weight=tgt_weights,
        mode=les_cfg.mode,
        num_cycles=les_cfg.num_cycles,
        lbfgs_max_iter=les_cfg.lbfgs_max_iter,
        lbfgs_lr=les_cfg.lbfgs_lr,
        lbfgs_lr_decay=les_cfg.lbfgs_lr_decay,
        lbfgs_history_size=les_cfg.lbfgs_history_size,
        lbfgs_tolerance_grad=les_cfg.lbfgs_tolerance_grad,
        lbfgs_tolerance_change=les_cfg.lbfgs_tolerance_change,
        random_features_normalization=cfg.rf_normalization,
        save_every_model=cfg.save_every_model,
        dtype=cfg.dtype,
        save_fmaps=cfg.save_fmaps,
        log_dir=log_dir,
        device=device,
        metrics=cfg.metrics,
    )
    trainer.val_dataloader = loaders.get("val")
    return trainer


def init_rf_model(
    cfg: AutotuneConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
):
    assert isinstance(loaders["train"].dataset, FrankenAtomsDataset)  # for typing
    return FrankenPotential(
        gnn_config=cfg.backbone,
        rf_config=cfg.rfs,
        scale_by_Z=cfg.scale_by_species,
        num_species=loaders["train"].dataset.num_species,
        atomic_energies=cfg.atomic_energies,
        jac_chunk_size=cfg.jac_chunk_size,
    )


def init_les_model(
    cfg: AutotuneConfig,
    loaders: dict[str, torch.utils.data.DataLoader],
):
    assert isinstance(loaders["train"].dataset, FrankenAtomsDataset)  # for typing
    assert cfg.les is not None
    is_periodic = None
    if all(ldr.dataset.is_all_periodic() for ldr in loaders.values()):
        is_periodic = True
    if all(ldr.dataset.is_all_non_periodic() for ldr in loaders.values()):
        is_periodic = False
    return LESFrankenPotential(
        gnn_config=cfg.backbone,
        rf_config=cfg.rfs,
        les_config=cfg.les,
        scale_by_Z=cfg.scale_by_species,
        num_species=loaders["train"].dataset.num_species,
        atomic_energies=cfg.atomic_energies,
        jac_chunk_size=cfg.jac_chunk_size,
        is_periodic=is_periodic,
    )


def autotune(cfg: AutotuneConfig):
    torch.manual_seed(cfg.seed)
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

        t_start = time.time()
        train_path, val_path, test_path = get_dataset_paths(
            cfg.dataset.train_path,
            cfg.dataset.val_path,
            cfg.dataset.test_path,
            cfg.dataset.name,
        )
        t_end = time.time()
        logger.debug(f"Fetched datasets in {t_end - t_start:.2f}s")

        t_start = time.time()
        loaders = init_loaders(
            cfg.backbone,
            train_path,
            val_path,
            test_path,
            cfg.dataset.max_train_samples,
            cfg.seed,
        )
        set_dataset_atomic_energies(loaders, cfg.atomic_energies)
        t_end = time.time()
        logger.debug(f"Initialized data-loaders in {t_end - t_start:.2f}s")

        run_autotune(
            auto_cfg=cfg,
            loaders=loaders,
            device=device,
            log_dir=run_dir,
        )
    except Exception as e:
        logger.error("Error encountered in autotune. Exiting.", exc_info=e)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return run_dir


def cli_entry_point():
    args = parse_cli(sys.argv[1:])
    autotune(args)


if __name__ == "__main__":
    cli_entry_point()


# For sphinx docs
get_parser_fn = lambda: build_parser()  # noqa: E731
