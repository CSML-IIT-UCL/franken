from contextlib import contextmanager
import gc
import inspect
import logging
import os
import sys
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import franken.utils.distributed as dist_utils


logger = logging.getLogger("franken")


def sanitize_init_dict(cls, params: dict) -> dict:
    signature = inspect.signature(cls.__init__)

    # Get valid parameter names, excluding 'self'
    valid_params = set(signature.parameters.keys()) - {"self"}

    # Filter the input params to only include valid parameters
    filtered_params = {
        key: value for key, value in params.items() if key in valid_params
    }

    invalid_params = {
        key: value for key, value in params.items() if key not in valid_params
    }
    if len(invalid_params) > 0:
        logger.warning(
            f"Tried to initialize {cls.__name__} with invalid parameters: {invalid_params}"
        )
    # Initialize and return the class instance
    return filtered_params


def params_grid(
    grid: Mapping[str, Sequence[Any]],
    split_distributed: bool = False,
    filter_function: Optional[
        Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    ] = None,
) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """Converts a map of hyperparameter names to their possible values to a list of individual hyperparameter combinations

    Args:
        grid (Mapping): maps hyperparameter names to all the values they can take
            in a hyperparameter sweep.
        split_distributed (bool, optional): The result is distributed among available
            processes (uses the pytorch rank and world-size for this). Defaults to False.
        filter_function (callable, optional): A function which takes each set of hyperparameters
            and returns the same hyperparameters, a modified set or None if they are invalid.

    Example::

        >>> hp_grid = {"hp1": ["val1", "val2"], "hp2": [1, 5, 10], "hp3": ["fixed"]}
        >>> list(params_grid(hp_grid))
        [
            (0, {'hp1': 'val1', 'hp2': 1, 'hp3': 'fixed'}),
            (1, {'hp1': 'val1', 'hp2': 5, 'hp3': 'fixed'}),
            (2, {'hp1': 'val1', 'hp2': 10, 'hp3': 'fixed'}),
            (3, {'hp1': 'val2', 'hp2': 1, 'hp3': 'fixed'}),
            (4, {'hp1': 'val2', 'hp2': 5, 'hp3': 'fixed'}),
            (5, {'hp1': 'val2', 'hp2': 10, 'hp3': 'fixed'})
        ]
    """
    # Always sort the keys of a dictionary, for reproducibility
    items = sorted(grid.items())
    if not items:
        return
    else:
        keys, values = zip(*items)
        for idx, v in enumerate(product(*values)):
            params = dict(zip(keys, v))
            if filter_function is not None:
                params = filter_function(params)
            if params is None:
                continue
            if split_distributed:
                if idx % dist_utils.get_world_size() == dist_utils.get_rank():
                    params = dict(zip(keys, v))
                    yield idx, params
                else:
                    continue
            else:
                yield idx, params


class throughput:
    def __init__(
        self,
        iterable,
        desc="",
        units="cfgs",
        update_interval=0.2,
        world_size: int | None = None,  # overrides world-size
        process_rank: int | None = None,  # overrides rank
        total: int | None = None,
        device=None,
        leave: bool = False,
    ):
        self.iterable = iterable
        self.update_interval = update_interval
        self.world_size = (
            dist_utils.get_world_size() if world_size is None else world_size
        )
        self.process_rank = (
            dist_utils.get_rank() if process_rank is None else process_rank
        )
        self.total = len(iterable) if hasattr(iterable, "__len__") else None
        # Initialize time-counters here to avoid errors with None later on.
        self.now = perf_counter()
        self.last_update = perf_counter()
        self.n_iterations = 0
        self._batch = 0
        self.desc = desc
        self.units = units
        self.hardware = get_device_name(device)
        self.total = total
        self.leave = leave

    def __iter__(self):
        if self.process_rank != 0:
            for obj in self.iterable:
                yield obj
            return

        self.last_update = perf_counter()

        try:
            for obj in self.iterable:
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                self._batch += 1
                if self.should_print():
                    self.display()
        finally:
            self.display()
            if self.leave:
                print()
            else:
                print("\r", end="")

    def should_print(self):
        self.now = perf_counter()
        return self.now - self.last_update > self.update_interval

    def display(self):
        time_elapsed = self.now - self.last_update
        num_done = self.world_size * self._batch
        throughput = num_done / (time_elapsed + 1e-9)
        self.n_iterations += num_done
        if self.total is not None:
            assert self.total > 0
            perc = 100 * self.n_iterations / self.total
            status = f"\r{self.desc:^20} | {throughput:.1f} {self.units}/s | {perc:>3.0f}% | {self.world_size} x {self.hardware}"
        else:
            status = f"\r{self.desc:^20} | {throughput:.1f} {self.units}/s | {self.world_size} x {self.hardware}"
        print(
            status,
            end="",
        )
        self._batch = 0
        self.last_update = self.now

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_device_name(device: str | None) -> str:
    if device is None:
        return ""
    if torch.device(device).type == "cpu":
        import os
        import platform
        import re
        import subprocess

        def get_processor_name():
            if platform.system() == "Windows":
                return platform.processor()
            elif platform.system() == "Darwin":
                os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
                command = "sysctl -n machdep.cpu.brand_string"
                return (
                    subprocess.check_output(command, shell=True)
                    .strip()
                    .decode(sys.stdout.encoding)
                )
            elif platform.system() == "Linux":
                command = "cat /proc/cpuinfo"
                all_info = subprocess.check_output(command, shell=True).decode().strip()
                for line in all_info.split("\n"):
                    if "model name" in line:
                        return re.sub(".*model name.*:", "", line, 1)
            return ""

        hardware = get_processor_name()
    elif torch.device(device).type == "cuda":
        if torch.cuda.is_available():
            hardware = torch.cuda.get_device_name(0)
        else:
            hardware = "None"
    else:
        raise ValueError(f"Unrecognized device {device}")
    return hardware


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_cuda_out_of_memory(exception):
            # Only handle OOM errors
            raise


class DDPRankFilter(logging.Filter):
    """Logging filter by rank. Pass `extra={"rank0": True}` when logging"""

    def filter(self, record: logging.LogRecord) -> bool:
        # noinspection PyUnresolvedReferences
        only_rank_0 = hasattr(record, "rank0") and record.rank0 is True

        if dist_utils.get_rank() != 0 and only_rank_0:
            return False
        return True


def remove_root_logger_handlers():
    # This is necessary since some libraries (MACE) configure root loggers
    # which causes double outputs! We remove handlers which are directly
    # attached to the root logger. This is behind a flag, set to true
    # when called from our scripts but potentially user-configurable
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


def setup_logger(
    level: Union[int, str] = logging.INFO,
    directory: Optional[str | Path] = None,
    logname: str = "franken",
    rank: Optional[int] = 0,
):
    # From https://github.com/ACEsuit/mace/blob/main/mace/tools/utils.py
    flogger = logging.getLogger("franken")
    flogger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Add rank-based filtering
    flogger.addFilter(DDPRankFilter())

    # Create formatters
    formatter = logging.Formatter(
        f"%(asctime)s.%(msecs)03d %(levelname)s (rank {rank}): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    flogger.addHandler(ch)

    if directory is not None:
        directory = str(directory)
        os.makedirs(name=directory, exist_ok=True)

        # Create file handler for non-debug logs
        main_log_path = os.path.join(directory, f"{logname}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(logging.DEBUG)  # Dump everything into the logfile
        fh_main.setFormatter(formatter)
        flogger.addHandler(fh_main)
    logging.captureWarnings(True)


def get_logger_stdout_level(log: logging.Logger):
    handlers = [
        h
        for h in log.handlers
        if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
    ]
    levels = [h.level for h in handlers]
    if len(levels) == 0:
        return 0
    min_level = min(levels)
    return min_level


def pprint_config(cfg: dict):
    tl_keys = sorted(list(cfg.keys()))
    tl_leaf_keys = [
        k
        for k in tl_keys
        if not isinstance(cfg[k], (DictConfig, dict, list, ListConfig))
    ]
    tl_struct_keys = [k for k in tl_keys if k not in tl_leaf_keys]
    s = ""
    for k in tl_leaf_keys:
        s += f"{k}: {str(cfg[k])}\n"
    yaml_content: dict[str, str] = {}
    for key in tl_struct_keys:
        yaml_content[key] = OmegaConf.to_yaml(cfg[key], resolve=True, sort_keys=True)
    for key, content in yaml_content.items():
        s += f"{key}:\n"
        aligned_content = [f"    {line}" for line in content.splitlines(True)]
        s += f"{''.join(aligned_content)}"
    s = s.replace("\n\n", "\n")
    # Output
    logger.debug(s, extra={"rank0": True})
    # If logger has output the string to stdout then we don't print it
    # otherwise we also print it!
    if get_logger_stdout_level(logger) > logging.DEBUG and dist_utils.get_rank() == 0:
        print(s)


def are_lists_equal(l1: Sequence, l2: Sequence) -> bool:
    if len(l1) != len(l2):
        return False
    for i1, i2 in zip(l1, l2):
        if isinstance(i1, dict):
            if not isinstance(i2, dict):
                return False
            if not are_dicts_equal(i1, i2):
                return False
        elif isinstance(i1, (list, tuple)):
            if not isinstance(i2, (list, tuple)):
                return False
            if not are_lists_equal(i1, i2):
                return False
        else:
            if i1 != i2:
                return False
    return True


def are_dicts_equal(d1: dict, d2: dict) -> bool:
    if set(d1.keys()) != set(d2.keys()):
        return False
    for k1, v1 in d1.items():
        if isinstance(v1, dict):
            if not isinstance(d2[k1], dict):
                return False
            if not are_dicts_equal(v1, d2[k1]):
                return False
        elif isinstance(v1, (list, tuple)):
            if not isinstance(d2[k1], (list, tuple)):
                return False
            if not are_lists_equal(v1, d2[k1]):
                return False
        else:
            if v1 != d2[k1]:
                return False
    return True


def torch_load_maybejit(path, map_location=None, weights_only=True):
    try:
        model = torch.jit.load(path, map_location=map_location)
    except RuntimeError as e:
        if "PytorchStreamReader" not in str(e):
            raise
        model = torch.load(path, map_location=map_location, weights_only=weights_only)
    return model


@contextmanager
def no_jit():
    # disable jit optimization
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(False)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)
