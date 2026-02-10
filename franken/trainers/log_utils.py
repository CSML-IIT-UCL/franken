import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

import franken.metrics
import franken.utils.distributed as dist_utils
from franken.config import DEFAULT_BEST_MODEL_SELECTION


logger = logging.getLogger("franken")


class DataSplit(Enum):
    UNDEFINED = -1
    TRAIN = 0
    TEST = 1
    VALIDATION = 2
    VAL = 2  # alias


@dataclass(frozen=True)
class MetricLog:
    split: DataSplit
    name: str
    value: float


@dataclass(frozen=True)
class HyperParameter:
    name: str
    value: Union[float, bool, str, int]


@dataclass(frozen=True)
class HyperParameterGroup:
    group_name: str
    hyperparameters: list[HyperParameter]

    @classmethod
    def from_dict(cls, group_name: str, hp_dict: dict):
        params = []
        for param_name, param_value in hp_dict.items():
            params.append(HyperParameter(name=param_name, value=param_value))
        return cls(group_name=group_name, hyperparameters=params)


@dataclass
class LogEntry:
    checkpoint_hash: str
    checkpoint_rf_weight_id: int
    timings_cov_coeffs: float
    timings_solve: float
    metrics: list[MetricLog] = field(default_factory=list)
    hyperparameters: list[HyperParameterGroup] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "LogEntry":
        """Create a LogEntry instance from a dict."""

        # Parse checkpoint
        checkpoint_hash = data["checkpoint"]["hash"]
        checkpoint_rf_weight_id = data["checkpoint"]["rf_weight_id"]

        # Parse timings
        timings_cov_coeffs = data["timings"]["cov_coeffs"]
        timings_solve = data["timings"]["solve"]

        # Parse metrics
        metrics = []
        for split_name, split_metrics in data["metrics"].items():
            try:
                split = DataSplit[split_name.upper()]
            except KeyError:
                split = DataSplit.UNDEFINED

            for metric_name, value in split_metrics.items():
                metrics.append(MetricLog(split=split, name=metric_name, value=value))

        # Parse hyperparameters
        hyperparameters = []
        for group_name, group_params in data["hyperparameters"].items():
            hp_group = HyperParameterGroup.from_dict(group_name, group_params)
            hyperparameters.append(hp_group)

        return cls(
            checkpoint_hash=checkpoint_hash,
            checkpoint_rf_weight_id=checkpoint_rf_weight_id,
            timings_cov_coeffs=timings_cov_coeffs,
            timings_solve=timings_solve,
            metrics=metrics,
            hyperparameters=hyperparameters,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert this LogEntry instance to dict."""

        # Convert metrics to the expected format
        metrics_dict = defaultdict(dict)
        for metric in self.metrics:
            metrics_dict[metric.split.name.lower()][metric.name] = metric.value

        # Convert hyperparameters to the expected format
        hyperparams_dict = {}
        for group in self.hyperparameters:
            hyperparams_dict[group.group_name] = {
                hp.name: hp.value for hp in group.hyperparameters
            }

        # Create the final dictionary structure
        data = {
            "checkpoint": {
                "hash": self.checkpoint_hash,
                "rf_weight_id": self.checkpoint_rf_weight_id,
            },
            "timings": {
                "cov_coeffs": self.timings_cov_coeffs,
                "solve": self.timings_solve,
            },
            "metrics": metrics_dict,
            "hyperparameters": hyperparams_dict,
        }

        return data

    def get_metric(self, name: str, split: Union[DataSplit, str] = DataSplit.TRAIN):
        if isinstance(split, str):
            try:
                split = DataSplit[split.upper()]
            except KeyError:
                raise KeyError(f"Unknown split {split=}")
        matches = []
        for metric in self.metrics:
            if metric.name == name and metric.split == split:
                matches.append(metric)
        if len(matches) == 0:
            raise KeyError(f"Unable to find the metric {name=} on {split=}")
        elif len(matches) == 1:
            return matches[0].value
        else:
            raise NameError(f"Multiple metrics with {name=} and {split=} found.")

    def get_hyperparameter(self, name: str, group: str):
        matches = []
        for hp_group in self.hyperparameters:
            for hp in hp_group.hyperparameters:
                if hp_group.group_name == group and hp.name == name:
                    matches.append(hp)
        if len(matches) == 0:
            raise KeyError(f"Unable to find the hyperparameter {group=}/{name=}")
        elif len(matches) == 1:
            return matches[0].value
        else:
            raise NameError(
                f"Multiple hyperparameters with {group=} and {name=} found."
            )

    def add_metric(self, name: str, value: float, split: Union[DataSplit, str]):
        if isinstance(split, str):
            try:
                split = DataSplit[split.upper()]
            except KeyError:
                raise KeyError(f"Unknown split {split=}")
        try:
            self.get_metric(name, split=split)
            raise ValueError(
                f"A metric with the same {name=} and {split=} already exists"
            )
        except KeyError:
            pass
        self.metrics.append(MetricLog(split=split, name=name, value=value))

    def __eq__(self, other) -> bool:
        try:
            same_hash = self.checkpoint_hash == other.checkpoint_hash
            same_id = self.checkpoint_rf_weight_id == other.checkpoint_rf_weight_id
            return same_hash and same_id
        except AttributeError:
            return False

    def __str__(self):
        return (
            f"<LogEntry hash={self.checkpoint_hash}, id={self.checkpoint_rf_weight_id}>"
        )


@dataclass
class LogCollection:
    logs: list[LogEntry]

    def __len__(self):
        return len(self.logs)

    def __iter__(self):
        return iter(self.logs)

    def __getitem__(self, idx):
        return self.logs[idx]

    def get_metric(
        self,
        name: str,
        split: Union[DataSplit, str] = DataSplit.TRAIN,
        default=float("nan"),
    ):
        values = []
        for entry_id, entry in enumerate(self.logs):
            try:
                metric_value = entry.get_metric(name, split=split)
            except KeyError:
                logger.warning(
                    f"Metric {name} not found for the {entry_id+1}-th log, returning {default}"
                )
                metric_value = default
            values.append(metric_value)
        return values

    def get_hyperparameter(
        self,
        name: str,
        group: str,
        default=float("nan"),
    ):
        values = []
        for entry_id, entry in enumerate(self.logs):
            try:
                hp_value = entry.get_hyperparameter(name, group)
            except KeyError:
                logger.warning(
                    f"The hyperparameter {group}/{name} was not found for the {entry_id+1}-th log, returning {default}"
                )
                hp_value = default
            values.append(hp_value)
        return values

    @classmethod
    def gather_from_ranks(cls, local_logs: dict[int, LogEntry]):
        idx_list = []
        entries_list = []
        for local_log in dist_utils.all_gather_object(local_logs):
            for idx, log_entry in local_log.items():
                idx_list.append(idx)
                entries_list.append(log_entry)
        assert sorted(idx_list) == list(
            range(len(idx_list))
        ), "The indices of the local logs fail to form a continuous list from 0 to sum(len(logs) for log in local_logs)"
        logger.debug(f"Retrieved {len(idx_list)} logs.")

        sorted_entries = [None for _ in idx_list]
        for idx, entry in zip(idx_list, entries_list):
            sorted_entries[idx] = entry
        return cls(logs=sorted_entries)

    def save_json(self, log_file: Path) -> None:
        """Append the log entry to a JSON file and save the full logs file"""
        log_batch = [log.to_dict() for log in self.logs]
        if log_file.exists():
            with open(log_file, "r+") as f:
                logs = json.load(f)
                logs.extend(log_batch)
                f.seek(0)
                json.dump(logs, f, indent=4, cls=dtypeJSONEncoder)
        else:
            with open(log_file, "w") as f:
                json.dump(log_batch, f, indent=4, cls=dtypeJSONEncoder)

    @classmethod
    def from_json(cls, log_file: Path):
        with open(log_file, "r") as f:
            logs = json.load(f)
        if not isinstance(logs, list):  # handle loading a single log!
            logs = [logs]
        return cls(logs=[LogEntry.from_dict(log) for log in logs])

    def get_best_model(
        self,
        metrics_to_minimize: list[str] | None = None,
        p: int = 1,
        split: DataSplit = DataSplit.TRAIN,
    ) -> LogEntry:
        """
        Find the best model from a list of log entries.
        The best model is chosen among the pareto frontier of ``metrics_to_minimize``,
        by minimizing their ``p``-norm.
        The function returns a dictionary with information about the best model.
        """

        if metrics_to_minimize is None:
            metrics_to_minimize = DEFAULT_BEST_MODEL_SELECTION.copy()
        if len(metrics_to_minimize) == 0:
            raise ValueError("`metrics_to_minimize` (in best_model_selection) must contain at least one metric.")

        available_metrics = {
            metric.name
            for entry in self.logs
            for metric in entry.metrics
            if metric.split == split
        }

        for metric in metrics_to_minimize:
            if metric.endswith("_species") and metric not in available_metrics:
                raise ValueError(
                    f"Metric '{metric}' is species-resolved and cannot be minimized directly. "
                    f"Use '{metric}_average' for the species average, or one of the per-species "
                    f"metrics like '{metric}_<Z>'."
                )
            assert (
                metric in available_metrics
            ), f"Unknown {metric=} for split={split}. Available: {sorted(available_metrics)}"

        costs = np.stack(
            [self.get_metric(m, split=split) for m in metrics_to_minimize], axis=-1
        )
        costs = np.nan_to_num(costs, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        err_norm = np.linalg.norm(costs, ord=p, axis=-1)
        err_norm[~franken.metrics.is_pareto_efficient(costs)] = np.inf
        best_model_idx = np.argmin(err_norm)
        return self.logs[best_model_idx]


class dtypeJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, torch.dtype):
            return str(o)
        return super().default(o)
