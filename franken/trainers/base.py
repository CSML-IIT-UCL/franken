import abc
import json
import logging
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.utils.data

from franken.config import asdict_with_classvar
from franken.rf.model import FrankenPotential
from franken.rf.scaler import Statistics, compute_dataset_statistics
from franken.trainers.log_utils import (
    DataSplit,
    LogCollection,
    LogEntry,
    dtypeJSONEncoder,
)
from franken.utils.misc import are_dicts_equal


logger = logging.getLogger("franken")


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        log_dir: Path | None = None,  # If None, logging is disabled
        save_every_model: bool = True,
        device: Union[torch.device, str, int] = "cpu",
        dtype: Union[str, torch.dtype] = torch.float32,
    ):
        self.log_dir = log_dir
        self.save_every_model = save_every_model
        self.train_dataloader = train_dataloader
        self.statistics_ = None
        if isinstance(dtype, str):
            if dtype.lower() == "float64" or dtype.lower == "double":
                dtype = torch.float64
            elif dtype.lower() == "float32" or dtype.lower == "float":
                dtype = torch.float32
            else:
                raise ValueError(
                    f"Invalid dtype {dtype}. Allowed values are 'float64', 'double', 'float32', 'single'."
                )
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError(
                f"Invalid dtype {dtype}. torch.float32 or torch.float64 are allowed."
            )
        self.buffer_dt = dtype
        self.device = torch.device(device)

    @torch.no_grad()
    def get_statistics(self, model: FrankenPotential) -> Tuple[Statistics, dict]:
        if self.statistics_ is None or not are_dicts_equal(
            self.statistics_[1], asdict_with_classvar(model.gnn_config)
        ):
            stat = compute_dataset_statistics(
                dataset=self.train_dataloader.dataset,  # type: ignore
                gnn=model.gnn,
                device=self.device,
            )
            stat_dict = asdict_with_classvar(model.gnn_config)
            self.statistics_ = (stat, stat_dict)

        return self.statistics_

    @abc.abstractmethod
    def on_fit_start(self, model: FrankenPotential):
        pass

    @abc.abstractmethod
    def fit(
        self,
        model: FrankenPotential,
        solver_params: dict,
    ) -> tuple[LogCollection, torch.Tensor]:
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        model: FrankenPotential,
        dataloader: torch.utils.data.DataLoader,
        log_collection: LogCollection,
        all_weights: torch.Tensor,
        metrics: list[str],
    ) -> LogCollection:
        pass

    def serialize_logs(
        self,
        model: FrankenPotential,
        log_collection: LogCollection,
        all_weights: torch.Tensor,
        best_model_split: DataSplit = DataSplit.TRAIN,
    ):
        assert self.log_dir is not None, "Log directory is not set"
        model_hash_set = set(log.checkpoint_hash for log in log_collection)
        assert len(model_hash_set) == 1
        model_hash = model_hash_set.pop()
        log_collection.save_json(self.log_dir / "log.json")

        # Save the model checkpoint
        if self.save_every_model:
            ckpt_dir = self.log_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = ckpt_dir / f"{model_hash}.pt"
            model.save(model_save_path, multi_weights=all_weights)
            logger.debug(
                f"Saved multiple models (hash={model_hash}) " f"to {model_save_path}"
            )
        # Log the best model
        self.serialize_best_model(model, all_weights, split=best_model_split)

    def serialize_best_model(
        self,
        model: FrankenPotential,
        all_weights: torch.Tensor,
        split: DataSplit = DataSplit.TRAIN,
    ) -> None:
        assert self.log_dir is not None, "Log directory is not set"
        log_collection = LogCollection.from_json(self.log_dir / "log.json")
        best_model = log_collection.get_best_model(split=split)

        best_model_file = self.log_dir / "best.json"
        should_save = True
        if best_model_file.exists():
            with open(best_model_file, "r") as f:
                current_best = LogEntry.from_dict(json.load(f))
            if best_model == current_best:
                should_save = False

        if should_save:
            logger.debug(f"Identified new best model: {best_model}")
            with open(best_model_file, "w") as f:
                json.dump(best_model.to_dict(), f, indent=4, cls=dtypeJSONEncoder)
            weights = all_weights[best_model.checkpoint_rf_weight_id]
            model.rf.weights = weights.reshape_as(model.rf.weights)
            model.save(self.log_dir / "best_ckpt.pt")
            logger.debug(
                f"Saved best model (within-experiment ID={best_model.checkpoint_rf_weight_id}) "
                f"to {self.log_dir / 'best_ckpt.pt'}"
            )
