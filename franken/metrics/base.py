import abc
from typing import Mapping
import typing
import torch

import franken.utils.distributed as dist_utils
from franken.data.base import Configuration, Target, TargetType


class BaseMetric(abc.ABC):
    target_type: typing.ClassVar[TargetType]
    name: typing.ClassVar[str]

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        units: Mapping[str, str | None] = {},
    ):
        self.device = device
        self.dtype = dtype
        self.buffer = None
        self.samples_counter = torch.zeros((1,), device=device, dtype=torch.int64)
        self.units = units

    def reset(self) -> None:
        """Reset the buffer to zeros"""
        self.buffer = None
        self.samples_counter.zero_()

    def buffer_add(self, value: torch.Tensor, num_samples: int = 1) -> None:
        if self.buffer is None:
            self.buffer = torch.zeros(value.shape, device=self.device, dtype=self.dtype)
        else:
            assert self.buffer.shape == value.shape
        self.buffer += value
        self.samples_counter += num_samples

    def update(self, predictions: Target, targets: Target, data: Configuration) -> None:
        """Update the metric buffer with new batch results"""
        raise NotImplementedError()

    def _compute_val(self) -> torch.Tensor:
        if self.buffer is None:
            raise ValueError(
                f"Cannot compute value for metric '{self.name}' "
                "because it was never updated."
            )
        dist_utils.all_sum(self.buffer)
        dist_utils.all_sum(self.samples_counter)
        error = self.buffer / self.samples_counter
        return error

    def compute(self) -> list[tuple[str, torch.Tensor]]:
        error = self._compute_val()
        self.reset()
        return [(self.name, error)]
