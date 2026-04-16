from typing import Mapping
import torch

import franken.utils.distributed as dist_utils
from franken.data.base import Configuration, Target


class BaseMetric:
    def __init__(
        self,
        name: str,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        units: Mapping[str, str | None] = {},
    ):
        self.name = name
        self.device = device
        self.dtype = dtype
        self.buffer = None
        self.samples_counter = torch.zeros((1,), device=device, dtype=dtype)
        self.units = units

    def reset(self) -> None:
        """Reset the buffer to zeros"""
        self.buffer = None
        self.samples_counter = torch.zeros((1,), device=self.device, dtype=torch.int64)

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

    def compute(self, reset: bool = True) -> torch.Tensor:
        if self.buffer is None:
            raise ValueError(
                f"Cannot compute value for metric '{self.name}' "
                "because it was never updated."
            )
        dist_utils.all_sum(self.buffer)
        dist_utils.all_sum(self.samples_counter)
        error = self.buffer / self.samples_counter
        if reset:
            self.reset()
        return error
