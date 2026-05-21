import torch

from franken.metrics.base import BaseMetric
from franken.metrics.functions import *  # noqa: F403
from franken.metrics.registry import metric_registry

__all__ = ["metric_registry"]


def available_metrics() -> list[str]:
    return metric_registry.available_metrics


def register_metric(metric_class: type) -> None:
    metric_registry.register()(metric_class)


def init_metric(
    name: str, device: torch.device, dtype: torch.dtype = torch.float32
) -> BaseMetric:
    return metric_registry.init_metric(name, device, dtype)
