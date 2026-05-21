import torch

from franken.metrics.base import BaseMetric
from franken.data.base import TargetType


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MetricRegistry(metaclass=Singleton):
    def __init__(self) -> None:
        self._metrics: dict[str, type[BaseMetric]] = {}

    def register(self):
        def inner(metric_cls: type[BaseMetric]):
            metric_name = metric_cls.name
            if metric_name in self._metrics:
                raise RuntimeError(
                    f"Attempting to re-register metric with name '{metric_name}' twice"
                )
            self._metrics[metric_name] = metric_cls
            return metric_cls

        return inner

    def init_metric(
        self,
        name: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> BaseMetric:
        """Create a new instance of a metric"""
        if name not in self._metrics:
            raise KeyError(
                f"Metric '{name}' not found. Available metrics: {self.available_metrics}"
            )
        metric_cls = self._metrics[name]
        return metric_cls(device, dtype)

    @property
    def available_metrics(self) -> list[str]:
        return self.available_metrics_for_target()

    def available_metrics_for_target(
        self, target_type: TargetType | None = None
    ) -> list[str]:
        all_metric_names = []
        for metric_name, metric_cls in self._metrics.items():
            if target_type is None:
                all_metric_names.append(metric_name)
            else:
                if metric_cls.target_type == target_type:
                    all_metric_names.append(metric_name)
        return all_metric_names


metric_registry = MetricRegistry()
