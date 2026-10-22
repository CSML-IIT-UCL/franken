from franken.datasets.registry import DATASET_REGISTRY

# Ensure all sub-datasets are imported so that they are registered.
from .water import water_dataset  # noqa: F401
from .water_transfer import water_transfer_dataset  # noqa: F401
from .TM23 import tm23_dataset  # noqa: F401
from .PtH2O import pth2o_dataset  # noqa: F401
from .test import test_dataset  # noqa: F401

__all__ = ("DATASET_REGISTRY",)
