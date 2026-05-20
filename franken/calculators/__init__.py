"""Deploy the franken model as a calculator through different interfaces.
"""

import importlib
import importlib.util

from .ase_calc import FrankenCalculator

__all__ = ("FrankenCalculator",)


