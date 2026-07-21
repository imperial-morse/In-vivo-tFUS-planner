"""Core simulation logic (grid sizing, transducer, GPU detection, calibration).
"""

from .params import SimParams, WATER
from .grid import GridSpec, build_grid_spec, round_even
from .gpu import detect_gpu, choose_solver

__all__ = [
    "SimParams",
    "WATER",
    "GridSpec",
    "build_grid_spec",
    "round_even",
    "detect_gpu",
    "choose_solver",
]
