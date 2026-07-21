"""Grid sizing.

We build a k-Wave grid that is just large enough to contain:

  * the transducer apex (with a small gap behind it),
  * the geometric focus (apex + radius_of_curvature along +x), and
  * the sensor box around the future skull region, plus margins.

Coordinate convention (matches k-Wave-python ``kgrid``): the grid is centred on
the origin. ``x`` is the axial / beam direction (transducer fires toward +x),
``y`` and ``z`` are the two lateral directions.

This module is pure NumPy - it does not import k-Wave, so it can be used for the
instant geometry preview and for unit tests without the solver installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .params import SimParams, WATER


def round_even(x: float) -> int:
    """Round to the nearest even integer."""
    return int(round(x / 2.0) * 2)


@dataclass
class GridSpec:
    # grid size
    dx: float                      # isotropic spacing [m]
    Nx: int
    Ny: int
    Nz: int

    # physical extents [m]
    x_size: float
    y_size: float
    z_size: float

    # key positions in the centred grid frame [m]
    apex_xyz: Tuple[float, float, float]    # bowl rear-surface centre (apex)
    focus_xyz: Tuple[float, float, float]   # geometric focus

    # sensor box (inclusive index bounds) around the focus / skull region
    box_ix: Tuple[int, int]
    box_iy: Tuple[int, int]
    box_iz: Tuple[int, int]

    # axis vectors [m], centred on origin
    x_vec: np.ndarray
    y_vec: np.ndarray
    z_vec: np.ndarray

    # ---- derived helpers ----
    @property
    def n_points(self) -> int:
        return self.Nx * self.Ny * self.Nz

    @property
    def box_shape(self) -> Tuple[int, int, int]:
        return (self.box_ix[1] - self.box_ix[0] + 1,
                self.box_iy[1] - self.box_iy[0] + 1,
                self.box_iz[1] - self.box_iz[0] + 1)

    @property
    def box_n_points(self) -> int:
        s = self.box_shape
        return s[0] * s[1] * s[2]

    def memory_per_field_gb(self) -> float:
        return self.n_points * 4 / 1e9   # float32

    def focus_in_grid(self) -> bool:
        fx, fy, fz = self.focus_xyz
        return (self.x_vec[0] <= fx <= self.x_vec[-1] and
                self.y_vec[0] <= fy <= self.y_vec[-1] and
                self.z_vec[0] <= fz <= self.z_vec[-1])


def build_grid_spec(p: SimParams) -> GridSpec:
    """Compute an auto-sized :class:`GridSpec` from user parameters."""
    p.validate()

    dx = p.grid_spacing_m()
    roc = p.focal_length_m
    aperture_r = p.aperture_diameter_m / 2.0

    back = p.back_gap_mm * 1e-3
    front = p.front_margin_mm * 1e-3
    lat_margin = p.lateral_margin_mm * 1e-3

    box_ax = p.skull_box_axial_mm * 1e-3
    box_lr = p.skull_box_lr_mm * 1e-3
    box_ap = p.skull_box_ap_mm * 1e-3

    # ----- axial extent: behind apex + ROC + half the box downstream + margin
    x_size_phys = back + roc + box_ax / 2.0 + front

    # ----- lateral extents: hold whichever is wider (aperture or box) + margin
    y_half = max(aperture_r, box_lr / 2.0) + lat_margin
    z_half = max(aperture_r, box_ap / 2.0) + lat_margin
    y_size_phys = 2.0 * y_half
    z_size_phys = 2.0 * z_half

    Nx = round_even(x_size_phys / dx)
    Ny = round_even(y_size_phys / dx)
    Nz = round_even(z_size_phys / dx)

    # centred axis vectors (k-Wave-python convention for even N)
    x_vec = (np.arange(Nx) - Nx // 2) * dx
    y_vec = (np.arange(Ny) - Ny // 2) * dx
    z_vec = (np.arange(Nz) - Nz // 2) * dx

    x_min = float(x_vec[0])

    # place the apex a "back_gap" inside the low-x boundary; focus = apex + ROC
    apex_x = x_min + back
    focus_x = apex_x + roc
    apex_xyz = (apex_x, 0.0, 0.0)
    focus_xyz = (focus_x, 0.0, 0.0)

    # ----- sensor box index bounds, centred on the focus -----
    def _bounds(center, half, vec, N):
        ic = int(np.argmin(np.abs(vec - center)))
        h = int(round(half / dx))
        lo = max(1, ic - h)
        hi = min(N - 2, ic + h)
        if hi < lo:
            lo, hi = ic, ic
        return lo, hi

    box_ix = _bounds(focus_x, box_ax / 2.0, x_vec, Nx)
    box_iy = _bounds(0.0, box_lr / 2.0, y_vec, Ny)
    box_iz = _bounds(0.0, box_ap / 2.0, z_vec, Nz)

    return GridSpec(
        dx=dx, Nx=Nx, Ny=Ny, Nz=Nz,
        x_size=Nx * dx, y_size=Ny * dx, z_size=Nz * dx,
        apex_xyz=apex_xyz, focus_xyz=focus_xyz,
        box_ix=box_ix, box_iy=box_iy, box_iz=box_iz,
        x_vec=x_vec, y_vec=y_vec, z_vec=z_vec,
    )
