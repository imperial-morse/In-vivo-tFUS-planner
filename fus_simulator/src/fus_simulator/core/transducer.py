"""Transducer source mask: focused bowl element with an optional central hole.

The bowl is built with ``kWaveArray.add_bowl_element``. The geometric focus of a focusing cap is its centre of
curvature, so:

    sphere centre  = geometric focus
    apex           = focus - ROC * axis_direction   (here axis = +x)

The central hole is modelled as a true annular aperture: every source voxel
whose radial distance from the beam axis (the y-z distance from centre) is less
than the hole radius is removed. 
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .params import SimParams
from .grid import GridSpec


# --------------------------------------------------------------------------- #
# Pure-NumPy preview (no k-Wave needed) - used by the GUI geometry panel
# --------------------------------------------------------------------------- #
def bowl_arc_xz(gspec: GridSpec, p: SimParams, n: int = 400):
    """Return (x, z) arrays [m] tracing the bowl cap in the x-z plane.

    Where a central hole is present, the inner part of the arc is replaced with
    NaN.
    """
    roc = p.focal_length_m
    fx = gspec.focus_xyz[0]
    aperture_r = p.aperture_diameter_m / 2.0
    theta_max = np.arcsin(np.clip(aperture_r / roc, -1.0, 1.0))
    theta = np.linspace(-theta_max, theta_max, n)
    x = fx - roc * np.cos(theta)
    z = roc * np.sin(theta)

    if p.hole_enabled and p.hole_diameter_m > 0:
        hole_r = p.hole_diameter_m / 2.0
        theta_min = np.arcsin(np.clip(hole_r / roc, -1.0, 1.0))
        x = x.copy()
        z = z.copy()
        x[np.abs(theta) < theta_min] = np.nan
        z[np.abs(theta) < theta_min] = np.nan
    return x, z


def bowl_surface_3d(gspec: GridSpec, p: SimParams, n_theta: int = 36, n_phi: int = 72):
    """Return (X, Y, Z) meshes [m] of the bowl cap surface for a 3-D plot.

    The cap is parametrised on the sphere of radius ROC centred at the focus.
    With a central hole, the polar angle starts at ``theta_min`` so the inner
    disc is left open
    """
    roc = p.focal_length_m
    fx = gspec.focus_xyz[0]
    aperture_r = p.aperture_diameter_m / 2.0
    theta_max = float(np.arcsin(np.clip(aperture_r / roc, -1.0, 1.0)))
    theta_min = 0.0
    if p.hole_enabled and p.hole_diameter_m > 0:
        theta_min = float(np.arcsin(np.clip((p.hole_diameter_m / 2.0) / roc, -1.0, 1.0)))

    th = np.linspace(theta_min, theta_max, n_theta)
    ph = np.linspace(0.0, 2.0 * np.pi, n_phi)
    TH, PH = np.meshgrid(th, ph)
    X = fx - roc * np.cos(TH)
    Y = roc * np.sin(TH) * np.cos(PH)
    Z = roc * np.sin(TH) * np.sin(PH)
    return X, Y, Z


# --------------------------------------------------------------------------- #
# Source-cost estimation + adaptive integration density
# --------------------------------------------------------------------------- #
def choose_upsampling(gspec: GridSpec, p: SimParams) -> int:
    """Pick kWaveArray ``upsampling_rate`` from how well the grid already
    resolves the aperture.

    kWaveArray integrates each element's band-limited interpolant using
    ``upsampling_rate`` sub-points per grid point per axis, so the work and
    memory scale as ``upsampling_rate**2`` times the aperture area in voxels.
    """
    ap_pts = p.aperture_diameter_m / gspec.dx          # grid points across aperture
    if ap_pts >= 60:
        return 2
    if ap_pts >= 30:
        return 3
    if ap_pts >= 15:
        return 5
    return 8


def estimate_source_points(gspec: GridSpec, p: SimParams, upsampling=None):
    """Rough (upper-bound) count of kWaveArray integration points and the
    upsampling that would be used."""
    if upsampling is None:
        upsampling = choose_upsampling(gspec, p)
    ap_r = p.aperture_diameter_m / 2.0
    disc_grid = np.pi * (ap_r / gspec.dx) ** 2          # projected aperture disc, voxels
    return int(disc_grid * upsampling ** 2), int(upsampling)


# --------------------------------------------------------------------------- #
# k-Wave source mask (requires k-wave-python)
# --------------------------------------------------------------------------- #
def build_source_mask(kgrid, gspec: GridSpec, p: SimParams,
                      focus_xyz=None, progress=None) -> Tuple[object, np.ndarray, int]:
    """Build the binary source mask for the bowl (+ optional hole).

    Returns ``(karray, source_p_mask, n_active_points)``.

    ``focus_xyz`` optionally moves the geometric focus to an arbitrary point
    (e.g. an off-axis protocol spot). The bowl keeps its +x orientation and is
    translated so its apex sits a distance ROC behind the focus on the beam
    axis; the central hole is taken about that same (shifted) axis.

    ``kgrid`` is a live ``kWaveGrid``; this function imports k-Wave lazily.
    """
    from kwave.utils.kwave_array import kWaveArray

    if focus_xyz is None:
        focus_xyz = gspec.focus_xyz
        apex = gspec.apex_xyz
    else:
        apex = (focus_xyz[0] - p.focal_length_m, focus_xyz[1], focus_xyz[2])

    ups = choose_upsampling(gspec, p)
    est_pts, _ = estimate_source_points(gspec, p, ups)
    if progress is not None:
        progress(f"Source: upsampling x{ups}, ~{est_pts:,} integration points "
                 f"(aperture spans {p.aperture_diameter_m / gspec.dx:.0f} grid points)")
    karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=ups, single_precision=True)
    karray.add_bowl_element(
        position=list(apex),               # bowl rear-surface centre (apex) [m]
        radius=p.focal_length_m,           # radius of curvature (= focal length)
        diameter=p.aperture_diameter_m,    # aperture diameter
        focus_pos=list(focus_xyz),         # geometric focus (sets orientation)
    )

    source_mask = karray.get_array_binary_mask(kgrid).astype(bool)

    if p.hole_enabled and p.hole_diameter_m > 0:
        source_mask = apply_central_hole(source_mask, gspec, p.hole_diameter_m,
                                         center_yz=(focus_xyz[1], focus_xyz[2]))

    n_active = int(source_mask.sum())
    if n_active == 0:
        raise RuntimeError(
            "Transducer source mask is empty - check geometry (aperture, ROC, "
            "hole, and grid spacing).")
    return karray, source_mask, n_active


def apply_central_hole(source_mask: np.ndarray, gspec: GridSpec, hole_diameter_m: float,
                       center_yz=(0.0, 0.0)) -> np.ndarray:
    """Remove source voxels within ``hole_radius`` of the beam axis.

    ``center_yz`` is the (y, z) of the beam axis (0,0 for an on-axis bowl).
    """
    hole_r = hole_diameter_m / 2.0
    yy = gspec.y_vec[None, :, None] - center_yz[0]
    zz = gspec.z_vec[None, None, :] - center_yz[1]
    radial = np.sqrt(yy ** 2 + zz ** 2)            # broadcasts to (1, Ny, Nz)
    keep_ring = radial >= hole_r                    # (1, Ny, Nz)
    return source_mask & keep_ring                  # broadcast over x
