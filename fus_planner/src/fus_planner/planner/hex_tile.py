"""Whole-brain (or any-mask) hex tiling planner.

Generates a hexagonal lattice of focal-spot centres in the focal plane,
sized so the (FWHM) circles or ellipses are tangent (no overlap), and keeps
only the spots whose footprint covers >= ``coverage_threshold`` of the
target mask.

For anisotropic FWHM (rx != ry) we stretch the hex lattice so the ellipses
remain tangent: spacing is 2*rx in the row direction and ry*sqrt(3) between
rows, with alternate rows shifted by rx. This is the affine image of the
isotropic packing under (x, y) -> (x, y * rx/ry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .footprint import Footprint
from .plane import FocalPlane


@dataclass
class FocalSpot:
    center_world_xy_mm: np.ndarray   # (2,) LPS world mm in the focal plane
    rx_mm: float
    ry_mm: float
    target_area_mm2: float
    offtarget_area_mm2: float        # non-target *brain* area inside the footprint
    footprint_area_mm2: float

    @property
    def roi_pct(self) -> float:
        return 100.0 * self.target_area_mm2 / self.footprint_area_mm2

    @property
    def offtarget_pct(self) -> float:
        return 100.0 * self.offtarget_area_mm2 / self.footprint_area_mm2

    @property
    def coverage_fraction(self) -> float:
        """target_area / footprint_area (0..1)."""
        return self.target_area_mm2 / self.footprint_area_mm2 if self.footprint_area_mm2 else 0.0


@dataclass
class HexTilePlan:
    spots: List[FocalSpot]
    coverage_threshold: float
    footprint: Footprint
    plane_z_lps_mm: float
    target_area_mm2: float
    target_area_covered_mm2: float

    @property
    def n_spots(self) -> int:
        return len(self.spots)

    @property
    def total_target_coverage_pct(self) -> float:
        return 100.0 * self.target_area_covered_mm2 / self.target_area_mm2 if self.target_area_mm2 else 0.0


def _hex_centres(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    rx: float, ry: float,
) -> np.ndarray:
    """Return (N, 2) hex-lattice centres covering the bbox.

    Even rows at x = x_min + 2*rx*i; odd rows offset by rx.
    Row pitch is ry*sqrt(3) so isotropically-spaced ellipses are tangent.
    """
    dx = 2.0 * rx
    dy = ry * np.sqrt(3.0)
    if dx <= 0 or dy <= 0:
        raise ValueError("rx and ry must be positive")

    # Pad bbox so we cover edges; +/- one cell.
    n_rows = int(np.ceil((y_max - y_min) / dy)) + 2
    n_cols = int(np.ceil((x_max - x_min) / dx)) + 2

    rows = np.arange(n_rows)
    cols = np.arange(n_cols)
    cc, rr = np.meshgrid(cols, rows, indexing="xy")
    xs = x_min + cc * dx + (rr % 2) * rx
    ys = y_min + rr * dy
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    keep = (
        (pts[:, 0] >= x_min - rx) & (pts[:, 0] <= x_max + rx)
        & (pts[:, 1] >= y_min - ry) & (pts[:, 1] <= y_max + ry)
    )
    return pts[keep]


def _evaluate_spot(
    cx: float, cy: float,
    plane: FocalPlane,
    footprint: Footprint,
    target_mask: np.ndarray,
    nontarget_brain_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Return (target_area, offtarget_area, footprint_area) all in mm^2.

    Computed by intersecting the elliptical footprint with the masks via a
    pixel test inside the (axis-aligned) bounding box of the ellipse — far
    cheaper than scanning the whole plane for every spot.
    """
    rx, ry = footprint.rx_mm, footprint.ry_mm
    sx, sy = plane.step_xy_mm
    ox, oy = plane.origin_xy_mm
    Ni, Nj = plane.shape

    # Pixel-index range covering [cx - rx, cx + rx] in world x.
    # Note step may be negative.
    i0 = int(np.floor((cx - rx - ox) / sx))
    i1 = int(np.ceil((cx + rx - ox) / sx))
    if sx < 0:
        i0, i1 = i1, i0
    j0 = int(np.floor((cy - ry - oy) / sy))
    j1 = int(np.ceil((cy + ry - oy) / sy))
    if sy < 0:
        j0, j1 = j1, j0
    i0 = max(0, i0); i1 = min(Ni - 1, i1)
    j0 = max(0, j0); j1 = min(Nj - 1, j1)
    if i0 > i1 or j0 > j1:
        return 0.0, 0.0, footprint.area_mm2

    ii = np.arange(i0, i1 + 1)
    jj = np.arange(j0, j1 + 1)
    II, JJ = np.meshgrid(ii, jj, indexing="ij")          # (di, dj)
    XX = ox + II * sx
    YY = oy + JJ * sy
    inside = footprint.contains(XX - cx, YY - cy)
    pix_area = plane.pixel_area_mm2
    foot_area = footprint.area_mm2

    sub_target = target_mask[i0:i1 + 1, j0:j1 + 1]
    sub_nontg  = nontarget_brain_mask[i0:i1 + 1, j0:j1 + 1]
    target_area    = float((inside & sub_target).sum() * pix_area)
    offtarget_area = float((inside & sub_nontg).sum() * pix_area)
    return target_area, offtarget_area, foot_area


def hex_tile_plan(
    plane: FocalPlane,
    footprint: Footprint,
    coverage_threshold: float = 0.80,
) -> HexTilePlan:
    """Cover ``plane.target_mask`` with non-overlapping FWHM footprints on a hex lattice.

    Spots are kept iff target_area / footprint_area >= ``coverage_threshold``.
    Off-target area is computed against any *brain* tissue not in the target
    (i.e. ``brain_mask & ~target_mask``).

    For "max-coverage" (no threshold) behaviour pass a tiny value such as
    ``1e-3`` so every spot that even nicks the target is kept.
    """
    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in (0, 1]")

    target_mask = plane.target_mask
    nontarget_brain_mask = plane.brain_mask & ~target_mask

    # World bbox of target mask
    if not target_mask.any():
        return HexTilePlan(
            spots=[], coverage_threshold=coverage_threshold,
            footprint=footprint, plane_z_lps_mm=plane.z_lps_mm,
            target_area_mm2=0.0, target_area_covered_mm2=0.0,
        )
    iis, jjs = np.where(target_mask)
    corners = plane.world_xy(
        np.array([iis.min(), iis.min(), iis.max(), iis.max()]),
        np.array([jjs.min(), jjs.max(), jjs.min(), jjs.max()]),
    )
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    centres = _hex_centres(x_min, x_max, y_min, y_max,
                           rx=footprint.rx_mm, ry=footprint.ry_mm)

    spots: List[FocalSpot] = []
    covered_pixels = np.zeros_like(target_mask, dtype=bool)
    for (cx, cy) in centres:
        ta, oa, fa = _evaluate_spot(cx, cy, plane, footprint,
                                    target_mask, nontarget_brain_mask)
        if fa <= 0 or ta / fa < coverage_threshold:
            continue
        # Mark pixels covered (for total coverage accounting)
        rx, ry = footprint.rx_mm, footprint.ry_mm
        sx, sy = plane.step_xy_mm; ox, oy = plane.origin_xy_mm
        Ni, Nj = plane.shape
        i0 = max(0, int(np.floor((cx - rx - ox) / sx)) if sx > 0 else int(np.floor((cx + rx - ox) / sx)))
        i1 = min(Ni - 1, int(np.ceil((cx + rx - ox) / sx)) if sx > 0 else int(np.ceil((cx - rx - ox) / sx)))
        j0 = max(0, int(np.floor((cy - ry - oy) / sy)) if sy > 0 else int(np.floor((cy + ry - oy) / sy)))
        j1 = min(Nj - 1, int(np.ceil((cy + ry - oy) / sy)) if sy > 0 else int(np.ceil((cy - ry - oy) / sy)))
        ii = np.arange(i0, i1 + 1); jj = np.arange(j0, j1 + 1)
        II, JJ = np.meshgrid(ii, jj, indexing="ij")
        inside = footprint.contains(ox + II * sx - cx, oy + JJ * sy - cy)
        covered_pixels[i0:i1 + 1, j0:j1 + 1] |= inside

        spots.append(FocalSpot(
            center_world_xy_mm=np.array([cx, cy]),
            rx_mm=footprint.rx_mm,
            ry_mm=footprint.ry_mm,
            target_area_mm2=ta,
            offtarget_area_mm2=oa,
            footprint_area_mm2=fa,
        ))

    target_area_mm2 = plane.target_area_mm2
    covered_mm2 = float((covered_pixels & target_mask).sum() * plane.pixel_area_mm2)
    return HexTilePlan(
        spots=spots,
        coverage_threshold=coverage_threshold,
        footprint=footprint,
        plane_z_lps_mm=plane.z_lps_mm,
        target_area_mm2=target_area_mm2,
        target_area_covered_mm2=covered_mm2,
    )
