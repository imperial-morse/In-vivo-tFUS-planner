"""Region-specific packing.

Two strategies, both honour the no-overlap (FWHM-tangent) constraint:

* `region_pack_centroid_seed` --- hex lattice anchored at the region centroid.
  Matches the user's spec: "one circle at least is positioned with centroid as
  the focus point, but the algorithm can add further ones to cover the
  remaining volume". Cheap, deterministic.

* `region_pack_max_coverage` --- enumerates N x N translated copies of the
  same hex lattice and returns whichever lattice yields the best total
  on-target coverage (with the same per-spot >= threshold rule). Often beats
  centroid-seeding for non-convex regions where the centroid lands in a
  poor spot or even off the region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .footprint import Footprint
from .hex_tile import FocalSpot, HexTilePlan, _evaluate_spot, _hex_centres
from .plane import FocalPlane


def _build_lattice(
    plane: FocalPlane,
    footprint: Footprint,
    anchor_xy: Tuple[float, float],
) -> np.ndarray:
    """Hex lattice covering the target bbox, snapped so that ``anchor_xy`` is on it."""
    ax, ay = anchor_xy
    rx, ry = footprint.rx_mm, footprint.ry_mm
    iis, jjs = np.where(plane.target_mask)
    if iis.size == 0:
        return np.empty((0, 2))
    corners = plane.world_xy(
        np.array([iis.min(), iis.min(), iis.max(), iis.max()]),
        np.array([jjs.min(), jjs.max(), jjs.min(), jjs.max()]),
    )
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    dx = 2.0 * rx
    dy = ry * np.sqrt(3.0)

    # Find lowest row index n_y such that y_anchor + n_y * dy <= y_min
    n_y = int(np.floor((y_min - ay) / dy)) - 1
    n_y_max = int(np.ceil((y_max - ay) / dy)) + 1
    pts = []
    for ny in range(n_y, n_y_max + 1):
        y = ay + ny * dy
        x_off = (ny % 2) * rx        # alternate-row shift, keep anchor row in phase
        n_x_min = int(np.floor((x_min - (ax + x_off)) / dx)) - 1
        n_x_max = int(np.ceil((x_max - (ax + x_off)) / dx)) + 1
        xs = ax + x_off + np.arange(n_x_min, n_x_max + 1) * dx
        for x in xs:
            pts.append((x, y))
    return np.asarray(pts) if pts else np.empty((0, 2))


def _filter_and_score(
    centres: np.ndarray,
    plane: FocalPlane,
    footprint: Footprint,
    coverage_threshold: float,
) -> Tuple[List[FocalSpot], float]:
    """Evaluate each candidate centre, keep those that meet the threshold,
    return (spots, total_target_area_covered_mm2)."""
    target_mask = plane.target_mask
    nontarget_brain_mask = plane.brain_mask & ~target_mask
    spots: List[FocalSpot] = []
    covered = np.zeros_like(target_mask, dtype=bool)

    rx, ry = footprint.rx_mm, footprint.ry_mm
    sx, sy = plane.step_xy_mm
    ox, oy = plane.origin_xy_mm
    Ni, Nj = plane.shape

    for cx, cy in centres:
        ta, oa, fa = _evaluate_spot(cx, cy, plane, footprint,
                                    target_mask, nontarget_brain_mask)
        if fa <= 0 or ta / fa < coverage_threshold:
            continue
        # mark covered pixels
        i_mid = (cx - ox) / sx
        j_mid = (cy - oy) / sy
        di = abs(rx / sx) + 1
        dj = abs(ry / sy) + 1
        i0 = max(0, int(np.floor(i_mid - di)))
        i1 = min(Ni - 1, int(np.ceil(i_mid + di)))
        j0 = max(0, int(np.floor(j_mid - dj)))
        j1 = min(Nj - 1, int(np.ceil(j_mid + dj)))
        if i0 > i1 or j0 > j1:
            continue
        ii = np.arange(i0, i1 + 1); jj = np.arange(j0, j1 + 1)
        II, JJ = np.meshgrid(ii, jj, indexing="ij")
        inside = footprint.contains(ox + II * sx - cx, oy + JJ * sy - cy)
        covered[i0:i1 + 1, j0:j1 + 1] |= inside
        spots.append(FocalSpot(
            center_world_xy_mm=np.array([cx, cy]),
            rx_mm=rx, ry_mm=ry,
            target_area_mm2=ta,
            offtarget_area_mm2=oa,
            footprint_area_mm2=fa,
        ))

    covered_mm2 = float((covered & target_mask).sum() * plane.pixel_area_mm2)
    return spots, covered_mm2


def region_pack_centroid_seed(
    plane: FocalPlane,
    footprint: Footprint,
    seed_xy_mm: Tuple[float, float],
    coverage_threshold: float = 0.80,
) -> HexTilePlan:
    """Hex lattice anchored on ``seed_xy_mm`` (typically the region centroid).

    Always emits at least the seed spot (even if its coverage is below the
    threshold) so the user can inspect the centroid hit; the seed is then
    flagged via its `roi_pct` field.
    """
    centres = _build_lattice(plane, footprint, seed_xy_mm)
    spots, covered = _filter_and_score(centres, plane, footprint, coverage_threshold)

    # Force-include the centroid seed if it is not already in the kept set.
    cx, cy = seed_xy_mm
    if not any(np.allclose(s.center_world_xy_mm, [cx, cy]) for s in spots):
        target_mask = plane.target_mask
        nontg = plane.brain_mask & ~target_mask
        ta, oa, fa = _evaluate_spot(cx, cy, plane, footprint, target_mask, nontg)
        if fa > 0:
            spots.insert(0, FocalSpot(
                center_world_xy_mm=np.array([cx, cy]),
                rx_mm=footprint.rx_mm, ry_mm=footprint.ry_mm,
                target_area_mm2=ta, offtarget_area_mm2=oa, footprint_area_mm2=fa,
            ))

    return HexTilePlan(
        spots=spots,
        coverage_threshold=coverage_threshold,
        footprint=footprint,
        plane_z_lps_mm=plane.z_lps_mm,
        target_area_mm2=plane.target_area_mm2,
        target_area_covered_mm2=covered,
    )


def region_pack_full_coverage(
    plane: FocalPlane,
    footprint: Footprint,
    candidate_step_factor: float = 0.4,
    min_target_pixels: int = 4,
) -> HexTilePlan:
    """Greedy max-coverage: cover every reachable target pixel with non-overlapping
    spots, ignoring the per-spot coverage threshold.

    Algorithm
    ---------
    1. Build a dense candidate grid (spacing = ``candidate_step_factor * min(rx, ry)``).
       This is *much* finer than a hex lattice, so non-convex regions like
       hippocampus / CA1 don't lose corners that fall between lattice points.
    2. Loop:
        a. Score every candidate by how many *currently uncovered* target pixels
           its footprint contains.
        b. Pick the candidate with the highest score that doesn't overlap any
           already-placed spot. Ellipse-overlap test:
           (dx / (2*rx))^2 + (dy / (2*ry))^2 < 1.
        c. Place that spot, mark its covered target pixels, repeat.
       Stop when no candidate would cover at least ``min_target_pixels`` more
       target pixels (a few pixels) without overlapping placed spots.

    This finds a non-overlapping packing that hits *every reachable* part of
    the target -- including isolated edges that lattice approaches miss.
    """
    target = plane.target_mask
    nontg = plane.brain_mask & ~target
    rx, ry = footprint.rx_mm, footprint.ry_mm
    sx, sy = plane.step_xy_mm
    ox, oy = plane.origin_xy_mm
    Ni, Nj = plane.shape

    if not target.any():
        return HexTilePlan(
            spots=[], coverage_threshold=0.0, footprint=footprint,
            plane_z_lps_mm=plane.z_lps_mm,
            target_area_mm2=0.0, target_area_covered_mm2=0.0,
        )

    # Target bbox + small padding (one footprint radius) so candidates can
    # cover the very edge.
    iis, jjs = np.where(target)
    corners = plane.world_xy(
        np.array([iis.min(), iis.min(), iis.max(), iis.max()]),
        np.array([jjs.min(), jjs.max(), jjs.min(), jjs.max()]),
    )
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    x_min -= rx; x_max += rx
    y_min -= ry; y_max += ry

    step_x = candidate_step_factor * rx
    step_y = candidate_step_factor * ry
    cand_xs = np.arange(x_min, x_max + step_x * 0.5, step_x)
    cand_ys = np.arange(y_min, y_max + step_y * 0.5, step_y)
    XX, YY = np.meshgrid(cand_xs, cand_ys, indexing="ij")
    candidates = np.column_stack([XX.ravel(), YY.ravel()])

    pixel_area = plane.pixel_area_mm2
    min_target_area = min_target_pixels * pixel_area

    spots: list[FocalSpot] = []
    covered = np.zeros_like(target, dtype=bool)

    def _footprint_inside_pixels(cx, cy):
        """Return (i_slice, j_slice, inside_mask) of pixels inside the footprint."""
        i_mid = (cx - ox) / sx; j_mid = (cy - oy) / sy
        di = abs(rx / sx) + 1; dj = abs(ry / sy) + 1
        i0 = max(0, int(np.floor(i_mid - di))); i1 = min(Ni - 1, int(np.ceil(i_mid + di)))
        j0 = max(0, int(np.floor(j_mid - dj))); j1 = min(Nj - 1, int(np.ceil(j_mid + dj)))
        if i0 > i1 or j0 > j1:
            return None, None, None
        ii = np.arange(i0, i1 + 1); jj = np.arange(j0, j1 + 1)
        II, JJ = np.meshgrid(ii, jj, indexing="ij")
        inside = footprint.contains(ox + II * sx - cx, oy + JJ * sy - cy)
        return slice(i0, i1 + 1), slice(j0, j1 + 1), inside

    while True:
        remaining = target & ~covered
        if not remaining.any():
            break

        best_score = 0
        best_cx = best_cy = None
        best_slices = None
        for cx, cy in candidates:
            # Non-overlap test against existing spots
            ok = True
            for s in spots:
                sxc, syc = s.center_world_xy_mm
                dx = cx - sxc; dy = cy - syc
                if (dx / (2 * rx)) ** 2 + (dy / (2 * ry)) ** 2 < 1.0:
                    ok = False; break
            if not ok:
                continue
            si, sj, inside = _footprint_inside_pixels(cx, cy)
            if inside is None:
                continue
            score = int((inside & remaining[si, sj]).sum())
            if score > best_score:
                best_score = score
                best_cx, best_cy = cx, cy
                best_slices = (si, sj, inside)

        if best_score == 0 or best_score * pixel_area < min_target_area:
            break

        # Score for the placed spot is computed against the FULL target / off-target
        cx, cy = best_cx, best_cy
        si, sj, inside = best_slices
        target_area = float((inside & target[si, sj]).sum() * pixel_area)
        offtg_area = float((inside & nontg[si, sj]).sum() * pixel_area)
        spots.append(FocalSpot(
            center_world_xy_mm=np.array([cx, cy]),
            rx_mm=rx, ry_mm=ry,
            target_area_mm2=target_area,
            offtarget_area_mm2=offtg_area,
            footprint_area_mm2=footprint.area_mm2,
        ))
        covered[si, sj] |= inside

    covered_mm2 = float((covered & target).sum() * pixel_area)
    return HexTilePlan(
        spots=spots,
        coverage_threshold=0.0,
        footprint=footprint,
        plane_z_lps_mm=plane.z_lps_mm,
        target_area_mm2=plane.target_area_mm2,
        target_area_covered_mm2=covered_mm2,
    )


def region_pack_max_coverage(
    plane: FocalPlane,
    footprint: Footprint,
    coverage_threshold: float = 0.80,
    n_offsets: int = 5,
) -> HexTilePlan:
    """Try ``n_offsets x n_offsets`` translated hex lattices, return the best.

    "Best" = highest total target area covered. This is significantly stronger
    than centroid-seeding for non-convex regions (e.g. hippocampus, striatum
    head), at the cost of n_offsets**2 extra evaluations.
    """
    if n_offsets < 1:
        raise ValueError("n_offsets must be >= 1")
    rx, ry = footprint.rx_mm, footprint.ry_mm
    iis, jjs = np.where(plane.target_mask)
    if iis.size == 0:
        return HexTilePlan(
            spots=[], coverage_threshold=coverage_threshold,
            footprint=footprint, plane_z_lps_mm=plane.z_lps_mm,
            target_area_mm2=0.0, target_area_covered_mm2=0.0,
        )
    corners = plane.world_xy(
        np.array([iis.min(), iis.min(), iis.max(), iis.max()]),
        np.array([jjs.min(), jjs.max(), jjs.min(), jjs.max()]),
    )
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)

    dx = 2.0 * rx
    dy = ry * np.sqrt(3.0)

    best_spots: List[FocalSpot] = []
    best_covered = -1.0
    for ix in range(n_offsets):
        for iy in range(n_offsets):
            ax = x_min + (ix / n_offsets) * dx
            ay = y_min + (iy / n_offsets) * dy
            centres = _build_lattice(plane, footprint, (ax, ay))
            spots, covered = _filter_and_score(centres, plane, footprint, coverage_threshold)
            if covered > best_covered:
                best_covered = covered
                best_spots = spots

    return HexTilePlan(
        spots=best_spots,
        coverage_threshold=coverage_threshold,
        footprint=footprint,
        plane_z_lps_mm=plane.z_lps_mm,
        target_area_mm2=plane.target_area_mm2,
        target_area_covered_mm2=max(best_covered, 0.0),
    )
