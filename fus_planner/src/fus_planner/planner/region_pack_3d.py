"""3D-aware region packing.

Background
----------
The earlier 2D planner extracted a single xy slice at the focal plane and
packed circles on it. For non-convex regions like hippocampus that extend
far above / below the focal plane, the 2D view missed parts that exist at
other depths.

This module instead:

1. Builds the **flat xy mask** of the target = union of the 3D target
   across z. That defines the candidate xy bbox we'll plan inside of.
2. **Scores each candidate (cx, cy)** by the 3D target volume inside the
   beam ellipsoid centred at (cx, cy, z_focus) with semi-axes (rx, ry, rz).
   Beam: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 + ((z-z_focus)/rz)^2 <= 1.
3. **Greedily places** the candidate with the largest uncovered 3D target
   volume, ensuring xy non-overlap with previously placed spots
   (ellipse non-overlap test on (cx, cy)).

Three strategies share the same machinery, differing only in how they
filter candidates:

* **centroid**: seed at the target centroid xy, greedy fill. Keep a spot
  iff 3D target_voxels / beam_voxels >= threshold (the centroid seed
  itself is always kept so the user can see what's hit, regardless).
* **coverage**: greedy from scratch. Keep a spot iff 3D ratio >=
  threshold.
* **max-coverage**: greedy from scratch. Threshold ignored; keep any
  spot that adds at least `min_target_voxels` of uncovered target.

For speed we downsample the labels volume by ``ds_factor`` (default 4 ->
0.06 mm at full DUKE resolution) before evaluating candidates. The
beam-axis math is unaffected because the downsampled grid still maps
linearly to world mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..io.volume import Volume
from .footprint import Footprint
from .hex_tile import FocalSpot, HexTilePlan
from .plane import FocalPlane


# ---------------------------------------------------------------------------
#                          downsampled 3D arrays
# ---------------------------------------------------------------------------

@dataclass
class Target3D:
    """Downsampled 3D target / non-target masks plus geometry."""

    target: np.ndarray         # bool (Ni', Nj', Nk')
    nontarget_brain: np.ndarray
    origin: np.ndarray         # (3,) world LPS mm
    spacing: np.ndarray        # (3,) signed step per voxel in world mm (matches direction*spacing*ds)
    ds_factor: int

    @property
    def shape(self):
        return self.target.shape

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(self.spacing[0] * self.spacing[1] * self.spacing[2]))

    def voxel_to_world_arrays(self, i_idx, j_idx, k_idx):
        x = self.origin[0] + i_idx * self.spacing[0]
        y = self.origin[1] + j_idx * self.spacing[1]
        z = self.origin[2] + k_idx * self.spacing[2]
        return x, y, z


def build_target_3d(
    labels: Volume,
    target_labels: Optional[Sequence[int]],
    ds_factor: int = 4,
) -> Target3D:
    """Build a downsampled 3D mask of the target and the rest of the brain.

    target_labels = None -> whole brain (target = labels > 0, no off-target).
    """
    if ds_factor < 1:
        ds_factor = 1
    sub = labels.data[::ds_factor, ::ds_factor, ::ds_factor]
    brain = sub > 0
    if target_labels is None:
        target = brain.copy()
        nontarget = np.zeros_like(brain)
    else:
        target = np.isin(sub, list(target_labels))
        nontarget = brain & ~target
    spacing = np.array([
        labels.direction[0, 0] * labels.spacing[0] * ds_factor,
        labels.direction[1, 1] * labels.spacing[1] * ds_factor,
        labels.direction[2, 2] * labels.spacing[2] * ds_factor,
    ])
    return Target3D(
        target=target,
        nontarget_brain=nontarget,
        origin=labels.origin.copy(),
        spacing=spacing,
        ds_factor=ds_factor,
    )


def flat_xy_mask(t3d: Target3D) -> np.ndarray:
    """xy projection: any voxel of the target along z."""
    return t3d.target.any(axis=2)


# ---------------------------------------------------------------------------
#                          per-spot 3D evaluation
# ---------------------------------------------------------------------------

def _beam_bbox_voxels(
    t3d: Target3D,
    cx: float, cy: float, z_focus: float,
    rx: float, ry: float, rz: float,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return (i0, i1, j0, j1, k0, k1) inclusive voxel ranges that bound the
    beam ellipsoid in the downsampled grid, or None if outside the grid."""
    nx, ny, nz = t3d.shape
    sx, sy, sz = t3d.spacing
    ox, oy, oz = t3d.origin
    i_c = (cx - ox) / sx
    j_c = (cy - oy) / sy
    k_c = (z_focus - oz) / sz
    di = abs(rx / sx) + 1
    dj = abs(ry / sy) + 1
    dk = abs(rz / sz) + 1
    i0 = max(0, int(np.floor(i_c - di))); i1 = min(nx - 1, int(np.ceil(i_c + di)))
    j0 = max(0, int(np.floor(j_c - dj))); j1 = min(ny - 1, int(np.ceil(j_c + dj)))
    k0 = max(0, int(np.floor(k_c - dk))); k1 = min(nz - 1, int(np.ceil(k_c + dk)))
    if i0 > i1 or j0 > j1 or k0 > k1:
        return None
    return i0, i1, j0, j1, k0, k1


def _beam_inside_mask(
    t3d: Target3D,
    bb: Tuple[int, int, int, int, int, int],
    cx: float, cy: float, z_focus: float,
    rx: float, ry: float, rz: float,
) -> np.ndarray:
    i0, i1, j0, j1, k0, k1 = bb
    sx, sy, sz = t3d.spacing
    ox, oy, oz = t3d.origin
    ii = np.arange(i0, i1 + 1); jj = np.arange(j0, j1 + 1); kk = np.arange(k0, k1 + 1)
    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing="ij")
    XX = ox + II * sx
    YY = oy + JJ * sy
    ZZ = oz + KK * sz
    return ((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2 + ((ZZ - z_focus) / rz) ** 2 <= 1.0


# ---------------------------------------------------------------------------
#                          candidate grid
# ---------------------------------------------------------------------------

def _candidate_grid(
    flat_mask: np.ndarray,
    t3d: Target3D,
    rx: float, ry: float,
    candidate_step_factor: float,
) -> np.ndarray:
    """xy candidate grid covering the bbox of the flat mask + one footprint pad."""
    iis, jjs = np.where(flat_mask)
    if iis.size == 0:
        return np.empty((0, 2))
    sx, sy, _ = t3d.spacing
    ox, oy, _ = t3d.origin
    x_a = ox + iis.min() * sx; x_b = ox + iis.max() * sx
    y_a = oy + jjs.min() * sy; y_b = oy + jjs.max() * sy
    x_min, x_max = min(x_a, x_b) - rx, max(x_a, x_b) + rx
    y_min, y_max = min(y_a, y_b) - ry, max(y_a, y_b) + ry
    step_x = candidate_step_factor * rx
    step_y = candidate_step_factor * ry
    xs = np.arange(x_min, x_max + step_x * 0.5, step_x)
    ys = np.arange(y_min, y_max + step_y * 0.5, step_y)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([XX.ravel(), YY.ravel()])


# ---------------------------------------------------------------------------
#                          core planner
# ---------------------------------------------------------------------------

@dataclass
class _ScoreCache:
    """Per-candidate scratch space."""
    bb: Optional[Tuple[int, int, int, int, int, int]]
    inside: Optional[np.ndarray]


def _ellipse_overlap_xy(cx, cy, sx, sy, rx, ry) -> bool:
    """True iff (cx, cy) lies inside the no-overlap ellipse around (sx, sy).
    Two beams (same shape) tangent in xy when ((dx/2rx)^2 + (dy/2ry)^2) == 1."""
    return ((cx - sx) / (2 * rx)) ** 2 + ((cy - sy) / (2 * ry)) ** 2 < 1.0


def plan_3d(
    labels: Volume,
    target_labels: Optional[Sequence[int]],
    z_focus_lps_mm: float,
    footprint: Footprint,
    fwhm_axial_mm: float,
    strategy: str,                  # "centroid" | "coverage" | "max-coverage"
    coverage_threshold: float = 0.80,
    seed_xy_lps_mm: Optional[Tuple[float, float]] = None,
    candidate_step_factor: float = 0.4,
    ds_factor: int = 4,
    min_target_voxels: int = 4,
) -> Tuple[HexTilePlan, FocalPlane]:
    """3D-aware planner. Returns (plan, focal_plane_for_viz).

    The returned ``focal_plane_for_viz`` carries the **flat xy mask** as its
    ``target_mask`` so the existing 2D viewers / report can render the plan
    consistently (spots overlaid on the projected target footprint).
    """
    if strategy not in ("centroid", "coverage", "max-coverage"):
        raise ValueError(f"unknown strategy {strategy!r}")

    rx, ry = footprint.rx_mm, footprint.ry_mm
    rz = fwhm_axial_mm / 2.0
    if rz <= 0:
        raise ValueError("fwhm_axial_mm must be > 0")

    # 1. 3D target arrays at downsampled resolution
    t3d = build_target_3d(labels, target_labels, ds_factor=ds_factor)
    flat = flat_xy_mask(t3d)
    voxel_volume = t3d.voxel_volume_mm3

    # 2. xy candidate grid bounded by flat mask
    candidates = _candidate_grid(flat, t3d, rx, ry, candidate_step_factor)

    # 3. For visualization, build a FocalPlane carrying the projected (flat)
    #    masks. We use the downsampled grid — it has plenty of resolution for
    #    overlay rendering and avoids allocating a full-resolution bool volume
    #    just to project it.
    target_proj_ds = flat                                     # already computed
    brain_proj_ds = (t3d.target | t3d.nontarget_brain).any(axis=2)
    plane_for_viz = FocalPlane(
        target_mask=target_proj_ds,
        brain_mask=brain_proj_ds,
        z_lps_mm=float(z_focus_lps_mm),
        origin_xy_mm=np.array([t3d.origin[0], t3d.origin[1]]),
        step_xy_mm=np.array([t3d.spacing[0], t3d.spacing[1]]),
        pixel_area_mm2=float(abs(t3d.spacing[0] * t3d.spacing[1])),
    )

    # 4. Score / packing
    spots: List[FocalSpot] = []
    covered = np.zeros_like(t3d.target, dtype=bool)

    def _evaluate(cx, cy, against_remaining: bool):
        """Return (target_voxels, nontarget_voxels, beam_voxels) for spot at (cx, cy).
        If against_remaining=True the target count uses (target & ~covered)."""
        bb = _beam_bbox_voxels(t3d, cx, cy, z_focus_lps_mm, rx, ry, rz)
        if bb is None:
            return 0, 0, 0, None
        inside = _beam_inside_mask(t3d, bb, cx, cy, z_focus_lps_mm, rx, ry, rz)
        i0, i1, j0, j1, k0, k1 = bb
        sub_t = t3d.target[i0:i1+1, j0:j1+1, k0:k1+1]
        sub_n = t3d.nontarget_brain[i0:i1+1, j0:j1+1, k0:k1+1]
        if against_remaining:
            sub_remaining = sub_t & ~covered[i0:i1+1, j0:j1+1, k0:k1+1]
            tn = int((inside & sub_remaining).sum())
        else:
            tn = int((inside & sub_t).sum())
        nn = int((inside & sub_n).sum())
        bn = int(inside.sum())
        return tn, nn, bn, (bb, inside)

    def _place(cx, cy, scratch):
        bb, inside = scratch
        i0, i1, j0, j1, k0, k1 = bb
        # Score against full target / nontarget for the spot's reported ROI
        sub_t = t3d.target[i0:i1+1, j0:j1+1, k0:k1+1]
        sub_n = t3d.nontarget_brain[i0:i1+1, j0:j1+1, k0:k1+1]
        target_full = int((inside & sub_t).sum())
        nontg_full  = int((inside & sub_n).sum())
        beam_full   = int(inside.sum())
        target_vol = target_full * voxel_volume
        nontg_vol  = nontg_full  * voxel_volume
        beam_vol   = beam_full   * voxel_volume
        spots.append(FocalSpot(
            center_world_xy_mm=np.array([cx, cy]),
            rx_mm=rx, ry_mm=ry,
            target_area_mm2=target_vol,        # 3D volume (mm^3) reused in the area slot
            offtarget_area_mm2=nontg_vol,
            footprint_area_mm2=beam_vol,
        ))
        covered[i0:i1+1, j0:j1+1, k0:k1+1] |= (inside & sub_t)

    # ---------------- strategy implementations ----------------

    if strategy == "centroid":
        # Seed at given centroid xy. We always include the seed (per user spec).
        if seed_xy_lps_mm is None:
            iis, jjs = np.where(flat)
            cx_seed = float(t3d.origin[0] + iis.mean() * t3d.spacing[0])
            cy_seed = float(t3d.origin[1] + jjs.mean() * t3d.spacing[1])
        else:
            cx_seed, cy_seed = float(seed_xy_lps_mm[0]), float(seed_xy_lps_mm[1])
        tn, nn, bn, scratch = _evaluate(cx_seed, cy_seed, against_remaining=False)
        if scratch is not None and bn > 0:
            _place(cx_seed, cy_seed, scratch)

        # Then greedily add more, applying threshold.
        while True:
            best = (0, None, None)
            for (cx, cy) in candidates:
                if any(_ellipse_overlap_xy(cx, cy, s.center_world_xy_mm[0],
                                           s.center_world_xy_mm[1], rx, ry)
                       for s in spots):
                    continue
                tn, nn, bn, scratch = _evaluate(cx, cy, against_remaining=True)
                if scratch is None or bn == 0:
                    continue
                # 3D ROI ratio: ratio against full beam target (not remaining)
                tn_full = (
                    t3d.target[scratch[0][0]:scratch[0][1]+1,
                               scratch[0][2]:scratch[0][3]+1,
                               scratch[0][4]:scratch[0][5]+1] & scratch[1]
                ).sum()
                ratio = tn_full / bn if bn else 0.0
                if ratio < coverage_threshold:
                    continue
                if tn > best[0]:
                    best = (tn, (cx, cy), scratch)
            if best[1] is None or best[0] < min_target_voxels:
                break
            _place(*best[1], best[2])

    else:
        # coverage and max-coverage start from scratch with greedy 3D scoring.
        threshold = coverage_threshold if strategy == "coverage" else 0.0
        while True:
            best = (0, None, None)
            for (cx, cy) in candidates:
                if any(_ellipse_overlap_xy(cx, cy, s.center_world_xy_mm[0],
                                           s.center_world_xy_mm[1], rx, ry)
                       for s in spots):
                    continue
                tn, nn, bn, scratch = _evaluate(cx, cy, against_remaining=True)
                if scratch is None or bn == 0 or tn == 0:
                    continue
                if threshold > 0:
                    bb, inside = scratch
                    tn_full = int((
                        t3d.target[bb[0]:bb[1]+1, bb[2]:bb[3]+1, bb[4]:bb[5]+1]
                        & inside).sum())
                    if tn_full / bn < threshold:
                        continue
                if tn > best[0]:
                    best = (tn, (cx, cy), scratch)
            if best[1] is None or best[0] < min_target_voxels:
                break
            _place(*best[1], best[2])

    # ---------------- summary ----------------
    target_total_vol = float(t3d.target.sum() * voxel_volume)
    covered_vol = float((covered & t3d.target).sum() * voxel_volume)
    plan = HexTilePlan(
        spots=spots,
        coverage_threshold=coverage_threshold if strategy != "max-coverage" else 0.0,
        footprint=footprint,
        plane_z_lps_mm=float(z_focus_lps_mm),
        target_area_mm2=target_total_vol,         # actually mm^3 here
        target_area_covered_mm2=covered_vol,
    )
    return plan, plane_for_viz
