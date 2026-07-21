"""Z-plane and transducer-face picker.

Pipeline
--------
1. Find the *brain apex*: the (x, y, z) of the highest +z brain voxel anywhere
   in the labels volume (not just along the focal column). For a symmetric
   atlas this is on the midline near the dorsal-most cortex / cerebellum.

2. Walk +z in the CT starting from ``brain_apex_z + skin_margin_mm`` (default
   0.05 mm) at the apex (x, y) until the CT intensity exceeds the skull
   threshold -- that's the inner skull surface. Continue until it drops back
   below the threshold -- that's the outer skull surface.

3. The transducer face goes ``focal_depth`` above the focal point so the
   free-field focus lands at the target centroid.

Why apex rather than focal column: the transducer enters from the dorsal
surface, and for FUS planning we care about the skull at the *highest* point
of the head (where the rig lives). Searching at the focal xy was producing
thickness numbers that varied with target -- a lateral target sees a thinner,
more curved skull edge -- which is not how the user actually couples the
transducer. The apex-based skull is a session-level constant: same skull
geometry for every plan run on the same dataset.

Convention: LPS, +z = superior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..io.volume import Volume


@dataclass
class ZPlanePlan:
    centroid_lps_mm: np.ndarray
    z_focus_mm: float
    z_inner_skull_mm: float
    z_outer_skull_mm: float
    skull_thickness_mm: float
    z_transducer_face_mm: float
    coupling_gap_mm: float
    focal_depth_mm: float
    skull_threshold_hu: float
    z_brain_top_mm: float = float("nan")     # apex z (session-level constant)
    apex_xy_lps_mm: Optional[Tuple[float, float]] = None   # apex xy (session-level)

    @property
    def is_geometrically_valid(self) -> bool:
        return self.coupling_gap_mm >= 0 and self.skull_thickness_mm > 0


# ---- helpers ---------------------------------------------------------------


def find_brain_apex_lps(labels: Volume) -> np.ndarray:
    """Return the LPS-mm coordinate of the brain apex.

    The apex is defined as: take the topmost z-slab (largest k) where any
    voxel of the labels image is > 0, and return the centroid of the brain
    voxels in that slab. For a symmetric atlas this lands on the midline at
    the dorsal-most cortex / cerebellum, which is what we want as the
    transducer's coupling reference.
    """
    brain = labels.data > 0
    has_brain_at_z = brain.any(axis=(0, 1))
    if not has_brain_at_z.any():
        raise RuntimeError("No brain voxels found in labels volume.")
    k_apex = int(np.where(has_brain_at_z)[0].max())
    slab = brain[:, :, k_apex]
    ii, jj = np.where(slab)
    i_c = float(ii.mean())
    j_c = float(jj.mean())
    return labels.voxel_to_world(np.array([i_c, j_c, k_apex], dtype=float))


def find_brain_top_z(
    labels: Volume,
    xy_lps_mm: Tuple[float, float],
) -> float:
    """Return the LPS z of the highest label>0 voxel along the (x, y) line.

    Used internally by tools that need the per-column brain top; the main
    skull search now uses ``find_brain_apex_lps`` instead.
    """
    x, y = xy_lps_mm
    nx, ny, nz = labels.shape
    z_axis = labels.origin[2] + np.arange(nz) * labels.direction[2, 2] * labels.spacing[2]
    pts = np.column_stack([np.full(nz, x), np.full(nz, y), z_axis])
    ijks = np.round(labels.world_to_voxel(pts)).astype(int)
    inb = (
        (ijks[:, 0] >= 0) & (ijks[:, 0] < nx)
        & (ijks[:, 1] >= 0) & (ijks[:, 1] < ny)
        & (ijks[:, 2] >= 0) & (ijks[:, 2] < nz)
    )
    if not inb.any():
        raise RuntimeError(f"xy=({x:.2f}, {y:.2f}) is outside the labels volume")
    vals = np.zeros(nz, dtype=np.int64)
    vals[inb] = labels.data[ijks[inb, 0], ijks[inb, 1], ijks[inb, 2]]
    on_brain = (vals > 0)
    if not on_brain.any():
        raise RuntimeError(f"No brain voxels found along xy=({x:.2f}, {y:.2f})")
    last_idx = int(np.where(on_brain)[0].max())
    return float(z_axis[last_idx])


def find_skull_along_z(
    ct: Volume,
    xy_lps_mm: Tuple[float, float],
    z_start_mm: float,
    z_max_mm: Optional[float] = None,
    skull_threshold_hu: float = 1000.0,
    z_step_mm: float = 0.025,
) -> Tuple[float, float]:
    """Walk +z from ``z_start_mm`` until we cross into / out of skull.

    Returns (z_inner, z_outer) in LPS mm. The first +z crossing into > threshold
    is taken as the inner skull surface and the next crossing back below as
    the outer surface.
    """
    x, y = xy_lps_mm
    if z_max_mm is None:
        _, mx = ct.world_bbox
        z_max_mm = mx[2]
    if z_max_mm <= z_start_mm:
        raise RuntimeError(
            f"z_max ({z_max_mm:.2f}) is not above z_start ({z_start_mm:.2f}); "
            f"the brain top may already exceed the CT bounds."
        )

    n_steps = int(np.ceil((z_max_mm - z_start_mm) / z_step_mm)) + 1
    zs = z_start_mm + np.arange(n_steps) * z_step_mm
    pts = np.column_stack([np.full_like(zs, x), np.full_like(zs, y), zs])
    ijks = np.round(ct.world_to_voxel(pts)).astype(int)
    nx, ny, nz = ct.shape
    inb = (
        (ijks[:, 0] >= 0) & (ijks[:, 0] < nx)
        & (ijks[:, 1] >= 0) & (ijks[:, 1] < ny)
        & (ijks[:, 2] >= 0) & (ijks[:, 2] < nz)
    )
    vals = np.zeros(n_steps, dtype=float)
    if inb.any():
        vals[inb] = ct.data[ijks[inb, 0], ijks[inb, 1], ijks[inb, 2]]

    above = vals > skull_threshold_hu
    if not above.any():
        raise RuntimeError(
            f"No skull crossing found above z={z_start_mm:.2f} mm at "
            f"(x={x:.2f}, y={y:.2f}) with threshold {skull_threshold_hu}."
        )
    inner_idx = int(np.argmax(above))
    after = above[inner_idx:]
    flip = np.where(~after)[0]
    if flip.size == 0:
        outer_idx = inner_idx + int(after.sum()) - 1
    else:
        outer_idx = inner_idx + int(flip[0]) - 1
    return float(zs[inner_idx]), float(zs[outer_idx])


def plan_z_for_centroid(
    centroid_lps_mm: np.ndarray,
    ct: Volume,
    labels: Volume,
    focal_depth_mm: float,
    skull_threshold_hu: float = 1000.0,
    skin_margin_mm: float = 0.05,
    z_step_mm: float = 0.025,
) -> ZPlanePlan:
    """Build a ZPlanePlan given a target centroid and the CT + labels volumes.

    The labels volume is used to anchor the skull search above the brain top
    rather than walking up from the centroid (which can hit false-positive
    skull pixels inside the brain due to registration noise).

    Parameters
    ----------
    centroid_lps_mm : array (3,)
        Target centre in LPS mm.
    ct, labels : Volume
        Both must be in the same physical (LPS) frame.
    focal_depth_mm : float
        Distance from transducer face to free-field focus (mm).
    skull_threshold_hu : float
        Raw CT value above which a voxel counts as skull bone.
    skin_margin_mm : float
        Step above the brain top before we begin sampling CT for skull. A small
        margin lets the search skip across any meninges / scalp without
        flagging them as skull.
    z_step_mm : float
        Sampling pitch along +z while searching CT.
    """
    centroid_lps_mm = np.asarray(centroid_lps_mm, dtype=float).reshape(3)
    if focal_depth_mm <= 0:
        raise ValueError("focal_depth_mm must be positive")

    z_focus = float(centroid_lps_mm[2])

    # Skull search anchored on the brain apex (session-level constant), not
    # the focal column. This is what the transducer actually couples through.
    apex_lps = find_brain_apex_lps(labels)
    apex_xy = (float(apex_lps[0]), float(apex_lps[1]))
    z_brain_apex = float(apex_lps[2])

    z_inner, z_outer = find_skull_along_z(
        ct,
        xy_lps_mm=apex_xy,
        z_start_mm=z_brain_apex + skin_margin_mm,
        skull_threshold_hu=skull_threshold_hu,
        z_step_mm=z_step_mm,
    )
    z_face = z_focus + focal_depth_mm
    return ZPlanePlan(
        centroid_lps_mm=centroid_lps_mm,
        z_focus_mm=z_focus,
        z_inner_skull_mm=z_inner,
        z_outer_skull_mm=z_outer,
        skull_thickness_mm=z_outer - z_inner,
        z_transducer_face_mm=z_face,
        coupling_gap_mm=z_face - z_outer,
        focal_depth_mm=focal_depth_mm,
        skull_threshold_hu=skull_threshold_hu,
        z_brain_top_mm=z_brain_apex,
        apex_xy_lps_mm=apex_xy,
    )
