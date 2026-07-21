"""Extract a 2D focal-plane mask from a labels volume.

We slice the labels image at the z plane closest to ``z_lps_mm`` and turn it
into a binary mask (or two masks: one for the target, one for everything-else
non-zero, for off-target accounting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from ..io.volume import Volume


@dataclass
class FocalPlane:
    """A 2D slice of the labels volume at a chosen z (LPS mm)."""

    target_mask: np.ndarray          # bool (Ni, Nj) - True = target tissue
    brain_mask: np.ndarray           # bool (Ni, Nj) - True = any brain tissue (label > 0)
    z_lps_mm: float                  # actual z of the slice (snapped to voxel)
    origin_xy_mm: np.ndarray         # (2,) world LPS xy of pixel (0, 0)
    step_xy_mm: np.ndarray           # (2,) world dx, dy per +1 in i, j (signed)
    pixel_area_mm2: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.target_mask.shape

    @property
    def target_area_mm2(self) -> float:
        return float(self.target_mask.sum() * self.pixel_area_mm2)

    @property
    def brain_area_mm2(self) -> float:
        return float(self.brain_mask.sum() * self.pixel_area_mm2)

    def world_xy(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Map pixel indices to world (x, y) mm. Accepts arrays."""
        x = self.origin_xy_mm[0] + i * self.step_xy_mm[0]
        y = self.origin_xy_mm[1] + j * self.step_xy_mm[1]
        return np.stack([x, y], axis=-1)

    def world_to_pixel(self, xy_mm: np.ndarray) -> np.ndarray:
        """Inverse of `world_xy`. Returns float pixel indices."""
        xy = np.asarray(xy_mm)
        i = (xy[..., 0] - self.origin_xy_mm[0]) / self.step_xy_mm[0]
        j = (xy[..., 1] - self.origin_xy_mm[1]) / self.step_xy_mm[1]
        return np.stack([i, j], axis=-1)


def extract_focal_plane(
    labels: Volume,
    z_lps_mm: float,
    target_labels: Optional[Iterable[int]] = None,
) -> FocalPlane:
    """Slice the labels volume at the z closest to ``z_lps_mm``.

    target_labels=None -> whole brain (all non-zero labels).
    """
    # Find the voxel z index that snaps to z_lps_mm. labels grid has identity
    # x and y direction signs (with optional flips), and direction[2,2] for z.
    # Compute via the affine for safety.
    cx = (labels.world_bbox[0][0] + labels.world_bbox[1][0]) / 2.0
    cy = (labels.world_bbox[0][1] + labels.world_bbox[1][1]) / 2.0
    ijk = labels.world_to_voxel(np.array([cx, cy, z_lps_mm]))
    iz = int(round(ijk[2]))
    nz = labels.shape[2]
    iz = max(0, min(nz - 1, iz))

    slab = labels.data[..., iz]                               # (Ni, Nj)
    brain_mask = slab > 0
    if target_labels is None:
        target_mask = brain_mask
    else:
        wanted = set(int(v) for v in target_labels)
        target_mask = np.isin(slab, list(wanted))

    # In-plane geometry from the affine.
    # World xy of pixel (i, j, iz) = origin + dir @ (spacing * (i, j, iz))
    # With axis-aligned NRRD this reduces to:
    step_x = labels.direction[0, 0] * labels.spacing[0]
    step_y = labels.direction[1, 1] * labels.spacing[1]
    actual_z = labels.voxel_to_world(np.array([0, 0, iz]))[2]
    origin_x = labels.origin[0] + 0 * step_x                 # i = 0
    origin_y = labels.origin[1] + 0 * step_y                 # j = 0

    return FocalPlane(
        target_mask=target_mask,
        brain_mask=brain_mask,
        z_lps_mm=float(actual_z),
        origin_xy_mm=np.array([origin_x, origin_y]),
        step_xy_mm=np.array([step_x, step_y]),
        pixel_area_mm2=float(abs(step_x * step_y)),
    )
