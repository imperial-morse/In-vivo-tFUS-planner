"""Centroid helpers.

For most planning we never need to scan the labels volume: the per-region
centroid table from DUKE includes voxel counts, so the centroid of any
subset of regions (whole brain, a parent ROI, or a custom selection) is
the voxel-weighted average of its members' centroids.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..io.centroids import Region, RegionCatalog
from ..io.volume import Volume


def whole_brain_centroid_voxel(catalog: RegionCatalog) -> np.ndarray:
    """Voxel-index centroid of every non-Exterior region, weighted by voxel count."""
    rs = [r for r in catalog if r.label != 0 and not np.isnan(r.centroid_voxel).any()]
    if not rs:
        raise ValueError("catalog has no usable brain regions")
    weights = np.array([r.voxels for r in rs], dtype=float)
    centers = np.stack([r.centroid_voxel for r in rs], axis=0)  # (N, 3)
    return (weights[:, None] * centers).sum(axis=0) / weights.sum()


def subset_centroid_voxel(regions: Iterable[Region]) -> np.ndarray:
    """Voxel-index centroid of an explicit set of regions, voxel-weighted."""
    rs = [r for r in regions if not np.isnan(r.centroid_voxel).any()]
    if not rs:
        raise ValueError("region list empty / no centroids")
    weights = np.array([r.voxels for r in rs], dtype=float)
    centers = np.stack([r.centroid_voxel for r in rs], axis=0)
    return (weights[:, None] * centers).sum(axis=0) / weights.sum()


def centroid_lps_mm(catalog_or_regions, labels_volume: Volume) -> np.ndarray:
    """Convenience: centroid of whole brain (catalog) or subset (iterable) in LPS mm."""
    if isinstance(catalog_or_regions, RegionCatalog):
        vox = whole_brain_centroid_voxel(catalog_or_regions)
    else:
        vox = subset_centroid_voxel(catalog_or_regions)
    return labels_volume.voxel_to_world(vox)
