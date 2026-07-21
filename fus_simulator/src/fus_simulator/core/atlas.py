"""DMBA atlas region centroids, for targeting a brain structure with the focus.

Only the small tab-separated centroid table is read. The 1.1 GB label volume is
never loaded: every region's centre of mass is already tabulated.

Coordinate convention (verified against the DMBA header + table):

    world_LPS = origin + direction * voxel_index          (axis-aligned NRRD)
    centroid_LR / _PA / _IS  relate to world LPS by
        x_LPS = -centroid_LR
        y_LPS = -centroid_PA
        z_LPS =  centroid_IS

Checked on ROI 0 (Exterior, voxel 0,0,0), ROI 1 and ROI 2: exact to 1e-3 mm.

The CT and the atlas share the DMBA "M4D" world space, so a region's world
coordinate can be mapped straight into the CT-derived skull volume. That only
holds for this dataset, hence :func:`same_space_as` before any targeting.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Region:
    roi: int
    name: str
    voxels: int
    lr_mm: float
    pa_mm: float
    is_mm: float

    @property
    def world_lps_mm(self) -> np.ndarray:
        """Centroid in world LPS millimetres."""
        return np.array([-self.lr_mm, -self.pa_mm, self.is_mm], float)

    @property
    def pretty(self) -> str:
        # "HPO__Hippocampus_uncharted_left" -> "Hippocampus uncharted left  [HPO]"
        n = self.name
        if "__" in n:
            abbrev, rest = n.split("__", 1)
            return f"{rest.replace('_', ' ')}  [{abbrev}]"
        return n.replace("_", " ")


_REQUIRED = ("Structure", "voxels", "centroid_LR", "centroid_PA", "centroid_IS")


def load_regions(path: str, min_voxels: int = 1) -> List[Region]:
    """Parse ``DMBA_RCCF_labels_centroids.txt`` into a list of regions.

    Rows without a usable centroid (NaN) or below ``min_voxels`` are skipped, as
    are the ``Exterior`` background rows.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    if not rows:
        return []
    hdr = list(rows[0])
    hdr[0] = hdr[0].lstrip("# ").strip()
    hdr = [h.strip() for h in hdr]
    missing = [c for c in _REQUIRED if c not in hdr]
    if missing:
        raise ValueError(f"Not a DMBA centroid table (missing columns: {missing}).")
    ix = {c: hdr.index(c) for c in _REQUIRED}
    ix_roi = hdr.index("ROI") if "ROI" in hdr else 0

    out: List[Region] = []
    for r in rows[1:]:
        if len(r) <= max(ix.values()):
            continue
        name = r[ix["Structure"]].strip()
        if not name or name.lower() == "exterior":
            continue
        try:
            vox = int(float(r[ix["voxels"]]))
            lr = float(r[ix["centroid_LR"]])
            pa = float(r[ix["centroid_PA"]])
            is_ = float(r[ix["centroid_IS"]])
            roi = int(float(r[ix_roi]))
        except (ValueError, IndexError):
            continue
        if not all(np.isfinite([lr, pa, is_])) or vox < min_voxels:
            continue
        out.append(Region(roi=roi, name=name, voxels=vox, lr_mm=lr, pa_mm=pa, is_mm=is_))
    out.sort(key=lambda x: -x.voxels)
    return out


def search(regions: List[Region], text: str) -> List[Region]:
    """Case-insensitive substring filter on the structure name."""
    t = (text or "").strip().lower()
    if not t:
        return regions
    return [r for r in regions if t in r.name.lower()]


def same_space_as(regions: List[Region], bbox_lo, bbox_hi) -> bool:
    """True if every region centroid falls inside the given world bounding box.

    Guards against targeting with a CT that is not in the DMBA/M4D space, where
    the atlas coordinates would be meaningless.
    """
    if bbox_lo is None or bbox_hi is None or not regions:
        return False
    lo = np.asarray(bbox_lo, float); hi = np.asarray(bbox_hi, float)
    pts = np.array([r.world_lps_mm for r in regions])
    return bool(np.all(pts >= lo - 1e-6) and np.all(pts <= hi + 1e-6))


def find(regions: List[Region], name_substr: str) -> Optional[Region]:
    hits = search(regions, name_substr)
    return hits[0] if hits else None
