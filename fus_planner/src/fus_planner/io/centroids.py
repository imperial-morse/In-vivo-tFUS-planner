"""Parser for the DMBA RCCF centroids / lookup table.

The centroids file is a tab-separated table with one row per ROI (label value
in the segmentation volume). The columns we actually use:

- `# ROI`            integer label value (0 = Exterior)
- `Structure`        machine-safe unique name
- `ARA_name`         human-readable Allen Reference Atlas name
- `level_1` ... `level_10`   ontology ancestors for tree display
- `voxels`           voxel count in the labels image
- `centroid_LR/PA/IS`  centroid in anatomical mm
- `centroid_X/Y/Z`     centroid in voxel index space of the labels volume
- `color_hex_triplet`  display colour
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


_LEVEL_COLS = [f"level_{i}" for i in range(1, 11)]


@dataclass
class Region:
    """A single ROI from the RCCF centroids table."""

    label: int
    name: str            # machine-safe `Structure`
    ara_name: str        # human-readable `ARA_name`
    voxels: int
    centroid_anat_mm: np.ndarray  # (LR, PA, IS) in mm
    centroid_voxel: np.ndarray    # (X, Y, Z) voxel indices in labels volume
    levels: List[str] = field(default_factory=list)
    color_hex: str = "#888888"

    @property
    def display_name(self) -> str:
        return self.ara_name or self.name

    def centroid_world_lps(self, labels_volume) -> np.ndarray:
        """Centroid in the labels volume's world-frame mm (LPS)."""
        return labels_volume.voxel_to_world(self.centroid_voxel)


@dataclass
class RegionCatalog:
    """All ROIs in the DMBA RCCF labels image, indexed by label value."""

    regions: Dict[int, Region]

    def __getitem__(self, label: int) -> Region:
        return self.regions[label]

    def __iter__(self) -> Iterable[Region]:
        return iter(self.regions.values())

    def __len__(self) -> int:
        return len(self.regions)

    def labels(self) -> List[int]:
        return sorted(self.regions.keys())

    def find_by_name(self, query: str) -> List[Region]:
        """Case-insensitive substring search over Structure / ARA_name."""
        q = query.strip().lower()
        if not q:
            return []
        return [
            r for r in self.regions.values()
            if q in r.name.lower() or q in r.ara_name.lower()
        ]

    def by_level(self, level: int, value: str) -> List[Region]:
        """All regions whose ontology level `level` equals `value`."""
        if not 1 <= level <= 10:
            raise ValueError("level must be 1..10")
        idx = level - 1
        return [r for r in self.regions.values()
                if len(r.levels) > idx and r.levels[idx] == value]


def load_region_catalog(path: str | Path) -> RegionCatalog:
    """Read a DMBA_RCCF_labels_centroids.txt file."""
    path = Path(path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    regions: Dict[int, Region] = {}
    for _, row in df.iterrows():
        roi = row.get("# ROI", "").strip()
        if not roi or not roi.lstrip("-").isdigit():
            continue
        label = int(roi)
        levels = [row.get(c, "").strip() for c in _LEVEL_COLS if c in df.columns]
        levels = [lv for lv in levels if lv]
        try:
            centroid_anat = np.array([
                float(row["centroid_LR"]),
                float(row["centroid_PA"]),
                float(row["centroid_IS"]),
            ])
            centroid_vox = np.array([
                float(row["centroid_X"]),
                float(row["centroid_Y"]),
                float(row["centroid_Z"]),
            ])
        except (KeyError, ValueError):
            # Some rows (e.g. Exterior) have placeholder centroids; skip silently.
            centroid_anat = np.array([np.nan] * 3)
            centroid_vox = np.array([np.nan] * 3)
        try:
            voxels = int(float(row.get("voxels", "0") or "0"))
        except ValueError:
            voxels = 0
        color = row.get("color_hex_triplet", "").strip() or "0x888888"
        # color_hex_triplet looks like "0xB0F0FF"; convert to "#B0F0FF"
        if color.lower().startswith("0x"):
            color = "#" + color[2:].upper()
        regions[label] = Region(
            label=label,
            name=row.get("Structure", "").strip(),
            ara_name=row.get("ARA_name", "").strip(),
            voxels=voxels,
            centroid_anat_mm=centroid_anat,
            centroid_voxel=centroid_vox,
            levels=levels,
            color_hex=color,
        )
    return RegionCatalog(regions=regions)


# ---- anatomical <-> LPS sign convention -----------------------------------

def lps_to_anat(xyz_lps_mm: np.ndarray) -> np.ndarray:
    """LPS world mm -> anatomical (LR, PA, IS) mm.

    LR positive = right side; PA positive = anterior; IS positive = superior.
    LPS has +x = left, +y = posterior, +z = superior, so:
        LR = -x, PA = -y, IS = +z
    """
    xyz = np.asarray(xyz_lps_mm, dtype=float)
    out = np.empty_like(xyz)
    out[..., 0] = -xyz[..., 0]
    out[..., 1] = -xyz[..., 1]
    out[..., 2] = xyz[..., 2]
    return out


def anat_to_lps(lr_pa_is_mm: np.ndarray) -> np.ndarray:
    """Anatomical (LR, PA, IS) mm -> LPS world mm (inverse of `lps_to_anat`)."""
    return lps_to_anat(lr_pa_is_mm)  # involution
