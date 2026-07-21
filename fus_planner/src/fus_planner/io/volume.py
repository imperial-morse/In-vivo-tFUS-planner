"""3D volume loader for NRRD/.nhdr files used in the DUKE DMBA dataset.

A `Volume` keeps the raw voxel array together with the affine that maps
voxel indices to physical millimetres in the LPS frame the dataset is stored in.

We deliberately do not depend on SimpleITK at import time because the GUI
should still launch on machines where it is missing (we can fall back to
pynrrd which is pure Python).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class Volume:
    """A 3D image with a voxel-to-world (LPS, mm) affine.

    Attributes
    ----------
    data : np.ndarray
        Voxel array in (i, j, k) order matching the NRRD `sizes` field.
    spacing : np.ndarray, shape (3,)
        Per-axis voxel size in mm (always positive).
    direction : np.ndarray, shape (3, 3)
        Column k is the world-frame direction of voxel axis k (unit length, may be negative).
    origin : np.ndarray, shape (3,)
        World-frame coordinates (mm) of voxel (0, 0, 0).
    space : str
        NRRD `space` field, e.g. "left-posterior-superior".
    source_path : Path
        Path the volume was loaded from.
    """

    data: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray
    origin: np.ndarray
    space: str
    source_path: Path

    # ---- coordinate helpers ------------------------------------------------

    def voxel_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """Map voxel indices (..., 3) to world-frame mm (..., 3)."""
        ijk = np.asarray(ijk, dtype=float)
        return ijk @ (self.direction * self.spacing).T + self.origin

    def world_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        """Map world-frame mm (..., 3) to (fractional) voxel indices (..., 3)."""
        xyz = np.asarray(xyz, dtype=float)
        m = (self.direction * self.spacing)
        return (xyz - self.origin) @ np.linalg.inv(m).T

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape)

    @property
    def world_bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """World-frame (mm) AABB of the volume corners."""
        ni, nj, nk = self.shape
        corners_ijk = np.array(
            [(i, j, k) for i in (0, ni - 1) for j in (0, nj - 1) for k in (0, nk - 1)],
            dtype=float,
        )
        corners_world = self.voxel_to_world(corners_ijk)
        return corners_world.min(axis=0), corners_world.max(axis=0)


# ---- loaders --------------------------------------------------------------


def load_volume(path: str | Path) -> Volume:
    """Load a NRRD/.nhdr file as a `Volume`.

    Tries SimpleITK first (handles every NRRD variant cleanly); falls back
    to pynrrd if SimpleITK is unavailable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        return _load_with_sitk(path)
    except ImportError:
        return _load_with_pynrrd(path)


def _load_with_sitk(path: Path) -> Volume:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    # SimpleITK arrays are (k, j, i); transpose to (i, j, k) to match NRRD `sizes`.
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    spacing = np.array(img.GetSpacing(), dtype=float)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    origin = np.array(img.GetOrigin(), dtype=float)
   
    return Volume(
        data=arr,
        spacing=spacing,
        direction=direction,
        origin=origin,
        space="left-posterior-superior",
        source_path=path,
    )


def _load_with_pynrrd(path: Path) -> Volume:
    import nrrd  # type: ignore

    data, header = nrrd.read(str(path))
    space_dirs = np.asarray(header["space directions"], dtype=float)
    spacing = np.linalg.norm(space_dirs, axis=1)
    # Avoid divide-by-zero on degenerate axes.
    spacing_safe = np.where(spacing > 0, spacing, 1.0)
    direction = (space_dirs / spacing_safe[:, None]).T  # columns are voxel-axis dirs
    origin = np.asarray(header["space origin"], dtype=float)
    space = header.get("space", "left-posterior-superior")
    return Volume(
        data=data,
        spacing=spacing,
        direction=direction,
        origin=origin,
        space=space,
        source_path=path,
    )


# ---- equality of grids -----------------------------------------------------


def same_grid(a: Volume, b: Volume, tol: float = 1e-6) -> bool:
    """True if two volumes share voxel grid (shape, spacing, direction, origin)."""
    return (
        a.data.shape == b.data.shape
        and np.allclose(a.spacing, b.spacing, atol=tol)
        and np.allclose(a.direction, b.direction, atol=tol)
        and np.allclose(a.origin, b.origin, atol=tol)
    )
