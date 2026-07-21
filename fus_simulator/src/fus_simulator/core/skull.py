"""Load and embed a CT-derived mouse skull (HDF5) into the simulation grid.

    bone_mask_ROI    (uint8)    binary bone mask
    sound_speed_map  (float32)  [m/s]
    density_map      (float32)  [kg/m^3]
    alpha_coeff_map  (float32)  [dB/(MHz^y cm)]
    ct_data_ROI      (float32)  raw CT (HU) - optional, for reference
    attrs: alpha_power, voxel_size_mm, ppw, frequency_hz, ...

"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np

from .grid import GridSpec
from .params import WATER

DEFAULT_SKULL_DIR = "."   # directory scanned for mouse_skull_seg_*.h5 files

@dataclass
class SkullData:
    bone_mask: np.ndarray        # uint8 / bool
    sound_speed: np.ndarray      # [m/s]
    density: np.ndarray          # [kg/m^3]
    alpha_coeff: np.ndarray      # [dB/(MHz^y cm)]
    ct_data: Optional[np.ndarray]
    voxel_size_mm: float
    alpha_power: float
    frequency_hz: Optional[float]
    ppw: Optional[int]
    path: str

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.bone_mask.shape)

    def summary(self) -> str:
        return (f"{os.path.basename(self.path)}  |  shape {self.shape}  |  "
                f"voxel {self.voxel_size_mm:.4f} mm  |  alpha_power {self.alpha_power}  |  "
                f"c [{self.sound_speed.min():.0f},{self.sound_speed.max():.0f}] m/s  |  "
                f"rho [{self.density.min():.0f},{self.density.max():.0f}] kg/m^3  |  "
                f"alpha [{self.alpha_coeff.min():.2f},{self.alpha_coeff.max():.2f}]")


# --------------------------------------------------------------------------- #
# Discovery / loading
# --------------------------------------------------------------------------- #
def find_default_skull(directory: str = DEFAULT_SKULL_DIR,
                       ppw: Optional[int] = None) -> Optional[str]:
    """Return the newest ``mouse_skull_seg_PPW{ppw}*.h5`` in *directory*.

    If a PPW is given, prefer an exact PPW match; otherwise (or if none match)
    fall back to the newest ``mouse_skull_seg_*`` file. Returns None if nothing
    is found or the directory does not exist.
    """
    if not directory or not os.path.isdir(directory):
        return None
    if ppw is not None:
        matches = sorted(glob.glob(os.path.join(directory, f"mouse_skull_seg_PPW{ppw}_*.h5")))
        matches += sorted(glob.glob(os.path.join(directory, f"mouse_skull_seg_PPW{ppw}*.h5")))
        matches = sorted(set(matches), key=os.path.getmtime)
        if matches:
            return matches[-1]
    any_matches = sorted(glob.glob(os.path.join(directory, "mouse_skull_seg_*.h5")),
                         key=os.path.getmtime)
    return any_matches[-1] if any_matches else None


def load_skull(path: str) -> SkullData:
    """Load a skull HDF5 into a :class:`SkullData`."""
    import h5py
    with h5py.File(path, "r") as f:
        bone = f["bone_mask_ROI"][:]
        c = f["sound_speed_map"][:].astype(np.float32)
        rho = f["density_map"][:].astype(np.float32)
        alpha = f["alpha_coeff_map"][:].astype(np.float32)
        ct = f["ct_data_ROI"][:].astype(np.float32) if "ct_data_ROI" in f else None
        attrs = dict(f.attrs)
    return SkullData(
        bone_mask=bone.astype(bool),
        sound_speed=c, density=rho, alpha_coeff=alpha, ct_data=ct,
        voxel_size_mm=float(attrs.get("voxel_size_mm", np.nan)),
        alpha_power=float(attrs.get("alpha_power", WATER.alpha_power)),
        frequency_hz=float(attrs["frequency_hz"]) if "frequency_hz" in attrs else None,
        ppw=int(attrs["ppw"]) if "ppw" in attrs else None,
        path=path,
    )


# --------------------------------------------------------------------------- #
# Embedding into the simulation grid
# --------------------------------------------------------------------------- #
def _rot_cw_xy(vol: np.ndarray) -> np.ndarray:
    """90 deg clockwise rotation in the x-y plane (np.rot90 k=-1)."""
    return np.rot90(vol, k=-1, axes=(0, 1))


def _resample(vol: np.ndarray, factor: float, order: int) -> np.ndarray:
    if abs(factor - 1.0) <= 0.005:
        return vol
    from scipy.ndimage import zoom as _zoom
    # mode="nearest" extends the edge value instead of padding with 0. The
    # default (mode="constant", cval=0) bleeds ZEROS into the resampled volume
    # at the boundaries - which for the sound-speed / density maps means c=0 or
    # rho=0 voxels that make k-Wave diverge. nearest avoids that.
    return _zoom(vol.astype(np.float32), factor, order=order, mode="nearest")


def _place_centered(vol: np.ndarray, target_shape, center_idx, fill: float) -> np.ndarray:
    """Place *vol* into a *target_shape* array so vol's centre lands on center_idx."""
    out = np.full(target_shape, fill, dtype=np.float32)
    sx, sy, sz = vol.shape
    ox = int(round(center_idx[0] - sx / 2.0))
    oy = int(round(center_idx[1] - sy / 2.0))
    oz = int(round(center_idx[2] - sz / 2.0))
    # source/target overlap with clipping
    def _span(o, s, N):
        t0 = max(0, o); t1 = min(N, o + s)
        s0 = t0 - o; s1 = s0 + (t1 - t0)
        return t0, t1, s0, s1
    tx0, tx1, sx0, sx1 = _span(ox, sx, target_shape[0])
    ty0, ty1, sy0, sy1 = _span(oy, sy, target_shape[1])
    tz0, tz1, sz0, sz1 = _span(oz, sz, target_shape[2])
    if tx0 < tx1 and ty0 < ty1 and tz0 < tz1:
        out[tx0:tx1, ty0:ty1, tz0:tz1] = vol[sx0:sx1, sy0:sy1, sz0:sz1].astype(np.float32)
    return out


@dataclass
class EmbeddedSkull:
    sound_speed: np.ndarray      # (Nx,Ny,Nz) water-filled outside skull
    density: np.ndarray
    alpha_coeff: np.ndarray
    bone_mask: np.ndarray        # bool
    alpha_power: float
    center_idx: Tuple[int, int, int]
    n_bone_voxels: int


def embed_skull(skull: SkullData,
                gspec: GridSpec,
                center_xyz_m: Tuple[float, float, float],
                rotate_cw90: bool = True) -> EmbeddedSkull:
    """Embed the skull maps into the simulation grid.

    Parameters
    ----------
    skull : loaded :class:`SkullData`.
    gspec : the simulation grid.
    center_xyz_m : physical point (grid frame, metres) the skull centre maps to
        (typically the geometric focus, optionally nudged by the user).
    rotate_cw90 : apply 90 deg clockwise x-y rotation.
    """
    c = skull.sound_speed
    rho = skull.density
    alpha = skull.alpha_coeff
    bone = skull.bone_mask.astype(np.float32)
    if rotate_cw90:
        c = _rot_cw_xy(c); rho = _rot_cw_xy(rho)
        alpha = _rot_cw_xy(alpha); bone = _rot_cw_xy(bone)

    factor = skull.voxel_size_mm / (gspec.dx * 1e3)
    c = _resample(c, factor, order=1)
    rho = _resample(rho, factor, order=1)
    alpha = _resample(alpha, factor, order=1)
    bone = _resample(bone, factor, order=0)

    target = (gspec.Nx, gspec.Ny, gspec.Nz)
    cx = int(np.argmin(np.abs(gspec.x_vec - center_xyz_m[0])))
    cy = int(np.argmin(np.abs(gspec.y_vec - center_xyz_m[1])))
    cz = int(np.argmin(np.abs(gspec.z_vec - center_xyz_m[2])))
    center_idx = (cx, cy, cz)

    c_e = _place_centered(c, target, center_idx, fill=WATER.sound_speed)
    rho_e = _place_centered(rho, target, center_idx, fill=WATER.density)
    alpha_e = _place_centered(alpha, target, center_idx, fill=0.0)
    bone_e = _place_centered(bone, target, center_idx, fill=0.0) > 0.5

    # Safety net: any non-physical voxels left by resampling become water so the
    # solver never sees c=0 / rho=0 (which would diverge).
    c_e[~np.isfinite(c_e) | (c_e <= 0)] = WATER.sound_speed
    rho_e[~np.isfinite(rho_e) | (rho_e <= 0)] = WATER.density
    alpha_e[~np.isfinite(alpha_e) | (alpha_e < 0)] = 0.0

    return EmbeddedSkull(
        sound_speed=c_e, density=rho_e, alpha_coeff=alpha_e, bone_mask=bone_e,
        alpha_power=skull.alpha_power, center_idx=center_idx,
        n_bone_voxels=int(bone_e.sum()),
    )
