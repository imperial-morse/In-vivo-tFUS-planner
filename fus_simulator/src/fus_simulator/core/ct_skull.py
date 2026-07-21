"""Build the skull medium straight from a high-resolution CT (Duke dataset).

Two-step workflow
-----------------
1. ``prepare_ct_bonefraction``. Read the CT NRRD, threshold bone, crop
   to the skull, area-downsample to a fine bone *fraction* (default 0.05 mm,
   finer than any practical sim grid) and save a compact HDF5.
2. :class:`CTSkullSource` : loads that fraction and, per simulation, downsamples
   it to the current ``dx`` and returns an :class:`~fus_simulator.core.skull.EmbeddedSkull`
   (binary bone/water, homogeneous skull properties) that drops straight into the
   existing simulate / thermal / protocol code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .grid import GridSpec
from .params import WATER, skull_props
from .skull import EmbeddedSkull, _place_centered


Progress = Optional[Callable[[str, float], None]]


def _to_sim_frame(vol: np.ndarray) -> np.ndarray:
    """Reorient a CT volume (LR, PA, IS) into the simulation frame so the
    transducer faces the top of the skull.
    """
    #   old axes (LR, PA, IS) -> transpose (2,0,1) -> (IS, LR, PA) = (x, y, z)
    #   then flip x so the vault (max IS) is at x = 0 (transducer side).
    return np.ascontiguousarray(np.transpose(vol, (2, 0, 1))[::-1, :, :])


def _say(progress: Progress, msg: str, frac: float) -> None:
    if progress is not None:
        try:
            progress(msg, float(frac))
        except Exception:  
            pass


# --------------------------------------------------------------------------- #
# NRRD (.nhdr + .raw) reader - minimal, for the detached uint16 Duke volumes
# --------------------------------------------------------------------------- #
_NRRD_DTYPE = {
    "uchar": "u1", "uint8": "u1", "int8": "i1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class CTVolume:
    data: np.ndarray                 # (nx, ny, nz)
    spacing_mm: Tuple[float, float, float]        # magnitudes (always positive)
    origin_mm: Tuple[float, float, float]         # world LPS of voxel (0,0,0)
    direction_mm: Tuple[float, float, float]      # SIGNED step per voxel per axis
    path: str

    def world_of_voxel(self, ijk) -> np.ndarray:
        """world_LPS = origin + direction * voxel_index (axis-aligned volumes)."""
        return np.asarray(self.origin_mm) + np.asarray(self.direction_mm) * np.asarray(ijk, float)

    def world_bbox(self):
        lo = self.world_of_voxel((0, 0, 0))
        hi = self.world_of_voxel(np.asarray(self.data.shape) - 1)
        return np.minimum(lo, hi), np.maximum(lo, hi)


def read_nrrd(nhdr_path: str, mmap: bool = True) -> CTVolume:
    """Read a detached NRRD (header ``.nhdr`` + ``.raw``) as (nx, ny, nz).

    Only the subset needed for the Duke CT (raw encoding, little-endian ints) is
    supported. Falls back to nothing fancy; raises on unexpected encodings.
    """
    hdr = {}
    with open(nhdr_path, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                hdr[key.strip().lower()] = val.strip()

    if hdr.get("encoding", "raw").lower() not in ("raw",):
        raise ValueError(f"Unsupported NRRD encoding: {hdr.get('encoding')!r} "
                         "(only 'raw' detached volumes are handled).")

    sizes = [int(s) for s in hdr["sizes"].split()]
    if len(sizes) != 3:
        raise ValueError(f"Expected a 3-D volume, got sizes={sizes}.")
    nx, ny, nz = sizes

    dtype_key = hdr.get("type", "uint16").strip().lower()
    base = _NRRD_DTYPE.get(dtype_key)
    if base is None:
        raise ValueError(f"Unsupported NRRD type: {dtype_key!r}.")
    endian = "<" if hdr.get("endian", "little").lower() == "little" else ">"
    dtype = np.dtype(endian + base)

    # spacing from the diagonal of 'space directions'
    def _floats(s: str):
        s = s.replace("(", " ").replace(")", " ").replace(",", " ")
        out = []
        for tok in s.split():
            try:
                out.append(float(tok))
            except ValueError:
                pass  # skip 'none' and other non-numeric tokens
        return out

    spacing = (1.0, 1.0, 1.0)
    direction = (1.0, 1.0, 1.0)
    if "space directions" in hdr:
        nums = _floats(hdr["space directions"])
        if len(nums) >= 9:
            m = np.array(nums[:9]).reshape(3, 3)
            spacing = tuple(float(np.linalg.norm(m[i])) for i in range(3))
            # axis-aligned volumes: the signed step is the diagonal element.
            # The sign matters - the Duke CT/atlas have NEGATIVE x and y steps
            # (LPS), so world = origin + direction*index, NOT origin + spacing*index.
            direction = tuple(float(m[i][i]) for i in range(3))
    origin = (0.0, 0.0, 0.0)
    if "space origin" in hdr:
        vals = _floats(hdr["space origin"])
        if len(vals) >= 3:
            origin = tuple(vals[:3])

    data_file = hdr.get("data file") or hdr.get("datafile")
    raw_path = os.path.join(os.path.dirname(nhdr_path), data_file) if data_file \
        else os.path.splitext(nhdr_path)[0] + ".raw"

    # NRRD raw stores the first axis fastest -> reshape (nz, ny, nx) then move to
    # (nx, ny, nz) so the volume axes line up with the sim grid convention.
    if mmap:
        flat = np.memmap(raw_path, dtype=dtype, mode="r", shape=(nz, ny, nx))
    else:
        flat = np.fromfile(raw_path, dtype=dtype).reshape(nz, ny, nx)
    vol = np.transpose(flat, (2, 1, 0))     # (nx, ny, nz), a view
    return CTVolume(data=vol, spacing_mm=spacing, origin_mm=origin,
                    direction_mm=direction, path=nhdr_path)


# --------------------------------------------------------------------------- #
# Area-preserving downsample (fine -> coarse), keeps thin structures alive
# --------------------------------------------------------------------------- #
def _area_downsample(vol: np.ndarray, in_dx_mm: float, out_dx_mm: float) -> np.ndarray:
    """Downsample a float volume from ``in_dx`` to ``out_dx`` by local averaging.

    Uses a box (uniform) pre-filter the size of one output voxel followed by
    linear resampling. Averaging (not point sampling) means a thin bone sheet
    that is sub-voxel on the coarse grid still contributes a non-zero fraction
    instead of being missed between sample points.
    """
    from scipy.ndimage import uniform_filter, zoom
    vol = np.asarray(vol, dtype=np.float32)
    factor = out_dx_mm / in_dx_mm            # >1 for a downsample
    if factor <= 1.02:                       # target as fine or finer: just resample
        z = in_dx_mm / out_dx_mm
        return zoom(vol, z, order=1, mode="nearest") if abs(z - 1) > 0.02 else vol
    win = max(1, int(round(factor)))
    smoothed = uniform_filter(vol, size=win, mode="nearest")
    return zoom(smoothed, 1.0 / factor, order=1, mode="nearest").astype(np.float32)


# --------------------------------------------------------------------------- #
# Step 1: prepare a compact bone-fraction file from the CT
# --------------------------------------------------------------------------- #
def prepare_ct_bonefraction(nhdr_path: str,
                            out_h5: str,
                            bone_threshold: float = 6000.0,
                            fine_dx_mm: float = 0.05,
                            margin_mm: float = 1.5,
                            progress: Progress = None) -> str:
    """Threshold bone in the CT, crop to the skull, area-downsample to a fine
    bone *fraction*, and save an HDF5.
    """
    import h5py
    _say(progress, "Reading CT header...", 0.02)
    ct = read_nrrd(nhdr_path, mmap=True)
    sx, sy, sz = ct.spacing_mm
    _say(progress, "Thresholding bone...", 0.15)

    # Materialise the bone mask (bool). For the Duke volumes this is <1 GB.
    bone = np.asarray(ct.data) > float(bone_threshold)
    if not bone.any():
        raise ValueError("No voxels exceed the bone threshold; lower it.")

    # Bounding box of bone + margin, then crop.
    _say(progress, "Cropping to skull...", 0.35)
    ax = np.where(bone.any(axis=(1, 2)))[0]
    ay = np.where(bone.any(axis=(0, 2)))[0]
    az = np.where(bone.any(axis=(0, 1)))[0]
    mx = int(round(margin_mm / sx)); my = int(round(margin_mm / sy)); mz = int(round(margin_mm / sz))
    x0, x1 = max(0, ax[0] - mx), min(bone.shape[0], ax[-1] + 1 + mx)
    y0, y1 = max(0, ay[0] - my), min(bone.shape[1], ay[-1] + 1 + my)
    z0, z1 = max(0, az[0] - mz), min(bone.shape[2], az[-1] + 1 + mz)
    bone_crop = np.ascontiguousarray(bone[x0:x1, y0:y1, z0:z1], dtype=np.float32)
    del bone

    _say(progress, "Downsampling to fine fraction...", 0.6)
    # spacing is isotropic for the Duke CT; use sx as the reference.
    frac = _area_downsample(bone_crop, sx, fine_dx_mm)
    frac = np.clip(frac, 0.0, 1.0)

    _say(progress, "Writing HDF5...", 0.85)
    # World origin of the crop corner. NOTE the SIGNED direction, not the
    # spacing magnitude: the Duke CT steps NEGATIVE in x and y, so using
    # +spacing here would mirror the volume in world space and send any
    # atlas-derived target to the wrong side of the head.
    origin = ct.world_of_voxel((x0, y0, z0))
    ct_lo, ct_hi = ct.world_bbox()
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("bone_fraction", data=(frac * 255).astype(np.uint8),
                         compression="gzip")
        f.attrs["fine_dx_mm"] = float(fine_dx_mm)
        f.attrs["bone_threshold"] = float(bone_threshold)
        f.attrs["ct_spacing_mm"] = list(ct.spacing_mm)
        f.attrs["ct_direction_mm"] = list(ct.direction_mm)
        f.attrs["origin_mm"] = [float(v) for v in origin]
        f.attrs["ct_bbox_lo"] = [float(v) for v in ct_lo]
        f.attrs["ct_bbox_hi"] = [float(v) for v in ct_hi]
        f.attrs["source"] = os.path.basename(nhdr_path)
        f.attrs["native_bone_voxels"] = int(bone_crop.sum())
        # v2 = stores the signed world affine (needed for atlas region targeting).
        # v1 files lack it and must be re-prepared.
        f.attrs["format_version"] = 2
    _say(progress, "Prepared.", 1.0)
    return out_h5


# --------------------------------------------------------------------------- #
# Step 2: the source object used by the GUI / simulations
# --------------------------------------------------------------------------- #
class CTSkullSource:
    """A prepared bone-fraction volume that can be embedded on any sim grid."""

    def __init__(self, fraction: np.ndarray, fine_dx_mm: float,
                 label: str = "CT skull", native_bone_voxels: int = 0,
                 origin_mm=None, ct_direction_mm=None,
                 ct_bbox_lo=None, ct_bbox_hi=None):
        self.fraction = np.asarray(fraction, dtype=np.float32)   # (i,j,k)=(LR,PA,IS), 0..1
        self.fine_dx_mm = float(fine_dx_mm)
        self.label = label
        self.native_bone_voxels = int(native_bone_voxels)
        # world geometry of the crop (None => atlas targeting unavailable)
        self.origin_mm = None if origin_mm is None else np.asarray(origin_mm, float)
        self.ct_direction_mm = None if ct_direction_mm is None else np.asarray(ct_direction_mm, float)
        self.ct_bbox_lo = None if ct_bbox_lo is None else np.asarray(ct_bbox_lo, float)
        self.ct_bbox_hi = None if ct_bbox_hi is None else np.asarray(ct_bbox_hi, float)

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, h5_path: str) -> "CTSkullSource":
        import h5py
        with h5py.File(h5_path, "r") as f:
            frac = f["bone_fraction"][:].astype(np.float32) / 255.0
            fine = float(f.attrs.get("fine_dx_mm", 0.05))
            nbone = int(f.attrs.get("native_bone_voxels", 0))
            src = str(f.attrs.get("source", os.path.basename(h5_path)))
            a = dict(f.attrs)
        return cls(frac, fine, label=src, native_bone_voxels=nbone,
                   origin_mm=a.get("origin_mm"),
                   ct_direction_mm=a.get("ct_direction_mm"),
                   ct_bbox_lo=a.get("ct_bbox_lo"), ct_bbox_hi=a.get("ct_bbox_hi"))

    # ---- world <-> sim-frame geometry (for atlas region targeting) --------- #
    @property
    def has_world_geometry(self) -> bool:
        return self.origin_mm is not None and self.ct_direction_mm is not None

    @property
    def sim_shape(self) -> Tuple[int, int, int]:
        """Shape after :func:`_to_sim_frame`: (x=IS', y=LR, z=PA)."""
        ni, nj, nk = self.fraction.shape
        return (nk, ni, nj)

    def world_to_sim_index(self, world_mm) -> np.ndarray:
        """World LPS (mm) -> continuous index (a, b, c) in the sim-frame volume."""
        if not self.has_world_geometry:
            raise RuntimeError("This prepared skull has no world geometry; re-Prepare from the CT.")
        d_fine = np.sign(self.ct_direction_mm) * self.fine_dx_mm
        i, j, k = (np.asarray(world_mm, float) - self.origin_mm) / d_fine
        nk = self.fraction.shape[2]
        return np.array([(nk - 1) - k, i, j], float)     # (a, b, c)

    def contains_world(self, world_mm) -> bool:
        if self.ct_bbox_lo is None:
            return True
        w = np.asarray(world_mm, float)
        return bool(np.all(w >= self.ct_bbox_lo - 1e-6) and np.all(w <= self.ct_bbox_hi + 1e-6))

    def target_offsets_mm(self, world_mm, fill_threshold: float = 0.30) -> Dict:
        """Skull AX/LR/AP offsets (mm) that put ``world_mm`` exactly on the focus.

        ``embed`` centres the volume on ``focus + offsets``, so a voxel at
        sim index ``p`` lands at ``focus + offsets + (p - centre)*fine_dx``.
        Setting that equal to the focus gives ``offsets = (centre - p)*fine_dx``.
        """
        a, b, c = self.world_to_sim_index(world_mm)
        na, nb, nc = self.sim_shape
        inside = (0 <= a < na) and (0 <= b < nb) and (0 <= c < nc)
        in_bone = None
        if inside:
            frac_sim = _to_sim_frame(self.fraction)
            in_bone = bool(frac_sim[int(round(a)), int(round(b)), int(round(c))] >= fill_threshold)
        return {
            "ok": inside,
            "inside_ct": self.contains_world(world_mm),
            "in_bone": in_bone,
            "ax_mm": (na / 2.0 - a) * self.fine_dx_mm,
            "lr_mm": (nb / 2.0 - b) * self.fine_dx_mm,
            "ap_mm": (nc / 2.0 - c) * self.fine_dx_mm,
            "sim_index": (float(a), float(b), float(c)),
        }

    @classmethod
    def from_ct(cls, nhdr_path: str, out_h5: str, **kw) -> "CTSkullSource":
        prepare_ct_bonefraction(nhdr_path, out_h5, **kw)
        return cls.load(out_h5)

    # ---- info -------------------------------------------------------------- #
    def summary(self) -> str:
        return (f"{self.label}  |  fraction {self.fraction.shape} @ "
                f"{self.fine_dx_mm:.3f} mm  |  native bone {self.native_bone_voxels:,} vox")

    # ---- the money method -------------------------------------------------- #
    def embed(self, gspec: GridSpec, center_xyz_m: Tuple[float, float, float],
              f0_hz: float, fill_threshold: float = 0.35) -> EmbeddedSkull:
        """Downsample the fine fraction to ``gspec.dx`` and place it on the grid.

        The skull is reoriented into the simulation frame so the transducer
        faces the top of the skull (see :func:`_to_sim_frame`).

        Parameters
        ----------
        fill_threshold : a coarse voxel becomes bone if its bone fraction is at
            least this. Low values keep a sub-voxel skull continuous (no holes)
            at the cost of slight over-thickening; 0.5 is geometrically faithful
            but can perforate a very thin vault on a coarse grid.
        """
        props = skull_props(f0_hz)
        dx_mm = gspec.dx * 1e3

        frac = _to_sim_frame(self.fraction)
        frac_grid = _area_downsample(frac, self.fine_dx_mm, dx_mm)
        frac_grid = np.clip(frac_grid, 0.0, 1.0)
        bone = frac_grid >= float(fill_threshold)

        shape = bone.shape
        c = np.full(shape, WATER.sound_speed, dtype=np.float32)
        rho = np.full(shape, WATER.density, dtype=np.float32)
        # non-bone voxels are coupling water, which is weakly absorbing (Kinsler)
        alpha = np.full(shape, WATER.alpha_coeff, dtype=np.float32)
        c[bone] = props.sound_speed
        rho[bone] = props.density
        alpha[bone] = props.alpha_coeff

        target = (gspec.Nx, gspec.Ny, gspec.Nz)
        cx = int(np.argmin(np.abs(gspec.x_vec - center_xyz_m[0])))
        cy = int(np.argmin(np.abs(gspec.y_vec - center_xyz_m[1])))
        cz = int(np.argmin(np.abs(gspec.z_vec - center_xyz_m[2])))
        center_idx = (cx, cy, cz)

        c_e = _place_centered(c, target, center_idx, fill=WATER.sound_speed)
        rho_e = _place_centered(rho, target, center_idx, fill=WATER.density)
        alpha_e = _place_centered(alpha, target, center_idx, fill=WATER.alpha_coeff)
        bone_e = _place_centered(bone.astype(np.float32), target, center_idx, fill=0.0) > 0.5

        # Safety net: never hand the solver a c=0 / rho=0 voxel.
        c_e[~np.isfinite(c_e) | (c_e <= 0)] = WATER.sound_speed
        rho_e[~np.isfinite(rho_e) | (rho_e <= 0)] = WATER.density
        alpha_e[~np.isfinite(alpha_e) | (alpha_e < 0)] = 0.0

        return EmbeddedSkull(
            sound_speed=c_e, density=rho_e, alpha_coeff=alpha_e, bone_mask=bone_e,
            alpha_power=props.alpha_power, center_idx=center_idx,
            n_bone_voxels=int(bone_e.sum()),
        )
