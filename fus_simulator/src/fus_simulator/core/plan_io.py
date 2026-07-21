"""Import a fus_planner plan CSV and turn it into a multi-spot thermal protocol.

The planner exports one row per target spot with (among others):

    world_x_lps_mm, world_y_lps_mm, world_z_lps_mm   focus in LPS world mm
    rx_mm, ry_mm                                     lateral footprint radii
    z_focus_mm, z_transducer_face_mm                 beam axial geometry

For an OVERLAP analysis we work directly in the planner's world-mm frame: each
spot deposits heat in an ellipsoidal focal region (lateral radii rx/ry, an axial
depth-of-field), and the cumulative thermal run shows where neighbouring spots
stack. Absolute temperature needs a drive/intensity assumption (the CSV carries
geometry, not pressure), so the heat magnitude is a user parameter; the overlap
pattern is fully determined by the geometry.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from .thermal import ThermalParams
from .protocol import run_protocol, ProtocolResult


@dataclass
class PlanSpot:
    spot_id: str
    focus_mm: Tuple[float, float, float]   # (x, y, z) world LPS mm
    rx_mm: float
    ry_mm: float
    z_transducer_mm: float = float("nan")
    extra: Dict = field(default_factory=dict)


def parse_plan_csv(path: str) -> List[PlanSpot]:
    """Parse a fus_planner plan CSV into a list of :class:`PlanSpot`."""
    spots: List[PlanSpot] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        def col(*names):
            for n in names:
                if n.lower() in cols:
                    return cols[n.lower()]
            return None

        cx = col("world_x_lps_mm", "focus_x_mm", "x_mm", "world_x")
        cy = col("world_y_lps_mm", "focus_y_mm", "y_mm", "world_y")
        cz = col("world_z_lps_mm", "focus_z_mm", "z_mm", "world_z")
        crx = col("rx_mm", "rx"); cry = col("ry_mm", "ry")
        cztx = col("z_transducer_face_mm", "z_transducer_mm")
        cid = col("spot_id", "id")
        if not (cx and cy and cz):
            raise ValueError(
                "CSV is missing focus coordinates (world_x/y/z_lps_mm). "
                f"Found columns: {reader.fieldnames}")
        for i, row in enumerate(reader):
            try:
                fx, fy, fz = float(row[cx]), float(row[cy]), float(row[cz])
            except (ValueError, KeyError):
                continue
            rx = float(row[crx]) if crx and row.get(crx) else 0.75
            ry = float(row[cry]) if cry and row.get(cry) else rx
            ztx = float(row[cztx]) if cztx and row.get(cztx) else float("nan")
            spots.append(PlanSpot(
                spot_id=str(row[cid]) if cid else str(i + 1),
                focus_mm=(fx, fy, fz), rx_mm=rx, ry_mm=ry,
                z_transducer_mm=ztx, extra=dict(row)))
    if not spots:
        raise ValueError("No valid spot rows found in the plan CSV.")
    return spots


@dataclass
class PlanProtocolParams:
    # ---- geometry / grid ----
    dx_mm: float = 0.3                 # shared grid spacing
    margin_mm: float = 4.0             # padding around the spot cloud
    axial_fwhm_mm: float = 6.0         # focal depth-of-field (beam axis = z)

    # ---- acoustic magnitude (physical) ----
    focal_pressure_pa: float = 0.35e6  # achieved peak focal pressure (per spot)
    tissue_alpha_db: float = 0.5       # tissue absorption [dB/(MHz^y cm)]
    alpha_power: float = 2.0
    f0_hz: float = 500e3
    duty_cycle: float = 0.2            # = PRF * pulse_duration
    transmission: float = 1.0          # optional through-skull factor (<=1)

    # ---- timing ----
    on_time_s: float = 6.0             # sonication time per spot
    cooldown_s: float = 3.0            # move/cool time BETWEEN spots


def build_and_run_plan(spots: List[PlanSpot], pp: PlanProtocolParams,
                       tp: ThermalParams, progress=None) -> ProtocolResult:
    """Build a shared world-frame grid + per-spot heat sources from a plan and
    run the cumulative thermal protocol with physical heating.

    Heat per spot:  Q = 2 alpha I,  I = p_rms^2/(rho c),  p_peak = achieved
    focal pressure (from calibration) x transmission, spatial shape from the
    plan footprint (rx, ry) and the axial depth-of-field. Q is time-averaged by
    the duty cycle. Sequential timing: each spot's on-time, then a cool-down gap.
    """
    from .thermal import CP_WATER, K_WATER, db2neper

    dx = pp.dx_mm * 1e-3
    foci = np.array([s.focus_mm for s in spots], dtype=float)  # mm
    lo = foci.min(0) - pp.margin_mm
    hi = foci.max(0) + pp.margin_mm
    lo[2] -= pp.axial_fwhm_mm
    hi[2] += pp.axial_fwhm_mm

    nx = max(8, int(np.ceil((hi[0] - lo[0]) / pp.dx_mm)))
    ny = max(8, int(np.ceil((hi[1] - lo[1]) / pp.dx_mm)))
    nz = max(8, int(np.ceil((hi[2] - lo[2]) / pp.dx_mm)))
    x_mm = lo[0] + np.arange(nx) * pp.dx_mm
    y_mm = lo[1] + np.arange(ny) * pp.dx_mm
    z_mm = lo[2] + np.arange(nz) * pp.dx_mm
    shape = (nx, ny, nz)

    # homogeneous tissue (brain modelled as water-like; absorption = tissue_alpha)
    rho, c = 1000.0, 1500.0
    density = np.full(shape, rho)
    cp = np.full(shape, CP_WATER)
    kcond = np.full(shape, K_WATER)
    bone = np.zeros(shape, dtype=bool)

    # physical peak heat rate at the focus [W/m^3]
    p_peak = pp.focal_pressure_pa * pp.transmission
    alpha_np = db2neper(pp.tissue_alpha_db, pp.alpha_power) * (2 * np.pi * pp.f0_hz) ** pp.alpha_power
    intensity_peak = (p_peak / np.sqrt(2.0)) ** 2 / (rho * c)
    q_peak = 2.0 * alpha_np * intensity_peak * pp.duty_cycle

    Xi = x_mm[:, None, None]; Yi = y_mm[None, :, None]; Zi = z_mm[None, None, :]
    s_ax = pp.axial_fwhm_mm / 2.3548

    q_fields = []
    focus_idx = []
    for s in spots:
        fx, fy, fz = s.focus_mm
        sx = max(s.rx_mm, pp.dx_mm) / 2.3548 * 2.0   # rx ~ FWHM radius -> sigma
        sy = max(s.ry_mm, pp.dx_mm) / 2.3548 * 2.0
        # pressure ~ Gaussian g; intensity ~ g^2; so Q ~ q_peak * g^2
        g = np.exp(-(((Xi - fx) ** 2) / (2 * sx * sx)
                     + ((Yi - fy) ** 2) / (2 * sy * sy)
                     + ((Zi - fz) ** 2) / (2 * s_ax * s_ax)))
        q_fields.append((q_peak * g ** 2).astype(np.float64))
        focus_idx.append((int(np.argmin(np.abs(x_mm - fx))),
                          int(np.argmin(np.abs(y_mm - fy))),
                          int(np.argmin(np.abs(z_mm - fz)))))

    grid = dict(dx=dx, density=density, specific_heat=cp,
                thermal_conductivity=kcond, bone_mask=bone,
                x_mm=x_mm, y_mm=y_mm, z_mm=z_mm)

    on_times = [pp.on_time_s] * len(spots)
    gaps = [pp.cooldown_s] * len(spots)

    if progress:
        progress(f"Running {len(spots)}-spot protocol (q_peak={q_peak:.2e} W/m^3)...")
    res = run_protocol(grid, q_fields, on_times, gaps, focus_idx, tp,
                       progress=progress)
    res.meta["spots_mm"] = [s.focus_mm for s in spots]
    res.meta["spot_ids"] = [s.spot_id for s in spots]
    res.meta["footprints_mm"] = [(s.rx_mm, s.ry_mm) for s in spots]
    res.meta["q_peak"] = q_peak
    res.meta["duty"] = pp.duty_cycle

    def _gf(s, key):
        try:
            return float(s.extra.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")
    res.meta["skull_z_mm"] = [
        (_gf(s, "z_brain_top_mm"), _gf(s, "z_inner_skull_mm"), _gf(s, "z_outer_skull_mm"))
        for s in spots]
    return res


# --------------------------------------------------------------------------- #
# Per-spot ACOUSTIC mode (skull included) - one k-Wave run per spot
# --------------------------------------------------------------------------- #
def _plan_offsets_to_sim(spots: List[PlanSpot]):
    """Map each plan focus (LPS world mm) to a sim-frame offset from the plan
    centroid. Plan beam axis = IS (z_world, deeper = more negative); sim beam
    axis = +x (deeper = more positive). Lateral LR->sim y, PA->sim z.

    Returns offsets in metres: list of (dx_axial, dy, dz).
    """
    foci = np.array([s.focus_mm for s in spots], dtype=float)
    c = foci.mean(0)
    offs = []
    for s in spots:
        d = np.array(s.focus_mm) - c          # (dLR, dPA, dIS) mm
        offs.append((-d[2] * 1e-3, d[0] * 1e-3, d[1] * 1e-3))  # (axial, y, z) [m]
    return offs


def build_and_run_plan_acoustic(spots, sim_params, skull, drive_pa,
                                pp: "PlanProtocolParams", tp: ThermalParams,
                                rotate_cw90: bool = True,
                                base_offset_mm=(0.0, 0.0, 0.0),
                                fill_threshold: float = 0.30,
                                progress=None):
    """ACCURATE protocol mode: one acoustic k-Wave run per spot, through the
    embedded skull (skull fixed, transducer focus moved to each spot), then the
    cumulative thermal accumulation - so skull heating and per-spot aberration
    are included.

    Requires a loaded skull (``skull`` = SkullData) and a calibrated ``drive_pa``.
    Heavy: N GPU acoustic runs. Q at each spot = 2*alpha*I with alpha = skull
    absorption map + ``pp.tissue_alpha_db`` in non-bone voxels, x duty.
    """
    from dataclasses import replace
    from .grid import build_grid_spec
    from .simulate import run_simulation
    from .thermal import compute_Q, CP_WATER, CP_SKULL, K_WATER, K_SKULL

    # widen the sensor box so it spans the whole spot cloud + margin
    foci = np.array([s.focus_mm for s in spots], dtype=float)
    spread = (foci.max(0) - foci.min(0))           # mm in (LR, PA, IS)
    lat = float(max(spread[0], spread[1])) + 2 * max(s.rx_mm for s in spots) + 8.0
    sp2 = replace(sim_params,
                  skull_box_lr_mm=max(sim_params.skull_box_lr_mm, lat),
                  skull_box_ap_mm=max(sim_params.skull_box_ap_mm, lat))
    gspec = build_grid_spec(sp2)

    # embed the skull ONCE (fixed), at the Skull-tab placement
    fx, fy, fz = gspec.focus_xyz
    center = (fx + base_offset_mm[0] * 1e-3,
              fy + base_offset_mm[1] * 1e-3,
              fz + base_offset_mm[2] * 1e-3)
    # skull may be a CTSkullSource (new CT path, has .embed) or a legacy
    # SkullData mask (embed_skull).
    if hasattr(skull, "embed"):
        emb = skull.embed(gspec, center_xyz_m=center, f0_hz=sim_params.f0,
                          fill_threshold=fill_threshold)
    else:
        from .skull import embed_skull
        emb = embed_skull(skull, gspec, center_xyz_m=center, rotate_cw90=rotate_cw90)

    (ix0, ix1), (iy0, iy1), (iz0, iz1) = gspec.box_ix, gspec.box_iy, gspec.box_iz
    sl = (slice(ix0, ix1 + 1), slice(iy0, iy1 + 1), slice(iz0, iz1 + 1))
    bone_box = emb.bone_mask[sl]
    dens_box = emb.density[sl]
    c_box = emb.sound_speed[sl]
    # effective absorption: skull map + tissue absorption in non-bone voxels
    alpha_box = emb.alpha_coeff[sl] + pp.tissue_alpha_db * (~bone_box)

    x_mm = gspec.x_vec[ix0:ix1 + 1] * 1e3
    y_mm = gspec.y_vec[iy0:iy1 + 1] * 1e3
    z_mm = gspec.z_vec[iz0:iz1 + 1] * 1e3
    grid = dict(dx=gspec.dx, density=dens_box.astype(float),
                specific_heat=(CP_WATER + (CP_SKULL - CP_WATER) * bone_box).astype(float),
                thermal_conductivity=(K_WATER + (K_SKULL - K_WATER) * bone_box).astype(float),
                bone_mask=bone_box, x_mm=x_mm, y_mm=y_mm, z_mm=z_mm)

    offs = _plan_offsets_to_sim(spots)
    q_fields = []
    focus_idx = []
    for k, (s, off) in enumerate(zip(spots, offs)):
        focus_k = (fx + off[0], fy + off[1], fz + off[2])
        if progress:
            progress(f"Acoustic spot {k+1}/{len(spots)} (focus offset "
                     f"{off[1]*1e3:+.1f},{off[2]*1e3:+.1f} mm)...")
        ff = run_simulation(sp2, gspec=gspec, source_amp_pa=drive_pa, skull=emb,
                            focus_override=focus_k, label=f"spot {s.spot_id}",
                            progress=progress)
        p_box_pa = ff.box_field_mpa * 1e6
        q = compute_Q(p_box_pa, alpha_box, emb.alpha_power, dens_box, c_box,
                      sim_params.f0) * pp.duty_cycle
        q_fields.append(q.astype(np.float64))
        focus_idx.append((int(np.argmin(np.abs(gspec.x_vec[ix0:ix1 + 1] - focus_k[0]))),
                          int(np.argmin(np.abs(gspec.y_vec[iy0:iy1 + 1] - focus_k[1]))),
                          int(np.argmin(np.abs(gspec.z_vec[iz0:iz1 + 1] - focus_k[2])))))

    on_times = [pp.on_time_s] * len(spots)
    gaps = [pp.cooldown_s] * len(spots)
    res = run_protocol(grid, q_fields, on_times, gaps, focus_idx, tp, progress=progress)
    res.meta["spots_mm"] = [s.focus_mm for s in spots]
    res.meta["spot_ids"] = [s.spot_id for s in spots]
    res.meta["footprints_mm"] = [(s.rx_mm, s.ry_mm) for s in spots]
    res.meta["duty"] = pp.duty_cycle
    res.meta["mode"] = "per-spot acoustic (skull)"
    return res
