"""Free-field (water-only) simulation used for pressure calibration.

  * auto-sizes the grid so the focus always fits (see :mod:`.grid`),
  * records ``p_max`` only inside the small sensor box around the future
    skull region (faster + less memory than the whole grid),
  * dispatches to the GPU or CPU solver based on detection / user choice.

The medium is homogeneous water, so the run
is the free-field baseline against which the source amplitude is calibrated.
"""

from __future__ import annotations

import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

import numpy as np

from .params import SimParams, WATER
from .grid import GridSpec, build_grid_spec
from .gpu import choose_solver
from .transducer import build_source_mask


@dataclass
class FreeFieldResult:
    peak_pressure_pa: float           # peak |p| inside the sensor box
    source_amp_pa: float              # drive amplitude used for this run
    gspec: GridSpec
    box_field_mpa: np.ndarray         # p_max in box [MPa], shape = box_shape
    # 2-D centre slices through the focus (for display), in MPa
    slice_xz_mpa: np.ndarray
    slice_xy_mpa: np.ndarray
    slice_xz_extent_mm: tuple         # (x0, x1, z0, z1)
    slice_xy_extent_mm: tuple         # (x0, x1, y0, y1)
    solver_msg: str
    used_gpu: bool
    dt: float
    Nt: int
    n_source_points: int
    runtime_s: float
    meta: Dict = field(default_factory=dict)
    focus_pressure_pa: float = 0.0    # p_max AT the geometric-focus voxel
    label: str = "free field (water)"  # description for plots/log


def _build_time_array(kgrid, gspec: GridSpec, p: SimParams, c_ref: float, cfl: float):
    """Design dt / Nt."""
    c0 = WATER.sound_speed
    ppw_ref = c_ref / (p.f0 * gspec.dx)
    ppp = int(np.ceil(ppw_ref / cfl))
    T = 1.0 / p.f0
    dt = T / ppp
    t_end = (p.num_cycles / p.f0) + (
        np.sqrt(kgrid.x_size ** 2 + kgrid.y_size ** 2) / c0)
    Nt = int(round(t_end / dt))
    kgrid.setTime(Nt, dt)
    return dt, Nt


def run_simulation(p: SimParams,
                   gspec: Optional[GridSpec] = None,
                   source_amp_pa: Optional[float] = None,
                   skull=None,
                   label: Optional[str] = None,
                   focus_override=None,
                   progress: Optional[Callable[[str], None]] = None) -> FreeFieldResult:
    """Run one k-Wave simulation and return the peak field in the sensor box.

    Parameters
    ----------
    p : SimParams
    gspec : optional pre-built grid spec (else built from ``p``).
    source_amp_pa : drive amplitude [Pa]; defaults to ``p.reference_source_amp_pa``.
    skull : optional :class:`~fus_simulator.core.skull.EmbeddedSkull`. If given,
        the medium is the heterogeneous skull (sound speed / density /
        attenuation maps); otherwise it is homogeneous water.
    label : description for plots/log.
    progress : optional callback for status strings (used by the GUI thread).
    """
    def _say(msg):
        if progress:
            progress(msg)

    # Lazy k-Wave imports so the module loads without the solver installed.
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D, kspaceFirstOrder3DG
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.utils.signals import tone_burst

    if gspec is None:
        gspec = build_grid_spec(p)
    if source_amp_pa is None:
        source_amp_pa = p.reference_source_amp_pa

    _say(f"Grid {gspec.Nx}x{gspec.Ny}x{gspec.Nz} "
         f"(dx={gspec.dx*1e3:.3f} mm, {gspec.memory_per_field_gb():.2f} GB/field)")

    kgrid = kWaveGrid(Vector([gspec.Nx, gspec.Ny, gspec.Nz]),
                      Vector([gspec.dx, gspec.dx, gspec.dx]))

    _say("Building bowl source mask...")
    _, source_p_mask, n_src = build_source_mask(kgrid, gspec, p,
                                                focus_xyz=focus_override, progress=_say)
    _say(f"Source active points: {n_src:,}")

    if skull is None:
        # Homogeneous water
        medium = kWaveMedium(
            sound_speed=WATER.sound_speed,
            density=WATER.density,
            alpha_coeff=WATER.alpha_coeff,
            alpha_power=WATER.alpha_power,
        )
        c_ref = WATER.sound_speed
        if label is None:
            label = "free field (water)"
    else:
        # ---- validate the embedded skull ----
        c_arr = np.asarray(skull.sound_speed)
        rho_arr = np.asarray(skull.density)
        if (not np.all(np.isfinite(c_arr)) or not np.all(np.isfinite(rho_arr))
                or float(c_arr.min()) <= 0.0 or float(rho_arr.min()) <= 0.0):
            raise RuntimeError(
                "Embedded skull medium has invalid values (NaN or <= 0 sound "
                "speed / density). The skull did not load correctly - reload it "
                "on the Skull tab and check the placement.")
        if int(skull.n_bone_voxels) == 0:
            _say("[WARNING] Embedded skull has 0 bone voxels - is a skull loaded "
                 "and placed over the focus? Running as near-water.")
        # Heterogeneous skull medium
        medium = kWaveMedium(
            sound_speed=skull.sound_speed,
            density=skull.density,
            alpha_coeff=skull.alpha_coeff,
            alpha_power=skull.alpha_power,
        )
        c_ref = float(np.nanmax(c_arr))
        _say(f"Skull medium: c [{c_arr.min():.0f},{c_arr.max():.0f}] m/s, "
             f"rho [{rho_arr.min():.0f},{rho_arr.max():.0f}] kg/m^3, "
             f"{skull.n_bone_voxels:,} bone voxels")
        if label is None:
            label = "through skull"

    # CFL is taken straight from the user (no forced cap).
    dt, Nt = _build_time_array(kgrid, gspec, p, c_ref, p.cfl)
    _say(f"dt = {dt*1e6:.4f} us, Nt = {Nt} (c_ref = {c_ref:.0f} m/s, CFL = {p.cfl:.3f})")

    # ---- auto-clamp dt to the medium's stability limit. 
    # # This keeps coarse/heterogeneous runs stable without
    # hardcoding a CFL: dt is reduced only if the medium actually requires it.
    try:
        from kwave.utils.checks import check_stability
        dt_limit = float(check_stability(kgrid, medium))
        if np.isfinite(dt_limit) and dt > 0.95 * dt_limit:
            safe_dt = 0.9 * dt_limit
            Nt = int(np.ceil(Nt * dt / safe_dt))
            dt = safe_dt
            kgrid.setTime(Nt, dt)
            _say(f"Auto-reduced dt to {dt*1e6:.4f} us for stability "
                 f"(limit {dt_limit*1e6:.4f} us); Nt = {Nt}.")
    except Exception as _err:  # check unavailable - rely on the sanity guard
        _say(f"[note] stability check skipped ({_err}).")

    source_signal = tone_burst(sample_freq=1.0 / kgrid.dt,
                               signal_freq=p.f0,
                               num_cycles=p.num_cycles)
    source = kSource()
    source.p_mask = source_p_mask
    source.p = source_amp_pa * source_signal

    # ---- sensor: record p_max only inside the skull-region box ----
    sensor = kSensor()
    box_mask = np.zeros((gspec.Nx, gspec.Ny, gspec.Nz), dtype=bool)
    (ix0, ix1), (iy0, iy1), (iz0, iz1) = gspec.box_ix, gspec.box_iy, gspec.box_iz
    box_mask[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = True
    sensor.mask = box_mask
    # IMPORTANT: use "p_max" (max pressure AT the sensor-mask points
    sensor.record = ["p_max"]

    use_gpu, solver_msg = choose_solver(p.compute_mode)
    _say(solver_msg)

    tmp_dir = tempfile.mkdtemp(prefix="fus_sim_")
    sim_opts = SimulationOptions(
        data_cast="single",
        data_recast=True,
        save_to_disk=True,
        input_filename="fus_input.h5",
        output_filename="fus_output.h5",
        save_to_disk_exit=False,
        data_path=tmp_dir,
        pml_inside=False,
    )
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=use_gpu,
        delete_data=True,
        verbose_level=1,
    )

    solver = kspaceFirstOrder3DG if use_gpu else kspaceFirstOrder3D
    _say("Running k-Wave simulation (this can take a while)...")
    t0 = time.time()
    # deepcopy inputs so the solver can't mutate our objects
    sensor_data = solver(
        medium=deepcopy(medium),
        kgrid=deepcopy(kgrid),
        source=deepcopy(source),
        sensor=deepcopy(sensor),
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    runtime = time.time() - t0
    _say(f"Simulation finished in {runtime:.1f} s")

    # ---- reshape masked p_max back into the box (column-major)
    box_shape = gspec.box_shape
    p_max_flat = np.asarray(sensor_data["p_max"]).ravel(order="F")
    expected = gspec.box_n_points
    if p_max_flat.size != expected:
        raise RuntimeError(
            f"Sensor returned {p_max_flat.size} points but the box has "
            f"{expected}. Check that record uses 'p_max' (masked), not "
            f"'p_max_all' (whole grid).")
    p_max_box = np.reshape(p_max_flat, box_shape, order="F")
    box_field_mpa = p_max_box / 1e6
    peak_pa = float(np.nanmax(p_max_box))

    # ---- sanity / stability guard ----
    # k-Wave initialises p_max to -FLT_MAX (~-3.4e38) and updates it with the
    # running maximum. If the run goes unstable the field becomes NaN, the
    # recorder never updates, and p_max stays at the sentinel -> a huge negative
    # peak. 
    n_bad = int(np.size(p_max_box) - np.isfinite(p_max_box).sum())
    if (not np.isfinite(peak_pa)) or peak_pa <= 0.0 or peak_pa > 1.0e9 or n_bad > 0:
        raise RuntimeError(
            "Simulation produced no valid pressure - this is numerical "
            "instability.\n\n"
            f"Recorded peak = {peak_pa:.3e} Pa, non-finite voxels = {n_bad}.\n\n"
            "Fix: lower the CFL number (e.g. 0.05) and/or "
            "raise points-per-wavelength. Heterogeneous skull media (sharp "
            "sound-speed jumps) need a smaller CFL than a water-only run.")

    # ---- pressure exactly at the geometric-focus voxel (within the box) ----
    fix = int(np.argmin(np.abs(gspec.x_vec - gspec.focus_xyz[0]))) - ix0
    fiy = int(np.argmin(np.abs(gspec.y_vec - gspec.focus_xyz[1]))) - iy0
    fiz = int(np.argmin(np.abs(gspec.z_vec - gspec.focus_xyz[2]))) - iz0
    if 0 <= fix < box_shape[0] and 0 <= fiy < box_shape[1] and 0 <= fiz < box_shape[2]:
        focus_pa = float(p_max_box[fix, fiy, fiz])
    else:
        focus_pa = peak_pa

    # ---- centre slices through the focus for display ----
    bx, by, bz = box_shape
    jy = by // 2
    jz = bz // 2
    slice_xz = box_field_mpa[:, jy, :]          # (bx, bz)
    slice_xy = box_field_mpa[:, :, jz]          # (bx, by)

    x_mm = gspec.x_vec[ix0:ix1 + 1] * 1e3
    y_mm = gspec.y_vec[iy0:iy1 + 1] * 1e3
    z_mm = gspec.z_vec[iz0:iz1 + 1] * 1e3
    xz_extent = (float(x_mm[0]), float(x_mm[-1]), float(z_mm[0]), float(z_mm[-1]))
    xy_extent = (float(x_mm[0]), float(x_mm[-1]), float(y_mm[0]), float(y_mm[-1]))

    return FreeFieldResult(
        peak_pressure_pa=peak_pa,
        source_amp_pa=source_amp_pa,
        gspec=gspec,
        box_field_mpa=box_field_mpa,
        slice_xz_mpa=slice_xz.T,                 # transpose -> rows=z for imshow
        slice_xy_mpa=slice_xy.T,                 # transpose -> rows=y for imshow
        slice_xz_extent_mm=xz_extent,
        slice_xy_extent_mm=xy_extent,
        solver_msg=solver_msg,
        used_gpu=use_gpu,
        dt=dt, Nt=Nt,
        n_source_points=n_src,
        runtime_s=runtime,
        meta={"tmp_dir": tmp_dir},
        focus_pressure_pa=focus_pa,
        label=label,
    )


def run_freefield(p: SimParams,
                  gspec: Optional[GridSpec] = None,
                  source_amp_pa: Optional[float] = None,
                  progress: Optional[Callable[[str], None]] = None) -> FreeFieldResult:
    """Backward-compatible wrapper: homogeneous water free-field run."""
    return run_simulation(p, gspec=gspec, source_amp_pa=source_amp_pa,
                          skull=None, label="free field (water)", progress=progress)
