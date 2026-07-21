"""Multi-spot sonication protocol - cumulative thermal simulation.

A protocol targets several spots sequentially: the transducer fires at spot 1
for its on-time, moves (a cooling gap with the source off), fires at spot 2, and
so on. The acoustic pressure from each spot is essentially instantaneous (it
fills/empties the domain in ~microseconds), so spots never overlap acoustically.
What overlaps is HEAT: tissue warmed by an earlier spot has not fully cooled
when a nearby later spot fires, so the shared region gets hotter than any single
spot. This module accumulates that on one shared grid.

Memory: only the current temperature field, a running max-temperature field, and
the CEM43 dose field are held (three 3-D arrays). Per-spot temperatures are 1-D
time traces. Nothing is stored per time step, so cost is set by grid size, not
duration - run the shared grid coarse (heat is smooth) to keep it light.

Per-spot heat sources ``Q`` [W/m^3] are computed from each spot's acoustic run
(via :func:`fus_simulator.core.thermal.compute_Q`) and placed on the shared
grid; this module just does the time accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Dict

import numpy as np

from .thermal import HeatDiffusion, ThermalParams


@dataclass
class ProtocolResult:
    times_s: np.ndarray                 # timeline samples
    spot_traces_c: List[np.ndarray]     # temperature at each spot focus vs time
    spot_windows_s: List[Tuple[float, float]]  # (on-start, on-end) per spot
    T_max: np.ndarray                   # cumulative max temperature [degC]
    cem43: np.ndarray                   # cumulative thermal dose
    overlap_count: np.ndarray           # # of spots depositing heat per voxel
    peak_temp_c: float
    peak_rise_c: float
    max_cem43: float
    lesion_volume_mm3: float
    overlap_voxels: int                 # voxels heated by >= 2 spots
    overlap_volume_mm3: float           # overlap region volume [mm^3]
    overlap_max_temp_c: float           # peak temp within overlap region
    # display slices through the hottest voxel
    extent_xz_mm: tuple
    extent_xy_mm: tuple
    slice_T_xz: np.ndarray
    slice_T_xy: np.ndarray
    overlap_xz: np.ndarray
    overlap_xy: np.ndarray
    bone_xz: np.ndarray
    bone_xy: np.ndarray
    hot_idx: Tuple[int, int, int]
    meta: Dict = field(default_factory=dict)


def run_protocol(grid: Dict,
                 q_fields: Sequence[np.ndarray],
                 on_times_s: Sequence[float],
                 gaps_s: Sequence[float],
                 focus_indices: Sequence[Tuple[int, int, int]],
                 tp: ThermalParams,
                 overlap_frac: float = 0.15,
                 progress: Optional[Callable[[str], None]] = None) -> ProtocolResult:
    """Accumulate heating across spots on the shared grid in ``grid``.

    ``grid`` keys: ``dx``, ``density``, ``specific_heat``, ``thermal_conductivity``,
    ``bone_mask``, ``x_mm``/``y_mm``/``z_mm`` (axis vectors of the shared grid).
    ``q_fields[s]`` is spot s's duty-averaged heat source [W/m^3] on the shared
    grid. ``on_times_s``/``gaps_s`` are the per-spot on-time and following gap.
    """
    def _say(m):
        if progress:
            progress(m)

    dx = grid["dx"]
    shape = grid["density"].shape
    zeroQ = np.zeros(shape, dtype=np.float64)

    solver = HeatDiffusion(
        shape, dx, grid["density"], grid["specific_heat"],
        grid["thermal_conductivity"], Q=zeroQ, T0=tp.baseline_temp_c,
        perfusion_coeff=tp.perfusion_rate, blood_temp=tp.blood_temp_c)

    dt = tp.dt_thermal_s
    n_spots = len(q_fields)
    times = [0.0]
    traces = [[float(solver.T[tuple(fi)])] for fi in focus_indices]
    windows: List[Tuple[float, float]] = []
    T_max = solver.T.copy()
    t_now = 0.0

    def _record():
        times.append(t_now)
        for s, fi in enumerate(focus_indices):
            traces[s].append(float(solver.T[tuple(fi)]))
        np.maximum(T_max, solver.T, out=T_max)

    def _run_phase(duration, Q):
        nonlocal t_now
        solver.set_Q(Q)
        n = max(1, int(round(duration / dt)))
        seg = max(1, int(round(n / 12)))
        done = 0
        while done < n:
            k = min(seg, n - done)
            solver.step(k, dt)
            done += k
            t_now += k * dt
            _record()

    for s in range(n_spots):
        _say(f"Spot {s+1}/{n_spots}: sonicating {on_times_s[s]:.2f} s...")
        start = t_now
        _run_phase(on_times_s[s], q_fields[s].astype(np.float64))
        windows.append((start, t_now))
        if gaps_s[s] > 0:
            _say(f"Spot {s+1}/{n_spots}: move/cool {gaps_s[s]:.2f} s...")
            _run_phase(gaps_s[s], zeroQ)

    # ---- overlap diagnostic: how many spots significantly heat each voxel ----
    overlap_count = np.zeros(shape, dtype=np.int16)
    for q in q_fields:
        qmax = float(q.max())
        if qmax > 0:
            overlap_count += (q >= overlap_frac * qmax).astype(np.int16)
    overlap_mask = overlap_count >= 2

    lesion = solver.cem43 >= 240.0
    hot = np.unravel_index(int(np.argmax(T_max)), shape)

    # slices through the hottest voxel
    x_mm, y_mm, z_mm = grid["x_mm"], grid["y_mm"], grid["z_mm"]
    bone = grid["bone_mask"]
    jx, jy, jz = hot
    ext_xz = (float(x_mm[0]), float(x_mm[-1]), float(z_mm[0]), float(z_mm[-1]))
    ext_xy = (float(x_mm[0]), float(x_mm[-1]), float(y_mm[0]), float(y_mm[-1]))

    res = ProtocolResult(
        times_s=np.array(times),
        spot_traces_c=[np.array(t) for t in traces],
        spot_windows_s=windows,
        T_max=T_max, cem43=solver.cem43, overlap_count=overlap_count,
        peak_temp_c=float(T_max.max()),
        peak_rise_c=float(T_max.max() - tp.baseline_temp_c),
        max_cem43=float(solver.cem43.max()),
        lesion_volume_mm3=float(lesion.sum() * (dx * 1e3) ** 3),
        overlap_voxels=int(overlap_mask.sum()),
        overlap_volume_mm3=float(overlap_mask.sum() * (dx * 1e3) ** 3),
        overlap_max_temp_c=float(T_max[overlap_mask].max()) if overlap_mask.any() else float("nan"),
        extent_xz_mm=ext_xz, extent_xy_mm=ext_xy,
        slice_T_xz=T_max[:, jy, :].T, slice_T_xy=T_max[:, :, jz].T,
        overlap_xz=overlap_count[:, jy, :].T, overlap_xy=overlap_count[:, :, jz].T,
        bone_xz=bone[:, jy, :].T, bone_xy=bone[:, :, jz].T,
        hot_idx=hot,
    )
    _say(f"Protocol peak {res.peak_temp_c:.2f} C (+{res.peak_rise_c:.2f}); "
         f"overlap {res.overlap_volume_mm3:.3f} mm^3")
    return res
