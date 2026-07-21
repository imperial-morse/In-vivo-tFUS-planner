"""Thermal (bioheat) simulation.

Solves the Pennes bioheat equation with a k-space pseudospectral method
(identical scheme to the ``kWaveDiffusion`` class):

    A dT/dt = div(Kt grad T) - B (T - Ta) + Q

with A = rho*Cp, Kt = thermal conductivity, B = perfusion*A, Q = volumetric
heat. Heat is deposited from the acoustic field via

    alpha_np = db2neper(alpha_coeff, y) * (2 pi f0)^y      [Np/m]
    I        = (p_peak / sqrt 2)^2 / (rho c)               [W/m^2]
    Q        = 2 * alpha_np * I                            [W/m^3]

For pulsed sonication the time-averaged source is ``Q * duty`` with
``duty = PRF * pulse_duration`` (e.g. a duty of 0.2). Tissue/skull
thermal properties follow published values:

    k_water=0.6, k_skull=0.32 W/(m K);  Cp_water=4180, Cp_skull=1300 J/(kg K)

This module is pure NumPy (no k-Wave / GPU), so it runs anywhere. It operates on
the small sensor-box subgrid around the focus to stay fast and memory-light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict

import numpy as np

# Thermal properties. Non-bone voxels are modelled as coupling WATER, not brain.
#
#   K_WATER, CP_WATER : liquid water at body temperature (37 C). Both are the
#       37 C values, not the room-temperature ones (k(20 C) = 0.60, k(37 C) =
#       0.623 W/(m K)). [CRC Handbook of Chemistry & Physics / IAPWS]
#   K_SKULL, CP_SKULL : cortical bone. Primary source Duck, "Physical Properties
#       of Tissues" (2013), as tabulated in Table 3 of Constans, Mateo, Tanter &
#       Aubry, Phys. Med. Biol. 63(2):025003 (2018):
#       thermal conductivity 0.440 W/(m K), specific heat 1300 J/(kg K).
K_WATER = 0.623    # W/(m K)   - liquid water @37 C [CRC/IAPWS]
K_SKULL = 0.44     # W/(m K)   - cortical bone [Duck 2013, via Constans 2018 T3]
CP_WATER = 4180.0  # J/(kg K)  - liquid water @37 C [CRC/IAPWS]
CP_SKULL = 1300.0  # J/(kg K)  - cortical bone [Duck 2013, via Constans 2018 T3]


def db2neper(alpha, y):
    """dB/(MHz^y cm) -> Np/((rad/s)^y m), k-Wave convention."""
    return 100.0 * np.asarray(alpha) * (1e-6 / (2.0 * np.pi)) ** y / (20.0 * np.log10(np.e))


def compute_Q(p_peak_pa, alpha_coeff, alpha_power, density, sound_speed, f0):
    """Volumetric heat deposition Q [W/m^3] from the peak-pressure field."""
    alpha_np = db2neper(alpha_coeff, alpha_power) * (2.0 * np.pi * f0) ** alpha_power
    intensity = (p_peak_pa / np.sqrt(2.0)) ** 2 / (density * sound_speed)
    return 2.0 * alpha_np * intensity


# --------------------------------------------------------------------------- #
# Pennes bioheat solver (k-space pseudospectral) 
# --------------------------------------------------------------------------- #
class HeatDiffusion:
    def __init__(self, shape, dx, density, specific_heat, thermal_conductivity,
                 Q, T0=37.0, perfusion_coeff=0.0, blood_temp=37.0):
        self.shape = tuple(shape)
        kx = 2 * np.pi * np.fft.fftfreq(shape[0], dx)
        ky = 2 * np.pi * np.fft.fftfreq(shape[1], dx)
        kz = 2 * np.pi * np.fft.fftfreq(shape[2], dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        self.k_sq = KX ** 2 + KY ** 2 + KZ ** 2
        self.deriv_x, self.deriv_y, self.deriv_z = 1j * KX, 1j * KY, 1j * KZ

        self.density = density
        self.specific_heat = specific_heat
        self.thermal_conductivity = thermal_conductivity
        self.diffusion_p1 = 1.0 / (density * specific_heat)      # 1/(rho Cp)
        self.diffusion_p2 = thermal_conductivity
        self.diffusion_coeff = self.diffusion_p1 * self.diffusion_p2
        self.diffusion_coeff_ref = float(np.max(self.diffusion_coeff))

        self.is_homogeneous = (
            np.allclose(self.diffusion_p1, np.asarray(self.diffusion_p1).flat[0]) and
            np.allclose(self.diffusion_p2, np.asarray(self.diffusion_p2).flat[0]))

        self.perfusion_coeff = perfusion_coeff
        self.blood_temp = blood_temp
        self.perfusion_ref = float(np.max(perfusion_coeff)) if np.ndim(perfusion_coeff) else float(perfusion_coeff)

        self.set_Q(Q)
        self.T = (T0 * np.ones(self.shape, dtype=np.float64)
                  if np.isscalar(T0) else T0.astype(np.float64))
        self.cem43 = np.zeros_like(self.T)
        self.steps_taken = 0

    def set_Q(self, Q):
        self.Q = Q
        self.q_scale = self.diffusion_p1 if np.any(Q != 0) else 0.0

    def _kappa(self, dt):
        arg = dt * (self.diffusion_coeff_ref * self.k_sq + self.perfusion_ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = (1.0 - np.exp(-arg)) / arg
            kappa = np.where(arg == 0, 1.0, kappa)
        return kappa

    def step(self, Nt, dt):
        kappa = self._kappa(dt)
        if np.any(self.Q != 0):
            q_term = self.q_scale * np.real(np.fft.ifftn(kappa * np.fft.fftn(self.Q)))
        else:
            q_term = 0.0
        for _ in range(int(Nt)):
            T_ft = np.fft.fftn(self.T)
            if self.is_homogeneous:
                d_term = self.diffusion_coeff_ref * np.real(
                    np.fft.ifftn(-self.k_sq * kappa * T_ft))
            else:
                dT_dx = np.real(np.fft.ifftn(self.deriv_x * kappa * T_ft))
                dT_dy = np.real(np.fft.ifftn(self.deriv_y * kappa * T_ft))
                dT_dz = np.real(np.fft.ifftn(self.deriv_z * kappa * T_ft))
                div_flux = (
                    np.real(np.fft.ifftn(self.deriv_x * np.fft.fftn(self.diffusion_p2 * dT_dx))) +
                    np.real(np.fft.ifftn(self.deriv_y * np.fft.fftn(self.diffusion_p2 * dT_dy))) +
                    np.real(np.fft.ifftn(self.deriv_z * np.fft.fftn(self.diffusion_p2 * dT_dz))))
                d_term = self.diffusion_p1 * div_flux
            if np.any(self.perfusion_coeff != 0):
                T_diff_ft = np.fft.fftn(self.T - self.blood_temp)
                p_term = -self.perfusion_coeff * np.real(np.fft.ifftn(kappa * T_diff_ft))
            else:
                p_term = 0.0
            self.T = self.T + dt * (d_term + p_term + q_term)
            R = np.where(self.T >= 43, 0.5, 0.25)
            self.cem43 = self.cem43 + (dt / 60.0) * np.power(R, 43.0 - self.T)
        self.steps_taken += int(Nt)


# --------------------------------------------------------------------------- #
# Parameters & result
# --------------------------------------------------------------------------- #
@dataclass
class ThermalParams:
    baseline_temp_c: float = 37.0
    prf_hz: float = 1000.0            # pulse repetition frequency
    pulse_duration_ms: float = 0.2    # per-pulse on-time -> duty = PRF*pulse
    sonication_time_s: float = 6.0    # total on-time
    cooling_time_s: float = 0.0       # optional post-sonication cooling
    dt_thermal_s: float = 0.1         # thermal time step
    perfusion_rate: float = 0.0       # blood perfusion coeff [1/s] (0 = off)
    blood_temp_c: float = 37.0

    @property
    def duty_cycle(self) -> float:
        return float(min(1.0, max(0.0, self.prf_hz * self.pulse_duration_ms * 1e-3)))


@dataclass
class ThermalResult:
    T_end: np.ndarray            # box temperature at end of sonication [degC]
    T_max: np.ndarray            # max-over-time box temperature [degC]
    cem43: np.ndarray            # thermal dose [equiv min @ 43C]
    times_s: np.ndarray          # time samples
    focus_temp_c: np.ndarray     # temperature at focus vs time
    peak_temp_c: float
    peak_rise_c: float
    focus_end_temp_c: float
    focus_rise_c: float
    max_cem43: float
    lesion_volume_mm3: float
    duty_cycle: float
    q_max: float
    extent_xz_mm: tuple
    extent_xy_mm: tuple
    slice_T_xz: np.ndarray       # T_max XZ slice (rows=z) for imshow
    slice_T_xy: np.ndarray       # T_max XY slice (rows=y)
    bone_xz: np.ndarray          # bone mask slices for overlay
    bone_xy: np.ndarray
    meta: Dict = field(default_factory=dict)


def run_thermal(box: Dict, pressure_box_pa: np.ndarray, tp: ThermalParams,
                f0: float, progress: Optional[Callable[[str], None]] = None) -> ThermalResult:
    """Run the bioheat simulation on the sensor-box subgrid.

    ``box`` keys: ``sound_speed``, ``density``, ``alpha_coeff`` (box arrays),
    ``bone_mask`` (box bool), ``alpha_power`` (float), ``dx`` (m),
    ``focus_idx`` (i,j,k local), ``x_mm``/``y_mm``/``z_mm`` (box axis vectors).
    """
    def _say(m):
        if progress:
            progress(m)

    rho = box["density"]
    c = box["sound_speed"]
    bone = box["bone_mask"].astype(np.float64)
    dx = box["dx"]
    duty = tp.duty_cycle

    _say(f"Computing heat source Q (duty = {duty*100:.1f} %)...")
    Q = compute_Q(pressure_box_pa, box["alpha_coeff"], box["alpha_power"],
                  rho, c, f0) * duty
    Q = Q.astype(np.float64)

    k = K_WATER + (K_SKULL - K_WATER) * bone
    cp = CP_WATER + (CP_SKULL - CP_WATER) * bone

    perf = tp.perfusion_rate
    solver = HeatDiffusion(Q.shape, dx, rho, cp, k, Q,
                           T0=tp.baseline_temp_c, perfusion_coeff=perf,
                           blood_temp=tp.blood_temp_c)

    fi = tuple(box["focus_idx"])
    times = [0.0]
    focus_temp = [float(solver.T[fi])]
    T_max = solver.T.copy()

    dt = tp.dt_thermal_s
    n_on = max(1, int(round(tp.sonication_time_s / dt)))
    seg = max(1, int(round(n_on / 30)))   # ~30 samples for the curve

    _say(f"Heating: {tp.sonication_time_s:.2f} s in {n_on} steps...")
    done = 0
    while done < n_on:
        s = min(seg, n_on - done)
        solver.step(s, dt)
        done += s
        times.append(done * dt)
        focus_temp.append(float(solver.T[fi]))
        T_max = np.maximum(T_max, solver.T)

    T_end = solver.T.copy()

    if tp.cooling_time_s > 0:
        _say(f"Cooling: {tp.cooling_time_s:.2f} s (source off)...")
        solver.set_Q(np.zeros_like(Q))
        n_off = max(1, int(round(tp.cooling_time_s / dt)))
        done = 0
        while done < n_off:
            s = min(seg, n_off - done)
            solver.step(s, dt)
            done += s
            times.append(tp.sonication_time_s + done * dt)
            focus_temp.append(float(solver.T[fi]))
            T_max = np.maximum(T_max, solver.T)

    # ---- slices through the focus (use T_max for the field display) ----
    bx, by, bz = T_max.shape
    jy, jz = fi[1], fi[2]
    bone_box = box["bone_mask"]
    x_mm, y_mm, z_mm = box["x_mm"], box["y_mm"], box["z_mm"]
    extent_xz = (float(x_mm[0]), float(x_mm[-1]), float(z_mm[0]), float(z_mm[-1]))
    extent_xy = (float(x_mm[0]), float(x_mm[-1]), float(y_mm[0]), float(y_mm[-1]))

    lesion = solver.cem43 >= 240.0
    res = ThermalResult(
        T_end=T_end, T_max=T_max, cem43=solver.cem43,
        times_s=np.array(times), focus_temp_c=np.array(focus_temp),
        peak_temp_c=float(T_max.max()),
        peak_rise_c=float(T_max.max() - tp.baseline_temp_c),
        focus_end_temp_c=float(T_end[fi]),
        focus_rise_c=float(np.max(focus_temp) - tp.baseline_temp_c),
        max_cem43=float(solver.cem43.max()),
        lesion_volume_mm3=float(lesion.sum() * (dx * 1e3) ** 3),
        duty_cycle=duty,
        q_max=float(Q.max()),
        extent_xz_mm=extent_xz, extent_xy_mm=extent_xy,
        slice_T_xz=T_max[:, jy, :].T, slice_T_xy=T_max[:, :, jz].T,
        bone_xz=bone_box[:, jy, :].T, bone_xy=bone_box[:, :, jz].T,
    )
    _say(f"Peak temperature {res.peak_temp_c:.2f} C (+{res.peak_rise_c:.2f} C)")
    return res
