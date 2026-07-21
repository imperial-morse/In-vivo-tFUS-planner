"""Exposure metrics derived from the simulated focal pressure.

For a linear plane progressive harmonic wave the instantaneous intensity is
``p(t)^2 / (rho c)``. Averaging ``sin^2`` over one cycle gives 1/2, so the
pulse-average (plateau) intensity at the spatial peak is

    I_SPPA = p_peak^2 / (2 * rho * c)                                    [W/m^2]

with ``p_peak`` the pressure AMPLITUDE (not rms) and ``rho c`` the impedance of
the medium AT THE FOCUS (water/brain here, not bone).

This factor of 2 is confirmed by Constans et al., Phys. Med. Biol. 63:025003
(2018), who quote "3 W/cm^2 ISPPA at focus ... corresponding to a 0.3 MPa
pressure in the brain": (0.3e6)^2 / (2 * 1000 * 1500) = 3.0e4 W/m^2 = 3.0 W/cm^2.
Dropping the 2 would give 6 W/cm^2. See also Kinsler et al., "Fundamentals of
Acoustics", plane-progressive-wave intensity.

The temporal-average intensity at the spatial peak scales by the duty cycle:

    duty   = pulse_duration * PRF
    I_SPTA = I_SPPA * duty                                              [W/m^2]

Mechanical index (FDA/AIUM):

    MI = p_rarefactional[MPa] / sqrt(f0[MHz])

In a linear simulation the peak rarefactional (negative) pressure equals the
peak compressional one, so the recorded focal amplitude is used directly.
"""

from __future__ import annotations

import math


def isppa_w_m2(p_peak_pa: float, rho: float, c: float) -> float:
    """Spatial-peak pulse-average intensity [W/m^2] from a pressure amplitude."""
    if p_peak_pa <= 0 or rho <= 0 or c <= 0:
        return 0.0
    return float(p_peak_pa) ** 2 / (2.0 * float(rho) * float(c))


def duty_cycle(prf_hz: float, pulse_duration_s: float) -> float:
    """Fraction of time the source is on; clamped to [0, 1]."""
    return min(1.0, max(0.0, float(prf_hz) * float(pulse_duration_s)))


def ispta_w_m2(isppa: float, duty: float) -> float:
    """Spatial-peak temporal-average intensity [W/m^2]."""
    return float(isppa) * float(duty)


def mechanical_index(p_rarefactional_pa: float, f0_hz: float) -> float:
    """MI = p[MPa] / sqrt(f0[MHz])."""
    if p_rarefactional_pa <= 0 or f0_hz <= 0:
        return 0.0
    return (float(p_rarefactional_pa) / 1e6) / math.sqrt(float(f0_hz) / 1e6)


def w_m2_to_w_cm2(i_w_m2: float) -> float:
    return float(i_w_m2) / 1e4
