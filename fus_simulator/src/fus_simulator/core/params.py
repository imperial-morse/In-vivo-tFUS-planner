"""User-editable parameters and fixed medium constants.

    c0   = 1500.0   # sound speed in water [m/s]
    rho0 = 1000.0   # density in water    [kg/m^3]
    alpha_coeff = 0.00217   # almost lossless water absorption [dB/(MHz^2 cm)]
    alpha_power = 2.0 (as only one frequency is used, the exponent is irrelevant, but 2 is commmonly used for monofrequency water absorption)

Everything in :class:`SimParams` is meant to be exposed in the GUI. Lengths are
stored in millimetres (what the user types); helper properties convert to SI
metres for the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict


# --------------------------------------------------------------------------- #
# Fixed medium constants (do NOT change)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MediumProps:
    sound_speed: float          # [m/s]
    density: float              # [kg/m^3]
    alpha_coeff: float          # [dB/(MHz^y cm)]
    alpha_power: float          # exponent y


# Water baseline (also used for every non-bone voxel: we model coupling water,
# not brain tissue).
#
# Absorption: pure water follows a clean square law
#     alpha(f) = 0.00217 * f[MHz]^2   [dB/cm]
# (Kinsler et al., "Fundamentals of Acoustics", 4th ed., p. 218). Note the
# exponent is exactly 2, which is precisely the ``alpha_power`` the solver runs
# with, so this coefficient is exact at every frequency - no rescaling needed.
# At 500 kHz it is 5.4e-4 dB/cm, i.e. essentially lossless, which is why the
# focus barely self-heats and the skull dominates the thermal response.
WATER = MediumProps(sound_speed=1500.0, density=1000.0,
                    alpha_coeff=0.00217, alpha_power=2.0)

# Homogeneous mouse-skull acoustic constants (binary bone/water model).
#
# Sound speed / density: Duck, "Physical Properties of Tissues" (2013), the
# generic cortical-bone entry, as tabulated in Table 3 of Constans, Mateo, Tanter
# & Aubry, Phys. Med. Biol. 63(2):025003 (2018), who applied it to rodent
# (rat/mouse) k-Wave + bioheat simulations:
#     c = 2400 m/s,  rho = 1850 kg/m^3
# NOTE none of the literature offers a mouse-MEASURED rho/c; this is a generic
# bone value. 
#
# ABSORPTION. We use the pure ABSORPTION coefficient of skull bone (Pinton et al.,
# "Attenuation, scattering, and absorption of ultrasound in the skull bone",
# Med. Phys. 39), which separates absorption from scattering. This is the input
# Q = 2*alpha*I actually wants:
#     alpha(f) = 2.7 dB/cm  *  f[MHz]^1.18        -> 1.19 dB/cm at 500 kHz
#
# The solver runs with alpha_power = 2 (kept at 2 to avoid k-Wave
# dispersion warnings), so skull_alpha_coeff() rescales the physical value into
# that exponent at the run frequency: alpha_coeff = alpha(f0) / f0[MHz]^2.
# At 500 kHz this gives 4.77 dB/(MHz^2 cm). Because the exponent is not the
# physical one, the absorption is exact at f0 and only approximate off it.
# This value should be adjusted if the user changes the source frequency. We aim 
# to add a GUI control for this in the future, but for now it is hard-coded to 500 kHz.

SKULL_SOUND_SPEED = 2400.0        # [m/s]      [Duck 2013, via Constans 2018 T3]
SKULL_DENSITY = 1850.0            # [kg/m^3]   [Duck 2013, via Constans 2018 T3]
SKULL_ALPHA_POWER = 2.0           # exponent handed to k-Wave
SKULL_ABS_DB_CM_MHZ = 2.7         # Pinton: absorption prefactor [dB/cm at 1 MHz]
SKULL_ABS_POWER = 1.18            # Pinton: physical frequency exponent


def skull_alpha_coeff(f0_hz: float,
                      abs_db_cm_mhz: float = SKULL_ABS_DB_CM_MHZ,
                      abs_power: float = SKULL_ABS_POWER,
                      alpha_power: float = SKULL_ALPHA_POWER) -> float:
    """Bone ``alpha_coeff`` [dB/(MHz^alpha_power cm)] for the k-Wave solver.

    Evaluates the physical absorption law ``alpha(f0) = a * f0^abs_power``
    [dB/cm] and re-expresses it in the exponent the solver is configured with,
    so the absorption is correct at the run frequency ``f0``.
    """
    fm = float(f0_hz) / 1e6
    if fm <= 0:
        return 0.0
    alpha_dbcm = abs_db_cm_mhz * fm ** abs_power       # physical dB/cm at f0
    return alpha_dbcm / fm ** alpha_power              # -> dB/(MHz^y cm)


def skull_props(f0_hz: float) -> MediumProps:
    """Homogeneous skull :class:`MediumProps` at a given source frequency."""
    return MediumProps(sound_speed=SKULL_SOUND_SPEED, density=SKULL_DENSITY,
                       alpha_coeff=skull_alpha_coeff(f0_hz),
                       alpha_power=SKULL_ALPHA_POWER)

# NOTE: skull thermal properties (k, Cp) live in core/thermal.py.


# --------------------------------------------------------------------------- #
# User-editable simulation parameters
# --------------------------------------------------------------------------- #
@dataclass
class SimParams:
    # ---- Source / transducer geometry (the section the user controls) ----
    frequency_hz: float = 500e3          # source centre frequency f0 [Hz]
    aperture_diameter_mm: float = 64.0   # bowl aperture diameter [mm]
    focal_length_mm: float = 63.2        # radius of curvature (ROC = focal length) [mm]

    hole_enabled: bool = False           # model a central hole (annular aperture)?
    hole_diameter_mm: float = 40.0       # central hole diameter [mm] (when enabled)

    desired_focal_pressure_mpa: float = 0.35   # target peak focal pressure [MPa]
    # Reference drive used for the single calibration run. k-Wave acoustics are
    # linear in source amplitude, so the value only needs to be in the linear
    # regime; the result is rescaled to the desired pressure afterwards.
    reference_source_amp_pa: float = 1.0e5     # [Pa] = 0.1 MPa

    # ---- Grid / numerics ----
    points_per_wavelength: int = 8       # PPW 
    cfl: float = 0.05                    # Courant number for dt design
    num_cycles: int = 60                 # tone-burst cycles (100 is also common)

    # ---- Sensor box (region that will later hold the skull, + buffer) ----
    # Cuboid centred on the geometric focus. Smaller box => faster post-proc and
    # less memory than recording p_max over the whole grid.
    skull_box_lr_mm: float = 22.0        # box size across beam (lateral / y) [mm]
    skull_box_ap_mm: float = 22.0        # box size across beam (elevation / z) [mm]
    skull_box_axial_mm: float = 24.0     # box size along beam (axial / x) [mm]

    # ---- Grid buffers (space added so the focus & aperture always fit) ----
    back_gap_mm: float = 6.0             # gap behind the transducer apex [mm]
    lateral_margin_mm: float = 8.0       # margin beyond aperture / box laterally [mm]
    front_margin_mm: float = 6.0         # margin beyond focus / box downstream [mm]

    # ---- Execution ----
    compute_mode: str = "auto"           # "auto" | "gpu" | "cpu"

    # --------------------------------------------------------- #
    @property
    def f0(self) -> float:
        return float(self.frequency_hz)

    @property
    def aperture_diameter_m(self) -> float:
        return self.aperture_diameter_mm * 1e-3

    @property
    def focal_length_m(self) -> float:
        return self.focal_length_mm * 1e-3

    @property
    def hole_diameter_m(self) -> float:
        return self.hole_diameter_mm * 1e-3 if self.hole_enabled else 0.0

    @property
    def desired_focal_pressure_pa(self) -> float:
        return self.desired_focal_pressure_mpa * 1e6

    def grid_spacing_m(self) -> float:
        """dx = c0 / (f0 * PPW)."""
        return WATER.sound_speed / (self.f0 * self.points_per_wavelength)

    def to_dict(self) -> Dict:
        return asdict(self)

    def validate(self) -> None:
        """Raise ValueError on physically impossible inputs."""
        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be positive.")
        if self.aperture_diameter_mm <= 0:
            raise ValueError("Aperture diameter must be positive.")
        if self.focal_length_mm <= 0:
            raise ValueError("Focal length (radius of curvature) must be positive.")
        if self.focal_length_mm < self.aperture_diameter_mm / 2.0:
            raise ValueError(
                "Focal length (ROC) must be >= aperture radius "
                f"({self.aperture_diameter_mm/2:.1f} mm); a spherical cap cannot "
                "have a radius of curvature smaller than its own radius.")
        if self.hole_enabled and self.hole_diameter_mm >= self.aperture_diameter_mm:
            raise ValueError("Central hole diameter must be smaller than the aperture.")
        if self.points_per_wavelength < 4:
            raise ValueError("Points-per-wavelength can't be this small.")
        if not (0 < self.cfl < 1):
            raise ValueError("CFL must be between 0 and 1.")
        if self.desired_focal_pressure_mpa <= 0:
            raise ValueError("Desired focal pressure must be positive.")
