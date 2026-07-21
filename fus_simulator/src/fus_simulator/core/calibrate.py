"""Pressure calibration.

When we drive the transducer with some input pressure, the peak pressure that
actually appears at the focus is generally different. k-Wave acoustics are
*linear* in the source amplitude, so a single free-field run is enough to find
the exact scaling - identical in spirit to a solver-based recalibration:

    source_amp_calibrated = source_amp_ref * (p_desired / p_measured)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

from .params import SimParams
from .grid import GridSpec, build_grid_spec
from .simulate import run_freefield, FreeFieldResult


@dataclass
class CalibrationResult:
    desired_focal_pressure_mpa: float
    measured_peak_mpa: float            # peak from the reference run
    reference_amp_pa: float             # drive used for the reference run
    calibrated_amp_pa: float            # drive needed for the desired pressure
    scale_factor: float                 # calibrated / reference = desired / measured
    transfer_gain_pa_per_pa: float      # measured_peak / reference_amp (output per input)
    free_field: FreeFieldResult         # full result of the reference run (for plots)

    # optional confirmation run at the calibrated amplitude
    confirmed_peak_mpa: Optional[float] = None
    confirm_error_pct: Optional[float] = None
    confirm_free_field: Optional[FreeFieldResult] = None

    def display_field(self):
        """Pressure slices to show the user, representing the CALIBRATED drive.

        If a confirmation run was performed, use its actual field. Otherwise
        scale the reference field by the (exact, linear) scale factor so the
        displayed peak equals the desired focal pressure.

        Returns ``(slice_xz_mpa, slice_xy_mpa, xz_extent, xy_extent, peak_mpa)``.
        """
        if self.confirm_free_field is not None:
            ff = self.confirm_free_field
            return (ff.slice_xz_mpa, ff.slice_xy_mpa,
                    ff.slice_xz_extent_mm, ff.slice_xy_extent_mm,
                    ff.peak_pressure_pa / 1e6)
        ff = self.free_field
        s = self.scale_factor
        return (ff.slice_xz_mpa * s, ff.slice_xy_mpa * s,
                ff.slice_xz_extent_mm, ff.slice_xy_extent_mm,
                self.measured_peak_mpa * s)

    def summary_text(self) -> str:
        lines = [
            "=== Pressure calibration (free field, water) ===",
            f"  Desired peak focal pressure : {self.desired_focal_pressure_mpa:.4f} MPa",
            f"  Reference drive             : {self.reference_amp_pa:.4e} Pa",
            f"  Measured peak (reference)   : {self.measured_peak_mpa:.4f} MPa",
            f"  Transfer gain (out/in)      : {self.transfer_gain_pa_per_pa:.4f} Pa/Pa",
            f"  Scale factor (in adjust)    : {self.scale_factor:.4f} x",
            f"  >>> Calibrated drive        : {self.calibrated_amp_pa:.4e} Pa",
            f"      ({self.calibrated_amp_pa/1e6:.4f} MPa source amplitude)",
        ]
        if self.confirmed_peak_mpa is not None:
            peak = self.confirmed_peak_mpa
            kind = "confirmed by re-run"
        else:
            peak = self.measured_peak_mpa * self.scale_factor   # exact (linear)
            kind = "predicted from linear scaling"
        lines += [
            f"  >>> Calibrated focal peak   : {peak:.4f} MPa  ({kind})",
        ]
        if self.confirmed_peak_mpa is not None:
            lines.append(
                f"      Confirmation error      : {self.confirm_error_pct:.2f} %")
        return "\n".join(lines)

    @property
    def calibrated_focal_peak_mpa(self) -> float:
        """Peak focal pressure expected at the calibrated drive [MPa].

        Equals the confirmation-run peak if a confirmation was performed, else
        the (exact, by linearity) prediction ``measured_peak * scale_factor``.
        """
        if self.confirmed_peak_mpa is not None:
            return self.confirmed_peak_mpa
        return self.measured_peak_mpa * self.scale_factor


def calibrate(p: SimParams,
              gspec: Optional[GridSpec] = None,
              confirm: bool = False,
              progress: Optional[Callable[[str], None]] = None) -> CalibrationResult:
    """Run the reference simulation and compute the calibrated source amplitude.

    Parameters
    ----------
    p : SimParams
    gspec : optional pre-built grid spec (reused for both runs).
    confirm : if True, run a second simulation at the calibrated amplitude to
        verify the desired pressure is hit (linear theory says it will be, to
        within numerical error).
    progress : optional status callback.
    """
    if gspec is None:
        gspec = build_grid_spec(p)

    ref_amp = p.reference_source_amp_pa
    if progress:
        progress("Calibration run 1/1 (reference amplitude)...")
    ff = run_freefield(p, gspec=gspec, source_amp_pa=ref_amp, progress=progress)

    measured_pa = ff.peak_pressure_pa
    if measured_pa <= 0:
        raise RuntimeError("Measured peak pressure is zero - the focus may be "
                           "outside the sensor box or the source mask is empty.")

    desired_pa = p.desired_focal_pressure_pa
    scale = desired_pa / measured_pa
    calibrated_amp = ref_amp * scale
    gain = measured_pa / ref_amp

    result = CalibrationResult(
        desired_focal_pressure_mpa=p.desired_focal_pressure_mpa,
        measured_peak_mpa=measured_pa / 1e6,
        reference_amp_pa=ref_amp,
        calibrated_amp_pa=calibrated_amp,
        scale_factor=scale,
        transfer_gain_pa_per_pa=gain,
        free_field=ff,
    )

    if confirm:
        if progress:
            progress("Confirmation run at calibrated amplitude...")
        ff2 = run_freefield(p, gspec=gspec, source_amp_pa=calibrated_amp, progress=progress)
        result.confirmed_peak_mpa = ff2.peak_pressure_pa / 1e6
        result.confirm_error_pct = abs(
            ff2.peak_pressure_pa - desired_pa) / desired_pa * 100.0
        result.confirm_free_field = ff2

    return result
