"""Focal-spot footprint geometry shared by every planner.

A footprint is an axis-aligned ellipse with semi-axes ``rx_mm`` and ``ry_mm``
(half of the lateral FWHM in each direction). For the isotropic case
``rx == ry``. We model the beam projected onto the focal plane as the area
where intensity >= half the peak — i.e. the FWHM contour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Footprint:
    rx_mm: float                # semi-axis along world +x (mm)
    ry_mm: float                # semi-axis along world +y (mm)

    @property
    def area_mm2(self) -> float:
        return float(np.pi * self.rx_mm * self.ry_mm)

    def contains(self, dx_mm: np.ndarray, dy_mm: np.ndarray) -> np.ndarray:
        """Vectorised inside-test for offsets relative to the footprint centre."""
        return (dx_mm / self.rx_mm) ** 2 + (dy_mm / self.ry_mm) ** 2 <= 1.0


def footprint_from_fwhm(
    fwhm_x_mm: float,
    fwhm_y_mm: float,
    mode: str = "isotropic",
    isotropic_choice: str = "min",
) -> Footprint:
    """Build a Footprint from horizontal/vertical FWHM.

    Parameters
    ----------
    fwhm_x_mm, fwhm_y_mm : float
        Lateral FWHM along the two transducer-aligned axes (mm).
    mode : {"isotropic", "anisotropic"}
        ``isotropic`` collapses the two FWHMs to a single value and uses a circle.
        ``anisotropic`` keeps both axes (ellipse).
    isotropic_choice : {"min", "max", "mean"}
        How to collapse fwhm_x and fwhm_y when ``mode == "isotropic"``.
        ``min`` is the conservative default (smaller footprint, less off-target).
    """
    if mode == "anisotropic":
        return Footprint(fwhm_x_mm / 2.0, fwhm_y_mm / 2.0)
    if mode != "isotropic":
        raise ValueError(f"unknown footprint mode {mode!r}")
    f = {
        "min": min(fwhm_x_mm, fwhm_y_mm),
        "max": max(fwhm_x_mm, fwhm_y_mm),
        "mean": 0.5 * (fwhm_x_mm + fwhm_y_mm),
    }[isotropic_choice]
    return Footprint(f / 2.0, f / 2.0)
