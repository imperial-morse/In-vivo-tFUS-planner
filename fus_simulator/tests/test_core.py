"""Unit tests for the pure-NumPy core (no k-Wave / GPU needed).

Run from the package root:
    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fus_simulator.core.params import SimParams, WATER
from fus_simulator.core.grid import build_grid_spec, round_even
from fus_simulator.core.transducer import bowl_arc_xz, apply_central_hole


def test_grid_spacing_matches_formula():
    p = SimParams()
    g = build_grid_spec(p)
    assert np.isclose(g.dx, WATER.sound_speed / (p.f0 * p.points_per_wavelength))


def test_grid_dims_even():
    g = build_grid_spec(SimParams())
    assert g.Nx % 2 == 0 and g.Ny % 2 == 0 and g.Nz % 2 == 0


def test_focus_is_apex_plus_roc_and_inside_grid():
    p = SimParams()
    g = build_grid_spec(p)
    assert np.isclose(g.focus_xyz[0] - g.apex_xyz[0], p.focal_length_m, atol=g.dx)
    assert g.focus_in_grid()


def test_focus_inside_sensor_box():
    g = build_grid_spec(SimParams())
    fix = int(np.argmin(np.abs(g.x_vec - g.focus_xyz[0])))
    assert g.box_ix[0] <= fix <= g.box_ix[1]


def test_sensor_box_smaller_than_grid():
    g = build_grid_spec(SimParams())
    assert g.box_n_points < g.n_points       # box is a subset -> faster


def test_central_hole_removes_axis_points():
    p = SimParams(hole_enabled=True, hole_diameter_mm=10.0)
    g = build_grid_spec(p)
    mask = np.zeros((g.Nx, g.Ny, g.Nz), bool)
    cy, cz = g.Ny // 2, g.Nz // 2
    mask[2, cy - 20:cy + 20, cz - 20:cz + 20] = True
    out = apply_central_hole(mask, g, p.hole_diameter_m)
    assert out.sum() < mask.sum()
    assert not out[2, cy, cz]                # axis voxel removed


def test_bowl_arc_rim_matches_aperture():
    p = SimParams()
    g = build_grid_spec(p)
    _, z = bowl_arc_xz(g, p)
    assert np.isclose(np.nanmax(np.abs(z)), p.aperture_diameter_m / 2, atol=g.dx)


def test_arc_has_gap_when_hole_enabled():
    p = SimParams(hole_enabled=True, hole_diameter_mm=40.0)
    g = build_grid_spec(p)
    x, _ = bowl_arc_xz(g, p)
    assert np.isnan(x).any()


@pytest.mark.parametrize("bad", [
    dict(frequency_hz=-1),
    dict(focal_length_mm=10, aperture_diameter_mm=64),       # ROC < aperture radius
    dict(hole_enabled=True, hole_diameter_mm=70, aperture_diameter_mm=64),
])
def test_validation_rejects_bad_geometry(bad):
    with pytest.raises(ValueError):
        SimParams(**bad).validate()


def test_calibration_is_linear_rescale(monkeypatch):
    """calibrated_amp should equal desired / transfer_gain."""
    import fus_simulator.core.calibrate as cal
    from fus_simulator.core.simulate import FreeFieldResult

    def fake_run(p, gspec=None, source_amp_pa=None, progress=None):
        if gspec is None:
            gspec = build_grid_spec(p)
        amp = source_amp_pa if source_amp_pa is not None else p.reference_source_amp_pa
        return FreeFieldResult(
            peak_pressure_pa=0.8 * amp, source_amp_pa=amp, gspec=gspec,
            box_field_mpa=np.zeros((2, 2, 2)),
            slice_xz_mpa=np.zeros((2, 2)), slice_xy_mpa=np.zeros((2, 2)),
            slice_xz_extent_mm=(0, 1, 0, 1), slice_xy_extent_mm=(0, 1, 0, 1),
            solver_msg="mock", used_gpu=False, dt=1e-7, Nt=10,
            n_source_points=100, runtime_s=0.0)

    monkeypatch.setattr(cal, "run_freefield", fake_run)
    p = SimParams(desired_focal_pressure_mpa=0.35, reference_source_amp_pa=1.0e5)
    r = cal.calibrate(p, confirm=True)
    assert np.isclose(r.transfer_gain_pa_per_pa, 0.8)
    assert np.isclose(r.calibrated_amp_pa, 0.35e6 / 0.8)
    assert np.isclose(r.confirmed_peak_mpa, 0.35)
