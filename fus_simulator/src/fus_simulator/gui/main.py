"""PyQt5 main window for the FUS simulator.

Single window, four tabs:

    1. Settings    - default file paths (CT / prepared skull, atlas centroids,
                     plan and figure folders), saved to disk and restored on the
                     next launch so paths only have to be picked once.
    2. Parameters  - transducer geometry (frequency, aperture, focal length,
                     optional central hole) and the sensor box, plus the skull
                     section: build the bone medium straight from a CT (see
                     core/ct_skull.py) and optionally aim the focus at a named
                     brain region from the DMBA atlas (core/atlas.py). Shows a
                     live preview of the skull with the transducer overlaid.
    3. Simulation  - calibrate in water for a target focal pressure, then run
                     one k-Wave sim through the skull. Reports the through-skull
                     pressure field (three orthogonal planes), transmission, and
                     the exposure metrics I_SPPA / I_SPTA / MI (core/metrics.py).
                     Below it, a thermal section solves Pennes' bioheat equation
                     for temperature rise and CEM43 dose.
    4. Protocol    - import a fus_planner plan CSV and accumulate heating over
                     the spots: a fast brain-only mode, or an accurate per-spot
                     acoustic mode that runs one sim per spot through the skull
                     (reusing the drive calibrated on the Simulation tab).

"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse

try:
    from ..core.params import SimParams, WATER
    from ..core.grid import build_grid_spec
    from ..core.transducer import bowl_arc_xz, bowl_surface_3d
    from ..core.gpu import detect_gpu
    from .skull_window import SkullWidget
    from . import settings
    from .viz import (StaticCanvas, draw_bowl_mesh,
                      set_ortho_static_view, set_equal_box_aspect)
    from ..core.thermal import run_thermal, ThermalParams
    from ..core.plan_io import parse_plan_csv, build_and_run_plan, PlanProtocolParams
except ImportError:  # pragma: no cover - script-style fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fus_simulator.core.params import SimParams, WATER
    from fus_simulator.core.grid import build_grid_spec
    from fus_simulator.core.transducer import bowl_arc_xz, bowl_surface_3d
    from fus_simulator.core.gpu import detect_gpu
    from fus_simulator.gui.skull_window import SkullWidget
    from fus_simulator.gui import settings
    from fus_simulator.gui.viz import (StaticCanvas, draw_bowl_mesh,
                                       set_ortho_static_view, set_equal_box_aspect)
    from fus_simulator.core.thermal import run_thermal, ThermalParams
    from fus_simulator.core.plan_io import parse_plan_csv, build_and_run_plan, PlanProtocolParams


# --------------------------------------------------------------------------- #
# Worker thread
# --------------------------------------------------------------------------- #
class CalibrationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, params: SimParams, confirm: bool):
        super().__init__()
        self._params = params
        self._confirm = confirm

    def run(self):
        try:
            from ..core.calibrate import calibrate
        except ImportError:
            from fus_simulator.core.calibrate import calibrate
        try:
            result = calibrate(self._params, confirm=self._confirm,
                               progress=self.progress.emit)
            self.finished_ok.emit(result)
        except ModuleNotFoundError as err:
            self.failed.emit(
                f"Could not import the solver ({err}).\n\n"
                "k-wave-python is not installed in this environment. Install it "
                "(pip install -r requirements-solver.txt) and try again. The "
                "geometry preview and skull viewer work without it.")
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"{err}\n\n{traceback.format_exc()}")


class SkullSimWorker(QtCore.QThread):
    """One press = calibrate in water for the CURRENT parameters, then run the
    simulation through the embedded skull at that calibrated drive. The
    calibration is used only to get the drive (and a free-field field to plot);
    nothing from it is persisted."""
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)     # dict(calib=..., skull=...)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, params: SimParams, gspec, emb):
        super().__init__()
        self._params = params
        self._gspec = gspec
        self._emb = emb

    def run(self):
        try:
            from ..core.calibrate import calibrate
            from ..core.simulate import run_simulation
        except ImportError:
            from fus_simulator.core.calibrate import calibrate
            from fus_simulator.core.simulate import run_simulation
        try:
            self.progress.emit("Step 1/2: calibrating in water...")
            calib = calibrate(self._params, gspec=self._gspec, confirm=False,
                              progress=self.progress.emit)
            self.progress.emit("Step 2/2: simulating through skull...")
            skull_res = run_simulation(
                self._params, gspec=self._gspec,
                source_amp_pa=calib.calibrated_amp_pa, skull=self._emb,
                label="through skull", progress=self.progress.emit)
            self.finished_ok.emit({"calib": calib, "skull": skull_res})
        except ModuleNotFoundError as err:
            self.failed.emit(
                f"Could not import the solver ({err}).\n\n"
                "Install k-wave-python (pip install -r requirements-solver.txt).")
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"{err}\n\n{traceback.format_exc()}")


class ThermalWorker(QtCore.QThread):
    """Run the bioheat simulation on the cached pressure box (pure NumPy)."""
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)     # ThermalResult
    failed = QtCore.pyqtSignal(str)

    def __init__(self, box, pressure_box_pa, tparams, f0):
        super().__init__()
        self._box = box
        self._p = pressure_box_pa
        self._tp = tparams
        self._f0 = f0

    def run(self):
        try:
            res = run_thermal(self._box, self._p, self._tp, self._f0,
                             progress=self.progress.emit)
            self.finished_ok.emit(res)
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"{err}\n\n{traceback.format_exc()}")


class ProtocolPlanWorker(QtCore.QThread):
    """Run the multi-spot plan thermal accumulation (pure NumPy)."""
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)     # ProtocolResult
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spots, pp, tp):
        super().__init__()
        self._spots = spots
        self._pp = pp
        self._tp = tp

    def run(self):
        try:
            res = build_and_run_plan(self._spots, self._pp, self._tp,
                                     progress=self.progress.emit)
            self.finished_ok.emit(res)
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"{err}\n\n{traceback.format_exc()}")


class AcousticPlanWorker(QtCore.QThread):
    """Per-spot acoustic protocol: one k-Wave run per spot through the skull."""
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spots, sim_params, skull, drive_pa, pp, tp,
                 rotate_cw90, base_offset_mm, fill_threshold=0.30):
        super().__init__()
        self._spots = spots; self._sp = sim_params; self._skull = skull
        self._drive = drive_pa; self._pp = pp; self._tp = tp
        self._rot = rotate_cw90; self._base = base_offset_mm
        self._fill = fill_threshold

    def run(self):
        try:
            from ..core.plan_io import build_and_run_plan_acoustic
        except ImportError:
            from fus_simulator.core.plan_io import build_and_run_plan_acoustic
        try:
            res = build_and_run_plan_acoustic(
                self._spots, self._sp, self._skull, self._drive, self._pp, self._tp,
                rotate_cw90=self._rot, base_offset_mm=self._base,
                fill_threshold=self._fill, progress=self.progress.emit)
            self.finished_ok.emit(res)
        except ModuleNotFoundError as err:
            self.failed.emit(f"Solver not installed ({err}). Install k-wave-python.")
        except Exception as err:  # noqa: BLE001
            self.failed.emit(f"{err}\n\n{traceback.format_exc()}")


# --------------------------------------------------------------------------- #
# Spinbox helpers
# --------------------------------------------------------------------------- #
def _dspin(lo, hi, val, decimals=2, step=1.0, suffix=""):
    sb = QtWidgets.QDoubleSpinBox()
    sb.setRange(lo, hi); sb.setDecimals(decimals); sb.setSingleStep(step)
    sb.setValue(val)
    if suffix:
        sb.setSuffix(suffix)
    return sb


def _ispin(lo, hi, val, step=1, suffix=""):
    sb = QtWidgets.QSpinBox()
    sb.setRange(lo, hi); sb.setSingleStep(step); sb.setValue(val)
    if suffix:
        sb.setSuffix(suffix)
    return sb


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUS Simulator (k-Wave)")
        self.resize(1320, 860)
        self._worker = None
        # We do NOT keep the calibration result object - only the drive value it
        # produced (and the target pressure), recomputed every skull run.
        self._last_drive_pa = None        # calibrated drive amplitude [Pa]
        self._last_desired_mpa = None     # target focal pressure [MPa]
        self._last_box = None             # cached pressure-box maps for thermal
        self._last_pressure_box_pa = None # cached peak pressure field (box) [Pa]
        self._plan_spots = None           # imported plan spots

        # run log: a timestamped record of every run's steps + result, written
        # to fus_sim_run.log next to run_gui.py so we can verify what happened.
        import datetime as _dt
        self._runlog_path = Path(__file__).resolve().parents[3] / "fus_sim_run.log"
        try:
            with open(self._runlog_path, "a") as f:
                f.write(f"\n===== app start {_dt.datetime.now()} =====\n")
        except Exception:
            self._runlog_path = None

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # skull widget is embedded in the Parameters tab, so build it first
        self.skull_widget = SkullWidget(self._gather_params)
        # Settings first: the saved CT path is pushed into the Parameters tab
        # before the user ever gets there.
        self.tabs.addTab(self._build_tab_settings(), "1. Settings")
        self.tabs.addTab(self._build_tab_parameters(), "2. Parameters")
        self.tabs.addTab(self._build_tab_simulation(), "3. Simulation")
        self.tabs.addTab(self._build_tab_protocol(), "4. Protocol")
        # applied after every tab exists (plan_path lives on the Protocol tab)
        self._apply_saved_paths()

        self._wire_signals()
        self._refresh_gpu_label()
        self.skull_widget.schedule_redraw()   # initial draw once all widgets exist

    # ===================================================== Tab 1: Parameters
    def _build_tab_parameters(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # ---- 1) transducer settings ----
        v.addWidget(self._section("1)  Transducer settings"))
        top = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(top)
        h.setContentsMargins(0, 0, 0, 0)

        g1 = QtWidgets.QGroupBox("Transducer & source"); f1 = QtWidgets.QFormLayout(g1)
        self.freq = _dspin(20.0, 5000.0, 500.0, 1, 10.0, " kHz")
        self.aperture = _dspin(1.0, 200.0, 64.0, 2, 1.0, " mm")
        self.focal = _dspin(1.0, 300.0, 63.2, 2, 1.0, " mm")
        f1.addRow("Frequency f0:", self.freq)
        f1.addRow("Aperture diameter:", self.aperture)
        f1.addRow("Focal length (ROC):", self.focal)
        h.addWidget(g1)

        g2 = QtWidgets.QGroupBox("Central hole (annular)"); f2 = QtWidgets.QFormLayout(g2)
        self.hole_on = QtWidgets.QCheckBox("Model a central hole")
        self.hole_d = _dspin(0.0, 199.0, 40.0, 2, 1.0, " mm"); self.hole_d.setEnabled(False)
        self.hole_on.toggled.connect(self.hole_d.setEnabled)
        f2.addRow(self.hole_on)
        f2.addRow("Hole diameter:", self.hole_d)
        h.addWidget(g2)

        g4 = QtWidgets.QGroupBox("Sensor box (skull region + buffer)"); f4 = QtWidgets.QFormLayout(g4)
        self.box_ax = _dspin(2.0, 120.0, 24.0, 1, 1.0, " mm")
        self.box_lr = _dspin(2.0, 120.0, 22.0, 1, 1.0, " mm")
        self.box_ap = _dspin(2.0, 120.0, 22.0, 1, 1.0, " mm")
        f4.addRow("Axial (along beam):", self.box_ax)
        f4.addRow("Lateral LR:", self.box_lr)
        f4.addRow("Lateral AP:", self.box_ap)
        h.addWidget(g4)
        h.addStretch(1)
        v.addWidget(top)

        # ---- 2) skull (below the transducer settings) ----
        v.addWidget(self._section("2)  Skull"))
        v.addWidget(self.skull_widget, 1)
        return w

    def _update_duty_label(self, *_):
        duty = min(1.0, max(0.0, self.t_prf.value() * self.t_pulse.value() * 1e-3))
        self.t_duty.setText(f"{duty*100:.1f} %   (PRF x pulse)")

    # ===================================================== Tab 5: Protocol
    def _build_tab_protocol(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter)
        left = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(left)

        # bold section header, same style as the Parameters / Simulation tabs
        v.addWidget(self._section("Plan (from fus_planner)"))
        gi = QtWidgets.QGroupBox(); fi = QtWidgets.QVBoxLayout(gi)
        row = QtWidgets.QHBoxLayout()
        self.plan_path = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton("Browse..."); b.clicked.connect(self._browse_plan)
        ld = QtWidgets.QPushButton("Load plan")
        ld.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #e74c3c; }")
        ld.clicked.connect(self._load_plan)
        row.addWidget(self.plan_path, 1); row.addWidget(b); row.addWidget(ld)
        fi.addLayout(row)
        self.plan_info = QtWidgets.QLabel("Import a plan CSV (focus coords per spot).")
        self.plan_info.setWordWrap(True); fi.addWidget(self.plan_info)
        v.addWidget(gi)

        gmode = QtWidgets.QGroupBox("Skull model"); fmode = QtWidgets.QFormLayout(gmode)
        self.pr_mode = QtWidgets.QComboBox()
        self.pr_mode.addItems(["Brain-only (fast, no skull)",
                               "Per-spot acoustic (skull, slow)"])
        self.pr_mode.setToolTip(
            "Brain-only: fast footprint approximation, ignores skull heating.\n"
            "Per-spot acoustic: one k-Wave run per spot through the loaded skull "
            "(needs a loaded skull + calibration; much slower).")
        fmode.addRow("Mode:", self.pr_mode)
        v.addWidget(gmode)

        ga = QtWidgets.QGroupBox("Acoustic / pulsing"); fa = QtWidgets.QFormLayout(ga)
        self.pr_pressure = _dspin(0.001, 20.0, 0.35, 3, 0.05, " MPa")
        self.pr_alpha = _dspin(0.0, 20.0, 0.5, 3, 0.05, " dB/MHz^y/cm")
        self.pr_prf = _dspin(0.1, 100000.0, 1000.0, 1, 100.0, " Hz")
        self.pr_pulse = _dspin(0.001, 1000.0, 0.2, 3, 0.05, " ms")
        self.pr_duty = QtWidgets.QLabel("-")
        fa.addRow("Focal pressure:", self.pr_pressure)
        fa.addRow("Tissue absorption:", self.pr_alpha)
        fa.addRow("PRF:", self.pr_prf)
        fa.addRow("Pulse duration:", self.pr_pulse)
        fa.addRow("Duty cycle:", self.pr_duty)
        v.addWidget(ga)

        gt = QtWidgets.QGroupBox("Sequence timing"); ft = QtWidgets.QFormLayout(gt)
        self.pr_on = _dspin(0.01, 600.0, 6.0, 2, 0.5, " s")
        self.pr_cool = _dspin(0.0, 600.0, 3.0, 2, 0.5, " s")
        ft.addRow("Sonication per spot:", self.pr_on)
        ft.addRow("Cool-down between spots:", self.pr_cool)
        v.addWidget(gt)

        gm = QtWidgets.QGroupBox("Thermal / grid"); fm = QtWidgets.QFormLayout(gm)
        self.pr_base = _dspin(0.0, 45.0, 37.0, 1, 0.5, " degC")
        self.pr_perf = _dspin(0.0, 1.0, 0.0, 4, 0.001, " 1/s")
        self.pr_dt = _dspin(0.01, 1.0, 0.2, 3, 0.05, " s")
        self.pr_axial = _dspin(0.5, 40.0, 6.0, 1, 0.5, " mm")
        self.pr_dx = _dspin(0.1, 1.0, 0.3, 2, 0.05, " mm")
        fm.addRow("Baseline temperature:", self.pr_base)
        fm.addRow("Perfusion coeff:", self.pr_perf)
        fm.addRow("Thermal time step:", self.pr_dt)
        fm.addRow("Focal depth (axial FWHM):", self.pr_axial)
        fm.addRow("Grid spacing:", self.pr_dx)
        v.addWidget(gm)

        for sb in (self.pr_prf, self.pr_pulse):
            sb.valueChanged.connect(self._update_pr_duty)

        self.run_proto_btn = QtWidgets.QPushButton("Run protocol thermal")
        self.run_proto_btn.clicked.connect(self._on_run_protocol)
        v.addWidget(self.run_proto_btn)
        self.proto_progress = QtWidgets.QProgressBar()
        self.proto_progress.setRange(0, 1); self.proto_progress.setValue(0); self.proto_progress.setFormat("idle")
        v.addWidget(self.proto_progress)
        self.proto_status = QtWidgets.QLabel("Load a plan, set parameters, run.")
        self.proto_status.setWordWrap(True); v.addWidget(self.proto_status)
        self.proto_text = QtWidgets.QPlainTextEdit(); self.proto_text.setReadOnly(True)
        self.proto_text.setStyleSheet("font-family: monospace;"); v.addWidget(self.proto_text, 1)

        right = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right)
        prow = QtWidgets.QHBoxLayout(); prow.addStretch(1)
        bpr = QtWidgets.QPushButton("Save figure..."); bpr.setMaximumWidth(140)
        bpr.clicked.connect(lambda: self._save_figure(self.proto_fig, "protocol"))
        prow.addWidget(bpr); rv.addLayout(prow)
        self.proto_fig = Figure(figsize=(9, 7))
        self.proto_canvas = FigureCanvas(self.proto_fig)
        rv.addWidget(self.proto_canvas)

        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 850])
        self._update_pr_duty()
        return w

    def _update_pr_duty(self, *_):
        duty = min(1.0, max(0.0, self.pr_prf.value() * self.pr_pulse.value() * 1e-3))
        self.pr_duty.setText(f"{duty*100:.1f} %   (PRF x pulse)")

    def _browse_plan(self):
        start = os.path.dirname(self.plan_path.text().strip()) or settings.get("plan_dir", "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select plan CSV", start, "CSV files (*.csv);;All files (*.*)")
        if path:
            self.plan_path.setText(path); self._load_plan()

    def _load_plan(self):
        path = self.plan_path.text().strip()
        if not path:
            return
        try:
            self._plan_spots = parse_plan_csv(path)
        except Exception as err:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Plan load failed", str(err)); return
        # prefill focal pressure from the last skull-run target, if available
        if self._last_desired_mpa is not None:
            self.pr_pressure.setValue(self._last_desired_mpa)
        coords = ", ".join(f"{s.spot_id}:({s.focus_mm[0]:.1f},{s.focus_mm[1]:.1f},{s.focus_mm[2]:.1f})"
                           for s in self._plan_spots)
        note = ("  Focal pressure prefilled from last skull run."
                if self._last_desired_mpa is not None else
                "  (Run a skull simulation first, or set focal pressure manually.)")
        self.plan_info.setText(f"Loaded {len(self._plan_spots)} spots: {coords}.{note}")

    def _on_run_protocol(self):
        if not self._plan_spots:
            QtWidgets.QMessageBox.information(self, "No plan", "Load a plan CSV first.")
            return
        duty = min(1.0, max(0.0, self.pr_prf.value() * self.pr_pulse.value() * 1e-3))
        pp = PlanProtocolParams(
            dx_mm=self.pr_dx.value(), axial_fwhm_mm=self.pr_axial.value(),
            focal_pressure_pa=self.pr_pressure.value() * 1e6,
            tissue_alpha_db=self.pr_alpha.value(), alpha_power=WATER.alpha_power,
            f0_hz=self._gather_params().f0, duty_cycle=duty,
            on_time_s=self.pr_on.value(), cooldown_s=self.pr_cool.value())
        tp = ThermalParams(baseline_temp_c=self.pr_base.value(),
                           dt_thermal_s=self.pr_dt.value(),
                           perfusion_rate=self.pr_perf.value(),
                           blood_temp_c=self.pr_base.value())

        per_spot = self.pr_mode.currentIndex() == 1
        if per_spot:
            if self._last_drive_pa is None:
                QtWidgets.QMessageBox.information(
                    self, "Run skull sim first",
                    "Per-spot acoustic mode reuses the drive from a skull run. "
                    "Run the simulation on the Simulation tab first.")
                return
            if not self.skull_widget.has_skull():
                QtWidgets.QMessageBox.information(
                    self, "No skull",
                    "Per-spot acoustic mode needs a skull loaded on the Skull tab.")
                self.tabs.setCurrentWidget(self.skull_widget); self.skull_widget.on_shown()
                return
            sw = self.skull_widget
            base = (sw.off_ax.value(), sw.off_lr.value(), sw.off_ap.value())
            self._worker = AcousticPlanWorker(
                self._plan_spots, self._gather_params(), sw._skull,
                self._last_drive_pa, pp, tp,
                True, base,
                fill_threshold=float(sw.fill_thr.value()))
        else:
            self._worker = ProtocolPlanWorker(self._plan_spots, pp, tp)

        self.run_proto_btn.setEnabled(False)
        self.proto_text.clear()
        self.proto_status.setText(
            "Running per-spot acoustic protocol (slow)..." if per_spot
            else "Running brain-only protocol...")
        self.proto_progress.setRange(0, 0); self.proto_progress.setFormat("running...")
        self._worker.progress.connect(self._on_proto_progress)
        self._worker.finished_ok.connect(self._on_protocol_done)
        self._worker.failed.connect(self._on_protocol_failed)
        self._worker.start()

    def _on_proto_progress(self, msg):
        self.proto_status.setText(msg)
        self.proto_text.appendPlainText(msg)
        self._runlog("protocol: " + msg)
        self.proto_progress.setFormat(msg if len(msg) <= 48 else msg[:45] + "...")

    def _proto_reset(self):
        self.run_proto_btn.setEnabled(True)
        self.proto_progress.setRange(0, 1); self.proto_progress.setValue(0)
        self.proto_progress.setFormat("idle")

    def _on_protocol_failed(self, msg):
        self._proto_reset()
        self.proto_status.setText("Protocol failed.")
        self.proto_text.appendPlainText("\n[ERROR]\n" + msg)
        self._runlog("PROTOCOL FAILED: " + msg.splitlines()[0])
        QtWidgets.QMessageBox.critical(self, "Protocol failed", msg.split("\n\n")[0])

    def _on_protocol_done(self, res):
        self._proto_reset()
        txt = [
            "=== Multi-spot protocol thermal ===",
            f"  Spots                 : {len(res.meta.get('spot_ids', []))}",
            f"  Duty cycle            : {res.meta.get('duty', 0)*100:.1f} %",
            f"  Peak temperature      : {res.peak_temp_c:.3f} degC (+{res.peak_rise_c:.3f})",
            f"  Overlap volume (>1)   : {res.overlap_volume_mm3:.4f} mm^3 ({res.overlap_voxels} voxels)",
            f"  Overlap-region peak   : {res.overlap_max_temp_c:.3f} degC",
            f"  Max thermal dose CEM43: {res.max_cem43:.3e} min",
            f"  Lesion volume (>=240) : {res.lesion_volume_mm3:.4f} mm^3",
        ]
        self.proto_status.setText(
            f"Done. Peak {res.peak_temp_c:.3f} degC (+{res.peak_rise_c:.3f}); "
            f"{res.overlap_volume_mm3:.4f} mm3 overlap.")
        self.proto_text.appendPlainText("\n" + "\n".join(txt))
        self._plot_protocol(res)

    def _plot_protocol(self, res):
        ids = res.meta.get("spot_ids", [])
        sp = res.meta.get("spots_mm", [])
        fp = res.meta.get("footprints_mm", [(0.75, 0.75)] * len(sp))
        skz = res.meta.get("skull_z_mm", [])

        def _draw_spots(ax, edge):
            for s, i, (rx, ry) in zip(sp, ids, fp):
                ax.add_patch(Ellipse((s[0], s[1]), 2 * rx, 2 * ry, fill=False,
                                     edgecolor=edge, lw=1.4))
                ax.annotate(str(i), (s[0], s[1]), color=edge, fontsize=9,
                            ha="center", va="center")

        def _draw_bone(ax, bone2d, ext, color="white"):
            # bone mask is present only in per-spot acoustic mode (skull); in
            # brain-only mode it is all zeros and nothing is drawn.
            bone2d = np.asarray(bone2d)
            if bone2d.size == 0 or bone2d.max() <= 0:
                return
            nr, nc = bone2d.shape
            hh = np.linspace(ext[0], ext[1], nc)
            vv = np.linspace(ext[2], ext[3], nr)
            ax.contour(hh, vv, bone2d, levels=[0.5], colors=color, linewidths=0.8)

        fig = self.proto_fig
        fig.clear()
        ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3); ax4 = fig.add_subplot(2, 2, 4)
        vmin = float(self.pr_base.value())
        vmax = float(max(res.slice_T_xy.max(), res.slice_T_xz.max(), vmin + 0.01))

        im = ax1.imshow(res.slice_T_xy, origin="lower", extent=res.extent_xy_mm,
                        cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="degC")
        _draw_bone(ax1, res.bone_xy, res.extent_xy_mm)
        _draw_spots(ax1, "cyan")
        ax1.set_title("Cumulative max T (focal plane) + plan", pad=16); ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")

        im = ax2.imshow(res.slice_T_xz, origin="lower", extent=res.extent_xz_mm,
                        cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="degC")
        # skull boundary from the plan (inner / outer skull z, beam axis = z)
        sk = np.array([s for s in skz if np.all(np.isfinite(s))]) if skz else np.empty((0, 3))
        if sk.size:
            z_inner = float(np.nanmean(sk[:, 1])); z_outer = float(np.nanmean(sk[:, 2]))
            ax2.axhline(z_outer, color="cyan", lw=1.2, ls="-", label="outer skull")
            ax2.axhline(z_inner, color="cyan", lw=1.2, ls="--", label="inner skull")
            ax2.legend(fontsize=7, loc="lower right")
        _draw_bone(ax2, res.bone_xz, res.extent_xz_mm)
        ax2.set_title("Cumulative max T (x-z) + skull", pad=16); ax2.set_xlabel("x [mm]"); ax2.set_ylabel("z [mm]")

        im = ax3.imshow(res.overlap_xy, origin="lower", extent=res.extent_xy_mm,
                        cmap="viridis", aspect="equal")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="# spots")
        _draw_bone(ax3, res.bone_xy, res.extent_xy_mm, color="red")
        _draw_spots(ax3, "red")
        ax3.set_title("Spatial overlap (>1 spot) + plan", pad=16); ax3.set_xlabel("x [mm]"); ax3.set_ylabel("y [mm]")

        for k in range(len(res.spot_traces_c)):
            lab = ids[k] if k < len(ids) else str(k + 1)
            ax4.plot(res.times_s, res.spot_traces_c[k], label=f"spot {lab}")
        for (a, b) in res.spot_windows_s:
            ax4.axvspan(a, b, color="orange", alpha=0.07)
        ax4.set_xlabel("time [s]"); ax4.set_ylabel("focus T [degC]")
        ax4.set_title("Temperature at each spot vs time", pad=16); ax4.legend(fontsize=8)
        fig.suptitle(f"Protocol: peak {res.peak_temp_c:.2f} degC "
                     f"(+{res.peak_rise_c:.2f}), {res.overlap_volume_mm3:.3f} mm3 overlap", y=0.99)
        # more right margin (right colorbar label was clipped), more vertical gap
        # (bottom plots were clashing), and titles sit a bit higher via top/hspace
        fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.10,
                            wspace=0.45, hspace=0.55)
        self.proto_canvas.draw_idle()

    # ===================================================== Tab 4: Settings
    def _build_tab_settings(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(self._section("Default paths"))
        v.addWidget(QtWidgets.QLabel(
            "Set the files/folders this app should use automatically at every launch.\n"
            "Saved defaults are restored on start, so you only pick them once."))

        form = QtWidgets.QFormLayout()

        def _row(filt, is_dir=False):
            box = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            edit = QtWidgets.QLineEdit(); h.addWidget(edit, 1)
            b = QtWidgets.QPushButton("Browse...")

            def _browse():
                if is_dir:
                    p = QtWidgets.QFileDialog.getExistingDirectory(
                        self, "Select folder", edit.text().strip())
                else:
                    p, _ = QtWidgets.QFileDialog.getOpenFileName(
                        self, "Select file", edit.text().strip(), filt)
                if p:
                    edit.setText(p)
            b.clicked.connect(_browse)
            h.addWidget(b)
            return box, edit

        ct_box, self.set_ct = _row("CT or prepared skull (*.nhdr *.nrrd *.h5 *.hdf5)")
        atlas_box, self.set_atlas = _row("Centroid table (*.txt *.csv);;All files (*.*)")
        plan_box, self.set_plan_dir = _row("", is_dir=True)
        fig_box, self.set_fig_dir = _row("", is_dir=True)
        form.addRow("CT / prepared skull:", ct_box)
        form.addRow("Atlas labels (centroids):", atlas_box)
        form.addRow("Plan CSV folder:", plan_box)
        form.addRow("Figure save folder:", fig_box)
        v.addLayout(form)
        hint = QtWidgets.QLabel(
            "Atlas = DMBA_RCCF_labels_centroids.txt. Needed only to target a brain region "
            "by name on the Parameters tab; manual offsets work without it.")
        hint.setWordWrap(True); hint.setStyleSheet("color: #666; font-size: 10px;")
        v.addWidget(hint)

        self._set_edits = {
            "ct_path": self.set_ct,
            "atlas_path": self.set_atlas,
            "plan_dir": self.set_plan_dir,
            "fig_dir": self.set_fig_dir,
        }

        row = QtWidgets.QHBoxLayout()
        b_save = QtWidgets.QPushButton("Save as default")
        b_save.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: bold; }")
        b_save.clicked.connect(self._on_save_defaults)
        b_copy = QtWidgets.QPushButton("Copy from current")
        b_copy.clicked.connect(self._on_copy_current)
        b_clear = QtWidgets.QPushButton("Forget saved defaults")
        b_clear.clicked.connect(self._on_clear_defaults)
        for b in (b_save, b_copy, b_clear):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        self.settings_status = QtWidgets.QLabel(""); self.settings_status.setWordWrap(True)
        v.addWidget(self.settings_status)
        v.addWidget(QtWidgets.QLabel(f"Settings file:  {settings.config_path()}"))
        v.addStretch(1)

        saved = settings.load()
        for k, e in self._set_edits.items():
            e.setText(saved.get(k, ""))
        return w

    def _apply_saved_paths(self):
        """Push saved defaults into the widgets that actually use them.

        Only the CT path fills a field. ``plan_dir`` / ``fig_dir`` are used as
        the starting directory for their file dialogs, read on demand.
        """
        saved = settings.load()
        if not saved:
            return
        msg = []
        ct = saved.get("ct_path", "")
        if ct and hasattr(self, "skull_widget"):
            self.skull_widget.path_edit.setText(ct)
            msg.append(f"CT: {os.path.basename(ct)}")
        atlas_path = saved.get("atlas_path", "")
        if hasattr(self, "skull_widget"):
            # always call: a blank/missing path just disables region targeting
            self.skull_widget.load_atlas(atlas_path)
            if atlas_path:
                msg.append(f"atlas: {os.path.basename(atlas_path)}")
        if msg:
            self.settings_status.setText(
                "Defaults restored -> " + ",  ".join(msg)
                + "   (Parameters tab -> press Prepare / Load)")

    def _on_save_defaults(self):
        vals = {k: e.text().strip() for k, e in self._set_edits.items()}
        settings.set_many(**vals)
        self.settings_status.setText(f"Saved to {settings.config_path()}")
        self._apply_saved_paths()

    def _on_copy_current(self):
        if hasattr(self, "skull_widget"):
            self.set_ct.setText(self.skull_widget.path_edit.text().strip())
        # atlas has no in-session widget of its own; keep whatever is set
        cur_plan = self.plan_path.text().strip() if hasattr(self, "plan_path") else ""
        if cur_plan:
            self.set_plan_dir.setText(os.path.dirname(cur_plan))
        self.settings_status.setText(
            "Copied from the current session. Press 'Save as default' to keep them.")

    def _on_clear_defaults(self):
        settings.clear()
        self.settings_status.setText(
            "Saved defaults removed. The app will not prefill on next launch.")

    # ===================================================== Tab 3: Simulation
    @staticmethod
    def _section(title):
        lab = QtWidgets.QLabel(title)
        lab.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 6px;")
        return lab

    def _build_tab_simulation(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(w)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter)

        left = QtWidgets.QScrollArea(); left.setWidgetResizable(True)
        inner = QtWidgets.QWidget(); left.setWidget(inner)
        v = QtWidgets.QVBoxLayout(inner)

        # ================= 1) Ultrasound simulation (pressure) =================
        v.addWidget(self._section("1)  Ultrasound simulation (pressure)"))

        gc = QtWidgets.QGroupBox("Calibration target"); fc = QtWidgets.QFormLayout(gc)
        self.desired_p = _dspin(0.001, 20.0, 0.35, 3, 0.05, " MPa")
        fc.addRow("Desired focal pressure:", self.desired_p)
        v.addWidget(gc)

        gn = QtWidgets.QGroupBox("Numerics"); fn = QtWidgets.QFormLayout(gn)
        self.ppw = _ispin(2, 20, 8)
        self.cfl = _dspin(0.01, 0.5, 0.05, 3, 0.01)
        self.cycles = _ispin(5, 400, 60)
        fn.addRow("Points per wavelength:", self.ppw)
        fn.addRow("CFL number:", self.cfl)
        fn.addRow("Tone-burst cycles:", self.cycles)
        v.addWidget(gn)

        gm = QtWidgets.QGroupBox("Compute"); fm = QtWidgets.QFormLayout(gm)
        self.compute = QtWidgets.QComboBox(); self.compute.addItems(["auto", "gpu", "cpu"])
        self.gpu_label = QtWidgets.QLabel("-"); self.gpu_label.setWordWrap(True)
        fm.addRow("Mode:", self.compute)
        fm.addRow("Detected:", self.gpu_label)
        v.addWidget(gm)

        self.run_skull_btn = QtWidgets.QPushButton("Run simulation (calibrate -> through skull)")
        self.run_skull_btn.setToolTip(
            "Calibrates in water for the current parameters, then runs the "
            "simulation through the loaded skull at the calibrated drive.")
        self.run_skull_btn.clicked.connect(self._on_run_skull)
        v.addWidget(self.run_skull_btn)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 1); self.progress_bar.setValue(0)
        self.progress_bar.setFormat("idle")
        v.addWidget(self.progress_bar)
        self.status = QtWidgets.QLabel("Ready."); self.status.setWordWrap(True)
        v.addWidget(self.status)
        self.result_text = QtWidgets.QPlainTextEdit(); self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("font-family: monospace;")
        self.result_text.setMaximumHeight(120)
        v.addWidget(self.result_text)

        # ========================= 2) Thermal (heating) =========================
        v.addWidget(self._section("2)  Thermal (heating)"))

        gp = QtWidgets.QGroupBox("Pulsing"); fp = QtWidgets.QFormLayout(gp)
        self.t_prf = _dspin(0.1, 100000.0, 1000.0, 1, 100.0, " Hz")
        self.t_pulse = _dspin(0.001, 1000.0, 0.2, 3, 0.05, " ms")
        self.t_duty = QtWidgets.QLabel("-")
        self.t_son = _dspin(0.01, 600.0, 6.0, 2, 0.5, " s")
        self.t_cool = _dspin(0.0, 600.0, 2.0, 2, 0.5, " s")
        fp.addRow("PRF:", self.t_prf)
        fp.addRow("Pulse duration:", self.t_pulse)
        fp.addRow("Duty cycle:", self.t_duty)
        fp.addRow("Sonication time:", self.t_son)
        fp.addRow("Cooling time:", self.t_cool)
        v.addWidget(gp)

        gt = QtWidgets.QGroupBox("Thermal model"); ft = QtWidgets.QFormLayout(gt)
        self.t_base = _dspin(0.0, 45.0, 37.0, 1, 0.5, " degC")
        self.t_dt = _dspin(0.0001, 1.0, 0.1, 4, 0.01, " s")
        self.t_perf = _dspin(0.0, 1.0, 0.0, 4, 0.001, " 1/s")
        ft.addRow("Baseline temperature:", self.t_base)
        ft.addRow("Thermal time step:", self.t_dt)
        ft.addRow("Perfusion coeff:", self.t_perf)
        v.addWidget(gt)
        for sb in (self.t_prf, self.t_pulse):
            sb.valueChanged.connect(self._update_duty_label)

        self.run_thermal_btn = QtWidgets.QPushButton("Run thermal (uses last pressure run)")
        self.run_thermal_btn.clicked.connect(self._on_run_thermal)
        v.addWidget(self.run_thermal_btn)
        self.t_progress = QtWidgets.QProgressBar()
        self.t_progress.setRange(0, 1); self.t_progress.setValue(0); self.t_progress.setFormat("idle")
        v.addWidget(self.t_progress)
        self.t_status = QtWidgets.QLabel("Run a pressure simulation first, then thermal.")
        self.t_status.setWordWrap(True); v.addWidget(self.t_status)
        self.t_result_text = QtWidgets.QPlainTextEdit(); self.t_result_text.setReadOnly(True)
        self.t_result_text.setStyleSheet("font-family: monospace;")
        self.t_result_text.setMaximumHeight(120)
        v.addWidget(self.t_result_text)
        v.addStretch(1)

        # ---- right: pressure figure (top) + temperature figure (bottom) ----
        rsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        pw = QtWidgets.QWidget(); pv = QtWidgets.QVBoxLayout(pw)
        pv.setContentsMargins(0, 0, 0, 0)
        ph = QtWidgets.QHBoxLayout()
        ph.addWidget(self._section("Pressure field")); ph.addStretch(1)
        bp = QtWidgets.QPushButton("Save figure..."); bp.setMaximumWidth(140)
        bp.clicked.connect(lambda: self._save_figure(self.res_fig, "pressure_field"))
        ph.addWidget(bp); pv.addLayout(ph)
        self.res_fig = Figure(figsize=(9, 4)); self.res_canvas = FigureCanvas(self.res_fig)
        pv.addWidget(self.res_canvas)
        tw = QtWidgets.QWidget(); tv = QtWidgets.QVBoxLayout(tw)
        tv.setContentsMargins(0, 0, 0, 0)
        th = QtWidgets.QHBoxLayout()
        th.addWidget(self._section("Temperature")); th.addStretch(1)
        bt = QtWidgets.QPushButton("Save figure..."); bt.setMaximumWidth(140)
        bt.clicked.connect(lambda: self._save_figure(self.t_fig, "temperature"))
        th.addWidget(bt); tv.addLayout(th)
        self.t_fig = Figure(figsize=(9, 4)); self.t_canvas = FigureCanvas(self.t_fig)
        tv.addWidget(self.t_canvas)
        rsplit.addWidget(pw); rsplit.addWidget(tw)

        splitter.addWidget(left); splitter.addWidget(rsplit)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 880])
        self._update_duty_label()
        return w

    # ===================================================== wiring
    def _wire_signals(self):
        # transducer/geometry changes -> redraw the skull+transducer preview on
        # the Parameters tab (debounced inside the skull widget)
        for w in (self.freq, self.aperture, self.focal, self.ppw,
                  self.hole_d, self.box_ax, self.box_lr, self.box_ap):
            w.valueChanged.connect(self.skull_widget.schedule_redraw)
        self.hole_on.toggled.connect(self.skull_widget.schedule_redraw)

    # ===================================================== params
    def _gather_params(self) -> SimParams:
        return SimParams(
            frequency_hz=self.freq.value() * 1e3,
            aperture_diameter_mm=self.aperture.value(),
            focal_length_mm=self.focal.value(),
            hole_enabled=self.hole_on.isChecked(),
            hole_diameter_mm=self.hole_d.value(),
            desired_focal_pressure_mpa=self.desired_p.value(),
            reference_source_amp_pa=1.0e5,   # fixed 0.1 MPa reference (hidden;
            # calibration is linear so this only sets the reference-run drive)
            points_per_wavelength=self.ppw.value(),
            cfl=self.cfl.value(),
            num_cycles=self.cycles.value(),
            skull_box_axial_mm=self.box_ax.value(),
            skull_box_lr_mm=self.box_lr.value(),
            skull_box_ap_mm=self.box_ap.value(),
            compute_mode=self.compute.currentText(),
        )

    def _runlog(self, msg: str):
        """Append a timestamped line to fus_sim_run.log (best-effort)."""
        if not self._runlog_path:
            return
        try:
            import datetime as _dt
            with open(self._runlog_path, "a") as f:
                f.write(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        except Exception:
            pass

    def _save_figure(self, fig, default_name="figure"):
        """Export a matplotlib figure to PNG/PDF/SVG via a file dialog."""
        start = os.path.join(settings.get("fig_dir", ""), default_name + ".png") \
            if settings.get("fig_dir", "") else default_name + ".png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", start,
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        try:
            # no bbox_inches='tight' - that recomputes tight bboxes and can
            # segfault matplotlib on this setup; a plain savefig is safe.
            fig.savefig(path, dpi=200)
            self.status.setText(f"Saved figure: {path}")
        except Exception as err:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save failed", str(err))

    def _refresh_gpu_label(self):
        _, desc = detect_gpu()
        self.gpu_label.setText(desc)
        self._runlog("GPU detect: " + desc)

    # ===================================================== run
    def _maybe_warn_gpu(self, params, gspec) -> bool:
        """Pre-flight size/memory check. Estimates the GPU field memory and the
        kWaveArray source-build cost and warns before launching, so a too-large
        run gives a dialog instead of a native crash. Returns True to proceed."""
        try:
            from ..core.gpu import choose_solver
            from ..core.transducer import estimate_source_points
        except ImportError:
            from fus_simulator.core.gpu import choose_solver
            from fus_simulator.core.transducer import estimate_source_points
        use_gpu, _ = choose_solver(params.compute_mode)

        # k-Wave GPU keeps ~16 single-precision fields of the full grid resident.
        gpu_gb = gspec.n_points * 16 * 4 / 1e9
        src_pts, ups = estimate_source_points(gspec, params)

        reasons = []
        if use_gpu and gpu_gb > 6.0:
            reasons.append(f"- GPU memory ~{gpu_gb:.1f} GB of fields "
                           f"({gspec.n_points/1e6:.0f} M grid points); may exceed the card "
                           "and abort, or trip the Windows TDR watchdog.")
        if src_pts > 8_000_000:
            reasons.append(f"- Source build ~{src_pts/1e6:.0f} M integration points "
                           f"(upsampling x{ups}); this runs on the CPU/host and can exhaust "
                           "RAM while 'Building bowl source mask'.")
        if not reasons or getattr(self, "_gpu_warned", False):
            return True

        self._gpu_warned = True
        r = QtWidgets.QMessageBox.question(
            self, "Large run - check before launching",
            "This run is heavy:\n\n" + "\n".join(reasons) +
            "\n\nIf it crashes with no error: lower PPW, shrink the aperture/domain, "
            "set Mode to 'cpu', or raise the Windows TdrDelay (see README).\n\nProceed?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes)
        return r == QtWidgets.QMessageBox.Yes

    def _set_running(self, running: bool):
        self.run_skull_btn.setEnabled(not running)
        if running:
            # indeterminate "busy" animation (k-Wave does not stream a %)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("running...")
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("idle")

    def _on_run_skull(self):
        """One press: calibrate in water for the current params, then run through
        the skull. Calibration is recomputed every time, so changing PPW/CFL/etc
        is always reflected."""
        try:
            params = self._gather_params(); params.validate()
        except Exception as err:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Invalid parameters", str(err)); return
        if not self.skull_widget.has_skull():
            QtWidgets.QMessageBox.information(
                self, "No skull loaded",
                "Load a skull on the '2. Skull' tab first (Auto-find or Browse).")
            self.tabs.setCurrentWidget(self.skull_widget)
            self.skull_widget.on_shown()
            return
        gspec = build_grid_spec(params)
        if not self._maybe_warn_gpu(params, gspec):
            return
        emb = self.skull_widget.current_embedding(gspec)
        if emb is None:
            QtWidgets.QMessageBox.warning(self, "Skull", "Could not embed the skull.")
            return
        self._pending_emb = emb           # keep maps for thermal / overlay
        self._pending_gspec = gspec
        self._runlog(f"RUN start: PPW={params.points_per_wavelength} CFL={params.cfl} "
                     f"cycles={params.num_cycles} f0={params.f0/1e3:.0f}kHz "
                     f"mode={params.compute_mode} grid {gspec.Nx}x{gspec.Ny}x{gspec.Nz} "
                     f"({gspec.n_points/1e6:.1f}M) bone_voxels={emb.n_bone_voxels}")
        self._set_running(True)
        self.result_text.clear()
        self.status.setText("Starting (calibrate -> skull)...")
        self._worker = SkullSimWorker(params, gspec, emb)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished_skull)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, msg: str):
        self.status.setText(msg)
        self.result_text.appendPlainText(msg)
        self._runlog(msg)
        # show the current step on the busy bar (truncated)
        short = msg if len(msg) <= 48 else msg[:45] + "..."
        self.progress_bar.setFormat(short)

    def _on_failed(self, msg: str):
        self._set_running(False)
        self.status.setText("Failed.")
        self.result_text.appendPlainText("\n[ERROR]\n" + msg)
        self._runlog("SIMULATION FAILED: " + msg.splitlines()[0])
        QtWidgets.QMessageBox.critical(self, "Simulation failed", msg.split("\n\n")[0])

    def _make_box(self, emb, gspec, pressure_box_pa):
        """Crop the embedded skull maps to the sensor box and bundle everything
        the thermal solver / overlays need."""
        (ix0, ix1), (iy0, iy1), (iz0, iz1) = gspec.box_ix, gspec.box_iy, gspec.box_iz
        sl = (slice(ix0, ix1 + 1), slice(iy0, iy1 + 1), slice(iz0, iz1 + 1))
        fi = (int(np.argmin(np.abs(gspec.x_vec - gspec.focus_xyz[0]))) - ix0,
              int(np.argmin(np.abs(gspec.y_vec - gspec.focus_xyz[1]))) - iy0,
              int(np.argmin(np.abs(gspec.z_vec - gspec.focus_xyz[2]))) - iz0)
        return {
            "sound_speed": emb.sound_speed[sl], "density": emb.density[sl],
            "alpha_coeff": emb.alpha_coeff[sl], "bone_mask": emb.bone_mask[sl],
            "alpha_power": emb.alpha_power, "dx": gspec.dx, "focus_idx": fi,
            "x_mm": gspec.x_vec[ix0:ix1 + 1] * 1e3,
            "y_mm": gspec.y_vec[iy0:iy1 + 1] * 1e3,
            "z_mm": gspec.z_vec[iz0:iz1 + 1] * 1e3,
        }

    def _on_finished_skull(self, payload):
        self._set_running(False)
        calib = payload["calib"]
        skull = payload["skull"]
        # keep ONLY the values needed (drive + target), not the calibration object
        self._last_drive_pa = calib.calibrated_amp_pa
        self._last_desired_mpa = calib.desired_focal_pressure_mpa
        # cache the pressure box + skull maps for the Thermal tab and overlays
        emb = getattr(self, "_pending_emb", None)
        gspec = getattr(self, "_pending_gspec", None)
        if emb is not None and gspec is not None:
            self._last_pressure_box_pa = skull.box_field_mpa * 1e6
            self._last_box = self._make_box(emb, gspec, self._last_pressure_box_pa)
            self.t_status.setText("Skull pressure ready - set pulsing and Run thermal.")
        desired = self._last_desired_mpa
        skull_focus = skull.focus_pressure_pa / 1e6
        skull_peak = skull.peak_pressure_pa / 1e6
        transmission = 100.0 * skull_focus / desired if desired > 0 else 0.0

        # ---- exposure metrics (ISPPA / ISPTA / MI) ----
        try:
            from ..core.metrics import (isppa_w_m2, duty_cycle, ispta_w_m2,
                                        mechanical_index, w_m2_to_w_cm2)
            from ..core.params import WATER
        except ImportError:
            from fus_simulator.core.metrics import (isppa_w_m2, duty_cycle, ispta_w_m2,
                                                    mechanical_index, w_m2_to_w_cm2)
            from fus_simulator.core.params import WATER
        f0 = self._gather_params().f0
        # the focus sits in water/brain, so use the water impedance (not bone)
        z_med = WATER.density * WATER.sound_speed
        duty = duty_cycle(self.t_prf.value(), self.t_pulse.value() * 1e-3)
        i_sppa = w_m2_to_w_cm2(isppa_w_m2(skull.focus_pressure_pa, WATER.density,
                                          WATER.sound_speed))
        i_spta = w_m2_to_w_cm2(ispta_w_m2(
            isppa_w_m2(skull.focus_pressure_pa, WATER.density, WATER.sound_speed), duty))
        # MI uses the in-situ (through-skull) focal pressure, per the FDA/AIUM
        # definition: the derated peak rarefactional pressure the tissue sees.
        # In a linear run |p_neg| = |p_pos|, so the recorded amplitude is used.
        mi = mechanical_index(skull.focus_pressure_pa, f0)

        txt = [
            "=== Calibrate -> through skull ===",
            f"  Drive (calibrated for {desired:.3f} MPa free field) : "
            f"{self._last_drive_pa/1e6:.4f} MPa",
            f"  Free-field focal peak (target)    : {desired:.4f} MPa",
            f"  Skull: pressure at geometric focus: {skull_focus:.4f} MPa",
            f"  Skull: peak pressure in box       : {skull_peak:.4f} MPa",
            f"  Transmission (focus/target)       : {transmission:.1f} %",
            f"  Skull run time                    : {skull.runtime_s:.1f} s",
            "",
            "--- Exposure metrics (at the focus) ---",
            f"  Impedance Z = rho*c (water)       : {z_med/1e6:.2f} MRayl",
            f"  I_SPPA = p^2/(2Z)                 : {i_sppa:.3f} W/cm^2",
            f"  Duty cycle = PRF x pulse          : {duty*100:.2f} %  "
            f"({self.t_prf.value():.0f} Hz x {self.t_pulse.value():.3f} ms)",
            f"  I_SPTA = I_SPPA x duty            : {i_spta:.4f} W/cm^2",
            f"  MI = p_focus / sqrt(f0)           : {mi:.3f}  "
            f"({skull_focus:.3f} MPa in brain / sqrt({f0/1e6:.3f} MHz))",
        ]
        self.status.setText(
            f"Done. Free-field {desired:.3f} MPa -> through-skull focus "
            f"{skull_focus:.4f} MPa ({transmission:.0f}% transmission).")
        self.result_text.appendPlainText("\n" + "\n".join(txt))
        self._runlog(f"RUN done: drive={self._last_drive_pa/1e6:.4f}MPa "
                     f"skull_focus={skull_focus:.4f}MPa transmission={transmission:.1f}% "
                     f"runtime={skull.runtime_s:.1f}s")
        self._plot_skull_result(calib, skull)

    @staticmethod
    def _add_cbar(fig, ax, vmin, vmax, cmap, label):
        """Colourbar pinned to the right of *ax*, matching its height exactly.

        ``make_axes_locatable`` sizes the bar to the axes box (a plain
        fig.colorbar sizes it to the un-shrunk cell, so it overshoots an
        equal-aspect panel). The divider's own anchor keeps the panel top-aligned
        - a plain ``ax.set_anchor`` is ignored once a locator is installed.
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.06)
        try:
            div.set_anchor("N")
        except Exception:  # pragma: no cover - older mpl
            pass
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        return cb

    @staticmethod
    def _field_over_bone(ax, f2d, bone2d, extent, vmin, vmax, cmap):
        """Paper-style panel: skull filled solid white, and the field laid over it
        with alpha ~ normalised amplitude, so weak regions reveal the white bone
        (and the dark background) underneath. Used for pressure and temperature."""
        ax.set_facecolor(cmap(0.0))                       # background = cmap's zero
        if bone2d is not None and np.asarray(bone2d).max() > 0:
            b = np.asarray(bone2d) > 0.5
            rgba_b = np.zeros(b.shape + (4,), dtype=float)
            rgba_b[b] = (1.0, 1.0, 1.0, 1.0)              # solid white bone
            ax.imshow(rgba_b, origin="lower", extent=extent,
                      aspect="equal", interpolation="nearest", zorder=1)
        span = max(float(vmax) - float(vmin), 1e-12)
        fn = np.clip((np.asarray(f2d, float) - float(vmin)) / span, 0.0, 1.0)
        rgba_f = cmap(fn)
        rgba_f[..., 3] = fn ** 0.7                        # transparent where weak
        ax.imshow(rgba_f, origin="lower", extent=extent,
                  aspect="equal", interpolation="bilinear", zorder=2)
        ax.tick_params(labelsize=7)
        ax.locator_params(axis="both", nbins=5)

    def _plot_skull_result(self, calib, skull):
        """Through-skull field in the three orthogonal planes through the focus.
        Bone is filled white, pressure is overlaid, focus marked with a star."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        try:                                   # matplotlib >= 3.9 removed cm.get_cmap
            from matplotlib import colormaps
            _jet = colormaps["jet"]
        except Exception:                      # pragma: no cover - older matplotlib
            from matplotlib import cm as _cm
            _jet = _cm.get_cmap("jet")

        box = self._last_box
        P = np.asarray(skull.box_field_mpa, dtype=float)          # (nx,ny,nz) MPa
        if box is None or P.ndim != 3:
            return
        bone = np.asarray(box["bone_mask"]).astype(bool)
        fx, fy, fz = box["focus_idx"]
        x, y, z = box["x_mm"], box["y_mm"], box["z_mm"]
        fx = int(np.clip(fx, 0, P.shape[0] - 1))
        fy = int(np.clip(fy, 0, P.shape[1] - 1))
        fz = int(np.clip(fz, 0, P.shape[2] - 1))
        vmax = float(max(P.max(), 1e-9))
        cmap = _jet

        ext_zy = [z[0], z[-1], y[0], y[-1]]      # lateral plane  (z horiz, y vert)
        ext_zx = [z[0], z[-1], x[0], x[-1]]      # coronal        (z horiz, x vert)
        ext_yx = [y[0], y[-1], x[0], x[-1]]      # sagittal       (y horiz, x vert)

        fig = self.res_fig
        fig.clear()
        a1 = fig.add_subplot(1, 3, 1)
        a2 = fig.add_subplot(1, 3, 2)
        a3 = fig.add_subplot(1, 3, 3)

        # Lateral (y-z) at the focal depth
        self._field_over_bone(a1, P[fx, :, :], bone[fx, :, :], ext_zy, 0.0, vmax, cmap)
        a1.plot([z[fz]], [y[fy]], marker="*", ms=6, color="w",
                markeredgecolor="k", markeredgewidth=0.4, zorder=3)
        a1.set_xlabel("Lateral position [z, mm]", fontsize=8)
        a1.set_ylabel("Lateral position [y, mm]", fontsize=8)
        a1.set_title("Lateral plane (y-z) at focus", fontsize=9, pad=10)

        # Coronal (x-z): beam axis x vertical, increasing downward = deeper
        self._field_over_bone(a2, P[:, fy, :], bone[:, fy, :], ext_zx, 0.0, vmax, cmap)
        a2.plot([z[fz]], [x[fx]], marker="*", ms=6, color="w",
                markeredgecolor="k", markeredgewidth=0.4, zorder=3)
        a2.set_xlabel("Lateral position [z, mm]", fontsize=8)
        a2.set_ylabel("Axial position [x, mm]", fontsize=8)
        a2.set_title("Coronal (x-z)", fontsize=9, pad=10)
        a2.invert_yaxis()

        # Sagittal (x-y)
        self._field_over_bone(a3, P[:, :, fz], bone[:, :, fz], ext_yx, 0.0, vmax, cmap)
        a3.plot([y[fy]], [x[fx]], marker="*", ms=6, color="w",
                markeredgecolor="k", markeredgewidth=0.4, zorder=3)
        a3.set_xlabel("Lateral position [y, mm]", fontsize=8)
        a3.set_ylabel("Axial position [x, mm]", fontsize=8)
        a3.set_title("Sagittal (x-y)", fontsize=9, pad=10)
        a3.invert_yaxis()

        # a colourbar beside every panel
        for _a in (a1, a2, a3):
            self._add_cbar(fig, _a, 0.0, 1.0, cmap, "Normalized pressure")

        # boxed data summary (top-left). Drive is deliberately not shown.
        sf = skull.focus_pressure_pa / 1e6
        summary = (f"Free field pressure : {calib.desired_focal_pressure_mpa:.3f} MPa\n"
                   f"Pressure at focus   : {sf:.3f} MPa\n"
                   f"Max pressure        : {vmax:.3f} MPa")
        fig.text(0.012, 0.985, summary, va="top", ha="left", fontsize=8,
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                           edgecolor="0.55", linewidth=0.8))

        # leave room above the panels for the summary box + the panel titles
        fig.subplots_adjust(left=0.06, right=0.95, top=0.78, bottom=0.12, wspace=0.50)
        self.res_canvas.draw_idle()

    def _overlay_bone(self, ax, plane, extent):
        """Overlay the skull bone outline on a box slice (uses cached box)."""
        box = self._last_box
        if not box:
            return
        bone = box["bone_mask"]
        bx, by, bz = bone.shape
        if plane == "xz":
            bslice = bone[:, by // 2, :].T
        else:
            bslice = bone[:, :, bz // 2].T
        if bslice.max() <= 0:
            return
        nrows, ncols = bslice.shape
        hh = np.linspace(extent[0], extent[1], ncols)
        vv = np.linspace(extent[2], extent[3], nrows)
        ax.contour(hh, vv, bslice, levels=[0.5], colors="white", linewidths=0.8)

    # ===================================================== thermal run
    def _on_run_thermal(self):
        if self._last_box is None or self._last_pressure_box_pa is None:
            QtWidgets.QMessageBox.information(
                self, "No pressure field",
                "Run a skull pressure simulation first on the Simulation tab "
                "('Run through skull'). The thermal step heats from that field.")
            return
        tp = ThermalParams(
            baseline_temp_c=self.t_base.value(),
            prf_hz=self.t_prf.value(),
            pulse_duration_ms=self.t_pulse.value(),
            sonication_time_s=self.t_son.value(),
            cooling_time_s=self.t_cool.value(),
            dt_thermal_s=self.t_dt.value(),
            perfusion_rate=self.t_perf.value(),
            blood_temp_c=self.t_base.value())
        f0 = self._gather_params().f0
        self.run_thermal_btn.setEnabled(False)
        self.t_result_text.clear()
        self.t_status.setText("Running thermal...")
        self.t_progress.setRange(0, 0); self.t_progress.setFormat("running...")
        self._tworker = ThermalWorker(self._last_box, self._last_pressure_box_pa, tp, f0)
        self._tworker.progress.connect(self._on_t_progress)
        self._tworker.finished_ok.connect(self._on_thermal_done)
        self._tworker.failed.connect(self._on_thermal_failed)
        self._tworker.start()

    def _on_t_progress(self, msg):
        self.t_status.setText(msg)
        self.t_result_text.appendPlainText(msg)
        self._runlog("thermal: " + msg)
        self.t_progress.setFormat(msg if len(msg) <= 48 else msg[:45] + "...")

    def _t_reset_bar(self):
        self.run_thermal_btn.setEnabled(True)
        self.t_progress.setRange(0, 1); self.t_progress.setValue(0)
        self.t_progress.setFormat("idle")

    def _on_thermal_failed(self, msg):
        self._t_reset_bar()
        self.t_status.setText("Thermal failed.")
        self.t_result_text.appendPlainText("\n[ERROR]\n" + msg)
        self._runlog("THERMAL FAILED: " + msg.splitlines()[0])
        QtWidgets.QMessageBox.critical(self, "Thermal failed", msg.split("\n\n")[0])

    def _on_thermal_done(self, res):
        self._t_reset_bar()
        txt = [
            "=== Thermal (bioheat) result ===",
            f"  Duty cycle              : {res.duty_cycle*100:.1f} %",
            f"  Max Q                   : {res.q_max:.3e} W/m^3",
            f"  Peak temperature        : {res.peak_temp_c:.3f} degC (+{res.peak_rise_c:.3f})",
            f"  Temperature at focus    : {res.focus_end_temp_c:.3f} degC "
            f"(+{res.focus_rise_c:.3f} peak)",
            f"  Max thermal dose (CEM43): {res.max_cem43:.3e} min",
            f"  Lesion volume (>=240)   : {res.lesion_volume_mm3:.4f} mm^3",
        ]
        self.t_status.setText(
            f"Thermal done. Peak {res.peak_temp_c:.2f} degC (+{res.peak_rise_c:.2f}).")
        self.t_result_text.appendPlainText("\n" + "\n".join(txt))
        self._plot_thermal(res)

    def _plot_thermal(self, res):
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        try:                                   # matplotlib >= 3.9 removed cm.get_cmap
            from matplotlib import colormaps
            cmap = colormaps["hot"]
        except Exception:                      # pragma: no cover - older matplotlib
            from matplotlib import cm as _cm
            cmap = _cm.get_cmap("hot")

        fig = self.t_fig
        fig.clear()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 1, 2)
        vmin = float(self.t_base.value())
        vmax = float(max(res.slice_T_xz.max(), res.slice_T_xy.max(), vmin + 0.01))

        # skull filled solid white, heat map shaded over it (alpha ~ rise)
        self._field_over_bone(ax1, res.slice_T_xz, res.bone_xz,
                              res.extent_xz_mm, vmin, vmax, cmap)
        ax1.set_title("Max temperature  XZ", fontsize=9, pad=10)
        ax1.set_xlabel("x [mm]"); ax1.set_ylabel("z [mm]")
        self._field_over_bone(ax2, res.slice_T_xy, res.bone_xy,
                              res.extent_xy_mm, vmin, vmax, cmap)
        ax2.set_title("Max temperature  XY", fontsize=9, pad=10)
        ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")

        for ax in (ax1, ax2):
            self._add_cbar(fig, ax, vmin, vmax, cmap, "degC")

        ax3.plot(res.times_s, res.focus_temp_c, color="tab:red", lw=2)
        ax3.axhline(self.t_base.value(), color="0.6", lw=0.8, ls="--")
        ax3.axvline(self.t_son.value(), color="0.6", lw=0.8, ls=":",
                    label="sonication end")
        ax3.set_xlabel("time [s]"); ax3.set_ylabel("focus T [degC]")
        ax3.set_title("Temperature at focus vs time", fontsize=9, pad=10)
        ax3.legend(fontsize=8)
        fig.suptitle(f"Thermal: peak {res.peak_temp_c:.2f} degC "
                     f"(+{res.peak_rise_c:.2f}), duty {res.duty_cycle*100:.0f}%  "
                     f"(white = skull)", fontsize=9, y=0.98)
        # extra top room + row gap so the suptitle and titles never clash
        fig.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.09,
                            wspace=0.35, hspace=0.55)
        self.t_canvas.draw_idle()

    def _plot_result(self, result):
        sxz, sxy, xz_ext, xy_ext, peak_mpa = result.display_field()
        confirmed = result.confirm_free_field is not None
        fig = self.res_fig
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1); ax2 = fig.add_subplot(1, 2, 2)
        vmax = float(max(sxz.max(), sxy.max(), 1e-9))
        im1 = ax1.imshow(sxz, origin="lower", extent=xz_ext, cmap="jet",
                         aspect="equal", vmin=0, vmax=vmax)
        ax1.set_title("Peak pressure  XZ (y=centre)")
        ax1.set_xlabel("x [mm]"); ax1.set_ylabel("z [mm]")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="MPa")
        im2 = ax2.imshow(sxy, origin="lower", extent=xy_ext, cmap="jet",
                         aspect="equal", vmin=0, vmax=vmax)
        ax2.set_title("Peak pressure  XY (z=centre)")
        ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="MPa")
        kind = "confirmation run" if confirmed else "calibrated (linearly scaled)"
        fig.suptitle(f"Calibrated field [{kind}]  |  peak {peak_mpa:.3f} MPa "
                     f"(target {result.desired_focal_pressure_mpa:.3f} MPa)  |  "
                     f"drive {result.calibrated_amp_pa/1e6:.4f} MPa")
        fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.10, wspace=0.30, hspace=0.35)
        self.res_canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
