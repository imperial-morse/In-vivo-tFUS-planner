"""PyQt5 main window for the FUS Planner.

Tabs
----
1. Data       : pick MRI / CT / labels / centroids; load.
2. Plan       : quadrant layout
                 - top left   : landmark coordinates
                 - bottom left: dorsal-skull picker (CT, height-shaded)
                 - top right  : transducer parameters
                 - bottom right: scenario / region search & multi-checkbox /
                                 strategy / threshold / Run plan
3. Results    : xy / xz / yz overlays + CT-with-grid panel; per-plane sliders;
                per-spot table with colour-coded headers; Save CSV / PDF;
                "Skull diagnostic..." button.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from . import settings
from ..io import load_volume, load_region_catalog
from ..io.centroids import RegionCatalog, lps_to_anat, anat_to_lps
from ..io.volume import Volume
from ..geometry.frames import build_stage_frame, StageFrame
from ..planner.centroid import whole_brain_centroid_voxel, subset_centroid_voxel
from ..planner.plane import extract_focal_plane, FocalPlane
from ..planner.footprint import footprint_from_fwhm, Footprint
from ..planner.hex_tile import hex_tile_plan, HexTilePlan
from ..planner.region_pack import (
    region_pack_centroid_seed, region_pack_max_coverage, region_pack_full_coverage)
from ..planner.region_pack_3d import plan_3d
from ..planner.zplane import plan_z_for_centroid, ZPlanePlan
from ..reporting import (
    plan_to_dataframe, write_csv, render_plan_pdf,
    plot_xy, plot_xz_or_yz, plot_ct_dorsal, plot_skull_diagnostic,
    plot_ct_axial_slice, ct_top_surface_depth, render_shaded_skull,
    COLUMN_GROUP_LOOKUP,
)


# ---------------------------------------------------------------- helpers

def _no_layout_engine(fig):
    """Guarantee this figure never runs an auto-layout pass.

    If the ``figure.autolayout`` rcParam is on (or a constrained/tight layout
    engine is attached), matplotlib re-runs tight_layout on EVERY draw. That
    walks all artists calling get_tightbbox(), which segfaults on Windows when
    it measures the curved Ellipse/legend paths (bezier.polynomial_coefficients).
    Removing our explicit tight_layout() calls is not enough on its own.
    """
    try:
        fig.set_layout_engine("none")
    except Exception:       # pragma: no cover - matplotlib < 3.6
        try:
            fig.set_tight_layout(False)
            fig.set_constrained_layout(False)
        except Exception:
            pass


def subset_plan(plan: HexTilePlan, keep: List[bool]) -> HexTilePlan:
    """Return a copy of ``plan`` holding only the spots where ``keep[i]`` is True.

    Coverage is recomputed for the kept subset. Both packers place spots whose
    footprints are mutually non-overlapping (``hex_tile_plan`` uses a hex lattice
    pitched at the footprint radii; ``plan_3d`` rejects any candidate that
    overlaps an existing spot via ``_ellipse_overlap_xy``). Because the beams are
    disjoint, the covered target area is exactly the sum of the per-spot target
    areas, and no re-rasterisation is needed.

    NOTE on units: for 3D region plans ``target_area_mm2`` actually holds a
    volume in mm^3 (see ``region_pack_3d.plan_3d``). Summing works either way,
    which is why this is done in the area/volume slot rather than by rasterising
    a 2D plane -- the latter would silently give a wrong number for 3D plans.
    """
    kept = [s for s, k in zip(plan.spots, keep) if k]
    covered = float(sum(s.target_area_mm2 for s in kept))
    # Guard against float drift pushing coverage past the total.
    covered = min(covered, plan.target_area_mm2) if plan.target_area_mm2 else covered
    return HexTilePlan(
        spots=kept,
        coverage_threshold=plan.coverage_threshold,
        footprint=plan.footprint,
        plane_z_lps_mm=plan.plane_z_lps_mm,
        target_area_mm2=plan.target_area_mm2,
        target_area_covered_mm2=covered,
    )


def _draw_excluded_spots(ax, spots, panel: str, rz: float, z_focus: float) -> None:
    """Overlay de-selected spots as faint grey dashed ellipses.

    Drawn after the normal plot so the user can see where an excluded spot sat
    and whether removing it opened a coverage hole. Uses the same anatomical
    conventions as ``_on_hover``.
    """
    for s in spots:
        cx, cy = s.center_world_xy_mm
        if panel in ("xy", "ct"):
            u, v, a, b = -cx, -cy, s.rx_mm, s.ry_mm
        elif panel == "xz":
            u, v, a, b = -cx, z_focus, s.rx_mm, rz
        elif panel == "yz":
            u, v, a, b = -cy, z_focus, s.ry_mm, rz
        else:
            return
        ax.add_patch(Ellipse((u, v), 2 * a, 2 * b, fill=False,
                             edgecolor="0.55", linestyle="--", linewidth=0.9,
                             zorder=2.5))


class FilePicker(QtWidgets.QWidget):
    def __init__(self, label, file_filter="All files (*.*)"):
        super().__init__()
        self._filter = file_filter
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel(label))
        self.edit = QtWidgets.QLineEdit(); h.addWidget(self.edit, 1)
        btn = QtWidgets.QPushButton("Browse..."); btn.clicked.connect(self._browse)
        h.addWidget(btn)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", self._filter)
        if path: self.edit.setText(path)

    def path(self):
        s = self.edit.text().strip()
        return Path(s) if s else None


class TripletInput(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, labels=("LR", "AP", "IS"), default=(0.0, 0.0, 0.0)):
        super().__init__()
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0)
        self._spins = []
        for lbl, val in zip(labels, default):
            h.addWidget(QtWidgets.QLabel(f"{lbl}:"))
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-100.0, 100.0); sb.setDecimals(3)
            sb.setSingleStep(0.1); sb.setSuffix(" mm"); sb.setValue(val)
            sb.valueChanged.connect(self._emit)
            h.addWidget(sb); self._spins.append(sb)

    def _emit(self, *_):
        self.valueChanged.emit(self.value())

    def value(self):
        return np.array([sb.value() for sb in self._spins], dtype=float)

    def set_value(self, vals):
        for sb, v in zip(self._spins, vals):
            sb.blockSignals(True); sb.setValue(float(v)); sb.blockSignals(False)
        self._emit()


# ----------- click-to-pick landmark view (skull / axial-slice modes) -------

class LandmarkPicker(QtWidgets.QWidget):
    """Two view modes:

    * **Skull rendering** - height-shaded dorsal skull (best for finding lambda
      and the cranial sutures).
    * **Axial slice (heatmap)** - single CT axial slice with a hot colormap and
      a z slider, so you can scroll through depths to find the bottom of the
      eye, the auditory meatus, etc.

    Either mode supports clicking to set the current landmark's (LR, AP).
    """

    landmarkPicked = QtCore.pyqtSignal(str, float, float)

    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)

        hdr = QtWidgets.QHBoxLayout()
        hdr.addWidget(QtWidgets.QLabel("Picking:"))
        self.pick_target = QtWidgets.QComboBox()
        self.pick_target.addItems(["lambda", "left_eye"])
        hdr.addWidget(self.pick_target)
        hdr.addWidget(QtWidgets.QLabel("View:"))
        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItems(["skull rendering", "axial slice (heatmap)"])
        self.view_mode.currentIndexChanged.connect(self._on_view_mode_change)
        hdr.addWidget(self.view_mode)
        hdr.addWidget(QtWidgets.QLabel("Skull thr.:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 100000.0); self.threshold_spin.setDecimals(0); self.threshold_spin.setValue(5000.0)
        self.threshold_spin.valueChanged.connect(lambda *_: self._draw())
        hdr.addWidget(self.threshold_spin)
        hdr.addStretch()
        v.addLayout(hdr)

        self.figure = Figure(figsize=(7, 9))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        v.addWidget(self.canvas, 1)

        # Slice slider for axial-slice mode. This slider is fully owned by the
        # picker widget; it does not share state with any other slider in the app.
        slice_row = QtWidgets.QHBoxLayout()
        self.slice_label = QtWidgets.QLabel("Picker slice IS:")
        slice_row.addWidget(self.slice_label)
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_slider.setObjectName("picker_slice_slider")
        self.slice_slider.setRange(0, 100); self.slice_slider.setValue(50)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_change)
        slice_row.addWidget(self.slice_slider, 1)
        self.slice_value_label = QtWidgets.QLabel("--")
        slice_row.addWidget(self.slice_value_label)
        v.addLayout(slice_row)

        self._labels_volume: Optional[Volume] = None
        self._ct_volume: Optional[Volume] = None
        self._lambda_anat: Optional[np.ndarray] = None
        self._eye_anat: Optional[np.ndarray] = None

        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(120)   # ms
        self._redraw_timer.timeout.connect(self._draw)
        self._drawing = False                  # guard against re-entrant draws

        self._on_view_mode_change()

    def _on_slice_slider_change(self, *_):
        # live label, debounced heavy redraw
        try:
            self.slice_value_label.setText(f"IS={self._ct_z_for_slider():+.2f} mm")
        except Exception:
            pass
        self._redraw_timer.start()

    def set_volumes(self, labels, ct):
        self._labels_volume = labels
        self._ct_volume = ct
        # Initialise slice slider range from CT z extent
        if ct is not None:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setRange(0, ct.shape[2] - 1)
            # default near the middle of CT (lower than skull, around eye level)
            self.slice_slider.setValue(int(ct.shape[2] * 0.55))
            self.slice_slider.blockSignals(False)
        self._draw()

    def set_landmarks(self, lambda_anat, eye_anat):
        self._lambda_anat = lambda_anat
        self._eye_anat = eye_anat
        self._draw()

    def threshold(self) -> float:
        return float(self.threshold_spin.value())

    def _on_view_mode_change(self, *_):
        is_axial = self.view_mode.currentText().startswith("axial")
        self.slice_slider.setEnabled(is_axial and self._ct_volume is not None)
        self.threshold_spin.setEnabled(not is_axial)
        self._draw()

    def _ct_z_for_slider(self) -> float:
        if self._ct_volume is None:
            return 0.0
        k = self.slice_slider.value()
        return float(self._ct_volume.voxel_to_world(np.array([0, 0, k]))[2])

    def _draw(self):
        # Prevent new draw while one is running
        if getattr(self, "_drawing", False):
            self._redraw_timer.start()
            return
        self._drawing = True
        try:
            self._draw_impl()
        except Exception as exc:  
            try:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f"draw error:\n{exc}", ha="center", va="center",
                        transform=ax.transAxes)
                self.canvas.draw_idle()
            except Exception:
                pass
        finally:
            self._drawing = False

    def _draw_impl(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        mode = self.view_mode.currentText()
        if self._ct_volume is None and self._labels_volume is None:
            ax.text(0.5, 0.5, "Load CT or labels on tab 1.",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw_idle(); return

        if mode.startswith("axial") and self._ct_volume is not None:
            z_lps = self._ct_z_for_slider()
            self.slice_value_label.setText(f"IS={z_lps:+.2f} mm")
            try:
                plot_ct_axial_slice(ax, self._ct_volume, z_lps_mm=z_lps,
                                    cmap="inferno",
                                    title="")
            except Exception as exc:
                ax.text(0.5, 0.5, f"CT slice error:\n{exc}",
                        ha="center", va="center", transform=ax.transAxes)
        elif self._ct_volume is not None:
            try:
                rgba, extent = render_shaded_skull(
                    self._ct_volume, self.threshold(), downsample=2)
                if rgba[..., 3].any():
                    ax.imshow(rgba, origin="lower", extent=extent,
                              interpolation="bilinear")
                else:
                    ax.text(0.5, 0.5, "No skull voxels above threshold",
                            ha="center", va="center", transform=ax.transAxes)
                LR_0, LR_1, AP_0, AP_1 = extent
                ax.set_xlim(min(LR_0, LR_1), max(LR_0, LR_1))
                ax.set_ylim(min(AP_0, AP_1), max(AP_0, AP_1))
            except Exception as exc:
                ax.text(0.5, 0.5, f"CT view error:\n{exc}",
                        ha="center", va="center", transform=ax.transAxes)
        else:
            vol = self._labels_volume
            ds = 4
            sub = vol.data[::ds, ::ds, ::ds] > 0
            mip = sub.any(axis=2).astype(float)
            Ni, Nj, _ = vol.shape
            sx = vol.direction[0, 0] * vol.spacing[0]
            sy = vol.direction[1, 1] * vol.spacing[1]
            LR_0 = -vol.origin[0]
            LR_1 = -(vol.origin[0] + (Ni - 1) * sx)
            AP_0 = -vol.origin[1]
            AP_1 = -(vol.origin[1] + (Nj - 1) * sy)
            ax.imshow(mip.T, origin="lower", extent=(LR_0, LR_1, AP_0, AP_1),
                      cmap="Greys", vmin=0, vmax=1.2)
            ax.set_xlim(min(LR_0, LR_1), max(LR_0, LR_1))
            ax.set_ylim(min(AP_0, AP_1), max(AP_0, AP_1))
            ax.set_title("Labels MIP (no CT loaded)", fontsize=9)

        ax.set_aspect("equal"); ax.set_xlabel("LR (mm)"); ax.set_ylabel("AP (mm)")
        if self._lambda_anat is not None:
            ax.plot(self._lambda_anat[0], self._lambda_anat[1],
                    "x", color="crimson", ms=10, mew=2, label="lambda")
        if self._eye_anat is not None:
            ax.plot(self._eye_anat[0], self._eye_anat[1],
                    "o", mfc="none", mec="dodgerblue", ms=8, mew=2, label="left eye")
        if self._lambda_anat is not None or self._eye_anat is not None:
            ax.legend(loc="upper right", fontsize=7)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is None: return
        which = self.pick_target.currentText()
        self.landmarkPicked.emit(which, float(event.xdata), float(event.ydata))


# ----------- region preview (single plane with toggle + slider) -----------

class RegionPreview(QtWidgets.QWidget):
    """Compact viewer that shows the currently-selected target region in one
    plane (xy / xz / yz) with a slider over the perpendicular axis. Lives in
    the Plan tab so the user can sanity-check what they are about to plan
    *before* running the planner."""

    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)

        hdr = QtWidgets.QHBoxLayout()
        hdr.addWidget(QtWidgets.QLabel("Plane:"))
        self.plane_combo = QtWidgets.QComboBox()
        self.plane_combo.addItems(["xy (dorsal)", "xz (coronal)", "yz (sagittal)"])
        self.plane_combo.currentIndexChanged.connect(lambda *_: self._init_slider_range_and_draw())
        hdr.addWidget(self.plane_combo); hdr.addStretch()
        v.addLayout(hdr)

        self.figure = Figure(figsize=(4, 3.6))
        self.canvas = FigureCanvas(self.figure)
        v.addWidget(self.canvas, 1)

        srow = QtWidgets.QHBoxLayout()
        self.slider_label_left = QtWidgets.QLabel("Preview slice:")
        srow.addWidget(self.slider_label_left)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setObjectName("region_preview_slider")
        self.slider.setRange(0, 100); self.slider.setEnabled(False)
        # Owned by RegionPreview; not connected to any other widget.
        self.slider.valueChanged.connect(self._on_preview_slider_change)
        srow.addWidget(self.slider, 1)
        self.slider_label_right = QtWidgets.QLabel("--")
        srow.addWidget(self.slider_label_right)
        v.addLayout(srow)

        self._labels: Optional[Volume] = None
        self._target_labels: Optional[List[int]] = None    # None = whole brain

    def _on_preview_slider_change(self, *_):
        self._draw()

    def set_labels_volume(self, labels):
        self._labels = labels
        self._init_slider_range_and_draw()

    def set_target(self, target_labels):
        """target_labels=None -> whole brain; otherwise list of label values."""
        self._target_labels = target_labels
        self._draw()

    def _axis_index(self) -> int:
        text = self.plane_combo.currentText()
        if text.startswith("xy"): return 2     # slider over k (z)
        if text.startswith("xz"): return 1     # slider over j (y)
        return 0                                # yz: slider over i (x)

    def _init_slider_range_and_draw(self):
        if self._labels is None:
            self._draw(); return
        ax_idx = self._axis_index()
        n = self._labels.shape[ax_idx]
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1); self.slider.setValue(n // 2)
        self.slider.setEnabled(True); self.slider.blockSignals(False)
        self._draw()

    def _draw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self._labels is None:
            ax.text(0.5, 0.5, "Load labels on tab 1.",
                    ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw(); return

        labels = self._labels
        plane_text = self.plane_combo.currentText()
        idx = self.slider.value()

        if plane_text.startswith("xy"):
            slab = labels.data[:, :, idx]                        # (Ni, Nj)
            sx = labels.direction[0, 0] * labels.spacing[0]
            sy = labels.direction[1, 1] * labels.spacing[1]
            LR_0 = -labels.origin[0]
            LR_1 = -(labels.origin[0] + (slab.shape[0] - 1) * sx)
            AP_0 = -labels.origin[1]
            AP_1 = -(labels.origin[1] + (slab.shape[1] - 1) * sy)
            extent = (LR_0, LR_1, AP_0, AP_1)
            xlabel, ylabel = "LR (mm)", "AP (mm)"
            slider_units = labels.voxel_to_world(np.array([0, 0, idx]))[2]
            self.slider_label_right.setText(f"IS={slider_units:+.2f} mm")
        elif plane_text.startswith("xz"):
            slab = labels.data[:, idx, :]                        # (Ni, Nk)
            sx = labels.direction[0, 0] * labels.spacing[0]
            sz = labels.direction[2, 2] * labels.spacing[2]
            LR_0 = -labels.origin[0]
            LR_1 = -(labels.origin[0] + (slab.shape[0] - 1) * sx)
            IS_0 = labels.origin[2]
            IS_1 = labels.origin[2] + (slab.shape[1] - 1) * sz
            extent = (LR_0, LR_1, IS_0, IS_1)
            xlabel, ylabel = "LR (mm)", "IS (mm)"
            slider_units = -labels.voxel_to_world(np.array([0, idx, 0]))[1]
            self.slider_label_right.setText(f"AP={slider_units:+.2f} mm")
        else:  # yz
            slab = labels.data[idx, :, :]                        # (Nj, Nk)
            sy = labels.direction[1, 1] * labels.spacing[1]
            sz = labels.direction[2, 2] * labels.spacing[2]
            AP_0 = -labels.origin[1]
            AP_1 = -(labels.origin[1] + (slab.shape[0] - 1) * sy)
            IS_0 = labels.origin[2]
            IS_1 = labels.origin[2] + (slab.shape[1] - 1) * sz
            extent = (AP_0, AP_1, IS_0, IS_1)
            xlabel, ylabel = "AP (mm)", "IS (mm)"
            slider_units = -labels.voxel_to_world(np.array([idx, 0, 0]))[0]
            self.slider_label_right.setText(f"LR={slider_units:+.2f} mm")

        brain = (slab > 0).astype(float) * 0.4
        if self._target_labels is None:
            target = (slab > 0).astype(float)
        else:
            target = np.isin(slab, list(self._target_labels)).astype(float)
        img = np.maximum(brain, target)
        ax.imshow(img.T, origin="lower", extent=extent, cmap="Greys",
                  vmin=0, vmax=1.2, interpolation="nearest")
        u0, u1, v0, v1 = extent
        ax.set_xlim(min(u0, u1), max(u0, u1))
        ax.set_ylim(min(v0, v1), max(v0, v1))
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title("Target preview", fontsize=9)
        self.canvas.draw()


# ----------- skull diagnostic dialog --------------------------------------

class SkullDiagnosticDialog(QtWidgets.QDialog):
    def __init__(self, parent, ct, labels, zplan):
        super().__init__(parent)
        self.setWindowTitle("Skull-search diagnostic")
        self.resize(720, 720)
        v = QtWidgets.QVBoxLayout(self)
        fig = Figure(figsize=(7, 7)); canvas = FigureCanvas(fig)
        v.addWidget(canvas)
        ax = fig.add_subplot(111)
        plot_skull_diagnostic(ax, ct, labels, zplan)
        # no tight_layout: it measures curved patches and can segfault (see
        # _render_plots). This figure also has an outside legend.
        _no_layout_engine(fig)
        fig.subplots_adjust(left=0.12, right=0.74, top=0.92, bottom=0.13)
        canvas.draw()


# ---------------------------------------------------------------- main window

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FUS Planner — in vivo mouse ultrasound treatment planning")
        self.resize(1500, 980)

        self.labels_volume: Optional[Volume] = None
        self.ct_volume: Optional[Volume] = None
        self.catalog: Optional[RegionCatalog] = None
        self.last_plan_bundle: Optional[dict] = None
        self._persistent_ticked: set = set()

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_data_tab(), "1. Data")
        tabs.addTab(self._build_plan_tab(), "2. Plan")
        tabs.addTab(self._build_results_tab(), "3. Results")
        self.setCentralWidget(tabs)
        # prefill the Data tab from the saved defaults (if the user ever saved any)
        self._apply_saved_paths()
        self.statusBar().showMessage("Ready.")

    # -------------------- default paths (persisted) -------------------------
    def _path_pickers(self):
        return {"mri": self.mri_picker, "ct": self.ct_picker,
                "labels": self.labels_picker, "centroids": self.centroids_picker}

    def _apply_saved_paths(self):
        """Prefill the Data tab pickers from the saved defaults, if present."""
        saved = settings.load()
        if not saved:
            return
        n = 0
        for k, picker in self._path_pickers().items():
            val = saved.get(k, "")
            if val:
                picker.edit.setText(val); n += 1
        if n:
            self.data_status.setText(
                f"Restored {n} default path(s) from settings. Press Load to read the volumes.")

    def _on_save_defaults(self):
        vals = {k: p.edit.text().strip() for k, p in self._path_pickers().items()}
        missing = [k for k, v in vals.items() if v and not Path(v).is_file()]
        settings.set_many(**vals)
        msg = f"Saved as default -> {settings.config_path()}"
        if missing:
            msg += f"   [warning] these do not exist: {', '.join(missing)}"
        self.data_status.setText(msg)

    def _on_clear_defaults(self):
        settings.clear()
        self.data_status.setText(
            "Saved defaults removed. The app will not prefill paths on next launch.")

    # -------------------- tab 1: data ---------------------------------------

    def _build_data_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            "Load the MRI, CT, segmentation labels and the DMBA centroids table."))
        nrrd_filter = "NRRD/.nhdr (*.nhdr *.nrrd);;All files (*.*)"
        self.mri_picker       = FilePicker("MRI:", nrrd_filter)
        self.ct_picker        = FilePicker("CT:",  nrrd_filter)
        self.labels_picker    = FilePicker("Labels:", nrrd_filter)
        self.centroids_picker = FilePicker(
            "Centroids:", "Centroids table (*.txt *.csv);;All files (*.*)")
        for x in [self.mri_picker, self.ct_picker, self.labels_picker, self.centroids_picker]:
            v.addWidget(x)

        btn_row = QtWidgets.QHBoxLayout()
        defaults_btn = QtWidgets.QPushButton("Use bundled paths")
        defaults_btn.setToolTip(
            "Fill all four paths to the DUKE DMBA dataset shipped alongside this toolbox.")
        defaults_btn.clicked.connect(self._on_use_defaults)
        btn_row.addWidget(defaults_btn)
        load_btn = QtWidgets.QPushButton("Load"); load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(load_btn)
        btn_row.addStretch()

        save_btn = QtWidgets.QPushButton("Save as default")
        save_btn.setToolTip(
            "Remember these four paths and prefill them automatically at every launch.")
        save_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #e74c3c; }")
        save_btn.clicked.connect(self._on_save_defaults)
        forget_btn = QtWidgets.QPushButton("Forget defaults")
        forget_btn.clicked.connect(self._on_clear_defaults)
        btn_row.addWidget(save_btn); btn_row.addWidget(forget_btn)
        v.addLayout(btn_row)

        self.data_status = QtWidgets.QLabel(""); self.data_status.setWordWrap(True)
        v.addWidget(self.data_status)
        cfg = QtWidgets.QLabel(f"Defaults are stored in:  {settings.config_path()}")
        cfg.setStyleSheet("color: #666; font-size: 10px;")
        v.addWidget(cfg)
        v.addStretch()
        return w

    # Default DUKE DMBA data paths. The "Mouse DUKE"
    # folder is expected to sit next to the fus_planner package, i.e. at the
    # project root:  <project>/Mouse DUKE/...  and  <project>/fus_planner/...
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
    _DUKE = _PROJECT_ROOT / "Mouse DUKE"
    DEFAULT_PATHS = {
        "mri":       str(_DUKE / "DMBA_N02_dwi_M4D.nhdr"),
        "ct":        str(_DUKE / "CT DUKE Mouse" / "DMBA_N20_230328-4-1_CT_M4D.nhdr"),
        "labels":    str(_DUKE / "DMBA_RCCF_labels_M4D.nhdr"),
        "centroids": str(_DUKE / "DMBA_RCCF_labels_M4D" / "DMBA_RCCF_labels_centroids.txt"),
    }

    def _on_use_defaults(self):
        self.mri_picker.edit.setText(self.DEFAULT_PATHS["mri"])
        self.ct_picker.edit.setText(self.DEFAULT_PATHS["ct"])
        self.labels_picker.edit.setText(self.DEFAULT_PATHS["labels"])
        self.centroids_picker.edit.setText(self.DEFAULT_PATHS["centroids"])

    def _on_load(self):
        try:
            if self.labels_picker.path():
                self.labels_volume = load_volume(self.labels_picker.path())
            if self.ct_picker.path():
                self.ct_volume = load_volume(self.ct_picker.path())
            if self.centroids_picker.path():
                self.catalog = load_region_catalog(self.centroids_picker.path())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc)); return

        msg = []
        if self.labels_volume is not None:
            msg.append(f"Labels: {self.labels_volume.shape}, spacing "
                       f"{tuple(round(s, 4) for s in self.labels_volume.spacing)} mm")
        if self.ct_volume is not None:
            msg.append(f"CT:     {self.ct_volume.shape}, spacing "
                       f"{tuple(round(s, 4) for s in self.ct_volume.spacing)} mm")
        if self.catalog is not None:
            msg.append(f"Catalog: {len(self.catalog)} regions")
            self._refresh_region_search()
        self.data_status.setText("\n".join(msg) or "Nothing loaded.")
        self.statusBar().showMessage("Loaded.")
        self.picker.set_volumes(self.labels_volume, self.ct_volume)
        self._sync_picker_landmarks()
        self.region_preview.set_labels_volume(self.labels_volume)
        self._update_region_preview()

    # -------------------- tab 2: plan ---------------------------------------

    def _build_plan_tab(self):
        outer = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(outer); h.setContentsMargins(8, 8, 8, 8)

        # Left half: picker (occupies the full vertical space).
        h.addWidget(self._build_picker_box(), 5)

        # Right half: top row puts landmarks + transducer side-by-side
        # (saves vertical space); plan section gets the rest.
        right = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self._build_landmarks_box(), 1)
        top_row.addWidget(self._build_transducer_box(), 1)
        rv.addLayout(top_row)

        # Plan section is a horizontal split: controls (left) + region preview (right).
        # IMPORTANT: build the preview *before* the controls so that
        # `_scenario_changed` (called at the end of _build_planning_controls)
        # can find `self.region_preview` already initialised.
        plan_section = QtWidgets.QGroupBox("Plan")
        ph = QtWidgets.QHBoxLayout(plan_section)
        preview_box = self._build_region_preview_box()
        controls_box = self._build_planning_controls()
        ph.addWidget(controls_box, 5)
        ph.addWidget(preview_box, 5)
        rv.addWidget(plan_section, 1)

        h.addWidget(right, 5)
        return outer

    def _build_region_preview_box(self):
        gb = QtWidgets.QGroupBox("Target preview")
        v = QtWidgets.QVBoxLayout(gb)
        self.region_preview = RegionPreview()
        v.addWidget(self.region_preview)
        return gb

    def _build_landmarks_box(self):
        gb = QtWidgets.QGroupBox(
            "Landmarks (anatomical mm: +LR right, +AP anterior, +IS superior)")
        v = QtWidgets.QVBoxLayout(gb)
        self.lambda_input = TripletInput(("LR", "AP", "IS"), (0.0, -3.0, 1.0))
        self.eye_input    = TripletInput(("LR", "AP", "IS"), (2.0,  8.0, -2.0))
        self.lambda_input.valueChanged.connect(lambda *_: self._sync_picker_landmarks())
        self.eye_input.valueChanged.connect(lambda *_: self._sync_picker_landmarks())
        form = QtWidgets.QFormLayout()
        form.addRow("Lambda suture:", self.lambda_input)
        form.addRow("Bottom of left eye:", self.eye_input)
        v.addLayout(form)
        v.addWidget(QtWidgets.QLabel(
            "Tip: click the dorsal CT view below to set LR / AP, then dial in IS by hand.\n"
            "Adjust the skull threshold spinner above the picker if sutures are not visible."))
        return gb

    def _build_picker_box(self):
        gb = QtWidgets.QGroupBox("Pick landmarks on the dorsal CT view (height-shaded)")
        v = QtWidgets.QVBoxLayout(gb)
        self.picker = LandmarkPicker()
        self.picker.landmarkPicked.connect(self._on_landmark_picked)
        v.addWidget(self.picker)
        return gb

    def _on_landmark_picked(self, which, lr, ap):
        if which == "lambda":
            cur = self.lambda_input.value(); cur[0] = lr; cur[1] = ap
            self.lambda_input.set_value(cur)
        else:
            cur = self.eye_input.value(); cur[0] = lr; cur[1] = ap
            self.eye_input.set_value(cur)

    def _sync_picker_landmarks(self):
        self.picker.set_landmarks(self.lambda_input.value(), self.eye_input.value())

    def _build_transducer_box(self):
        gb = QtWidgets.QGroupBox("Transducer")
        form = QtWidgets.QFormLayout(gb)
        self.fwhm_x = QtWidgets.QDoubleSpinBox(); self.fwhm_x.setRange(0.05, 20.0); self.fwhm_x.setDecimals(3); self.fwhm_x.setValue(1.5); self.fwhm_x.setSuffix(" mm")
        self.fwhm_y = QtWidgets.QDoubleSpinBox(); self.fwhm_y.setRange(0.05, 20.0); self.fwhm_y.setDecimals(3); self.fwhm_y.setValue(1.5); self.fwhm_y.setSuffix(" mm")
        self.fwhm_z = QtWidgets.QDoubleSpinBox(); self.fwhm_z.setRange(0.1,  50.0); self.fwhm_z.setDecimals(3); self.fwhm_z.setValue(4.0); self.fwhm_z.setSuffix(" mm")
        self.focal_depth = QtWidgets.QDoubleSpinBox(); self.focal_depth.setRange(1.0, 100.0); self.focal_depth.setDecimals(2); self.focal_depth.setValue(20.0); self.focal_depth.setSuffix(" mm")
        self.skull_threshold = QtWidgets.QDoubleSpinBox(); self.skull_threshold.setRange(0.0, 100000.0); self.skull_threshold.setDecimals(0); self.skull_threshold.setValue(5000.0)
        # Keep the picker's threshold in sync with the planner's threshold
        self.skull_threshold.valueChanged.connect(
            lambda v: (self.picker.threshold_spin.blockSignals(True),
                       self.picker.threshold_spin.setValue(v),
                       self.picker.threshold_spin.blockSignals(False),
                       self.picker._draw()))
        self.skin_margin = QtWidgets.QDoubleSpinBox(); self.skin_margin.setRange(0.0, 5.0); self.skin_margin.setDecimals(3); self.skin_margin.setValue(0.05); self.skin_margin.setSuffix(" mm")
        self.z_offset = QtWidgets.QDoubleSpinBox(); self.z_offset.setRange(-20.0, 20.0); self.z_offset.setDecimals(3); self.z_offset.setSuffix(" mm")
        self.z_offset.setValue(0.0)
        self.mode_combo = QtWidgets.QComboBox(); self.mode_combo.addItems(["isotropic", "anisotropic"])
        self.iso_choice = QtWidgets.QComboBox(); self.iso_choice.addItems(["min", "mean", "max"])
        for label, w in [
            ("FWHM horizontal (lateral x):", self.fwhm_x),
            ("FWHM vertical (lateral y):",   self.fwhm_y),
            ("FWHM axial (along beam):",     self.fwhm_z),
            ("Focal depth (face → focus):",  self.focal_depth),
            ("Skull threshold (raw CT value):", self.skull_threshold),
            ("Skin margin above brain top:", self.skin_margin),
            ("z offset for skull lines:",    self.z_offset),
            ("Footprint mode:",              self.mode_combo),
            ("Isotropic FWHM choice:",       self.iso_choice),
        ]:
            form.addRow(label, w)
        return gb

    def _build_planning_controls(self):
        wrap = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(wrap); v.setContentsMargins(0, 0, 0, 0)

        h = QtWidgets.QHBoxLayout()
        self.scenario = QtWidgets.QButtonGroup(self)
        rb_wb = QtWidgets.QRadioButton("Whole brain"); rb_wb.setChecked(True)
        rb_lh = QtWidgets.QRadioButton("Left hemi")
        rb_rh = QtWidgets.QRadioButton("Right hemi")
        rb_rg = QtWidgets.QRadioButton("Region(s)")
        self.scenario.addButton(rb_wb, 0)
        self.scenario.addButton(rb_lh, 1)
        self.scenario.addButton(rb_rh, 2)
        self.scenario.addButton(rb_rg, 3)
        self.scenario.idClicked.connect(self._scenario_changed)
        for rb in (rb_wb, rb_lh, rb_rh, rb_rg):
            h.addWidget(rb)
        h.addStretch()
        v.addLayout(h)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Search:"))
        self.region_search = QtWidgets.QLineEdit()
        self.region_search.setPlaceholderText("e.g. striatum, hippocamp")
        self.region_search.textChanged.connect(self._refresh_region_search)
        h.addWidget(self.region_search, 1)
        self.clear_ticks_btn = QtWidgets.QPushButton("Clear")
        self.clear_ticks_btn.clicked.connect(self._clear_ticks)
        h.addWidget(self.clear_ticks_btn)
        v.addLayout(h)
        self.region_list = QtWidgets.QListWidget()
        self.region_list.itemChanged.connect(self._on_region_item_changed)
        v.addWidget(self.region_list, 1)

        opts = QtWidgets.QFormLayout()
        self.strategy_combo = QtWidgets.QComboBox()
        self.strategy_combo.addItems(["centroid", "coverage", "max-coverage"])
        self.strategy_combo.setToolTip(
            "centroid: hex lattice anchored on the region centroid; threshold filters spots.\n"
            "coverage: best of N x N translated lattices; threshold filters spots.\n"
            "max-coverage: greedy non-overlapping placement on a fine candidate grid;\n"
            "              threshold IGNORED, every spot that helps is kept.")
        self.strategy_combo.currentIndexChanged.connect(self._strategy_changed)
        opts.addRow("Strategy:", self.strategy_combo)
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0); self.threshold.setSingleStep(0.05); self.threshold.setDecimals(2); self.threshold.setValue(0.80)
        opts.addRow("Per-spot threshold:", self.threshold)
        v.addLayout(opts)

        run_btn = QtWidgets.QPushButton("Run plan")
        run_btn.setStyleSheet(
            "QPushButton { background-color: #c1272d; color: white; font-weight: bold; padding: 6px 14px; }"
            "QPushButton:hover { background-color: #a4181e; }")
        run_btn.clicked.connect(self._on_run)
        v.addWidget(run_btn)

        # Apply initial enable/disable state for whole-brain default
        self._scenario_changed(self.scenario.checkedId())
        return wrap

    def _scenario_changed(self, scenario_id: int):
        # Region search/tick UI is only relevant in scenario 3 ("Region(s)").
        is_region_mode = (scenario_id == 3)
        for w in (self.region_search, self.region_list, self.clear_ticks_btn):
            w.setEnabled(is_region_mode)
        self._update_region_preview()

    def _update_region_preview(self):
        if not hasattr(self, "region_preview"):
            return  # called before the preview widget exists
        sid = self.scenario.checkedId()
        if sid == 0:                              # whole brain
            self.region_preview.set_target(None)
        elif sid == 1 and self.catalog is not None:   # left hemisphere
            self.region_preview.set_target(self._hemisphere_labels("left"))
        elif sid == 2 and self.catalog is not None:   # right hemisphere
            self.region_preview.set_target(self._hemisphere_labels("right"))
        else:                                     # region(s)
            self.region_preview.set_target(self._ticked_labels())

    def _hemisphere_labels(self, side: str) -> List[int]:
        """Labels with anatomical centroid on the requested side.

        side='left'  -> centroid LR < 0  (subject's left, +LPS_x)
        side='right' -> centroid LR > 0  (subject's right, -LPS_x)
        Midline-centred structures (|LR| < 50 um) are excluded from both.
        """
        if self.catalog is None:
            return []
        sign = -1 if side == "left" else +1
        return [
            r.label for r in self.catalog
            if r.label != 0
            and not np.isnan(r.centroid_anat_mm).any()
            and sign * r.centroid_anat_mm[0] > 0.05
        ]

    def _strategy_changed(self, *_):
        # Disable threshold spinner in max-coverage mode.
        is_max = self.strategy_combo.currentText() == "max-coverage"
        self.threshold.setEnabled(not is_max)

    def _refresh_region_search(self):
        """Rebuild the region list. Ticked items always show at the top
        regardless of search text, so the user never loses a selection by
        searching for a different region."""
        self._sync_persistent_ticked_from_visible()
        self.region_list.blockSignals(True)
        self.region_list.clear()
        if self.catalog is None:
            self.region_list.blockSignals(False)
            return

        q = self.region_search.text().strip()
        ticked_regions = [self.catalog[lbl] for lbl in sorted(self._persistent_ticked)
                          if lbl in self.catalog.regions]
        if q:
            search_regions = self.catalog.find_by_name(q)
        else:
            search_regions = [r for r in self.catalog if r.label != 0][:300]

        seen, ordered = set(), []
        for r in ticked_regions + search_regions:
            if r.label in seen: continue
            seen.add(r.label); ordered.append(r)

        for r in ordered[:500]:
            label_str = f"[{r.label:5d}] {r.display_name}  ({r.voxels} voxels)"
            if r.label in self._persistent_ticked:
                label_str = "* " + label_str
            item = QtWidgets.QListWidgetItem(label_str)
            item.setData(QtCore.Qt.UserRole, r.label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if r.label in self._persistent_ticked
                               else QtCore.Qt.Unchecked)
            self.region_list.addItem(item)
        self.region_list.blockSignals(False)

    def _sync_persistent_ticked_from_visible(self):
        for i in range(self.region_list.count()):
            it = self.region_list.item(i)
            label = int(it.data(QtCore.Qt.UserRole))
            if it.checkState() == QtCore.Qt.Checked:
                self._persistent_ticked.add(label)
            else:
                self._persistent_ticked.discard(label)

    def _on_region_item_changed(self, item):
        label = int(item.data(QtCore.Qt.UserRole))
        if item.checkState() == QtCore.Qt.Checked:
            self._persistent_ticked.add(label)
        else:
            self._persistent_ticked.discard(label)
        self._update_region_preview()

    def _ticked_labels(self):
        self._sync_persistent_ticked_from_visible()
        return sorted(self._persistent_ticked)

    def _clear_ticks(self):
        self._persistent_ticked.clear()
        self._refresh_region_search()
        self._update_region_preview()

    # -------------------- tab 3: results ------------------------------------

    def _build_results_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        self.summary_label = QtWidgets.QLabel(
            "Run a plan from tab 2 to see the xy / xz / yz overlays here.")
        self.summary_label.setWordWrap(True)
        v.addWidget(self.summary_label)

        # Top row: xy / xz / yz with sliders.
        top_row = QtWidgets.QHBoxLayout(); v.addLayout(top_row, 3)
        self.xy_canvas, self.xy_slider, self.xy_slider_label = self._make_plot_panel("Slice IS:", top_row)
        self.xz_canvas, self.xz_slider, self.xz_slider_label = self._make_plot_panel("Slice AP:", top_row)
        self.yz_canvas, self.yz_slider, self.yz_slider_label = self._make_plot_panel("Slice LR:", top_row)
        self.xy_slider.valueChanged.connect(lambda *_: self._render_plots())
        self.xz_slider.valueChanged.connect(lambda *_: self._render_plots())
        self.yz_slider.valueChanged.connect(lambda *_: self._render_plots())

        # Per-panel hover handlers -- look up the spot under the cursor and
        # select the matching row in the coordinate table.
        self._hover_last_id = {}        # canvas id -> last spot id reported
        self.xy_canvas.mpl_connect("motion_notify_event",
                                    lambda e: self._on_hover(e, "xy"))
        self.xz_canvas.mpl_connect("motion_notify_event",
                                    lambda e: self._on_hover(e, "xz"))
        self.yz_canvas.mpl_connect("motion_notify_event",
                                    lambda e: self._on_hover(e, "yz"))

        # Bottom row: CT dorsal + stars on the left, coordinate table on the right.
        bottom = QtWidgets.QHBoxLayout(); v.addLayout(bottom, 4)
        ct_wrap = QtWidgets.QWidget(); ct_v = QtWidgets.QVBoxLayout(ct_wrap)
        ct_v.setContentsMargins(0, 0, 0, 0)
        ct_v.addWidget(QtWidgets.QLabel("CT dorsal skull + stimulation grid"))
        ct_fig = Figure(figsize=(5.5, 5.0))
        self.ct_canvas = FigureCanvas(ct_fig); ct_v.addWidget(self.ct_canvas, 1)
        self.ct_canvas.mpl_connect("motion_notify_event",
                                    lambda e: self._on_hover(e, "ct"))
        bottom.addWidget(ct_wrap, 4)

        table_wrap = QtWidgets.QWidget(); tv = QtWidgets.QVBoxLayout(table_wrap)
        tv.setContentsMargins(0, 0, 0, 0)

        thdr = QtWidgets.QHBoxLayout()
        thdr.addWidget(QtWidgets.QLabel("Spot coordinates"))
        thdr.addStretch()
        self.check_all_btn = QtWidgets.QPushButton("Check all")
        self.check_all_btn.setToolTip("Include every spot in the plots and the export.")
        self.check_all_btn.clicked.connect(lambda: self._set_all_kept(True))
        self.check_all_btn.setEnabled(False)
        self.uncheck_all_btn = QtWidgets.QPushButton("Uncheck all")
        self.uncheck_all_btn.setToolTip("Exclude every spot, then tick the ones you want.")
        self.uncheck_all_btn.clicked.connect(lambda: self._set_all_kept(False))
        self.uncheck_all_btn.setEnabled(False)
        thdr.addWidget(self.check_all_btn); thdr.addWidget(self.uncheck_all_btn)
        tv.addLayout(thdr)

        self.table = QtWidgets.QTableWidget(0, 0); tv.addWidget(self.table, 1)
        # Guard so programmatic population does not re-enter the toggle handler.
        self._table_populating = False
        self.table.itemChanged.connect(self._on_table_item_changed)
        bottom.addWidget(table_wrap, 5)

        # Action buttons.
        h = QtWidgets.QHBoxLayout()
        self.diag_btn = QtWidgets.QPushButton("Skull diagnostic...")
        self.diag_btn.setToolTip(
            "Cross-section view of CT skull + brain mask along the focal column.\n"
            "Use it to verify the skull threshold and the inner / outer-skull positions.")
        self.diag_btn.clicked.connect(self._on_skull_diag); self.diag_btn.setEnabled(False)
        self.save_csv_btn = QtWidgets.QPushButton("Save CSV...")
        self.save_csv_btn.clicked.connect(self._on_save_csv); self.save_csv_btn.setEnabled(False)
        self.save_pdf_btn = QtWidgets.QPushButton("Save PDF...")
        self.save_pdf_btn.clicked.connect(self._on_save_pdf); self.save_pdf_btn.setEnabled(False)
        h.addWidget(self.diag_btn); h.addStretch()
        h.addWidget(self.save_csv_btn); h.addWidget(self.save_pdf_btn)
        v.addLayout(h)
        return w

    def _make_plot_panel(self, slider_label, parent_layout, has_slider=True):
        wrap = QtWidgets.QWidget()
        vv = QtWidgets.QVBoxLayout(wrap); vv.setContentsMargins(0, 0, 0, 0)
        fig = Figure(figsize=(4.0, 3.6)); canvas = FigureCanvas(fig)
        vv.addWidget(canvas, 1)
        slider = val_label = None
        if has_slider:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(slider_label))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100); slider.setValue(50); slider.setEnabled(False)
            row.addWidget(slider, 1)
            val_label = QtWidgets.QLabel("--"); row.addWidget(val_label)
            vv.addLayout(row)
        else:
            vv.addWidget(QtWidgets.QLabel(slider_label))
        parent_layout.addWidget(wrap, 1)
        return canvas, slider, val_label

    # -------------------- run -----------------------------------------------

    def _stage_frame(self):
        return build_stage_frame(
            anat_to_lps(self.lambda_input.value()),
            anat_to_lps(self.eye_input.value()))

    def _footprint(self):
        return footprint_from_fwhm(self.fwhm_x.value(), self.fwhm_y.value(),
                                   mode=self.mode_combo.currentText(),
                                   isotropic_choice=self.iso_choice.currentText())

    def _selected_targets(self):
        sid = self.scenario.checkedId()
        if sid == 0:
            return None, whole_brain_centroid_voxel(self.catalog), "Whole brain"
        if sid in (1, 2):
            side = "left" if sid == 1 else "right"
            label_list = self._hemisphere_labels(side)
            if not label_list:
                raise ValueError(f"No regions on the {side} side of the catalog.")
            regions = [self.catalog[l] for l in label_list]
            centroid = subset_centroid_voxel(regions)
            return label_list, centroid, f"{side.capitalize()} hemisphere"
        # sid == 3: explicit region selection
        ticked = self._ticked_labels()
        if not ticked:
            raise ValueError("Tick at least one region in the list, or switch to Whole brain / hemisphere.")
        regions = [self.catalog[lbl] for lbl in ticked]
        centroid = subset_centroid_voxel(regions)
        if len(regions) == 1:
            title = f"Region: {regions[0].display_name}"
        else:
            title = "Regions: " + ", ".join(r.display_name for r in regions[:4])
            if len(regions) > 4:
                title += f", +{len(regions) - 4} more"
        return ticked, centroid, title

    def _on_run(self):
        if self.labels_volume is None or self.catalog is None:
            QtWidgets.QMessageBox.warning(self, "Missing data",
                "Load labels and centroids on tab 1 first."); return
        try:
            stage = self._stage_frame()
            target_labels, centroid_vox, title = self._selected_targets()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot run", str(exc)); return

        centroid_lps = self.labels_volume.voxel_to_world(centroid_vox)

        if self.ct_volume is None:
            zplan = ZPlanePlan(
                centroid_lps_mm=centroid_lps,
                z_focus_mm=float(centroid_lps[2]),
                z_inner_skull_mm=float("nan"), z_outer_skull_mm=float("nan"),
                skull_thickness_mm=float("nan"),
                z_transducer_face_mm=float(centroid_lps[2]) + self.focal_depth.value(),
                coupling_gap_mm=float("nan"),
                focal_depth_mm=self.focal_depth.value(),
                skull_threshold_hu=self.skull_threshold.value(),
                z_brain_top_mm=float("nan"),
            )
        else:
            try:
                zplan = plan_z_for_centroid(
                    centroid_lps, self.ct_volume, self.labels_volume,
                    focal_depth_mm=self.focal_depth.value(),
                    skull_threshold_hu=self.skull_threshold.value(),
                    skin_margin_mm=self.skin_margin.value())
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Skull search failed", str(exc)); return
            zo = self.z_offset.value()
            if zo != 0.0 and not np.isnan(zplan.z_outer_skull_mm):
                zplan = ZPlanePlan(
                    centroid_lps_mm=zplan.centroid_lps_mm,
                    z_focus_mm=zplan.z_focus_mm,
                    z_inner_skull_mm=zplan.z_inner_skull_mm + zo,
                    z_outer_skull_mm=zplan.z_outer_skull_mm + zo,
                    skull_thickness_mm=zplan.skull_thickness_mm,
                    z_transducer_face_mm=zplan.z_transducer_face_mm,
                    coupling_gap_mm=zplan.z_transducer_face_mm - (zplan.z_outer_skull_mm + zo),
                    focal_depth_mm=zplan.focal_depth_mm,
                    skull_threshold_hu=zplan.skull_threshold_hu,
                    z_brain_top_mm=zplan.z_brain_top_mm,
                )

        fp = self._footprint()
        strategy = self.strategy_combo.currentText()
        sid = self.scenario.checkedId()

        if sid in (0, 1, 2):
            # Whole brain or whole hemisphere -- the target mass fills the focal
            # slice, so 2D hex tiling is appropriate (and much faster than 3D).
            plane = extract_focal_plane(self.labels_volume, z_lps_mm=zplan.z_focus_mm,
                                        target_labels=target_labels)
            t = 1e-3 if strategy == "max-coverage" else self.threshold.value()
            plan = hex_tile_plan(plane, fp, coverage_threshold=t)
        else:
            # Region targets: 3D-aware planning. Spots are placed on the flat
            # xy projection of the 3D target, scored by 3D target volume inside
            # the beam ellipsoid (semi-axes rx, ry, rz=axial_FWHM/2).
            plan, plane = plan_3d(
                labels=self.labels_volume,
                target_labels=target_labels,
                z_focus_lps_mm=zplan.z_focus_mm,
                footprint=fp,
                fwhm_axial_mm=self.fwhm_z.value(),
                strategy=strategy,
                coverage_threshold=self.threshold.value(),
                seed_xy_lps_mm=(centroid_lps[0], centroid_lps[1]),
                ds_factor=4,
            )

        df = plan_to_dataframe(plan, stage, zplan)
        # ``plan`` / ``df`` stay the FULL plan for the lifetime of the bundle:
        # table rows and hover spot-ids index into them. ``keep`` is the user's
        # per-spot include/exclude mask; everything exported or plotted is
        # derived from it via _kept_plan() / _kept_df().
        self.last_plan_bundle = dict(
            plane=plane, plan=plan, zplan=zplan, df=df,
            keep=[True] * plan.n_spots,
            target_labels=target_labels, title=title,
            centroid_lps=centroid_lps, stage=stage)

        self._init_sliders()
        self._render_table(df)
        self._render_plots()

        self.save_csv_btn.setEnabled(True); self.save_pdf_btn.setEnabled(True)
        self.check_all_btn.setEnabled(True); self.uncheck_all_btn.setEnabled(True)
        self.diag_btn.setEnabled(self.ct_volume is not None)
        self._update_status_and_summary()

    # -------------------- per-spot include / exclude -------------------------

    def _kept_plan(self) -> Optional[HexTilePlan]:
        """The plan restricted to ticked spots, with coverage recomputed."""
        b = self.last_plan_bundle
        if not b:
            return None
        return subset_plan(b["plan"], b["keep"])

    def _excluded_spots(self):
        b = self.last_plan_bundle
        if not b:
            return []
        return [s for s, k in zip(b["plan"].spots, b["keep"]) if not k]

    def _kept_df(self):
        """The coordinate table restricted to ticked spots, renumbered 1..N."""
        b = self.last_plan_bundle
        df = b["df"][np.asarray(b["keep"], dtype=bool)].copy()
        df["spot_id"] = np.arange(1, len(df) + 1)
        return df.reset_index(drop=True)

    def _on_table_item_changed(self, item):
        if self._table_populating or item.column() != 0:
            return
        b = self.last_plan_bundle
        if not b:
            return
        row = item.row()
        if not (0 <= row < len(b["keep"])):
            return
        b["keep"][row] = item.checkState() == QtCore.Qt.Checked
        self._grey_row(row, kept=b["keep"][row])
        self._render_plots()
        self._update_status_and_summary()

    def _set_all_kept(self, kept: bool):
        b = self.last_plan_bundle
        if not b:
            return
        state = QtCore.Qt.Checked if kept else QtCore.Qt.Unchecked
        self._table_populating = True
        for r in range(len(b["keep"])):
            b["keep"][r] = kept
            self.table.item(r, 0).setCheckState(state)
            self._grey_row(r, kept)
        self._table_populating = False
        self._render_plots()
        self._update_status_and_summary()

    def _grey_row(self, row: int, kept: bool):
        # setForeground() emits itemChanged, which would re-enter
        # _on_table_item_changed and replot a second time. Suppress it.
        colour = QtGui.QColor("#202020") if kept else QtGui.QColor("#A0A0A0")
        was = self._table_populating
        self._table_populating = True
        try:
            for c in range(self.table.columnCount()):
                it = self.table.item(row, c)
                if it is not None:
                    it.setForeground(colour)
        finally:
            self._table_populating = was

    def _update_status_and_summary(self):
        b = self.last_plan_bundle
        if not b:
            return
        plan = self._kept_plan()
        n_off = len(b["keep"]) - plan.n_spots
        msg = (f"{b['title']}: {plan.n_spots} spots, "
               f"coverage {plan.total_target_coverage_pct:.1f}%")
        if n_off:
            msg += f"  ({n_off} excluded)"
        self.statusBar().showMessage(msg)
        self.summary_label.setText(self._format_summary())
        any_kept = plan.n_spots > 0
        self.save_csv_btn.setEnabled(any_kept)
        self.save_pdf_btn.setEnabled(any_kept)

    def _format_summary(self):
        b = self.last_plan_bundle
        if not b: return ""
        plan = self._kept_plan(); z = b["zplan"]
        gap = "n/a" if np.isnan(z.coupling_gap_mm) else f"{z.coupling_gap_mm:.2f} mm"
        sk  = "n/a" if np.isnan(z.skull_thickness_mm) else f"{z.skull_thickness_mm:.2f} mm"
        bt  = "n/a" if np.isnan(z.z_brain_top_mm) else f"{z.z_brain_top_mm:+.2f} mm"
        n_off = len(b["keep"]) - plan.n_spots
        excl = f" ({n_off} excluded)" if n_off else ""
        return (f"{b['title']}  |  {plan.n_spots} spots{excl}  |  coverage "
                f"{plan.total_target_coverage_pct:.1f}%  |  z_focus="
                f"{z.z_focus_mm:+.3f}  |  brain_top@focus={bt}  |  "
                f"z_face={z.z_transducer_face_mm:+.3f}  |  skull={sk}  |  gap={gap}")

    # -------------------- sliders + redraw ----------------------------------

    def _init_sliders(self):
        b = self.last_plan_bundle
        if not b or self.labels_volume is None: return
        labels = self.labels_volume
        Ni, Nj, Nk = labels.shape
        c_lps = b["centroid_lps"]
        ic, jc, kc = np.round(labels.world_to_voxel(c_lps)).astype(int)
        for sl, val, hi in [(self.xy_slider, kc, Nk - 1),
                            (self.xz_slider, jc, Nj - 1),
                            (self.yz_slider, ic, Ni - 1)]:
            sl.blockSignals(True)
            sl.setRange(0, hi); sl.setValue(int(np.clip(val, 0, hi)))
            sl.setEnabled(True); sl.blockSignals(False)

    def _vox_to_world_axis(self, k, axis):
        v = np.zeros(3); v[axis] = k
        return float(self.labels_volume.voxel_to_world(v)[axis])

    def _render_plots(self):
        b = self.last_plan_bundle
        if not b or self.labels_volume is None: return
        labels = self.labels_volume
        # Only ticked spots are drawn normally; de-selected ones are overlaid
        # afterwards as grey dashed outlines so gaps stay visible.
        plan, zplan = self._kept_plan(), b["zplan"]
        excluded = self._excluded_spots()
        rz = self.fwhm_z.value() / 2.0
        z_focus = b["plan"].plane_z_lps_mm
        target_labels = b["target_labels"]
        stage = b["stage"]
        fwhm_z = self.fwhm_z.value()

        # xy
        k = self.xy_slider.value()
        z_lps = self._vox_to_world_axis(k, 2); is_anat = z_lps
        plane = extract_focal_plane(labels, z_lps_mm=z_lps, target_labels=target_labels)
        self.xy_slider_label.setText(f"IS={is_anat:+.2f} mm")
        fig = self.xy_canvas.figure; fig.clear(); ax = fig.add_subplot(111)
        plot_xy(ax, plane, plan, fwhm_axial_mm=fwhm_z,
                slice_z_lps_mm=z_lps, stage_frame=stage,
                title=f"xy @ IS={is_anat:+.2f} mm")
        _draw_excluded_spots(ax, excluded, "xy", rz, z_focus)

        _no_layout_engine(fig)
        fig.subplots_adjust(left=0.12, right=0.78, top=0.92, bottom=0.13)
        self.xy_canvas.draw()

        # xz
        j = self.xz_slider.value()
        y_lps = self._vox_to_world_axis(j, 1); ap = -y_lps
        self.xz_slider_label.setText(f"AP={ap:+.2f} mm")
        fig = self.xz_canvas.figure; fig.clear(); ax = fig.add_subplot(111)
        plot_xz_or_yz(ax, labels, plan, zplan, fwhm_z, target_labels,
                      axis="xz", fixed_anat_mm=ap, title="xz",
                      show_legend=False, stage_frame=stage)
        _draw_excluded_spots(ax, excluded, "xz", rz, z_focus)
        _no_layout_engine(fig)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.13)
        self.xz_canvas.draw()

        # yz
        i = self.yz_slider.value()
        x_lps = self._vox_to_world_axis(i, 0); lr = -x_lps
        self.yz_slider_label.setText(f"LR={lr:+.2f} mm")
        fig = self.yz_canvas.figure; fig.clear(); ax = fig.add_subplot(111)
        plot_xz_or_yz(ax, labels, plan, zplan, fwhm_z, target_labels,
                      axis="yz", fixed_anat_mm=lr, title="yz",
                      show_legend=False, stage_frame=stage)
        _draw_excluded_spots(ax, excluded, "yz", rz, z_focus)
        _no_layout_engine(fig)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.13)
        self.yz_canvas.draw()

        # CT dorsal + grid
        fig = self.ct_canvas.figure; fig.clear(); ax = fig.add_subplot(111)
        if self.ct_volume is not None:
            plot_ct_dorsal(ax, self.ct_volume,
                           skull_threshold=self.skull_threshold.value(),
                           plan=plan, stage_frame=stage,
                           title="CT dorsal + stimulation grid")
            _draw_excluded_spots(ax, excluded, "ct", rz, z_focus)
        else:
            ax.text(0.5, 0.5, "Load CT for this view.",
                    ha="center", va="center", transform=ax.transAxes)
        _no_layout_engine(fig)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.13)
        self.ct_canvas.draw()

    def _on_hover(self, event, panel: str):
        """When the cursor enters a focal-spot ellipse on a panel, select that
        spot's row in the coordinate table. ``panel`` is one of
        ``xy`` / ``xz`` / ``yz`` / ``ct``."""
        if event.inaxes is None or self.last_plan_bundle is None:
            return
        plan = self.last_plan_bundle["plan"]
        if not plan.spots:
            return
        rz = self.fwhm_z.value() / 2.0
        z_focus = plan.plane_z_lps_mm

        hit_id = None
        for k, s in enumerate(plan.spots, start=1):
            cx, cy = s.center_world_xy_mm
            if panel in ("xy", "ct"):
                u, v = -cx, -cy            # anatomical (LR, AP)
                a, b = s.rx_mm, s.ry_mm
            elif panel == "xz":
                u, v = -cx, z_focus        # (LR, IS)
                a, b = s.rx_mm, rz
            elif panel == "yz":
                u, v = -cy, z_focus        # (AP, IS)
                a, b = s.ry_mm, rz
            else:
                return
            if ((event.xdata - u) / a) ** 2 + ((event.ydata - v) / b) ** 2 <= 1.0:
                hit_id = k
                break

        last = self._hover_last_id.get(panel)
        if hit_id == last:
            return
        self._hover_last_id[panel] = hit_id
        if hit_id is not None:
            self.table.selectRow(hit_id - 1)
            self.statusBar().showMessage(f"Spot {hit_id}")

    def _render_table(self, df):
        # Column 0 is the include/exclude checkbox; data columns are shifted +1.
        # itemChanged fires on every setItem, so suppress the handler while we
        # populate, otherwise each cell would trigger a full replot.
        self._table_populating = True
        try:
            self.table.clear()
            self.table.setRowCount(len(df))
            self.table.setColumnCount(len(df.columns) + 1)

            use_hdr = QtWidgets.QTableWidgetItem("use")
            use_hdr.setToolTip("Untick to drop this spot from the plots, the CSV and the PDF.")
            self.table.setHorizontalHeaderItem(0, use_hdr)
            # Coloured headers
            for c, col in enumerate(df.columns):
                group, colour = COLUMN_GROUP_LOOKUP.get(col, ("?", "#FFFFFF"))
                hi = QtWidgets.QTableWidgetItem(col)
                hi.setBackground(QtGui.QColor(colour))
                hi.setToolTip(f"group: {group}")
                self.table.setHorizontalHeaderItem(c + 1, hi)

            for r in range(len(df)):
                chk = QtWidgets.QTableWidgetItem()
                chk.setFlags(QtCore.Qt.ItemIsUserCheckable |
                             QtCore.Qt.ItemIsEnabled |
                             QtCore.Qt.ItemIsSelectable)
                chk.setCheckState(QtCore.Qt.Checked)
                chk.setToolTip("Include this spot in the export.")
                self.table.setItem(r, 0, chk)
                for c, col in enumerate(df.columns):
                    v = df.iat[r, c]
                    if isinstance(v, (int, np.integer)):
                        txt = str(int(v))
                    elif isinstance(v, (float, np.floating)):
                        txt = "n/a" if np.isnan(v) else f"{v:.4f}"
                    else:
                        txt = str(v)
                    self.table.setItem(r, c + 1, QtWidgets.QTableWidgetItem(txt))
            self.table.resizeColumnsToContents()
        finally:
            self._table_populating = False

    def _on_skull_diag(self):
        b = self.last_plan_bundle
        if not b or self.ct_volume is None: return
        dlg = SkullDiagnosticDialog(self, self.ct_volume, self.labels_volume, b["zplan"])
        dlg.exec_()

    # -------------------- save ----------------------------------------------

    def _no_spots_warning(self) -> bool:
        """True (and warns) if the user has unticked every spot."""
        if self._kept_plan().n_spots == 0:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to export",
                "Every spot is unticked. Tick at least one spot in the table.")
            return True
        return False

    def _on_save_csv(self):
        if not self.last_plan_bundle: return
        if self._no_spots_warning(): return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plan CSV", "fus_plan.csv", "CSV (*.csv)")
        if path:
            write_csv(self._kept_df(), path)
            n_off = len(self.last_plan_bundle["keep"]) - self._kept_plan().n_spots
            extra = f" ({n_off} spots excluded)" if n_off else ""
            self.statusBar().showMessage(f"CSV saved to {path}{extra}")

    def _on_save_pdf(self):
        if not self.last_plan_bundle: return
        if self._no_spots_warning(): return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plan PDF", "fus_plan.pdf", "PDF (*.pdf)")
        if not path: return
        b = self.last_plan_bundle
        plan = self._kept_plan()
        n_off = len(b["keep"]) - plan.n_spots
        title = b["title"] + (f"  ({n_off} spots excluded by user)" if n_off else "")
        render_plan_pdf(path, b["plane"], plan, b["zplan"], self.labels_volume,
                        fwhm_axial_mm=self.fwhm_z.value(),
                        target_labels=b["target_labels"], title=title,
                        stage_frame=b["stage"])
        extra = f" ({n_off} spots excluded)" if n_off else ""
        self.statusBar().showMessage(f"PDF saved to {path}{extra}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()