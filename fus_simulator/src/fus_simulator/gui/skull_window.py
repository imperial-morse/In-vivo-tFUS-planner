"""Skull section (embeddable widget) - lives under the transducer controls on the
Parameters tab.

Builds the skull medium straight from the high-resolution CT (Duke dataset):
you pick a CT (.nhdr/.nrrd) once, "Prepare" thresholds bone and downsamples it to
a compact fine bone-fraction file, and every simulation grid is then built by
downsampling that fraction (never upsampling a coarse mask). The preview shows the
resulting bone mask with the transducer bowl and focus on top.

"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.patches import Rectangle

try:
    from ..core.params import SimParams
    from ..core.grid import build_grid_spec
    from ..core.ct_skull import CTSkullSource, prepare_ct_bonefraction
    from ..core.transducer import bowl_arc_xz
    from ..core import atlas
    from .viz import StaticCanvas
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fus_simulator.core.params import SimParams
    from fus_simulator.core.grid import build_grid_spec
    from fus_simulator.core.ct_skull import CTSkullSource, prepare_ct_bonefraction
    from fus_simulator.core.transducer import bowl_arc_xz
    from fus_simulator.core import atlas
    from fus_simulator.gui.viz import StaticCanvas


# Default CT location: the Duke set inside the project, if present.
def _default_ct_dir() -> str:
    try:
        root = Path(__file__).resolve().parents[4]
        d = root / "Mouse DUKE" / "CT DUKE Mouse"
        if d.is_dir():
            return str(d)
    except Exception:
        pass
    return ""


class _PrepareWorker(QtCore.QThread):
    """Runs the (slow) CT -> bone-fraction preparation off the GUI thread."""
    progress = QtCore.pyqtSignal(str, float)
    done = QtCore.pyqtSignal(str)          # out_h5 path
    failed = QtCore.pyqtSignal(str)

    def __init__(self, nhdr_path: str, out_h5: str, threshold: float):
        super().__init__()
        self._nhdr = nhdr_path; self._out = out_h5; self._thr = threshold

    def run(self):
        try:
            prepare_ct_bonefraction(
                self._nhdr, self._out, bone_threshold=self._thr,
                progress=lambda m, f: self.progress.emit(m, f))
            self.done.emit(self._out)
        except Exception as err:  # noqa: BLE001
            self.failed.emit(str(err))


class SkullWidget(QtWidgets.QWidget):
    def __init__(self, get_params: Callable[[], SimParams], parent=None):
        super().__init__(parent)
        self._get_params = get_params
        self._skull: Optional[CTSkullSource] = None
        self._emb = None
        self._gspec = None
        self._busy = False
        self._worker: Optional[_PrepareWorker] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None
        self._regions: list = []          # atlas regions (empty = targeting off)
        self._targeted: str = ""          # name of the region currently targeted

        # debounce: dragging offsets / changing transducer fires rapidly
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self._redraw_now)

        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._build_controls())
        srow = QtWidgets.QHBoxLayout(); srow.addStretch(1)
        sbtn = QtWidgets.QPushButton("Save figure..."); sbtn.setMaximumWidth(140)
        sbtn.clicked.connect(self._save_figure)
        srow.addWidget(sbtn); v.addLayout(srow)
        self.view = StaticCanvas(figsize=(9.0, 4.6), min_size=(600, 300))
        v.addWidget(self.view, 1)
        self.info = QtWidgets.QLabel(
            "Pick a CT (.nhdr/.nrrd) and press Prepare, or Browse to a prepared "
            "*_bonefrac.h5, to overlay the skull on the transducer.")
        self.info.setWordWrap(True)
        v.addWidget(self.info)

        # NOTE: no auto-detection here. The CT path comes from the Settings tab
        # (MainWindow._apply_saved_paths fills path_edit), so the box stays empty
        # until the user has actually saved a default or browsed to a file.
        self.schedule_redraw()   # draw transducer geometry even before a skull

    # ------------------------------------------------------------ controls
    def _build_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Skull (from CT)")
        v = QtWidgets.QVBoxLayout(box)

        # --- file row: CT / prepared file + Browse + red Prepare/Load ---
        r1 = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse..."); browse.clicked.connect(self._browse)
        self.load_btn = QtWidgets.QPushButton("Prepare / Load")
        self.load_btn.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #e74c3c; }")
        self.load_btn.clicked.connect(self._prepare_or_load)
        r1.addWidget(QtWidgets.QLabel("CT / prepared:")); r1.addWidget(self.path_edit, 1)
        r1.addWidget(browse); r1.addWidget(self.load_btn)
        v.addLayout(r1)

        # --- build row: CT threshold + bone fill + rotate ---
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("CT bone thr:"))
        self.ct_thr = QtWidgets.QDoubleSpinBox()
        self.ct_thr.setRange(0.0, 65535.0); self.ct_thr.setDecimals(0)
        self.ct_thr.setSingleStep(250.0); self.ct_thr.setValue(6000.0)
        self.ct_thr.setToolTip("CT intensity above which a voxel is bone (used when preparing).")
        r2.addWidget(self.ct_thr)

        r2.addWidget(QtWidgets.QLabel("Bone fill:"))
        self.fill_thr = QtWidgets.QDoubleSpinBox()
        self.fill_thr.setRange(0.05, 0.95); self.fill_thr.setDecimals(2)
        self.fill_thr.setSingleStep(0.05); self.fill_thr.setValue(0.30)
        self.fill_thr.setToolTip(
            "A grid voxel becomes bone if this fraction of it is bone.\n"
            "Lower keeps a thin skull continuous on coarse grids; higher is\n"
            "geometrically tighter but can perforate the vault at low PPW.")
        self.fill_thr.valueChanged.connect(self.schedule_redraw)
        r2.addWidget(self.fill_thr)
        r2.addStretch(1)
        v.addLayout(r2)

        # --- placement row: offsets (one box) + preview ppw ---
        r3 = QtWidgets.QHBoxLayout()
        obox = QtWidgets.QGroupBox("Skull offset (mm)")
        oh = QtWidgets.QHBoxLayout(obox); oh.setContentsMargins(6, 2, 6, 2)

        def _off(label):
            oh.addWidget(QtWidgets.QLabel(label))
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-60.0, 60.0); sb.setDecimals(1); sb.setSingleStep(0.5); sb.setValue(0.0)
            sb.valueChanged.connect(self.schedule_redraw)
            oh.addWidget(sb)
            return sb
        self.off_ax = _off("AX")   # axial (along beam)
        self.off_lr = _off("LR")   # left-right
        self.off_ap = _off("AP")   # anterior-posterior
        r3.addWidget(obox)

        r3.addWidget(QtWidgets.QLabel("Preview PPW:"))
        self.preview_ppw = QtWidgets.QSpinBox()
        self.preview_ppw.setRange(3, 12); self.preview_ppw.setValue(5)
        self.preview_ppw.setToolTip("Grid resolution for THIS preview only (low = light).")
        self.preview_ppw.valueChanged.connect(self.schedule_redraw)
        r3.addWidget(self.preview_ppw)
        r3.addStretch(1)
        v.addLayout(r3)

        # --- atlas region targeting (optional; manual offsets still work) ---
        v.addWidget(self._build_target_box())
        return box

    def _build_target_box(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Target a brain region (optional)")
        g = QtWidgets.QVBoxLayout(box)

        top = QtWidgets.QHBoxLayout()
        self.region_search = QtWidgets.QLineEdit()
        self.region_search.setPlaceholderText(
            "Search regions (e.g. hippocampus, somatosensory, striatum)...")
        self.region_search.textChanged.connect(self._refresh_region_list)
        top.addWidget(self.region_search, 1)
        top.addWidget(QtWidgets.QLabel("min voxels:"))
        self.region_minvox = QtWidgets.QSpinBox()
        self.region_minvox.setRange(0, 10_000_000); self.region_minvox.setSingleStep(1000)
        self.region_minvox.setValue(5000)
        self.region_minvox.setToolTip(
            "Hide tiny structures. The atlas has 811 rows including every level\n"
            "of the ontology and both hemispheres.")
        self.region_minvox.valueChanged.connect(self._refresh_region_list)
        top.addWidget(self.region_minvox)
        g.addLayout(top)

        self.region_list = QtWidgets.QListWidget()
        self.region_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.region_list.setMaximumHeight(110)
        self.region_list.itemDoubleClicked.connect(lambda _i: self._on_target_region())
        g.addWidget(self.region_list)

        row = QtWidgets.QHBoxLayout()
        self.target_btn = QtWidgets.QPushButton("Target selected region")
        self.target_btn.setToolTip(
            "Move the skull so the selected region's centroid sits exactly on the focus.")
        self.target_btn.clicked.connect(self._on_target_region)
        self.manual_btn = QtWidgets.QPushButton("Manual (clear target)")
        self.manual_btn.clicked.connect(self._on_manual)
        row.addWidget(self.target_btn); row.addWidget(self.manual_btn); row.addStretch(1)
        g.addLayout(row)

        self.target_info = QtWidgets.QLabel(
            "No atlas loaded. Set the centroid table on the Settings tab to enable targeting.")
        self.target_info.setWordWrap(True)
        g.addWidget(self.target_info)
        self.target_btn.setEnabled(False)
        return box

    # ------------------------------------------------------------ atlas
    def load_atlas(self, path: str):
        """Load the DMBA centroid table. Safe to call with a bad/blank path."""
        self._regions = []
        if not path or not os.path.isfile(path):
            self.target_info.setText(
                "No atlas loaded. Set the centroid table on the Settings tab to enable targeting.")
            self._update_target_enabled(); return
        try:
            self._regions = atlas.load_regions(path, min_voxels=1)
        except Exception as err:  # noqa: BLE001
            self.target_info.setText(f"<span style='color:#b00'>Atlas not loaded: {err}</span>")
            self._update_target_enabled(); return
        self._refresh_region_list()
        self._update_target_enabled()

    def _refresh_region_list(self):
        self.region_list.clear()
        if not self._regions:
            return
        mv = int(self.region_minvox.value())
        hits = [r for r in atlas.search(self._regions, self.region_search.text())
                if r.voxels >= mv]
        for r in hits[:400]:
            it = QtWidgets.QListWidgetItem(f"{r.pretty}   ({r.voxels:,} vox)")
            it.setData(QtCore.Qt.UserRole, r.roi)
            self.region_list.addItem(it)
        if not hits:
            self.region_list.addItem("(no match)")

    def _region_by_roi(self, roi):
        for r in self._regions:
            if r.roi == roi:
                return r
        return None

    def _update_target_enabled(self):
        """Enable targeting only when an atlas, a skull, and a shared world space exist."""
        if not self._regions:
            self.target_btn.setEnabled(False); return
        if self._skull is None:
            self.target_btn.setEnabled(False)
            self.target_info.setText(
                f"{len(self._regions)} regions loaded. Prepare/Load a CT skull to enable targeting.")
            return
        if not self._skull.has_world_geometry:
            self.target_btn.setEnabled(False)
            self.target_info.setText(
                f"<span style='color:#b00'>{len(self._regions)} regions loaded, but this prepared "
                "skull predates region targeting (no world geometry). Press 'Prepare / Load' on "
                "the CT to rebuild it.</span>")
            return
        same = atlas.same_space_as(self._regions, self._skull.ct_bbox_lo, self._skull.ct_bbox_hi)
        self.target_btn.setEnabled(bool(same))
        if same:
            self.target_info.setText(
                f"{len(self._regions)} regions loaded. Pick one and press Target "
                "(or just use the manual offsets).")
        else:
            self.target_info.setText(
                "<span style='color:#b00'>This CT is not in the atlas world space, so region "
                "targeting is disabled. Use the manual offsets.</span>")

    def _on_manual(self):
        self.region_list.clearSelection()
        self._targeted = ""
        self.target_info.setText("Manual mode: offsets are yours to set.")
        self.schedule_redraw()

    def _on_target_region(self):
        if self._skull is None or not self._regions:
            return
        # double-clicking a row calls this directly, bypassing the disabled
        # button, so the precondition has to be re-checked here.
        if not self._skull.has_world_geometry:
            QtWidgets.QMessageBox.information(
                self, "Re-prepare needed",
                "This prepared skull was made before region targeting existed and does "
                "not store the CT's world geometry.\n\nGo to the CT path above and press "
                "'Prepare / Load' again - you will be offered to rebuild it.")
            return
        items = self.region_list.selectedItems()
        if not items:
            QtWidgets.QMessageBox.information(self, "Target", "Select a region first.")
            return
        roi = items[0].data(QtCore.Qt.UserRole)
        reg = self._region_by_roi(roi) if roi is not None else None
        if reg is None:
            return
        try:
            t = self._skull.target_offsets_mm(
                reg.world_lps_mm, fill_threshold=float(self.fill_thr.value()))
        except Exception as err:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Target failed", str(err)); return

        if not t["inside_ct"]:
            QtWidgets.QMessageBox.warning(
                self, "Region outside CT",
                f"'{reg.pretty}' lies outside this CT's world extent. "
                "Is this the DMBA/M4D CT the atlas was built on?")
            return
        if not t["ok"]:
            QtWidgets.QMessageBox.warning(
                self, "Region outside skull crop",
                f"'{reg.pretty}' falls outside the prepared skull volume (the CT crop "
                "does not cover it). Pick a more central region, or prepare from the full CT.")
            return

        # apply without firing three separate redraws
        for sb, val in ((self.off_ax, t["ax_mm"]), (self.off_lr, t["lr_mm"]),
                        (self.off_ap, t["ap_mm"])):
            sb.blockSignals(True); sb.setValue(round(float(val), 1)); sb.blockSignals(False)
        self._targeted = reg.pretty
        note = ""
        if t["in_bone"]:
            note = ("  <span style='color:#b00'>[warning] the centroid landed in BONE - "
                    "check the placement.</span>")
        self.target_info.setText(
            f"Focus on <b>{reg.pretty}</b>  ->  AX {t['ax_mm']:+.2f}, LR {t['lr_mm']:+.2f}, "
            f"AP {t['ap_mm']:+.2f} mm{note}")
        self.schedule_redraw()

    # ------------------------------------------------------------ public API
    def has_skull(self) -> bool:
        return self._skull is not None

    def _center_m(self, gspec):
        fx, fy, fz = gspec.focus_xyz
        return (fx + self.off_ax.value() * 1e-3,
                fy + self.off_lr.value() * 1e-3,
                fz + self.off_ap.value() * 1e-3)

    def current_embedding(self, gspec):
        """Embed the skull into *gspec* (the SIMULATION grid) with the current
        placement. Returns an EmbeddedSkull or None."""
        if self._skull is None:
            return None
        p = self._safe_params()
        f0 = p.f0 if p else 500e3
        return self._skull.embed(
            gspec, center_xyz_m=self._center_m(gspec), f0_hz=f0,
            fill_threshold=float(self.fill_thr.value()))

    def on_shown(self):
        self.schedule_redraw()

    def schedule_redraw(self, *_):
        self._timer.start()

    def _save_figure(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", "skull_transducer.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        try:
            self.view.figure.savefig(path, dpi=200)
        except Exception as err:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save failed", str(err))

    # ------------------------------------------------------------ helpers
    def _safe_params(self) -> Optional[SimParams]:
        try:
            p = self._get_params(); p.validate(); return p
        except Exception:
            return None

    def _browse(self):
        start = os.path.dirname(self.path_edit.text().strip()) or _default_ct_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select CT or prepared skull", start,
            "CT or skull (*.nhdr *.nrrd *.h5 *.hdf5)")
        if path:
            self.path_edit.setText(path)

    def _prepared_path_for(self, ct_path: str) -> str:
        stem = os.path.splitext(ct_path)[0]
        return f"{stem}_bonefrac.h5"

    def _prepare_or_load(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Skull", "Pick a valid CT or .h5 file first.")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in (".h5", ".hdf5"):
            self._load_prepared(path); return
        # It's a CT. If a prepared file already exists, load it (fast); else prepare.
        out = self._prepared_path_for(path)
        if os.path.isfile(out):
            self._load_prepared(out)
            # A file made before the world-affine fix cannot be used for region
            # targeting. Offer to rebuild it rather than making the user hunt
            # down and delete the .h5 by hand.
            if self._skull is not None and not self._skull.has_world_geometry:
                r = QtWidgets.QMessageBox.question(
                    self, "Re-prepare skull?",
                    f"{os.path.basename(out)} was made by an older version and does not "
                    "store the CT's world geometry, so targeting a brain region is "
                    "disabled.\n\nRe-prepare it now from the CT? (takes a few seconds)",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes)
                if r == QtWidgets.QMessageBox.Yes:
                    self._start_prepare(path, out)
                    return
            self.info.setText(f"Loaded existing {os.path.basename(out)} "
                              "(delete it to re-prepare with a new threshold).")
            return
        self._start_prepare(path, out)

    def _load_prepared(self, h5_path: str):
        try:
            self._skull = CTSkullSource.load(h5_path)
        except Exception as err:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load failed", str(err)); return
        self._update_target_enabled()
        self.schedule_redraw()

    def _start_prepare(self, nhdr_path: str, out_h5: str):
        if self._worker is not None:
            return
        self._progress = QtWidgets.QProgressDialog(
            "Preparing skull from CT...", None, 0, 100, self)
        self._progress.setWindowModality(QtCore.Qt.WindowModal)
        self._progress.setMinimumDuration(0); self._progress.setCancelButton(None)
        self._progress.setValue(0); self._progress.show()
        self.load_btn.setEnabled(False)

        self._worker = _PrepareWorker(nhdr_path, out_h5, float(self.ct_thr.value()))
        self._worker.progress.connect(self._on_prep_progress)
        self._worker.done.connect(self._on_prep_done)
        self._worker.failed.connect(self._on_prep_failed)
        self._worker.start()

    def _on_prep_progress(self, msg: str, frac: float):
        if self._progress is not None:
            self._progress.setLabelText(msg); self._progress.setValue(int(frac * 100))

    def _cleanup_worker(self):
        if self._progress is not None:
            self._progress.close(); self._progress = None
        self.load_btn.setEnabled(True)
        self._worker = None

    def _on_prep_done(self, out_h5: str):
        self._cleanup_worker()
        self._load_prepared(out_h5)
        self.info.setText(f"Prepared {os.path.basename(out_h5)}. Skull built from CT.")

    def _on_prep_failed(self, err: str):
        self._cleanup_worker()
        QtWidgets.QMessageBox.critical(self, "Prepare failed", err)

    # ------------------------------------------------------------ draw
    def _redraw_now(self):
        if self._busy:
            self._timer.start(); return
        self._busy = True
        try:
            self._draw_impl()
        except Exception as err:  # noqa: BLE001
            self.info.setText(f"<span style='color:#b00'>draw error: {err}</span>")
        finally:
            self._busy = False

    def _draw_impl(self):
        p = self._safe_params()
        if p is None:
            self.info.setText("Fix the transducer parameters above first.")
            return
        # preview at a coarse, light resolution (independent of simulation PPW)
        pv = replace(p, points_per_wavelength=int(self.preview_ppw.value()))
        gspec = build_grid_spec(pv)
        self._gspec = gspec

        emb = None
        if self._skull is not None:
            emb = self._skull.embed(
                gspec, center_xyz_m=self._center_m(gspec), f0_hz=pv.f0,
                fill_threshold=float(self.fill_thr.value()))
        self._emb = emb

        xmm = gspec.x_vec * 1e3; ymm = gspec.y_vec * 1e3; zmm = gspec.z_vec * 1e3
        fx, fy, fz = gspec.focus_xyz
        xa, za = bowl_arc_xz(gspec, pv)          # transducer bowl arc (x, lateral)
        (ix0, ix1), (iy0, iy1), (iz0, iz1) = gspec.box_ix, gspec.box_iy, gspec.box_iz

        fig = self.view.figure
        fig.clear()

        def _panel(ax, bone_slice, lat_mm, lat_lo, lat_hi, latlabel, arc_lat):
            if bone_slice is not None:
                ax.imshow(bone_slice.T, origin="lower",
                          extent=[xmm[0], xmm[-1], lat_mm[0], lat_mm[-1]],
                          cmap="gray", vmin=0, vmax=1, aspect="equal", interpolation="nearest")
            else:
                ax.set_xlim(xmm[0], xmm[-1]); ax.set_ylim(lat_mm[0], lat_mm[-1])
                ax.set_aspect("equal")
            # sensor box (skull region)
            ax.add_patch(Rectangle((xmm[ix0], lat_lo), xmm[ix1] - xmm[ix0], lat_hi - lat_lo,
                                   fill=False, edgecolor="tab:orange", lw=1.0, ls="--"))
            # transducer bowl + focus
            ax.plot(xa * 1e3, arc_lat * 1e3, color="deepskyblue", lw=2.0)
            ax.axhline(0, color="0.4", lw=0.5, zorder=0)
            ax.set_xlabel("x  (axial) [mm]"); ax.set_ylabel(latlabel)
            # keep the text-rendering load small: fewer ticks, smaller labels.
            ax.tick_params(labelsize=7)
            ax.locator_params(axis="both", nbins=4)

        a1 = fig.add_subplot(1, 2, 1)
        _panel(a1, emb.bone_mask[:, emb.center_idx[1], :] if emb else None,
               zmm, zmm[iz0], zmm[iz1], "z  (lateral) [mm]", za)
        a1.plot([fx * 1e3], [fz * 1e3], marker="*", color="red", ms=13,
                markeredgecolor="white", markeredgewidth=0.5)
        a1.set_title("Coronal (x-z): bone mask + transducer", fontsize=9)
        # NOTE: no ax.legend() - matplotlib's legend/offsetbox text layout
        # segfaults under Agg on Windows during rapid redraws. Use text labels.
        a1.text(0.02, 0.98, "transducer", transform=a1.transAxes, fontsize=7,
                color="deepskyblue", va="top", ha="left")
        a1.text(0.02, 0.90, "focus", transform=a1.transAxes, fontsize=7,
                color="red", va="top", ha="left")

        a2 = fig.add_subplot(1, 2, 2)
        _panel(a2, emb.bone_mask[:, :, emb.center_idx[2]] if emb else None,
               ymm, ymm[iy0], ymm[iy1], "y  (lateral) [mm]", za)
        a2.plot([fx * 1e3], [fy * 1e3], marker="*", color="red", ms=13,
                markeredgecolor="white", markeredgewidth=0.5)
        a2.set_title("Axial (x-y): bone mask + transducer", fontsize=9)

        fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.14, wspace=0.30)
        self.view.render_figure()

        if emb is not None:
            warn = ("  <span style='color:#b00'>&lt;- very few bone voxels; lower "
                    "the Bone fill or raise PPW</span>") if emb.n_bone_voxels < 50 else ""
            self.info.setText(
                self._skull.summary() + f"   ||   {emb.n_bone_voxels:,} bone voxels on "
                f"the preview grid (PPW {self.preview_ppw.value()}, fill "
                f"{self.fill_thr.value():.2f}). Simulation uses the Numerics PPW.{warn}")
        else:
            self.info.setText("Transducer geometry shown. Prepare/Load a CT to overlay the bone mask.")
