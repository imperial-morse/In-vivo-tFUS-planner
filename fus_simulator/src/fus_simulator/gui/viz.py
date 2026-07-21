"""Shared 3-D drawing helpers + a non-interactive static canvas.
"""

from __future__ import annotations

import numpy as np
import matplotlib
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


# --------------------------------------------------------------------------- #
# Static (non-interactive) matplotlib view
# --------------------------------------------------------------------------- #
class StaticCanvas(QtWidgets.QLabel):
    """A matplotlib Figure rendered to a static image (QLabel pixmap).
    """

    def __init__(self, figsize=(7.0, 6.0), dpi=100, min_size=(420, 360)):
        super().__init__()
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self._agg = FigureCanvasAgg(self.figure)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(*min_size)
        self.setStyleSheet("background: white;")
        self.setScaledContents(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self._rendered_once = False
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._fit_figure_to_widget)

    # -- keep the label's own size hint from fighting the pixmap size --------
    def sizeHint(self):
        return self.minimumSize()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start()

    def _fit_figure_to_widget(self):
        """Match the Figure to the widget in pixels, then re-rasterise."""
        if not self._rendered_once:
            return
        dpi = float(self.figure.get_dpi()) or 100.0
        w = max(int(self.width()), 50)
        h = max(int(self.height()), 50)
        new_w, new_h = w / dpi, h / dpi
        cur_w, cur_h = self.figure.get_size_inches()
        if abs(new_w - cur_w) < 0.05 and abs(new_h - cur_h) < 0.05:
            return
        try:
            self.figure.set_size_inches(new_w, new_h, forward=False)
            self.render_figure()
        except Exception:   # never let a resize kill the app
            pass

    def render_figure(self):
        self._agg.draw()
        w, h = self._agg.get_width_height()
        self._buf = self._agg.buffer_rgba().tobytes()   # keep alive
        img = QtGui.QImage(self._buf, w, h, QtGui.QImage.Format_RGBA8888)
        self.setPixmap(QtGui.QPixmap.fromImage(img))
        first = not self._rendered_once
        self._rendered_once = True
        if first:
            # grow to the widget right after the first draw; _fit_figure_to_widget
            # is a no-op once the sizes agree, so this cannot loop.
            self._resize_timer.start()


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_bowl_mesh(ax, X, Y, Z, cmap: str = "viridis"):
    """Draw the transducer cap as a colour-graded mesh.

    Faces are coloured by axial position (X) -> concentric bands following the
    bowl curvature; thin edges show the mesh. Coordinates m -> mm here.
    """
    rng = float(np.ptp(X)) or 1.0
    Xn = (X - X.min()) / rng
    colors = matplotlib.colormaps[cmap](Xn)
    ax.plot_surface(
        X * 1e3, Y * 1e3, Z * 1e3,
        facecolors=colors,
        edgecolor=(0, 0, 0, 0.35),
        linewidth=0.25,
        antialiased=True,
        shade=False,
    )


def set_ortho_static_view(ax, elev: float = 22.0, azim: float = -55.0):
    """Orthographic projection at a fixed 3/4 viewpoint (clean, undistorted)."""
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)


def set_equal_box_aspect(ax, xs, ys, zs):
    """Set the box aspect to the true data proportions (no squashing)."""
    def _rng(v):
        v = np.asarray(v, dtype=float)
        r = float(np.ptp(v))
        return r if r > 1e-9 else 1.0
    try:
        ax.set_box_aspect((_rng(xs), _rng(ys), _rng(zs)))
    except Exception:
        pass
