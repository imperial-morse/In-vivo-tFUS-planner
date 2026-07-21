"""Entry point for the FUS Planner GUI, with hard-crash forensics.

Start it with:

    .venv\\Scripts\\python run_gui.py

Every run creates ONE log file:

    logs\\fus_<YYYYmmdd>_<HHMMSS>.log

It contains, in order:
  1. environment banner  - python / numpy / matplotlib / pandas / Qt versions,
                           interpreter path, argv, relevant env vars.
  2. breadcrumbs         - a line for every button click, tab change, list
                           selection and key press, written and fsync'd
                           immediately.
  3. the failure         - either a Python traceback (soft error) or a
                           faulthandler dump (native crash / access violation).

Because breadcrumbs are flushed to disk on every event, the LAST line in the
file is the last thing the app did before it died. That is the line to read.

Safe mode
---------
    set FUS_SAFE=1

Disables OpenBLAS threading and numpy's AVX-512 dispatch before numpy is
imported. Use this to test whether a native crash is caused by CPU dispatch or
BLAS threading. If the app is stable with FUS_SAFE=1 and crashes without it,
the fault is in the numpy binary, not in this application.
"""
import os
import sys
import datetime
from pathlib import Path

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Environment knobs. MUST run before numpy is imported by anything.
# ---------------------------------------------------------------------------
if os.environ.get("FUS_SAFE") == "1":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NPY_DISABLE_CPU_FEATURES"] = (
        "AVX512F AVX512CD AVX512_SKX AVX512_ICL AVX512_KNL AVX512_KNM"
    )

sys.path.insert(0, str(_HERE / "src"))

# ---------------------------------------------------------------------------
# 2. Open the session log and turn on the fault handler.
# ---------------------------------------------------------------------------
import faulthandler  # noqa: E402
import traceback     # noqa: E402

_LOG_DIR = _HERE / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_PATH = _LOG_DIR / f"fus_{_stamp}.log"

# buffering=1 -> line buffered. We additionally fsync after each breadcrumb.
_log = open(_LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")


def log(msg):
    """Write one line and force it to the platter. Survives a segfault."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _log.write(f"{ts}  {msg}\n")
    try:
        _log.flush()
        os.fsync(_log.fileno())
    except Exception:
        pass


# faulthandler writes to the raw fd, so a native crash lands in this same file.
faulthandler.enable(file=_log, all_threads=True)

log(f"===== session start {datetime.datetime.now()} =====")
log(f"log file   : {_LOG_PATH}")
log(f"python     : {sys.version.split()[0]}  ({sys.executable})")
log(f"argv       : {sys.argv}")
log(f"cwd        : {os.getcwd()}")
log(f"FUS_SAFE   : {os.environ.get('FUS_SAFE', '<unset>')}")
for _v in ("OPENBLAS_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES", "MPLBACKEND"):
    log(f"{_v:<11}: {os.environ.get(_v, '<unset>')}")

# ---------------------------------------------------------------------------
# 3. Imports, each announced so a crash *during import* is attributable.
# ---------------------------------------------------------------------------
log("importing numpy ...")
import numpy as _np  # noqa: E402
log(f"numpy      : {_np.__version__}  ({_np.__file__})")

log("importing matplotlib ...")
import matplotlib  # noqa: E402
matplotlib.use("Qt5Agg")
# figure.autolayout / constrained_layout re-run a layout pass on EVERY draw,
# which measures every artist via get_tightbbox(). Keep them off: the layout in
# gui/main.py is done with explicit subplots_adjust fractions.
matplotlib.rcParams["figure.autolayout"] = False
matplotlib.rcParams["figure.constrained_layout.use"] = False
log(f"matplotlib : {matplotlib.__version__}  (backend Qt5Agg)")

try:
    import pandas as _pd
    log(f"pandas     : {_pd.__version__}")
except Exception as _e:                                    # pragma: no cover
    log(f"pandas     : NOT IMPORTABLE - {_e!r}")

from PyQt5 import QtCore, QtWidgets  # noqa: E402
log(f"Qt         : {QtCore.QT_VERSION_STR}  PyQt {QtCore.PYQT_VERSION_STR}")
log("-" * 70)

# ---------------------------------------------------------------------------
# 4. Python-level uncaught exceptions (main thread + worker threads + Qt slots).
# ---------------------------------------------------------------------------
def _log_uncaught(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    _log.write(f"\n===== UNCAUGHT EXCEPTION {datetime.datetime.now()} =====\n")
    traceback.print_exception(exc_type, exc, tb, file=_log)
    _log.flush()
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _log_uncaught

import threading  # noqa: E402
threading.excepthook = lambda a: _log_uncaught(a.exc_type, a.exc_value, a.exc_traceback)


# ---------------------------------------------------------------------------
# 5. Qt's own warnings (they often precede a native crash).
# ---------------------------------------------------------------------------
def _qt_message(mode, ctx, message):
    log(f"QT[{int(mode)}] {message}")


QtCore.qInstallMessageHandler(_qt_message)


# ---------------------------------------------------------------------------
# 6. Breadcrumbs: log every user interaction, app-wide, without touching
#    gui/main.py. An application-level event filter sees every event first.
# ---------------------------------------------------------------------------
def _describe(obj):
    try:
        cls = obj.__class__.__name__
        name = obj.objectName() or ""
        text = ""
        for getter in ("text", "currentText", "title"):
            if hasattr(obj, getter):
                try:
                    text = str(getattr(obj, getter)())[:40]
                    if text:
                        break
                except Exception:
                    pass
        bits = [cls]
        if name:
            bits.append(f"name={name!r}")
        if text:
            bits.append(f"text={text!r}")
        return " ".join(bits)
    except Exception:
        return "<undescribable>"


class _Breadcrumbs(QtCore.QObject):
    _WATCH = {
        QtCore.QEvent.MouseButtonPress: "CLICK",
        QtCore.QEvent.KeyPress: "KEY",
    }

    def eventFilter(self, obj, ev):
        kind = self._WATCH.get(ev.type())
        if kind == "CLICK":
            log(f"CLICK  {_describe(obj)}")
        elif kind == "KEY":
            try:
                log(f"KEY    {ev.text()!r} on {_describe(obj)}")
            except Exception:
                pass
        return False


_orig_qapp_init = QtWidgets.QApplication.__init__


def _qapp_init(self, *args, **kwargs):
    _orig_qapp_init(self, *args, **kwargs)
    self._fus_breadcrumbs = _Breadcrumbs()
    self.installEventFilter(self._fus_breadcrumbs)
    log("breadcrumb event filter installed")


QtWidgets.QApplication.__init__ = _qapp_init


# ---------------------------------------------------------------------------
# 7. Go.
# ---------------------------------------------------------------------------
log("importing fus_planner.gui.main ...")
from fus_planner.gui.main import main  # noqa: E402

if __name__ == "__main__":
    log("calling main()")
    try:
        main()
    finally:
        log("===== session end (clean exit of main()) =====")
