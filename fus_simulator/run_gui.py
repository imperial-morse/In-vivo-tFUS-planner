"""
Crash recording: writes two logs next to this file so you can see what happened
if the app dies -
  * fus_sim_crash.log   - low-level fault handler (catches hard/native crashes,
                          e.g. a matplotlib/Qt segfault).
  * fus_sim_errors.log  - any uncaught Python exception with full traceback.
"""
import sys
import faulthandler
import datetime
import traceback
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

_crash_log = open(_HERE / "fus_sim_crash.log", "a", buffering=1)
_crash_log.write(f"\n===== session start {datetime.datetime.now()} =====\n")
faulthandler.enable(file=_crash_log, all_threads=True)


def _log_uncaught(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    with open(_HERE / "fus_sim_errors.log", "a") as f:
        f.write(f"\n===== {datetime.datetime.now()} =====\n")
        traceback.print_exception(exc_type, exc, tb, file=f)
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _log_uncaught

from fus_simulator.gui.main import main

if __name__ == "__main__":
    main()

# https://www.youtube.com/watch?v=dQw4w9WgXcQ 
