"""GPU detection and solver selection.

The simulation can run on an NVIDIA GPU (``kspaceFirstOrder3DG``) or on the CPU
(``kspaceFirstOrder3D``). We auto-detect a usable NVIDIA GPU via ``nvidia-smi``;
the user can override this from the GUI.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Tuple


def detect_gpu() -> Tuple[bool, str]:
    """Return ``(has_gpu, description)``.

    Detection strategy: look for ``nvidia-smi`` on PATH and run it. If it returns
    cleanly and reports at least one GPU, we assume a CUDA-capable device that
    the k-Wave GPU binary can use.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, "nvidia-smi not found - no NVIDIA GPU detected (will use CPU)."
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4,
        )
    except Exception as err:  # pragma: no cover - environment dependent
        return False, f"nvidia-smi failed to run ({err}); will use CPU."

    if out.returncode != 0:
        return False, "nvidia-smi returned an error; will use CPU."

    names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not names:
        return False, "No GPU reported by nvidia-smi; will use CPU."
    return True, "GPU detected: " + ", ".join(names)


def choose_solver(compute_mode: str = "auto") -> Tuple[bool, str]:
    """Decide whether to run on GPU.

    Parameters
    ----------
    compute_mode : "auto" | "gpu" | "cpu"

    Returns
    -------
    (use_gpu, message)
    """
    mode = (compute_mode or "auto").lower()
    if mode == "cpu":
        return False, "Forced CPU (kspaceFirstOrder3D)."
    if mode == "gpu":
        has, desc = detect_gpu()
        if not has:
            # Respect the user's explicit choice with warning.
            return True, "Forced GPU (kspaceFirstOrder3DG) - " + desc
        return True, "GPU (kspaceFirstOrder3DG). " + desc

    # auto
    has, desc = detect_gpu()
    if has:
        return True, "Auto -> GPU (kspaceFirstOrder3DG). " + desc
    return False, "Auto -> CPU (kspaceFirstOrder3D). " + desc
