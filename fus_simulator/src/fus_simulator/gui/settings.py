"""Persistent user settings (default data paths).

Stored as a small JSON file in the user's config directory so the app remembers
the files you picked instead of asking every launch:

    Windows : %APPDATA%\\fus_simulator\\settings.json
    Linux   : ~/.config/fus_simulator/settings.json
    macOS   : ~/.config/fus_simulator/settings.json

Nothing here raises: a missing or corrupt settings file just yields defaults, so
a bad config can never stop the GUI from starting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

APP_NAME = "fus_simulator"


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") \
        or str(Path.home() / ".config")
    d = Path(base) / APP_NAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def config_path() -> Path:
    return config_dir() / "settings.json"


def load() -> Dict[str, str]:
    p = config_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save(data: Dict[str, str]) -> Path:
    p = config_path()
    try:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    return p


def get(key: str, default: str = "") -> str:
    return str(load().get(key, default))


def set_many(**kwargs) -> Path:
    data = load()
    data.update({k: str(v) for k, v in kwargs.items()})
    return save(data)


def clear() -> None:
    try:
        config_path().unlink()
    except Exception:
        pass
