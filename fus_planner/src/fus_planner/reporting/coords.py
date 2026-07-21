"""Convert focal-spot plans into stage-frame coordinate tables.

For each spot we report the LPS world position plus offsets from the two
anatomical landmarks (lambda suture, bottom of left eye), the per-spot
ROI% and off-target%, and the constant transducer-face z.

Columns are organised in named groups so the GUI can colour the headers
consistently and so external consumers know which columns belong together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..geometry.frames import StageFrame
from ..planner.hex_tile import HexTilePlan
from ..planner.zplane import ZPlanePlan


# (group_name, [columns]) -- order here defines the column order.
# Each group also carries a header background colour for the GUI.
COLUMN_GROUPS: List[Tuple[str, List[str], str]] = [
    ("id",       ["spot_id"],                                                      "#E0E0E0"),
    ("world",    ["world_x_lps_mm", "world_y_lps_mm", "world_z_lps_mm"],            "#CFE8FC"),
    ("lambda",   ["from_lambda_LR_mm", "from_lambda_PA_mm", "from_lambda_IS_mm"],   "#D6F5D6"),
    ("eye",      ["from_eye_LR_mm", "from_eye_PA_mm", "from_eye_IS_mm"],            "#FFF3CD"),
    ("coverage", ["footprint_area_mm2", "ROI_pct", "off_target_pct"],               "#FAD7C7"),
    ("beam",     ["rx_mm", "ry_mm"],                                                "#F8D6E8"),
    ("zplumbing",["z_focus_mm", "z_brain_top_mm", "z_inner_skull_mm",
                  "z_outer_skull_mm", "skull_thickness_mm",
                  "z_transducer_face_mm", "coupling_gap_mm"],                       "#E5DCEF"),
]

COLUMNS: List[str] = [c for _, cols, _ in COLUMN_GROUPS for c in cols]

# Lookup: column name -> (group_name, group_colour_hex)
COLUMN_GROUP_LOOKUP: Dict[str, Tuple[str, str]] = {
    c: (group, colour)
    for (group, cols, colour) in COLUMN_GROUPS
    for c in cols
}


def plan_to_dataframe(
    plan: HexTilePlan,
    stage: StageFrame,
    z_plan: ZPlanePlan,
) -> pd.DataFrame:
    """Build a per-spot coordinate table from a plan + stage + z-plan.

    Spot IDs are **1-indexed** for human readability (1..N).
    """
    z_brain_top = getattr(z_plan, "z_brain_top_mm", float("nan"))
    rows = []
    for k, s in enumerate(plan.spots):
        cx, cy = s.center_world_xy_mm
        cz = plan.plane_z_lps_mm
        xyz_lps = np.array([cx, cy, cz])
        d_lambda = stage.offset_from_lambda(xyz_lps)
        d_eye = stage.offset_from_left_eye(xyz_lps)
        rows.append({
            "spot_id": k + 1,
            "world_x_lps_mm": cx, "world_y_lps_mm": cy, "world_z_lps_mm": cz,
            "from_lambda_LR_mm": d_lambda[0], "from_lambda_PA_mm": d_lambda[1], "from_lambda_IS_mm": d_lambda[2],
            "from_eye_LR_mm":    d_eye[0],    "from_eye_PA_mm":    d_eye[1],    "from_eye_IS_mm":    d_eye[2],
            "footprint_area_mm2": s.footprint_area_mm2,
            "ROI_pct": s.roi_pct,
            "off_target_pct": s.offtarget_pct,
            "rx_mm": s.rx_mm, "ry_mm": s.ry_mm,
            "z_focus_mm": z_plan.z_focus_mm,
            "z_brain_top_mm": z_brain_top,
            "z_inner_skull_mm": z_plan.z_inner_skull_mm,
            "z_outer_skull_mm": z_plan.z_outer_skull_mm,
            "skull_thickness_mm": z_plan.skull_thickness_mm,
            "z_transducer_face_mm": z_plan.z_transducer_face_mm,
            "coupling_gap_mm": z_plan.coupling_gap_mm,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    df.to_csv(path, index=False, float_format="%.4f")
    return path
