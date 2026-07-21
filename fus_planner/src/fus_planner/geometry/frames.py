"""Stage coordinate frame anchored on lambda suture and bottom of left eye.

The transducer rig moves in xy and is fixed in z (sonication is always from
above). Spot positions are reported as offsets from each of two anatomical
landmarks the user picks at session start:

    1. Lambda suture (primary origin)
    2. Bottom of the left eye (secondary origin)

Offsets are returned in anatomical millimetres:

    +LR  = right side  (millimetres right of the landmark)
    +PA  = anterior    (millimetres anterior to the landmark)
    +IS  = superior    (millimetres above the landmark)

We keep both LPS landmark positions plus an ``axis`` derived from them
(lambda -> eye), useful later if anyone wants a stage frame whose +x is
forced to point at the eye instead of pure anatomical right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..io.centroids import lps_to_anat


@dataclass
class StageFrame:
    """Two anatomical landmarks defining the rig coordinate references."""

    lambda_lps_mm: np.ndarray         # (3,) LPS mm
    left_eye_lps_mm: np.ndarray       # (3,) LPS mm

    @property
    def lambda_anat_mm(self) -> np.ndarray:
        return lps_to_anat(self.lambda_lps_mm)

    @property
    def left_eye_anat_mm(self) -> np.ndarray:
        return lps_to_anat(self.left_eye_lps_mm)

    def offset_from_lambda(self, xyz_lps_mm: np.ndarray) -> np.ndarray:
        """Anatomical (dLR, dPA, dIS) mm offset from lambda."""
        return lps_to_anat(np.asarray(xyz_lps_mm)) - self.lambda_anat_mm

    def offset_from_left_eye(self, xyz_lps_mm: np.ndarray) -> np.ndarray:
        """Anatomical (dLR, dPA, dIS) mm offset from bottom-of-left-eye."""
        return lps_to_anat(np.asarray(xyz_lps_mm)) - self.left_eye_anat_mm

    def both_offsets(self, xyz_lps_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(offset_from_lambda, offset_from_left_eye) for the same point."""
        return self.offset_from_lambda(xyz_lps_mm), self.offset_from_left_eye(xyz_lps_mm)

    @property
    def lambda_eye_axis(self) -> np.ndarray:
        """Unit vector pointing from lambda to bottom-of-left-eye in anatomical mm.

        Useful if a user later wants a stage frame whose +x axis is locked to this
        line instead of pure anatomical right; for now it is exposed for diagnostics.
        """
        v = self.left_eye_anat_mm - self.lambda_anat_mm
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def build_stage_frame(
    lambda_lps_mm: np.ndarray,
    left_eye_lps_mm: np.ndarray,
) -> StageFrame:
    """Construct a StageFrame from two LPS-mm landmark coordinates.

    Both landmarks come from clicks the user makes on the CT (lambda — clearly
    visible on the skull) and the MRI (left eye — easy on T2/dwi). They are
    expected in the dataset's native LPS frame.
    """
    lam = np.asarray(lambda_lps_mm, dtype=float).reshape(3)
    eye = np.asarray(left_eye_lps_mm, dtype=float).reshape(3)
    if lam.shape != (3,) or eye.shape != (3,):
        raise ValueError("landmark coordinates must be 3-vectors")
    if np.allclose(lam, eye):
        raise ValueError("lambda and eye landmarks are identical")
    return StageFrame(lambda_lps_mm=lam, left_eye_lps_mm=eye)
