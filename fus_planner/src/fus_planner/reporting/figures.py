"""Overlay figures (xy / xz / yz) for a focal-spot plan.

All axes are drawn in **anatomical** coordinates (LR, AP, IS) so the brain
shows up the way a neuroscientist expects:

* xy panel: dorsal view, anterior at top, animal's right on viewer's right.
* xz panel: coronal-style slice through one AP; LR on x axis, IS on y axis.
* yz panel: sagittal-style slice through one LR; AP on x axis, IS on y axis.

The beam is modeled as an ellipsoid with semi-axes (rx, ry, rz) =
(FWHM_x/2, FWHM_y/2, FWHM_axial/2). At any slice that cuts the beam, the
cross-section is a scaled ellipse whose semi-axes shrink as we move away
from the focal centre along the slice's perpendicular axis. We render that
cross-section on every slice the beam reaches, with line weight scaled by
the cross-section's relative size so the focal-plane slice still pops.

Anatomical to / from LPS:  LR = -LPS_x   AP = -LPS_y   IS = +LPS_z
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from ..io.volume import Volume
from ..planner.hex_tile import HexTilePlan
from ..planner.plane import FocalPlane
from ..planner.zplane import ZPlanePlan


# ----------- CT dorsal-surface depth map -----------------------------------

def ct_top_surface_depth(
    ct: Volume,
    skull_threshold: float,
    downsample: int = 6,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Compute a dorsal-surface depth map of the skull.

    For each (x, y) column we find the highest z where ``CT > threshold`` and
    return that z value. The result is a 2D image where intensity encodes
    skull height; cranial sutures show up as small height steps because the
    parietal / frontal / interparietal bones overlap differently across them.

    Returns
    -------
    z_top_anat : (Ni', Nj') array, NaN where there is no skull voxel
        Anatomical IS (= +LPS z) of the dorsal skull surface in mm.
    extent : (LR_0, LR_1, AP_0, AP_1)
        Anatomical extent in mm, in the order matplotlib expects.
    """
    ds = max(1, int(downsample))
    sub = ct.data[::ds, ::ds, ::ds]
    skull = sub > skull_threshold
    has_any = skull.any(axis=2)
    flipped = skull[:, :, ::-1]
    top_k_rev = flipped.argmax(axis=2)
    top_k_full_sub = (skull.shape[2] - 1) - top_k_rev
    # Map to physical z (using the un-downsampled spacing scaled by ds)
    z_step_full = ct.direction[2, 2] * ct.spacing[2]
    z_top_lps = ct.origin[2] + top_k_full_sub * (z_step_full * ds)
    z_top = np.where(has_any, z_top_lps, np.nan)

    # Anatomical extent
    Ni, Nj, _ = ct.shape
    sx = ct.direction[0, 0] * ct.spacing[0]
    sy = ct.direction[1, 1] * ct.spacing[1]
    LR_0 = -ct.origin[0]
    LR_1 = -(ct.origin[0] + (Ni - 1) * sx)
    AP_0 = -ct.origin[1]
    AP_1 = -(ct.origin[1] + (Nj - 1) * sy)
    return z_top, (LR_0, LR_1, AP_0, AP_1)


def render_shaded_skull(
    ct: Volume,
    skull_threshold: float = 5000.0,
    downsample: int = 2,
    vert_exag: float = 200.0,
    azdeg: float = 315.0,
    altdeg: float = 35.0,
    cmap: str = "gray",
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Hill-shaded RGBA image of the dorsal skull surface.

    The shading is intentionally very high-contrast: ``vert_exag`` is large
    (sub-mm sutures need significant exaggeration to register visually) and
    the light source sits at a low altitude so each suture casts a small
    shadow. We also clamp the height range to roughly the top 1.2 mm of the
    skull so the parietal/interparietal sutures get the full dynamic range
    of the colour map instead of being squashed by the much-lower margins.
    """
    from matplotlib.colors import LightSource
    z_top, extent = ct_top_surface_depth(ct, skull_threshold, downsample)
    finite = np.isfinite(z_top)
    if not finite.any():
        return np.zeros((*z_top.T.shape, 4), dtype=float), extent

    # Tight value range -- compress the top 1.2 mm of the skull so sutures pop.
    hi = float(np.nanpercentile(z_top, 99))
    lo = hi - 1.2
    fill = lo
    z_filled = np.where(finite, z_top, fill).T

    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    rgb = ls.shade(
        z_filled, cmap=plt.get_cmap(cmap),
        vert_exag=vert_exag, blend_mode="overlay",
        vmin=lo, vmax=hi,
    )
    alpha = np.where(finite.T, 1.0, 0.0).astype(rgb.dtype)
    rgba = np.dstack([rgb[..., :3], alpha])
    return rgba, extent


def plot_ct_dorsal(
    ax,
    ct: Volume,
    skull_threshold: float = 5000.0,
    downsample: int = 2,
    plan: Optional[HexTilePlan] = None,
    stage_frame=None,
    title: str = "CT dorsal skull (height-shaded)",
    vert_exag: float = 90.0,
    altdeg: float = 50.0,
) -> None:
    """Hill-shaded top-down skull view -- cranial sutures show as relief.

    If ``plan`` is given, each spot is overlaid as a green asterisk.
    If ``stage_frame`` is given, lambda and left-eye are marked.
    Defaults are gentler than the picker so the Results overlay is not
    too dark on the edges.
    """
    rgba, extent = render_shaded_skull(
        ct, skull_threshold, downsample,
        vert_exag=vert_exag, altdeg=altdeg)
    if rgba[..., 3].any():
        ax.imshow(rgba, origin="lower", extent=extent, interpolation="bilinear")
    else:
        ax.text(0.5, 0.5, "No skull voxels above threshold",
                ha="center", va="center", transform=ax.transAxes)

    LR_0, LR_1, AP_0, AP_1 = extent
    ax.set_xlim(min(LR_0, LR_1), max(LR_0, LR_1))
    ax.set_ylim(min(AP_0, AP_1), max(AP_0, AP_1))
    ax.set_aspect("equal")
    ax.set_xlabel("LR (mm)"); ax.set_ylabel("AP (mm)")
    ax.set_title(title, fontsize=10)

    if plan is not None:
        for s in plan.spots:
            cx, cy = s.center_world_xy_mm
            ax.plot(-cx, -cy, marker="*", color="lime", ms=11,
                    mec="black", mew=0.6)
    if stage_frame is not None:
        ax.plot(stage_frame.lambda_anat_mm[0], stage_frame.lambda_anat_mm[1],
                "x", color="crimson", ms=10, mew=2, label="lambda")
        ax.plot(stage_frame.left_eye_anat_mm[0], stage_frame.left_eye_anat_mm[1],
                "o", mfc="none", mec="dodgerblue", ms=8, mew=2, label="left eye")
        ax.legend(loc="upper right", fontsize=7)


# ----------- axial-slice heatmap (for finding eye / soft tissue) -----------

def plot_ct_axial_slice(
    ax,
    ct: Volume,
    z_lps_mm: float,
    cmap: str = "inferno",
    title: Optional[str] = None,
) -> None:
    """Single CT axial slice with a hot colormap. Shows soft tissue (eyes,
    muscles, cartilage) clearly -- useful for picking the bottom of the eye
    when the skull-only view hides it."""
    ijk = ct.world_to_voxel(np.array([0.0, 0.0, z_lps_mm]))
    k = max(0, min(ct.shape[2] - 1, int(round(ijk[2]))))
    slab = ct.data[:, :, k].astype(float)        # (Ni, Nj)
    Ni, Nj = slab.shape
    sx = ct.direction[0, 0] * ct.spacing[0]
    sy = ct.direction[1, 1] * ct.spacing[1]
    LR_0 = -ct.origin[0]
    LR_1 = -(ct.origin[0] + (Ni - 1) * sx)
    AP_0 = -ct.origin[1]
    AP_1 = -(ct.origin[1] + (Nj - 1) * sy)
    actual_z = ct.voxel_to_world(np.array([0, 0, k]))[2]
    # Robust contrast over the slice's own histogram
    lo, hi = np.percentile(slab, [2, 99])
    ax.imshow(slab.T, origin="lower", extent=(LR_0, LR_1, AP_0, AP_1),
              cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
    ax.set_xlim(min(LR_0, LR_1), max(LR_0, LR_1))
    ax.set_ylim(min(AP_0, AP_1), max(AP_0, AP_1))
    ax.set_aspect("equal")
    ax.set_xlabel("LR (mm)"); ax.set_ylabel("AP (mm)")
    ax.set_title(title or f"CT axial slice @ IS={actual_z:+.2f} mm", fontsize=10)


# ----------- skull-search diagnostic ---------------------------------------

def plot_skull_diagnostic(
    ax,
    ct: Volume,
    labels: Volume,
    z_plan: ZPlanePlan,
    fwhm_axial_mm: Optional[float] = None,
    half_width_mm: float = 4.0,
) -> None:
    """Side-by-side cross-section of CT and labels through the focus column.

    The figure shows a coronal-style slice (LR vs IS) through the focal point's
    AP. CT skull voxels (above threshold) are shaded brown; brain mask is grey.
    Horizontal lines mark brain top, inner / outer skull and the transducer face;
    a vertical red line marks the LR of the focal point (the column the search
    actually walked).

    This is the right view to debug a "skull threshold looks wrong" complaint:
    you can directly see the CT bone, the brain mass and where the algorithm
    placed each interface.
    """
    cx, cy, cz = z_plan.centroid_lps_mm
    LR_c = -cx
    AP_c = -cy

    # Slice CT and labels at AP = AP_c, restricted to LR window for clarity.
    # CT slice
    ijk_ct = ct.world_to_voxel(np.array([0.0, cy, 0.0]))
    j_ct = max(0, min(ct.shape[1] - 1, int(round(ijk_ct[1]))))
    ct_slab = ct.data[:, j_ct, :]                                 # (Ni_ct, Nk_ct)
    sx_ct = ct.direction[0, 0] * ct.spacing[0]
    sz_ct = ct.direction[2, 2] * ct.spacing[2]
    LR_ct_0 = -ct.origin[0]
    LR_ct_1 = -(ct.origin[0] + (ct.shape[0] - 1) * sx_ct)
    IS_ct_0 = ct.origin[2]
    IS_ct_1 = ct.origin[2] + (ct.shape[2] - 1) * sz_ct

    # Labels slice
    ijk_lb = labels.world_to_voxel(np.array([0.0, cy, 0.0]))
    j_lb = max(0, min(labels.shape[1] - 1, int(round(ijk_lb[1]))))
    lb_slab = labels.data[:, j_lb, :] > 0
    sx_lb = labels.direction[0, 0] * labels.spacing[0]
    sz_lb = labels.direction[2, 2] * labels.spacing[2]
    LR_lb_0 = -labels.origin[0]
    LR_lb_1 = -(labels.origin[0] + (labels.shape[0] - 1) * sx_lb)
    IS_lb_0 = labels.origin[2]
    IS_lb_1 = labels.origin[2] + (labels.shape[2] - 1) * sz_lb

    # Draw CT skull as semi-transparent brown overlay
    skull_mask_ct = (ct_slab > z_plan.skull_threshold_hu).astype(float)
    ct_image = np.where(skull_mask_ct > 0, 1.0, 0.0)
    ax.imshow(
        ct_image.T, origin="lower",
        extent=(LR_ct_0, LR_ct_1, IS_ct_0, IS_ct_1),
        cmap="copper", vmin=0, vmax=1.5, alpha=0.8, interpolation="nearest",
    )
    # Brain mask outline as grey overlay
    ax.contour(
        np.linspace(LR_lb_0, LR_lb_1, lb_slab.shape[0]),
        np.linspace(IS_lb_0, IS_lb_1, lb_slab.shape[1]),
        lb_slab.T.astype(float),
        levels=[0.5], colors="dimgray", linewidths=0.8,
    )

    # Horizontal guides
    if not np.isnan(getattr(z_plan, "z_brain_top_mm", float("nan"))):
        ax.axhline(z_plan.z_brain_top_mm, color="seagreen", lw=0.8, ls=":",
                   label="brain top @ focus xy")
    if not np.isnan(z_plan.z_inner_skull_mm):
        ax.axhline(z_plan.z_inner_skull_mm, color="saddlebrown", lw=0.8, ls=":",
                   label=f"inner skull = {z_plan.z_inner_skull_mm:+.2f} mm")
    if not np.isnan(z_plan.z_outer_skull_mm):
        ax.axhline(z_plan.z_outer_skull_mm, color="goldenrod", lw=0.8, ls="--",
                   label=f"outer skull = {z_plan.z_outer_skull_mm:+.2f} mm")
    if not np.isnan(z_plan.z_transducer_face_mm):
        ax.axhline(z_plan.z_transducer_face_mm, color="blue", lw=0.8,
                   label=f"face = {z_plan.z_transducer_face_mm:+.2f} mm")

    # Vertical line at search column (focal LR)
    ax.axvline(LR_c, color="red", lw=0.8,
               label=f"search column LR={LR_c:+.2f} mm")

    # Zoom around the focus
    ax.set_xlim(LR_c - half_width_mm, LR_c + half_width_mm)
    is_lo = (z_plan.z_brain_top_mm if not np.isnan(getattr(z_plan, "z_brain_top_mm", float("nan")))
             else z_plan.z_focus_mm) - 1.0
    is_hi = (z_plan.z_transducer_face_mm if not np.isnan(z_plan.z_transducer_face_mm)
             else z_plan.z_focus_mm + 5.0) + 1.0
    ax.set_ylim(is_lo, is_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("LR (mm)"); ax.set_ylabel("IS (mm)")
    ax.set_title(f"Skull-search diagnostic @ AP={AP_c:+.2f} mm "
                 f"(threshold={z_plan.skull_threshold_hu:.0f})", fontsize=10)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.85,
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


# ----------- ellipsoid cross-section helper --------------------------------

def _cross_section_axes(
    r_in_plane: tuple[float, float],   # semi-axes in the slice's two displayed axes (mm)
    r_perpendicular: float,             # beam semi-axis along the slice's perpendicular direction (mm)
    perp_offset: float,                 # signed distance of the slice from the beam centre (mm)
) -> Optional[tuple[float, float]]:
    """Return the slice-plane semi-axes of an ellipsoidal beam, or None if outside.

    For an ellipsoid (u/a)^2 + (v/b)^2 + (w/c)^2 = 1, slicing at w = w0 yields an
    ellipse with (u/(a*s))^2 + (v/(b*s))^2 = 1 where s = sqrt(1 - (w0/c)^2).
    """
    if r_perpendicular <= 0:
        return None
    if abs(perp_offset) >= r_perpendicular:
        return None
    s = float(np.sqrt(1.0 - (perp_offset / r_perpendicular) ** 2))
    a, b = r_in_plane
    return (a * s, b * s)


def _draw_landmarks(
    ax,
    stage_frame,
    panel: str,                         # "xy" | "xz" | "yz"
    slice_anat_mm: Optional[float] = None,   # IS for xy, AP for xz, LR for yz
    tolerance_mm: float = 0.5,
):
    """Mark lambda + left-eye on the panel; fade if the slice is far from their depth."""
    if stage_frame is None:
        return

    def _project(p_anat: np.ndarray) -> tuple[float, float]:
        if panel == "xy":
            return (p_anat[0], p_anat[1])
        if panel == "xz":
            return (p_anat[0], p_anat[2])
        return (p_anat[1], p_anat[2])

    def _faded(p_anat):
        if slice_anat_mm is None:
            return False
        if panel == "xy":
            return abs(p_anat[2] - slice_anat_mm) > tolerance_mm
        if panel == "xz":
            return abs(p_anat[1] - slice_anat_mm) > tolerance_mm
        return abs(p_anat[0] - slice_anat_mm) > tolerance_mm

    for p, marker, colour, name in [
        (stage_frame.lambda_anat_mm, "x", "crimson", "lambda"),
        (stage_frame.left_eye_anat_mm, "o", "dodgerblue", "left eye"),
    ]:
        u, v = _project(p)
        alpha = 0.35 if _faded(p) else 1.0
        if marker == "x":
            ax.plot(u, v, "x", color=colour, ms=10, mew=2, alpha=alpha, label=name)
        else:
            ax.plot(u, v, "o", mfc="none", mec=colour, ms=8, mew=2, alpha=alpha, label=name)


# ----------- xy (focal plane, dorsal view) ---------------------------------

def plot_xy(
    ax,
    plane: FocalPlane,
    plan: HexTilePlan,
    fwhm_axial_mm: float,
    slice_z_lps_mm: Optional[float] = None,
    stage_frame=None,
    title: str = "xy (focal plane, anterior up)",
) -> None:
    """xy dorsal view at slice ``slice_z_lps_mm`` (default = focal plane)."""
    img = np.zeros(plane.shape, dtype=float)
    img[plane.brain_mask] = 0.4
    img[plane.target_mask] = 1.0

    Ni, Nj = plane.shape
    sx, sy = plane.step_xy_mm
    ox, oy = plane.origin_xy_mm
    LR_0 = -ox
    LR_1 = -(ox + (Ni - 1) * sx)
    AP_0 = -oy
    AP_1 = -(oy + (Nj - 1) * sy)
    extent = (LR_0, LR_1, AP_0, AP_1)
    ax.imshow(img.T, origin="lower", extent=extent, cmap="Greys",
              vmin=0, vmax=1.2, interpolation="nearest")

    z_slice = slice_z_lps_mm if slice_z_lps_mm is not None else plan.plane_z_lps_mm
    rz = fwhm_axial_mm / 2.0
    z_focus = plan.plane_z_lps_mm
    dz = z_slice - z_focus
    for s in plan.spots:
        cs = _cross_section_axes((s.rx_mm, s.ry_mm), rz, dz)
        if cs is None:
            continue
        a, b = cs
        is_focal = abs(dz) < 1e-3
        ax.add_patch(Ellipse(
            (-s.center_world_xy_mm[0], -s.center_world_xy_mm[1]),
            width=2 * a, height=2 * b,
            edgecolor="red", facecolor="none",
            lw=1.4 if is_focal else 0.9,
            ls="-" if is_focal else "--",
            alpha=1.0 if is_focal else 0.85,
        ))
        if is_focal:
            ax.plot(-s.center_world_xy_mm[0], -s.center_world_xy_mm[1], "r.", ms=2)

    _draw_landmarks(ax, stage_frame, "xy",
                    slice_anat_mm=z_slice, tolerance_mm=0.5)
    if stage_frame is not None:
        # Outside the axes (to the right) so it never covers the brain. Same
        # placement as plot_xz_or_yz. The PDF export uses bbox_inches="tight";
        # the GUI reserves the right margin with a fixed subplots_adjust
        # (tight_layout there segfaults measuring curved patches on Windows).
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85,
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ax.set_xlim(min(LR_0, LR_1), max(LR_0, LR_1))
    ax.set_ylim(min(AP_0, AP_1), max(AP_0, AP_1))
    ax.set_aspect("equal")
    ax.set_xlabel("LR (mm)  (right of animal +)")
    ax.set_ylabel("AP (mm)  (anterior +)")
    ax.set_title(title, fontsize=10)


# ----------- xz / yz side views --------------------------------------------

def _slice_for_axis(
    labels: Volume,
    axis: str,
    fixed_anat_value_mm: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    """Return (slab_2d, anat_extent, abscissa_label) for an xz / yz slice.

    For xz, fixed_anat_value_mm is AP (mm).
    For yz, fixed_anat_value_mm is LR (mm).
    """
    if axis == "xz":
        lps_y = -fixed_anat_value_mm
        ijk = labels.world_to_voxel(np.array([0.0, lps_y, 0.0]))
        j_idx = max(0, min(labels.shape[1] - 1, int(round(ijk[1]))))
        slab = labels.data[:, j_idx, :]
        Ni, Nk = slab.shape
        sx = labels.direction[0, 0] * labels.spacing[0]
        sz = labels.direction[2, 2] * labels.spacing[2]
        ox, oz = labels.origin[0], labels.origin[2]
        LR_0 = -ox
        LR_1 = -(ox + (Ni - 1) * sx)
        IS_0 = oz
        IS_1 = oz + (Nk - 1) * sz
        return slab, (LR_0, LR_1, IS_0, IS_1), "LR (mm)"
    if axis == "yz":
        lps_x = -fixed_anat_value_mm
        ijk = labels.world_to_voxel(np.array([lps_x, 0.0, 0.0]))
        i_idx = max(0, min(labels.shape[0] - 1, int(round(ijk[0]))))
        slab = labels.data[i_idx, :, :]
        Nj, Nk = slab.shape
        sy = labels.direction[1, 1] * labels.spacing[1]
        sz = labels.direction[2, 2] * labels.spacing[2]
        oy, oz = labels.origin[1], labels.origin[2]
        AP_0 = -oy
        AP_1 = -(oy + (Nj - 1) * sy)
        IS_0 = oz
        IS_1 = oz + (Nk - 1) * sz
        return slab, (AP_0, AP_1, IS_0, IS_1), "AP (mm)"
    raise ValueError(f"unknown axis {axis!r}")


def plot_xz_or_yz(
    ax,
    labels: Volume,
    plan: HexTilePlan,
    z_plan: ZPlanePlan,
    fwhm_axial_mm: float,
    target_labels: Optional[Sequence[int]],
    axis: str,
    fixed_anat_mm: Optional[float] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    stage_frame=None,
) -> None:
    if axis not in ("xz", "yz"):
        raise ValueError("axis must be 'xz' or 'yz'")
    cx, cy, cz = z_plan.centroid_lps_mm
    LR_c = -cx
    AP_c = -cy
    if fixed_anat_mm is None:
        fixed_anat_mm = AP_c if axis == "xz" else LR_c

    slab, extent, xlabel = _slice_for_axis(labels, axis, fixed_anat_mm)
    brain = (slab > 0).astype(float) * 0.4
    if target_labels is None:
        target = (slab > 0).astype(float)
    else:
        target = np.isin(slab, list(target_labels)).astype(float)
    img = np.maximum(brain, target)

    ax.imshow(img.T, origin="lower", extent=extent, cmap="Greys",
              vmin=0, vmax=1.2, interpolation="nearest")
    u0, u1, v0, v1 = extent
    ax.set_xlim(min(u0, u1), max(u0, u1))
    ax.set_ylim(min(v0, v1), max(v0, v1))

    rz = fwhm_axial_mm / 2.0
    for s in plan.spots:
        sx, sy = s.center_world_xy_mm
        s_LR, s_AP = -sx, -sy
        if axis == "xz":
            d = fixed_anat_mm - s_AP
            cs = _cross_section_axes((s.rx_mm, rz), s.ry_mm, d)
            if cs is None:
                continue
            a, b = cs
            u_centre = s_LR
        else:
            d = fixed_anat_mm - s_LR
            cs = _cross_section_axes((s.ry_mm, rz), s.rx_mm, d)
            if cs is None:
                continue
            a, b = cs
            u_centre = s_AP
        is_focal = abs(d) < 1e-3
        ax.add_patch(Ellipse(
            (u_centre, z_plan.z_focus_mm),
            width=2 * a, height=2 * b,
            edgecolor="red", facecolor="none",
            lw=1.2 if is_focal else 0.7,
            ls="-" if is_focal else "--",
            alpha=1.0 if is_focal else 0.85,
        ))
        if is_focal:
            ax.plot(u_centre, z_plan.z_focus_mm, "r.", ms=2)

    # Skull / face guides. These are evaluated at the focal xy (where the focus
    # actually lives), not at the slice's xy. Mouse skull is curved, so at other
    # x/y the skull may be higher or lower; that is expected and not a bug.
    if not np.isnan(getattr(z_plan, "z_brain_top_mm", float("nan"))):
        ax.axhline(z_plan.z_brain_top_mm, color="seagreen", lw=0.6, ls=":",
                   label="brain top @ focus xy")
    if not np.isnan(z_plan.z_inner_skull_mm):
        ax.axhline(z_plan.z_inner_skull_mm, color="saddlebrown", lw=0.7, ls=":",
                   label="inner skull @ focus xy")
    if not np.isnan(z_plan.z_outer_skull_mm):
        ax.axhline(z_plan.z_outer_skull_mm, color="goldenrod", lw=0.7, ls="--",
                   label="outer skull @ focus xy")
    if not np.isnan(z_plan.z_transducer_face_mm):
        ax.axhline(z_plan.z_transducer_face_mm, color="blue", lw=0.7,
                   label="transducer face")

    _draw_landmarks(ax, stage_frame, axis,
                    slice_anat_mm=fixed_anat_mm, tolerance_mm=0.5)

    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("IS (mm)  (superior +)")
    fixed_label = "AP" if axis == "xz" else "LR"
    ax.set_title(f"{title or axis} @ {fixed_label}={fixed_anat_mm:+.2f} mm", fontsize=10)

    if show_legend:
        ax.legend(loc="upper left", fontsize=6, framealpha=0.85,
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


# ----------- PDF report -----------------------------------------------------

def render_plan_pdf(
    pdf_path: str | Path,
    plane: FocalPlane,
    plan: HexTilePlan,
    z_plan: ZPlanePlan,
    labels: Volume,
    fwhm_axial_mm: float,
    target_labels: Optional[Sequence[int]] = None,
    title: str = "FUS Plan",
    stage_frame=None,
) -> Path:
    pdf_path = Path(pdf_path)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(13.5, 5.5))
        gs = fig.add_gridspec(1, 3, wspace=0.45)

        ax_xy = fig.add_subplot(gs[0, 0])
        plot_xy(ax_xy, plane, plan, fwhm_axial_mm,
                stage_frame=stage_frame, title="xy (focal plane)")

        ax_xz = fig.add_subplot(gs[0, 1])
        plot_xz_or_yz(ax_xz, labels, plan, z_plan, fwhm_axial_mm,
                      target_labels=target_labels, axis="xz", title="xz",
                      stage_frame=stage_frame)

        ax_yz = fig.add_subplot(gs[0, 2])
        plot_xz_or_yz(ax_yz, labels, plan, z_plan, fwhm_axial_mm,
                      target_labels=target_labels, axis="yz", title="yz",
                      stage_frame=stage_frame)

        gap_str = "n/a" if np.isnan(z_plan.coupling_gap_mm) else f"{z_plan.coupling_gap_mm:.2f} mm"
        sk_str  = "n/a" if np.isnan(z_plan.skull_thickness_mm) else f"{z_plan.skull_thickness_mm:.2f} mm"
        summary = (
            f"{title}    spots = {plan.n_spots}    threshold = {plan.coverage_threshold:.0%}    "
            f"footprint = {plan.footprint.rx_mm * 2:.2f} x {plan.footprint.ry_mm * 2:.2f} mm    "
            f"total target coverage = {plan.total_target_coverage_pct:.1f}%\n"
            f"focal plane z = {plan.plane_z_lps_mm:+.3f} mm     "
            f"transducer face z = {z_plan.z_transducer_face_mm:+.3f} mm     "
            f"skull thickness = {sk_str}     coupling gap = {gap_str}"
        )
        fig.suptitle(summary, fontsize=8.5, y=1.02, ha="center")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return pdf_path
