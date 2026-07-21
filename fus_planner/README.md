# FUS Planner

Interactive treatment planning toolbox for in vivo ultrasound neuromodulation in mice.

## What it does

Given an MRI / CT / segmentation of a mouse and the calibration data of an ultrasound transducer
(lateral FWHM in two axes + focal depth), the toolbox computes:

1. The transverse plane (z) where the transducer should focus,
2. A set of non-overlapping focal-spot positions in (x, y) that cover either the whole brain or a
   chosen Allen-CCF region above a coverage threshold (default 80%),
3. A coordinate table relative to two anatomical landmarks (lambda suture and bottom of left eye),
4. Per-spot ROI% and off-target ROI% reports, plus xy / xz / yz overlay figures.

## Layout

```
fus_planner/
├── io/          NRRD/.nhdr loaders, centroid CSV parser, label lookup
├── geometry/    voxel <-> world (LPS) <-> stage (lambda+eye) transforms
├── planner/     z-plane picker, hex tiling, region packing, no-overlap checks
├── reporting/   xy/xz/yz overlay figures, ROI% computation, CSV / PDF export
├── gui/         PyQt5 main window + tabs (Data, Transducer, Plan, Results)
└── tests/       unit tests + sanity checks against the sample DUKE dataset
```

## Data assumptions (DUKE DMBA)

- MRI + label volumes share grid: 750 x 1260 x 600 voxels at 0.015 mm isotropic,
  LPS orientation, origin (5.6175, 11.9575, -8.0385).
- CT lives in the same physical (LPS) frame but on its own grid: 716 x 1300 x 712 at 0.02501 mm.
- Label 0 = Exterior; positive labels = brain ROIs per `DMBA_RCCF_labels_centroids.txt`.

## Coordinate conventions

All figures display in **anatomical** coordinates:
* +LR = right side of the animal (animal's right is on the viewer's right in xy)
* +AP = anterior (anterior is up in xy and yz panels)
* +IS = superior (up in xz and yz panels)

CSV outputs report each spot's offsets from the two landmarks (lambda suture
and bottom of left eye) in those same anatomical mm.

## CT-vs-labels registration note

The CT and the atlas labels are nominally in the same LPS frame, but in this
dataset (DMBA) we observed that the CT-detected outer skull surface lands
about 1-2 mm *inside* the labels' dorsal extent -- i.e. the side views show
the inner / outer skull lines drawn through the upper part of the cerebrum.

This is **not** a bug in the planner; it is a residual offset between the
per-subject CT and the atlas-space labels. Two ways to handle it in the GUI:

1. **Quick fix:** use the *"z offset for outer skull"* spinner on the Plan
   tab (under Transducer) to nudge the detected skull up by 1-2 mm. The
   inner skull and coupling gap are recomputed automatically.
2. **Proper fix:** re-co-register the CT to the labels frame in 3D Slicer
   (rigid registration, points-or-mutual-information based) and re-export.
   Then leave the offset at 0.
