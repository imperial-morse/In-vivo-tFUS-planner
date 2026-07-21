"""fus_simulator - k-Wave focused-ultrasound simulation GUI.

Task 1 scope
------------
* Let the user choose the transducer (aperture diameter, focal length / radius
  of curvature, frequency) and an optional central hole (annular aperture).
* Auto-size the simulation grid so the geometric focus is always inside it,
  with a buffer.
* Detect an NVIDIA GPU and run with ``kspaceFirstOrder3DG`` when available,
  otherwise fall back to ``kspaceFirstOrder3D`` (CPU).
* Record pressure only inside a small sensor box around the region that will
  later hold the mouse skull (plus a buffer) - this keeps the run fast.
* Calibrate: find the factor by which the input source pressure must be
  scaled to obtain the desired peak focal pressure in free field (water).

"""

__version__ = "0.1.0"
