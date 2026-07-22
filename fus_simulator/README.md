# FUS Simulator

A GUI for running mouse-skull focused-ultrasound (FUS) k-Wave
simulations, built so that non-coders can pick a transducer and calibrate the
focal pressure. It is the simulation companion to `fus_planner`.

## Task 1

Select files to use for the simulation. Once the files are selected you can save 
them as default so next time you opne the app it will remember them. Use the 
cropped version of the skull from the DUKE brain atlas.

## Task 2: Parameters

Choose the transducer and source:

1. **Transducer & source** - set aperture diameter, focal length (radius of
   curvature), frequency, and the desired peak focal pressure.
2. **Central hole** - optionally model an annular aperture (hole in the middle
   of the bowl) and set its diameter.
3. **Grid auto-sizing** - the grid is expanded automatically so the geometric
   focus always sit inside it.
4. **Small sensor box** - pressure is recorded only inside a cuboid around the
   future skull region (plus buffer), not the whole grid, which keeps memory and
   post-processing small.

Click **"Prepare/Load"** to load a CT-derived mouse skull and see it against your 
transducer geometry. 


## Task 3: Simulation tab

* **Numerics** - points-per-wavelength (PPW), CFL, and tone-burst cycles. 
PPW still drives the grid spacing shown in the Transducer preview.
* **Run simulation** - the focal pressure you ask for is usually *not*
   what a given input drive produces. Because k-Wave acoustics are linear in the
   source amplitude, one free-field (water-only) run is enough:

   ```
   transfer gain  g = measured_peak / reference_drive        (output per input)
   calibrated drive = reference_drive * (desired / measured)  = desired / g
   ```
  Ater this the simulation is run through the mouse skull.

  The result reports the pressure at the geometric focus through the skull, the
  peak in the box, and the transmission (focus / free-field target), with XZ/XY
  pressure-field slices. It also report ISPTA, ISPPA and Mechanical index.

NOTE: at the moment the simulator's attenuation values reflect those for 500kHz centre 
frequency. We aim to extend to more frequencies after a more in depth research on
frequency dependency attenuation values in the murine skull.

## Thermal tab

Calculates skull/tissue heating from the pressure field:

* First run **Run through skull** on the Simulation tab - the thermal step heats
  from that pressure field and the embedded skull maps.
* Set the pulsing: **PRF**, **pulse duration** (duty cycle = PRF x pulse, shown
  live), **sonication time**, and optional **cooling time**. Plus baseline
  temperature, thermal time step, and an optional blood **perfusion** coefficient.
* Heat source `Q = 2 alpha I`, `I = (p_peak/sqrt2)^2 / (rho c)`, time-averaged
  by the duty cycle; tissue/skull `k` (0.6 / 0.32) and `Cp` (4180 / 1300) follow
  the notebook. CEM43 thermal dose and lesion volume (>= 240 min) are reported.
* Output: max-temperature XZ/XY maps, and peak temperature / rise / CEM43 / lesion metrics.


## Layout

```
fus_simulator/
├── run_gui.py            launch the GUI:  python run_gui.py
├── requirements.txt
├── pyproject.toml        console script: fus-simulator
├── src/fus_simulator/
│   ├── core/
│   │   ├── params.py       user parameters + fixed water/skull constants
│   │   ├── grid.py         auto grid sizing so the focus fits (pure NumPy)
│   │   ├── transducer.py   bowl + optional central hole (kWaveArray) + preview
│   │   ├── gpu.py          NVIDIA detection -> 3D vs 3DG
│   │   ├── simulate.py     free-field run, small sensor box, p_max
│   │   └── calibrate.py    linear pressure calibration
│   └── gui/main.py         PyQt5 window (Geometry + Calibration tabs)
└── tests/test_core.py      pure-NumPy unit tests (no GPU needed)
```

## Install & launch

Use a dedicated virtual environment. **Use Python 3.10 - 3.12** (numpy, PyQt5 and
k-wave-python do not yet ship wheels for 3.13/3.14, so those will try to compile
from source and fail):

```powershell
cd fus_simulator
py -3.12 -m venv .venv          # Windows; pick a 3.10-3.12 interpreter
.venv\Scripts\activate
python -m pip install --upgrade pip

# 1) GUI + live geometry preview (numpy / matplotlib / PyQt5)
pip install -r requirements.txt
python run_gui.py

# 2) Acoustic solver, installed separately so a build issue here
#    cannot block the GUI above
pip install --no-deps -r requirements-solver.txt

The **geometry preview** works with just `requirements.txt`. The **calibration
run** additionally needs `requirements-solver.txt` (`k-wave-python`, which uses
the CPU and CUDA binaries); the GPU path needs an NVIDIA GPU.
