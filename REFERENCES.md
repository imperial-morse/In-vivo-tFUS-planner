# References

---

## 1. Software

### Core 

| Package | Version | Citation |
|---|---|---|
| Python | 3.12 | Python Software Foundation. <https://www.python.org> |
| NumPy | 2.0.2 | Harris CR, Millman KJ, van der Walt SJ, et al. **Array programming with NumPy.** *Nature* 585:357–362 (2020). doi:10.1038/s41586-020-2649-2 |
| SciPy | 1.15.3 | Virtanen P, Gommers R, Oliphant TE, et al. **SciPy 1.0: fundamental algorithms for scientific computing in Python.** *Nature Methods* 17:261–272 (2020). doi:10.1038/s41592-019-0686-2 |
| Matplotlib | 3.9.2 | Hunter JD. **Matplotlib: A 2D graphics environment.** *Computing in Science & Engineering* 9(3):90–95 (2007). doi:10.1109/MCSE.2007.55 |
| pandas | 2.2.3 | McKinney W. **Data structures for statistical computing in Python.** *Proc. 9th Python in Science Conf.*, 56–61 (2010). doi:10.25080/Majora-92bf1922-00a |
| h5py | 3.16.0 | Collette A. **Python and HDF5.** O'Reilly Media (2013). <https://www.h5py.org> |

### Acoustic solver

| Package | Version | Citation |
|---|---|---|
| k-Wave (the method) | – | Treeby BE, Cox BT. **k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields.** *J. Biomedical Optics* 15(2):021314 (2010). doi:10.1117/1.3360308 |
| k-Wave (heterogeneous, power-law absorption) | – | Treeby BE, Jaros J, Rendell AP, Cox BT. **Modeling nonlinear ultrasound propagation in heterogeneous media with power law absorption using a k-space pseudospectral method.** *JASA* 131(6):4324–4336 (2012). doi:10.1121/1.4712021 |
| `kWaveArray` off-grid bowl source | – | Wise ES, Cox BT, Jaros J, Treeby BE. **Representing arbitrary acoustic source and sensor distributions in Fourier collocation methods.** *JASA* 146(1):278–288 (2019). doi:10.1121/1.5116132 |
| k-wave-python | 0.6.2 | Yagubbayli F, Sinden D, Simson W. **k-Wave-python** (software). Zenodo, doi:10.5281/zenodo.10719460. <https://github.com/waltsims/k-wave-python>. Licensed LGPL-3.0. This is the Python interface; the numerics are k-Wave, so cite the Treeby papers above as well. |


### Imaging IO and GUI

| Package | Version | Citation |
|---|---|---|
| SimpleITK | 2.5.5 | Lowekamp BC, Chen DT, Ibáñez L, Blezek D. **The design of SimpleITK.** *Frontiers in Neuroinformatics* 7:45 (2013). doi:10.3389/fninf.2013.00045 <br> Yaniv Z, Lowekamp BC, Johnson HJ, Beare R. **SimpleITK image-analysis notebooks.** *J. Digital Imaging* 31:290–303 (2018). doi:10.1007/s10278-017-0037-8 |
| pynrrd | 1.1.3 | Software: <https://github.com/mhe/pynrrd> |
| PyQt5 | 5.15.11 | Riverbank Computing Ltd. <https://riverbankcomputing.com/software/pyqt> |
| OpenCV (`opencv-python`) | 4.13.0.92 | Bradski G. **The OpenCV Library.** *Dr. Dobb's Journal of Software Tools* (2000). Pulled in by k-wave-python. |

Support libraries:
Pillow 10.4.0, contourpy 1.2.1, kiwisolver 1.4.5, tqdm 4.68.2, beartype 0.22.9,
deepdiff 9.0.0, jaxtyping 0.3.7.

> **Licensing.** The project is released under GPL-3.0-or-later because it links
> **PyQt5 (GPL v3)**. k-wave-python is LGPL-3.0 (weak copyleft) and does not by
> itself impose GPL. 

---

## 2. Acoustic and thermal properties

### Water (also used for every non-bone voxel)

| Quantity | Value | Source |
|---|---|---|
| Density ρ | 1000 kg/m³ | Constans et al. (2018), Table 3 |
| Sound speed c | 1500 m/s | Constans et al. (2018), Table 3 |
| Absorption α | 0.00217·f²  dB/cm (f in MHz) | **Kinsler et al.**, *Fundamentals of Acoustics*, 4th ed., p. 218 |
| Thermal conductivity k | 0.623 W/(m·K) at 37 °C | CRC Handbook of Chemistry and Physics / IAPWS |
| Specific heat Cp | 4180 J/(kg·K) at 37 °C | CRC Handbook of Chemistry and Physics / IAPWS |

`alpha_power = 2` since only one frequency is used in the simulations.

### Skull (cortical bone, binary bone/water model)

| Quantity | Value | Source |
|---|---|---|
| Density ρ | 1850 kg/m³ | **Duck (2013)**, via Constans et al. (2018) Table 3 |
| Sound speed c | 2400 m/s | **Duck (2013)**, via Constans et al. (2018) Table 3 |
| Thermal conductivity k | 0.44 W/(m·K) | **Duck (2013)**, via Constans et al. (2018) Table 3 |
| Specific heat Cp | 1300 J/(kg·K) | **Duck (2013)**, via Constans et al. (2018) Table 3 |
| Absorption α | 2.7·f^1.18 dB/cm → 1.19 dB/cm at 500 kHz | **Pinton et al. (2012)** |


### Thermal model

| Item | Source |
|---|---|
| Bioheat equation | Pennes HH. **Analysis of tissue and arterial blood temperatures in the resting human forearm.** *J. Applied Physiology* 1(2):93–122 (1948). |
| Thermal dose CEM43 | Sapareto SA, Dewey WC. **Thermal dose determination in cancer therapy.** *Int. J. Radiation Oncology Biology Physics* 10(6):787–800 (1984). |
| CEM43, form used here (R = 0.25 below 43 °C, 0.5 above) | Dewey WC. **Arrhenius relationships from the molecule and cell to the clinic.** *Int. J. Hyperthermia* 25(1):3–20 (2009). As applied in Constans et al. (2018). |

### Exposure metrics

| Metric | Definition | Source |
|---|---|---|
| I_SPPA | p_peak² / (2ρc) | Plane progressive wave intensity, Kinsler et al. Verified against Constans et al. (2018): 0.3 MPa → 3.0 W/cm² in brain. |
| I_SPTA | I_SPPA × duty cycle | AIUM/NEMA UD 2, *Acoustic Output Measurement Standard*. |
| MI | p_rarefactional [MPa] / √(f₀ [MHz]) | Apfel RE, Holland CK. **Gauging the likelihood of cavitation from short-pulse, low-duty cycle diagnostic ultrasound.** *Ultrasound in Medicine & Biology* 17(2):179–185 (1991). Codified in the AIUM/NEMA Output Display Standard. |

---

## 3. Literature the parameters were drawn from or checked against

1. **Constans C, Mateo P, Tanter M, Aubry J-F.** Potential impact of thermal
   effects during ultrasonic neurostimulation: retrospective numerical estimation
   of temperature elevation in seven rodent setups. *Physics in Medicine and
   Biology* 63(2):025003 (2018). doi:10.1088/1361-6560/aaa15c
   *Primary source for the bone/tissue property table, the bioheat formulation and
   the CEM43 form. The only one of these papers that actually solves bioheat.*

2. **Pinton G, Aubry J-F, Bossy E, Muller M, Pernot M, Tanter M.** Attenuation,
   scattering, and absorption of ultrasound in the skull bone. *Medical Physics*
   39(1):299–307 (2012). doi:10.1118/1.3668316
   *Separates absorption from scattering. Source of the bone absorption law.*

3. **Duck FA.** *Physical Properties of Tissues: A Comprehensive Reference Book.*
   Academic Press (1990; 2013 reprint).
   *Source of bone ρ, c, k and Cp.*

4. **Kinsler LE, Frey AR, Coppens AB, Sanders JV.** *Fundamentals of Acoustics*,
   4th ed. Wiley (2000).
   *Water absorption (p. 218) and the plane-wave intensity relation.*

5. **Li Z, Wang H, Zhang X, Lu C, Yu S, Yu B, Li Y, Zeng K, Li X.** Cross-species
   characterization of transcranial ultrasound propagation. *Brain Stimulation*
   18:164–172 (2025).
   *Source of the often-quoted 19.57 % mouse-skull attenuation. Note this is an
   INTENSITY transmission loss through the whole skull, dominated by reflection,
   and must not be used directly as an absorption coefficient.*

6. **Kim MG, Yeh C-Y, Yu K, Li Z, Gupta K, He B.** Analgesic effect of
   simultaneously targeting multiple pain processing brain circuits in an aged
   humanized mouse model of chronic pain by transcranial focused ultrasound.
   *APL Bioengineering* 9:016108 (2025). doi:10.1063/5.0236108
   *Comparable mouse k-Wave setup; used as a cross-check on skull absorption.*

7. **Goss SA, Frizzell LA, Dunn F.** Ultrasonic absorption and attenuation in
   mammalian tissues. *Ultrasound in Medicine & Biology* 5(2):181–186 (1979).
   *Brain absorption. Cited by Constans et al.; not used here because non-bone
   voxels are modelled as coupling water, not brain.*

---

## 4. Data

**Duke Mouse Brain Atlas (DMBA) / RCCF labels and CT**
`Mouse DUKE/` (not distributed with this repository).

The label volume's own `README.txt` states the labels were derived from the
**Allen Institute Common Coordinate Framework (CCF2017)**, transformed into an
atlas-oriented, symmetrised 15 µm template, with manual editing. 

- Allen CCF: **Wang Q, Ding S-L, Li Y, et al.** The Allen Mouse Brain Common
  Coordinate Framework: A 3D reference atlas. *Cell* 181(4):936–953 (2020).
  doi:10.1016/j.cell.2020.04.007
- DMBA / CT acquisition (Duke Center for In Vivo Microscopy): **Harrison Mansour et al.**,
  The Duke Mouse Brain Atlas: MRI and light sheet microscopy stereotaxic atlas of the 
  mouse brain.Sci. Adv.11,eadq8089(2025).DOI:10.1126/sciadv.adq8089

Volumes used:

| File | Purpose | Resolution |
|---|---|---|
| `DMBA_N20_..._CT_M4D.nhdr` | full CT, planner | 25 µm isotropic |
| `DMBA_N20_..._CT-cropped_M4D.nhdr` | cropped CT, simulator skull | 25 µm isotropic |
| `DMBA_RCCF_labels_M4D.nhdr` | atlas labels | 15 µm isotropic |
| `DMBA_RCCF_labels_centroids.txt` | region centroids, used for targeting | – |
| `DMBA_N02_dwi_M4D.nhdr` | MRI, planner | – |

Skull thickness measured from the CT in this project: median **0.18 mm**

---
