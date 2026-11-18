# MRI as a Thermometer

Simulating PRF-shift MR thermometry for monitoring MRI-guided thermal ablation
(Medical Imaging I, SBE 4120).

**Idea:** heating changes the resonance frequency of water protons. In a gradient-echo
MRI image this shows up in the *phase*. Comparing the phase before heating with the phase
during heating gives a temperature map of every pixel. We simulate the whole process on a
virtual brain with a tumor, and measure how accurate the MRI thermometer is, because in a
simulation we know the true temperature.

## Project status

| Block | Owner | Status |
|---|---|---|
| 1. Virtual patient, scan conditions, pipeline driver | Member 1 | Done |
| 2. Heat model, thermal dose, true ablation zone | Member 2 | Done (report section left) |
| 3. MRI sequence and simulation (Pypulseq + KomaMRI) | Member 3 | To do |
| 4. Reconstruction and PRF thermometry | Member 4 | To do |
| 5. Evaluation, experiments, figures, animation | Member 5 | To do |

## Quick start

```bash
git clone https://github.com/Ayaahmed211/MRI-as-a-Thermometer.git
cd MRI-as-a-Thermometer
pip install -r requirements.txt

python run_pipeline.py --config configs/baseline.yaml
```

This runs stages 1 and 2 for real (stages 3 to 5 print "not implemented yet") and writes
everything to `results/baseline/`. The `results/` folder is **not** in Git, so everyone
generates it locally with the command above.

Julia and KomaMRI are only needed to regenerate the raw brain
(`julia phantom/export_phantom.jl`). `phantom/phantom_raw.npz` is already in the repo.

## Repository layout

```
configs/        one yaml file per experiment (all numbers live here)
phantom/        Member 1: virtual brain, tumor, noise / motion / field-error tools
heat/           Member 2: thermal properties, bioheat solver, heat source, thermal dose
mri/            Member 3 (empty for now)
recon/          Member 4 (empty for now)
eval/           Member 5 (empty for now)
results/        generated output, not in Git
run_pipeline.py runs the five stages in order from a config file
```

## Conventions everyone must follow

- **Grid:** 103 rows x 86 columns, pixel size 2.0 mm (about 20.6 x 17.2 cm field of view).
  Arrays are indexed `[row, col]` = `[y, x]`.
- **Plots:** use `origin="lower"` in `imshow`, so the brain is the right way up.
- **Units:** temperature in degC, time in seconds, T1 and T2 in **seconds** (not ms),
  frequency shifts in Hz, lengths in meters (millimeters where stated).
- **Frames:** one image every 3 s (`frame_dt_s` in the config).
- **No hard-coded numbers in code:** read them from the yaml config.
- **Branches:** work in your own folder; run `git pull` before every push.

## Member 1: virtual patient and scan conditions (`phantom/`)

### Virtual brain

| File | What it does |
|---|---|
| `export_phantom.jl` | Julia: takes the 2D brain slice from KomaMRI (`brain_phantom2D()`, 6506 points) and saves `phantom_raw.npz`. |
| `make_phantom.py` | Python: turns the raw points into 103 x 86 images, labels the tissues, inserts a 1 cm tumor, saves one phantom per tumor location. |
| `view_phantom.py` | Quick plot of PD, T1, T2 for checking. |

The brain has 6 tissue types, found from their (T1, T2) values:

| Label | T1 / T2 (ms) | Tissue |
|---|---|---|
| 0 | 500 / 70 | white matter |
| 1 | 833 / 83 | gray matter |
| 2 | 350 / 70 | fat |
| 3 | 569 / 329 | skin |
| 4 | 900 / 47 | muscle |
| 5 | 2569 / 329 | CSF |

The skull has no signal in this phantom (it is the empty ring around the brain).

**Tumor:** disk of radius 5 mm (1 cm diameter, 21 pixels), centered at row 50, column 49,
next to the ventricle. To make it visible it gets PD x 1.15, T1 x 1.30, T2 x 1.40,
T2* x 1.30 compared with normal tissue (**our assumption**, not measured values).

**`phantom/phantom.npz` (the default phantom)** contains:
`PD, T1, T2, T2s` (maps), `label` (tissue number per pixel, -1 outside), `valid` (True inside
the head), `tumor_mask`, `tumor_center_idx` (row, col), `tumor_center_m` (x, y in meters),
`x_axis, y_axis` (meters), `pixel_size_m`.

Other tumor positions (each file has **one** tumor in an otherwise identical brain) are
available for extra experiments: `phantom_deep_wm.npz`, `phantom_gray_matter.npz`,
`phantom_edge_wm.npz`. A picture of all four is in `phantom_locations.png`.

### Scan-condition tools (`phantom/scan_conditions.py`)

Run `python phantom/scan_conditions.py` to execute the self-tests.

| Function | What it does |
|---|---|
| `add_noise(kspace, snr, seed, ref_mask)` | Adds complex Gaussian noise to k-space so that the image SNR equals `snr`. Apply it **after** the MRI simulation. |
| `breathing_shift(t, amp_mm, period_s, phase)` | Displacement in mm at time `t` (sine wave). |
| `apply_motion(phantom, dy_mm, dx_mm)` | Returns a copy of the phantom shifted by that amount. Use it **before** simulating each frame. |
| `b0_map(valid, pixel_size_m, amp_hz, smooth_mm, seed)` | Smooth static field-error map in Hz (zero outside the head). |
| `b0_drift(t, rate_hz_per_s)` | Slow uniform field drift in Hz at time `t`. |
| `hz_per_degC(B0_T, alpha_ppm=-0.01)` | Frequency change per degC: about -1.28 Hz at 3 T and -0.64 Hz at 1.5 T. |

Test results:

- Noise: measured SNR 10.1 / 25.1 / 50.3 / 100.5 for requested 10 / 25 / 50 / 100.
- Motion: the tumor moves exactly as commanded (for example +2.00 mm for +2 mm).
- Field map: peak 20.0 Hz, mean 0, largest jump between neighbouring pixels 0.6 Hz (smooth).
- **Baseline subtraction removes the static field error:** a +10 degC tumor is measured as
  +10.00 degC with 0 error elsewhere (3 T, TE = 20 ms).
- **Drift is not removed by subtraction:** it looks like fake heating. After 30 / 60 / 120 s
  the fake error is 0.47 / 0.94 / 1.88 degC at 3 T and twice that at 1.5 T.

Default values and assumptions (all in the config): motion amplitude 2 mm, breathing period
4 s, along the row axis; field error 20 Hz; drift 0.02 Hz/s (set to 0 in the baseline).
Because the breathing period (4 s) and frame time (3 s) do not match, the brain is scanned in
only 4 repeating positions (0, -2, 0, +2 mm).

## Member 2: heat model and ground truth (`heat/`)

Run order: `properties.py` (once), then `run_pipeline.py` (calls `run_heat.py` and `dose.py`).

| File | What it does |
|---|---|
| `properties.py` | Maps each tissue to density, specific heat, conductivity and blood perfusion; saves `thermal_props.npz`. Values are approximate literature values (similar to the IT'IS database); cite the source in the report. The tumor gets its own values (assumption: slightly less perfused than gray matter). |
| `solver.py` | 2D Pennes bioheat equation, finite differences. The time step is reduced automatically to stay stable. The border and everything outside the head stay at 37 degC. |
| `run_heat.py` | Adds a Gaussian focal heat source centered on the tumor, heats for `duration_s`, then cools for `cooldown_s`. The source power is found automatically so the peak reaches `peak_C` (the equation is linear in the power). Saves the true temperature. |
| `dose.py` | `cem43(temp, t, cumulative=False)`: thermal dose in equivalent minutes at 43 degC (Sapareto-Dewey, R = 0.5 above 43 degC, 0.25 below). Also writes the true ablation masks. |

Solver tests (`python heat/solver.py`):

- No heating: no change from 37 degC (deviation 0).
- Diffusion of a Gaussian spot compared with the exact solution: peak error 0.38 %.
- Cooling by blood flow only: 5.468 degC (solver) versus 5.473 degC (exact).

Dose tests (`python heat/dose.py`): 43 degC for 10 min gives 10, 44 degC gives 20, 47 degC gives 160, and
47 degC for 1 min gives 16 equivalent minutes. All match the exact values.

### Outputs written to `results/<name>/`

| File | Contents |
|---|---|
| `temp_true.npy` | True temperature, shape (41, 103, 86), degC, one frame every 3 s (0 to 120 s). **This is the answer sheet.** |
| `t.npy` | Frame times in seconds, shape (41,). |
| `cem43_true.npy` | True thermal dose map (equivalent minutes at 43 degC). |
| `ablation_true.npy` | True ablation mask: dose >= 240 min (boolean). |
| `ablation_true_18.npy` | Secondary mask: dose >= 18 min (the lower threshold shown in the reference paper). |
| `heat_preview.png`, `dose_preview.png` | Quick-look figures. |

### Heating plans (configs)

| Config | Focus width (FWHM) | Peak | Tumor pixels ablated (of 21) | Healthy pixels ablated | Healthy pixels with dose >= 18 min |
|---|---|---|---|---|---|
| `baseline.yaml` | 9 mm | 65 degC | 21 | 0 | 12 |
| `narrow_focus.yaml` | 6 mm | 65 degC | 9 | 0 | 0 |
| `mild.yaml` | 9 mm | 45 degC | 0 | 0 | 0 |

- **baseline:** the whole tumor is ablated (84 mm2) and no healthy pixel reaches the ablation
  dose. Source power 3.21 MW/m3. After the 60 s of cooling the peak is 46.9 degC.
- **narrow_focus:** a focus narrower than the tumor spares healthy tissue but leaves most of the
  tumor under-treated.
- **mild:** warming without damage, useful to test whether accuracy depends on the temperature range.

Note for Members 3 and 4: in `baseline` the tumor warms by about 28 degC. At 3 T with TE = 20 ms the
phase changes by about 0.16 rad per degC, so it passes pi after about 19.6 degC. **The phase will wrap
and must be unwrapped.**

## How to use these outputs

**Member 3 (MRI simulation)** needs `phantom/phantom.npz`, `results/<name>/temp_true.npy`, `t.npy`, and the tools in
`phantom/scan_conditions.py`. Frequency shift per frame:

```python
import numpy as np
from phantom.scan_conditions import hz_per_degC, b0_map, apply_motion

T = np.load("results/baseline/temp_true.npy")        # (41, 103, 86) degC
df_heat = hz_per_degC(3.0) * (T - 37.0)              # Hz, per frame
# add b0_map(...) and b0_drift(...), put the total into the phantom's off-resonance, simulate each frame
```

Save k-space as `results/<name>/kspace.h5` with `B0, TE, TR, flip angle` and frame times, and replace
`stage3_mri` in `run_pipeline.py`.

**Member 4 (thermometry)** should reuse the dose function so the estimated and true ablation use the same formula:

```python
from heat.dose import cem43
dose_est = cem43(temp_est, t)           # temp_est: (Nt, 103, 86)
ablation_est = dose_est >= 240
```

Save `temp_est.npy` and `ablation_est.npy` in `results/<name>/` and replace `stage4_recon`.

**Member 5 (evaluation)** compares `temp_est.npy` with `temp_true.npy`, and `ablation_est.npy` with
`ablation_true.npy` (RMSE, bias, noise in unheated tissue, Dice). Replace `stage5_eval`.
The experiment list is in the work-division document.

## Assumptions to mention in the report

- Tumor MRI properties are scaled from normal tissue (x 1.15, 1.30, 1.40, 1.30), not measured.
- Thermal tissue properties are approximate literature values.
- Skull is not modeled (no signal in the phantom); heat equation uses the soft tissues only.
- Field error amplitude (20 Hz), drift rate (0.02 Hz/s) and breathing amplitude (2 mm) are chosen values.
- Temperature is assumed constant during one image acquisition.
- PRF coefficient: -0.01 ppm/degC (the paper reports -0.009 to -0.01).
- No fat or tissue-property changes during coagulation are modeled in the heat equation.

## Reference

V. Rieke, K. Butts Pauly, "MR thermometry," J. Magn. Reson. Imaging 27(2):376-390, 2008.
