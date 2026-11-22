import argparse
from pathlib import Path

import numpy as np
import yaml


# ---------- the five stages (only stage 1 is real for now) ----------
def stage1_phantom(cfg, out):
    d = np.load(cfg["phantom"])
    print("  phantom loaded, grid =", d["PD"].shape,
          "| tumor pixels =", int(d["tumor_mask"].sum()))


def stage2_heat(cfg, out):
    from heat.run_heat import run_heat
    from heat.dose import run_dose
    run_heat(cfg, out=out)
    run_dose(cfg, out)


def stage3_mri(cfg, out):
    from mri.simulate_mri import run_mri_stage
    run_mri_stage(cfg, out)


def stage4_recon(cfg, out):
    print("  [not implemented yet]  Member 4: phase -> temperature map")


def stage5_eval(cfg, out):
    print("  [not implemented yet]  Member 5: errors, plots, animation")


STAGES = [
    ("1. virtual patient", stage1_phantom),
    ("2. heat model", stage2_heat),
    ("3. MRI simulation", stage3_mri),
    ("4. reconstruction + thermometry", stage4_recon),
    ("5. evaluation", stage5_eval),
]


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a yaml file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = Path("results") / cfg["name"]      # results/baseline/
    out.mkdir(parents=True, exist_ok=True)

    print("experiment:", cfg["name"], "| B0 =", cfg["field_T"], "T | SNR =", cfg["scan"]["snr"])
    for title, stage in STAGES:
        print("\nstage", title)
        stage(cfg, out)
    print("\ndone. results folder:", out)


if __name__ == "__main__":
    main()