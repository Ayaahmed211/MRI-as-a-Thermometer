import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solver import solve, TA
from properties import build_props


def run_heat(cfg, out=None, show=False):
    h = cfg["heating"]
    dur, cool = h["duration_s"], h["cooldown_s"]
    fwhm_m = h["focus_fwhm_mm"] / 1000.0
    peak_C = h.get("peak_C", 65)
    frame_dt = cfg["frame_dt_s"]

    props = build_props(cfg["phantom"])
    ph = np.load(cfg["phantom"])
    xs, ys = ph["x_axis"], ph["y_axis"]
    cx, cy = ph["tumor_center_m"]
    tumor = ph["tumor_mask"]
    valid = ph["valid"]

    # Gaussian focal spot centered on the tumor
    X, Y = np.meshgrid(xs, ys)
    sigma = fwhm_m / 2.3548
    g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)) * valid

    # ---- step A: trial run with a guess, to learn how hot it gets ----
    Q0 = 1e6                                          # W/m3 (a guess)
    Q = lambda t: Q0 * g if t < dur else 0.0
    _, fr = solve(props, Q=Q, t_end=dur, frame_dt=frame_dt)
    rise = fr[-1].max() - TA
    Q0 = Q0 * (peak_C - TA) / rise                    # the equation is linear in Q
    print(f"heater power found : {Q0/1e6:.2f} MW/m3 at the focus (peak target {peak_C} C)")

    # ---- step B: the real run (heating, then cooling) ----
    Q = lambda t: Q0 * g if t < dur else 0.0
    times, frames = solve(props, Q=Q, t_end=dur + cool, frame_dt=frame_dt)

    # ---- numbers to check ----
    peak_t = frames.max(axis=(1, 2))
    i_end = int(round(dur / frame_dt))
    print(f"peak at end of heating ({dur:.0f} s) : {frames[i_end].max():.1f} C")
    print(f"peak at the very end ({times[-1]:.0f} s)  : {frames[-1].max():.1f} C")
    print(f"tumor mean temperature at {dur:.0f} s    : {frames[i_end][tumor].mean():.1f} C")
    mx = frames.max(axis=0)                            # hottest each pixel ever got
    for thr in (43, 50, 60):
        print(f"tumor pixels that reached >= {thr} C : {int((mx[tumor] >= thr).sum())} of {int(tumor.sum())}")
    print(f"healthy pixels that reached >= 50 C : {int(((mx >= 50) & valid & ~tumor).sum())}")

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "temp_true.npy", frames)
        np.save(out / "t.npy", times)
        print("saved", out / "temp_true.npy", "and t.npy")

        fig, ax = plt.subplots(1, 4, figsize=(17, 4))
        for a, tt in zip(ax[:3], (30, dur, dur + cool)):
            im = a.imshow(np.where(valid, frames[int(round(tt / frame_dt))], np.nan),
                          origin="lower", cmap="hot", vmin=TA, vmax=peak_C)
            a.contour(tumor, levels=[0.5], colors="c", linewidths=1)
            a.set_title(f"t = {tt:.0f} s")
            plt.colorbar(im, ax=a, fraction=0.046)
        ci, cj = ph["tumor_center_idx"]
        ax[3].plot(times, frames[:, ci, cj], label="tumor center")
        ax[3].plot(times, peak_t, "--", label="hottest pixel")
        ax[3].axvline(dur, color="gray", lw=0.8)
        ax[3].set_xlabel("time (s)")
        ax[3].set_ylabel("temperature (C)")
        ax[3].legend()
        plt.tight_layout()
        plt.savefig(out / "heat_preview.png", dpi=120)
        if show:
            plt.show()
    return times, frames


if __name__ == "__main__":
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    run_heat(cfg, out=Path("results") / cfg["name"], show=True)