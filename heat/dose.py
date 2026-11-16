from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def cem43(temp, t, cumulative=False):
    """Thermal dose in equivalent minutes at 43 C (Sapareto-Dewey).

    temp : array (Nt, rows, cols) in degC
    t    : times in seconds, shape (Nt,)
    cumulative=True returns the dose after every frame (Nt, rows, cols),
    otherwise only the final dose (rows, cols).
    R = 0.5 for T >= 43 C and 0.25 below.
    """
    temp = np.asarray(temp, dtype=float)
    t = np.asarray(t, dtype=float)
    R = np.where(temp >= 43.0, 0.5, 0.25)
    rate = R ** (43.0 - temp)                       # minutes of 43 C per minute of real time
    dt_min = np.diff(t) / 60.0
    inc = 0.5 * (rate[1:] + rate[:-1]) * dt_min.reshape(-1, *([1] * (temp.ndim - 1)))
    cum = np.concatenate([np.zeros((1,) + temp.shape[1:]), np.cumsum(inc, axis=0)])
    return cum if cumulative else cum[-1]


def run_dose(cfg, out):
    out = Path(out)
    frames = np.load(out / "temp_true.npy")
    times = np.load(out / "t.npy")
    ph = np.load(cfg["phantom"])
    tumor, valid = ph["tumor_mask"], ph["valid"]
    dx_mm = float(ph["pixel_size_m"]) * 1000

    dose = cem43(frames, times)
    abl240 = dose >= 240
    abl18 = dose >= 18
    np.save(out / "cem43_true.npy", dose)
    np.save(out / "ablation_true.npy", abl240)
    np.save(out / "ablation_true_18.npy", abl18)

    n_t = int(tumor.sum())
    print(f"ablation (dose >= 240 min) : {int(abl240.sum())} pixels = {abl240.sum()*dx_mm**2:.0f} mm2")
    print(f"  tumor pixels ablated     : {int((abl240 & tumor).sum())} of {n_t}")
    print(f"  healthy pixels ablated   : {int((abl240 & valid & ~tumor).sum())}")
    print(f"ablation (dose >= 18 min)  : {int(abl18.sum())} pixels,"
          f"  healthy among them: {int((abl18 & valid & ~tumor).sum())}")
    print("saved cem43_true.npy, ablation_true.npy, ablation_true_18.npy")

    mx = frames.max(axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = ax[0].imshow(np.where(valid, mx, np.nan), origin="lower", cmap="hot", vmin=37, vmax=mx.max())
    ax[0].set_title("hottest each pixel ever got (C)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(np.where(valid, np.log10(np.maximum(dose, 1e-3)), np.nan),
                       origin="lower", cmap="viridis")
    ax[1].set_title("thermal dose, log10(CEM43 minutes)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    for a in ax:
        a.contour(tumor, levels=[0.5], colors="r", linewidths=1)
        a.contour(abl240, levels=[0.5], colors="c", linewidths=1.5)
        a.contour(abl18, levels=[0.5], colors="w", linewidths=1, linestyles="--")
    plt.tight_layout()
    plt.savefig(out / "dose_preview.png", dpi=120)
    return dose, abl240


if __name__ == "__main__":
    # ---- self-tests: results we can calculate by hand ----
    print("--- dose tests (exact answers known) ---")
    tt = np.arange(0, 601, 3.0)                                  # 10 minutes
    for T, exact in ((43.0, 10.0), (44.0, 20.0), (47.0, 160.0)):
        d = cem43(np.full((len(tt), 1, 1), T), tt)[0, 0]
        print(f"{T:.0f} C for 10 min : solver {d:9.3f}   exact {exact:9.3f}")
    tt = np.arange(0, 61, 3.0)
    d = cem43(np.full((len(tt), 1, 1), 47.0), tt)[0, 0]
    print(f"47 C for 1 min  : solver {d:9.3f}   exact {16.0:9.3f}")
    d = cem43(np.full((len(tt), 1, 1), 37.0), tt)[0, 0]
    print(f"37 C for 1 min  : {d:.2e} (should be tiny: body temperature does no damage)")

    import yaml
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)
    print("\n--- real result (baseline) ---")
    run_dose(cfg, Path("results") / cfg["name"])
    plt.show()