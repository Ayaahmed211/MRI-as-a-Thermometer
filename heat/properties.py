import numpy as np

# name: (rho kg/m3, c J/kg/K, k W/m/K, perfusion 1/s)
# approximate typical literature values (similar to the IT'IS tissue database).
# Check them against the database and cite the source in the report.
TISSUES = {
    "white matter": (1041, 3583, 0.50, 0.0043),
    "gray matter":  (1045, 3696, 0.56, 0.0105),
    "CSF":          (1007, 4096, 0.57, 0.0),
    "fat":          (911,  2348, 0.21, 0.0003),
    "muscle":       (1090, 3421, 0.49, 0.0007),
    "skin":         (1109, 3391, 0.37, 0.0018),
    "tumor":        (1050, 3600, 0.55, 0.0080),   # assumption: a bit less perfused than gray matter
}

# (T1 ms, T2 ms) of each tissue class in the phantom (BrainWeb values used by KomaMRI)
KNOWN = {
    (500, 70):   "white matter",
    (833, 83):   "gray matter",
    (350, 70):   "fat",
    (569, 329):  "skin",          # KomaMRI name: SKIN/MUSCLE
    (900, 47):   "muscle",
    (2569, 329): "CSF",
}


def classify(t1_ms, t2_ms):
    """Pick the closest known (T1, T2) pair."""
    key = min(KNOWN, key=lambda p: abs(p[0] - t1_ms) + abs(p[1] - t2_ms))
    return KNOWN[key]


def build_props(phantom_path="phantom/phantom.npz", verbose=False):
    """Thermal property maps for the phantom in `phantom_path`."""
    d = np.load(phantom_path)
    label, valid, tumor = d["label"], d["valid"], d["tumor_mask"]
    T1, T2 = d["T1"], d["T2"]                         # seconds
    fat = d["fat_mask"] if "fat_mask" in d.files else np.zeros(valid.shape, bool)

    rho = np.zeros(label.shape)
    c = np.zeros(label.shape)
    k = np.zeros(label.shape)
    perf = np.zeros(label.shape)

    if verbose:
        print("label  tissue          pixels   T1(ms)  T2(ms)")
    for i in np.unique(label[valid]):
        m = (label == i) & ~tumor
        t1ms = np.median(T1[m]) * 1000
        t2ms = np.median(T2[m]) * 1000
        name = classify(t1ms, t2ms)
        r, cc, kk, pp = TISSUES[name]
        sel = (label == i) & valid
        rho[sel], c[sel], k[sel], perf[sel] = r, cc, kk, pp
        if verbose:
            print(f"{i:>4}   {name:<14} {sel.sum():>6}   {t1ms:>6.0f}  {t2ms:>6.0f}")

    # fat pixels that share their (PD, T1, T2) with another tissue (FAT2 looks like white matter)
    f = fat & ~tumor
    r, cc, kk, pp = TISSUES["fat"]
    rho[f], c[f], k[f], perf[f] = r, cc, kk, pp
    if verbose:
        print(f"fat pixels (all): {int(fat.sum())}")

    # the tumor gets its own values
    r, cc, kk, pp = TISSUES["tumor"]
    rho[tumor], c[tumor], k[tumor], perf[tumor] = r, cc, kk, pp

    return dict(rho=rho, c=c, k=k, perf=perf, valid=valid, tumor_mask=tumor,
                pixel_size_m=float(d["pixel_size_m"]))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    props = build_props("phantom/phantom.npz", verbose=True)
    rho, c, k, perf = props["rho"], props["c"], props["k"], props["perf"]
    valid, tumor, dx = props["valid"], props["tumor_mask"], props["pixel_size_m"]

    # thermal diffusivity (how fast heat spreads) and a stability check for the solver
    alpha = np.where(valid, k / np.maximum(rho * c, 1), 0)
    print(f"\nfastest heat spreading : {alpha.max():.2e} m2/s")
    print(f"pixel size             : {dx*1000:.1f} mm")
    print(f"largest safe time step : {dx**2 / (4 * alpha.max()):.1f} s  (for the simple solver)")

    np.savez("heat/thermal_props.npz", **props)
    print("saved heat/thermal_props.npz")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, img, name in zip(ax, [k, perf, rho * c / 1e6],
                            ["conductivity k (W/m/K)", "perfusion (1/s)", "rho*c (MJ/m3/K)"]):
        im = a.imshow(np.where(valid, img, np.nan), origin="lower", cmap="viridis")
        a.contour(tumor, levels=[0.5], colors="r", linewidths=1)
        a.set_title(name)
        plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    plt.savefig("heat/thermal_props.png", dpi=120)
    plt.show()