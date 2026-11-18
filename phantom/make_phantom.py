import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ---------- 1) read the raw brain ----------
d = np.load("phantom/phantom_raw.npz")
x, y = d["x"], d["y"]
xs, ys = np.unique(x), np.unique(y)
ix = np.searchsorted(xs, x)
iy = np.searchsorted(ys, y)
shape = (len(ys), len(xs))

def to_image(v):
    img = np.zeros(shape)
    img[iy, ix] = v
    return img

valid = np.zeros(shape, bool)
valid[iy, ix] = True
PD, T1, T2, T2s = [to_image(d[k]) for k in ("PD", "T1", "T2", "T2s")]

# off-resonance from the phantom (rad/s): fat has a chemical shift of -220 Hz
DW = to_image(d["dw"]) if "dw" in d.files else np.zeros(shape)
fat = valid & (DW != 0)                      # FAT1 + FAT2
print("fat pixels (chemical-shifted):", int(fat.sum()))

dx = np.median(np.diff(xs))                  # pixel size in meters
print(f"grid: {shape[0]} x {shape[1]}   pixel size: {dx*1000:.2f} mm")

# ---------- 2) tissue labels ----------
combos = np.round(np.stack([PD[valid], T1[valid], T2[valid]], axis=1), 6)
uniq, inv = np.unique(combos, axis=0, return_inverse=True)
inv = inv.ravel()
label = np.full(shape, -1, int)
label[valid] = inv

# Identify tissues from their (PD, T1 [s], T2 [s]) values, NOT from pixel counts.
# Values confirmed in the KomaMRI source (BrainWeb): WM 0.77, GM 0.86, CSF 1.0.
STANDARD = {"wm":  (0.77, 0.500, 0.070),
            "gm":  (0.86, 0.833, 0.083),
            "csf": (1.00, 2.569, 0.329)}

def find(name):
    ref = np.array(STANDARD[name])
    i = int(np.argmin(np.abs(uniq - ref).sum(axis=1)))
    assert np.abs(uniq[i] - ref).max() < 0.01, f"{name}: no matching tissue in the phantom"
    return i

wm_label, gm_label, csf_label = find("wm"), find("gm"), find("csf")
print(f"tissue labels   WM: {wm_label}   GM: {gm_label}   CSF: {csf_label}")
print("label  PD     T1(ms)  T2(ms)  pixels")
for i, (pd_, t1_, t2_) in enumerate(uniq):
    print(f"{i:>4}  {pd_:5.2f}  {t1_*1000:7.0f} {t2_*1000:7.0f}  {int(np.sum(inv == i))}")
print(f"fat pixels inside the WM-valued label (FAT2): {int((fat & (label == wm_label)).sum())}")

wm = (label == wm_label) & ~fat               # white matter without FAT2/marrow-like fat
gm = label == gm_label
csf = label == csf_label                      # CSF only (skin has the same T2 but is not fluid)
dist_nonwm = ndimage.distance_transform_edt(wm)          # pixels to nearest non-white-matter
dist_csf_mm = ndimage.distance_transform_edt(~csf) * dx * 1000
cm = np.array(ndimage.center_of_mass(valid))

# ---------- 3) tumor geometry ----------
R = 0.005                                     # 5 mm radius = 1 cm diameter
r_pix = R / dx
n = int(np.ceil(r_pix))
yy, xx = np.ogrid[-n:n + 1, -n:n + 1]
disk = ((xx**2 + yy**2) <= r_pix**2).astype(float)
X, Y = np.meshgrid(xs, ys)

def pick(region, score_fn, thrs=(1.0, 0.9, 0.8, 0.7)):
    """Find a place where the tumor disk fits inside `region`; best score wins."""
    frac = ndimage.correlate(region.astype(float), disk, mode="constant") / disk.sum()
    for thr in thrs:
        ok = (frac >= thr) & region
        if ok.any():
            idx = np.argwhere(ok)
            best = idx[np.argmax(score_fn(idx, frac))]
            return int(best[0]), int(best[1]), thr
    raise RuntimeError("no place found")

dist_c = lambda idx: np.sum((idx - cm) ** 2, axis=1)

LOCATIONS = {
    # near the ventricle, close to the center (the default phantom)
    "center":      pick(wm, lambda idx, f: -dist_c(idx)),
    # deepest white matter, as far as possible from fluid and cortex
    "deep_wm":     pick(wm, lambda idx, f: dist_nonwm[idx[:, 0], idx[:, 1]]),
    # gray matter (cortex)
    "gray_matter": pick(gm, lambda idx, f: f[idx[:, 0], idx[:, 1]] * 1000 - dist_c(idx)),
    # white matter far from the center, close to the brain edge
    "edge_wm":     pick(wm, lambda idx, f: dist_c(idx)),
}
# To add your own spot: LOCATIONS["my_spot"] = (row, col, 1.0)

# ---------- 4) build + save one phantom per location ----------
def build(ci, cj):
    cy, cx = ys[ci], xs[cj]
    tumor = ((X - cx) ** 2 + (Y - cy) ** 2 <= R**2) & valid
    PD_t, T1_t, T2_t, T2s_t = PD.copy(), T1.copy(), T2.copy(), T2s.copy()
    PD_t[tumor] *= 1.15          # assumption: tumor slightly different from normal tissue
    T1_t[tumor] *= 1.30
    T2_t[tumor] *= 1.40
    T2s_t[tumor] *= 1.30
    return tumor, PD_t, T1_t, T2_t, T2s_t, cx, cy

print("\nname          row  col  disk-inside  distance to CSF   tumor pixels")
built = {}
for name, (ci, cj, thr) in LOCATIONS.items():
    tumor, PD_t, T1_t, T2_t, T2s_t, cx, cy = build(ci, cj)
    built[name] = (tumor, T2_t)
    print(f"{name:<12} {ci:>4} {cj:>4}   {thr*100:>5.0f}%       {dist_csf_mm[ci, cj]:>6.1f} mm        {tumor.sum():>4}")
    fname = "phantom/phantom.npz" if name == "center" else f"phantom/phantom_{name}.npz"
    np.savez(fname,
             PD=PD_t, T1=T1_t, T2=T2_t, T2s=T2s_t,
             label=label, valid=valid, tumor_mask=tumor,
             tumor_center_idx=np.array([ci, cj]),
             tumor_center_m=np.array([cx, cy]),
             x_axis=xs, y_axis=ys, pixel_size_m=dx,
             dw=DW, fat_mask=fat)
print("\nsaved. 'center' is saved as phantom/phantom.npz (the default).")

# ---------- 5) picture: all locations ----------
fig, ax = plt.subplots(1, len(built), figsize=(4 * len(built), 4.5))
for a, (name, (tumor, T2_t)) in zip(np.atleast_1d(ax), built.items()):
    a.imshow(np.where(valid, T2_t, np.nan), cmap="gray", origin="lower")
    a.contour(tumor, levels=[0.5], colors="r", linewidths=1.5)
    a.set_title(name)
plt.tight_layout()
plt.savefig("phantom/phantom_locations.png", dpi=120)
plt.show()