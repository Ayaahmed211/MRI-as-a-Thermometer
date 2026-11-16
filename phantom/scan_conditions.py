import numpy as np
from scipy import ndimage


# ======================================================================
# 1) NOISE  (already tested)
# ======================================================================
def add_noise(kspace, snr, seed=0, ref_mask=None):
    """Add complex Gaussian noise to k-space so the image SNR equals `snr`."""
    rng = np.random.default_rng(seed)
    img = np.fft.ifft2(kspace, axes=(-2, -1), norm="ortho")
    mag = np.abs(img)
    mag0 = mag[0] if mag.ndim == 3 else mag
    if ref_mask is None:
        ref_mask = mag0 > 0.5 * mag0.max()
    signal = mag0[ref_mask].mean()
    sigma = signal / snr
    noise = sigma * (rng.standard_normal(kspace.shape) + 1j * rng.standard_normal(kspace.shape))
    return kspace + noise


# ======================================================================
# 2) BREATHING MOTION
# ======================================================================
def breathing_shift(t, amp_mm=2.0, period_s=4.0, phase=0.0):
    """Displacement (mm) at time t (seconds): a sine wave.

    amp_mm   : peak displacement (0 = no motion)
    period_s : time of one breath (4 s = 15 breaths per minute)
    Returns an array of the same shape as t.
    """
    t = np.asarray(t, dtype=float)
    return amp_mm * np.sin(2 * np.pi * t / period_s + phase)


def shift_map(img, valid, dy_mm, dx_mm, pixel_size_m, order=1):
    """Move one map (PD, T1, ...) by (dy, dx) in mm. Positive dy = toward higher row index.

    The shift is done on (map * valid) and on `valid`, and then divided,
    so the zero background does not leak into tissue values at the brain edge.
    """
    s = (dy_mm / (pixel_size_m * 1000), dx_mm / (pixel_size_m * 1000))
    v = ndimage.shift(valid.astype(float), s, order=1, mode="constant", cval=0.0)
    m = ndimage.shift(img * valid, s, order=order, mode="constant", cval=0.0)
    out = np.where(v > 0.5, m / np.maximum(v, 1e-6), 0.0)
    return out, v > 0.5


def apply_motion(phantom, dy_mm, dx_mm=0.0):
    """Return a shifted copy of the phantom dictionary (what the scanner 'sees' in that frame).

    phantom : dict-like loaded from phantom.npz
    """
    dx = float(phantom["pixel_size_m"])
    valid = phantom["valid"]
    out = {}
    for k in ("PD", "T1", "T2", "T2s"):
        out[k], new_valid = shift_map(phantom[k], valid, dy_mm, dx_mm, dx)
    out["valid"] = new_valid
    # tumor mask and labels are shifted too (nearest neighbour for labels)
    tm, _ = shift_map(phantom["tumor_mask"].astype(float), valid, dy_mm, dx_mm, dx)
    out["tumor_mask"] = tm > 0.5
    lab = ndimage.shift(phantom["label"].astype(float),
                        (dy_mm / (dx * 1000), dx_mm / (dx * 1000)),
                        order=0, mode="constant", cval=-1)
    out["label"] = np.where(new_valid, lab, -1).astype(int)
    out["pixel_size_m"] = dx
    return out


# ======================================================================
# self-tests
# ======================================================================
if __name__ == "__main__":
    d = np.load("phantom/phantom.npz")

    # ---- test 1: noise ----
    print("--- noise test ---")
    clean_img = d["PD"] * d["valid"]
    k = np.fft.fft2(clean_img, norm="ortho")
    lab = d["label"]
    wm = lab == np.bincount(lab[d["valid"]]).argmax()
    for want in (10, 25, 50, 100):
        noisy_k = add_noise(k, want, seed=1, ref_mask=wm)
        noisy_img = np.fft.ifft2(noisy_k, norm="ortho")
        measured = np.abs(clean_img)[wm].mean() / np.std((noisy_img - clean_img).real)
        print(f"wanted SNR {want:>3}   measured SNR {measured:6.1f}")

    # ---- test 2: breathing shift over time ----
    print("\n--- breathing: shift at each frame (amp 2 mm, period 4 s, frame every 3 s) ---")
    frames_t = np.arange(0, 30, 3.0)
    shifts = breathing_shift(frames_t, amp_mm=2.0, period_s=4.0)
    for tt, s in zip(frames_t, shifts):
        print(f"t = {tt:4.0f} s   shift = {s:+.2f} mm")

    # ---- test 3: does the tumor really move by the commanded amount? ----
    print("\n--- motion test: tumor centre-of-mass moves as commanded ---")
    dx_mm = float(d["pixel_size_m"]) * 1000
    ref_row = ndimage.center_of_mass(d["tumor_mask"].astype(float))[0]
    for cmd in (0.0, 1.0, 2.0, 3.0, -2.0):
        moved = apply_motion(d, dy_mm=cmd)
        row = ndimage.center_of_mass(moved["tumor_mask"].astype(float))[0]
        print(f"commanded {cmd:+.1f} mm   measured {(row - ref_row) * dx_mm:+.2f} mm")