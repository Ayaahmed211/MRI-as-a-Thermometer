import numpy as np
from scipy import ndimage


# ======================================================================
# 1) NOISE
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
    """Displacement (mm) at time t (seconds): a sine wave."""
    t = np.asarray(t, dtype=float)
    return amp_mm * np.sin(2 * np.pi * t / period_s + phase)


def shift_map(img, valid, dy_mm, dx_mm, pixel_size_m, order=1):
    """Move one map by (dy, dx) mm without leaking the zero background into tissue."""
    s = (dy_mm / (pixel_size_m * 1000), dx_mm / (pixel_size_m * 1000))
    v = ndimage.shift(valid.astype(float), s, order=1, mode="constant", cval=0.0)
    m = ndimage.shift(img * valid, s, order=order, mode="constant", cval=0.0)
    out = np.where(v > 0.5, m / np.maximum(v, 1e-6), 0.0)
    return out, v > 0.5


def apply_motion(phantom, dy_mm, dx_mm=0.0):
    """Return a shifted copy of the phantom (what the scanner 'sees' in that frame)."""
    dx = float(phantom["pixel_size_m"])
    valid = phantom["valid"]
    out = {}
    for k in ("PD", "T1", "T2", "T2s"):
        out[k], new_valid = shift_map(phantom[k], valid, dy_mm, dx_mm, dx)
    out["valid"] = new_valid
    tm, _ = shift_map(phantom["tumor_mask"].astype(float), valid, dy_mm, dx_mm, dx)
    out["tumor_mask"] = tm > 0.5
    lab = ndimage.shift(phantom["label"].astype(float),
                        (dy_mm / (dx * 1000), dx_mm / (dx * 1000)),
                        order=0, mode="constant", cval=-1)
    out["label"] = np.where(new_valid, lab, -1).astype(int)
    out["pixel_size_m"] = dx
    return out


# ======================================================================
# 3) B0 FIELD ERRORS
# ======================================================================
def hz_per_degC(B0_T, alpha_ppm=-0.01):
    """Frequency change (Hz) of water protons for +1 degC.  3 T -> about -1.28 Hz/degC."""
    return alpha_ppm * 1e-6 * 42.58e6 * B0_T


def b0_map(valid, pixel_size_m, amp_hz=20.0, smooth_mm=50.0, seed=0):
    """Static smooth field error (Hz) for every pixel (zero outside the brain).

    amp_hz    : largest error in the brain (an assumption: tens of Hz after shimming)
    smooth_mm : how smooth it is (bigger = smoother)
    """
    rng = np.random.default_rng(seed)
    sigma_pix = smooth_mm / (pixel_size_m * 1000)
    f = ndimage.gaussian_filter(rng.standard_normal(valid.shape), sigma_pix, mode="reflect")
    f = f - f[valid].mean()
    f = f * (amp_hz / np.abs(f[valid]).max())
    return f * valid


def b0_drift(t, rate_hz_per_s=0.02):
    """Slow uniform drift (Hz) at time t (seconds): same value in every pixel (an assumption)."""
    return rate_hz_per_s * np.asarray(t, dtype=float)


# ======================================================================
# self-tests
# ======================================================================
if __name__ == "__main__":
    d = np.load("phantom/phantom.npz")
    valid = d["valid"]
    dxm = float(d["pixel_size_m"])

    # ---- test 1: noise ----
    print("--- noise test ---")
    clean_img = d["PD"] * valid
    k = np.fft.fft2(clean_img, norm="ortho")
    lab = d["label"]
    wm = lab == np.bincount(lab[valid]).argmax()
    for want in (10, 25, 50, 100):
        noisy_k = add_noise(k, want, seed=1, ref_mask=wm)
        noisy_img = np.fft.ifft2(noisy_k, norm="ortho")
        measured = np.abs(clean_img)[wm].mean() / np.std((noisy_img - clean_img).real)
        print(f"wanted SNR {want:>3}   measured SNR {measured:6.1f}")

    # ---- test 2: motion ----
    print("\n--- motion test: tumor moves as commanded ---")
    dx_mm = dxm * 1000
    ref_row = ndimage.center_of_mass(d["tumor_mask"].astype(float))[0]
    for cmd in (0.0, 1.0, 2.0, 3.0, -2.0):
        moved = apply_motion(d, dy_mm=cmd)
        row = ndimage.center_of_mass(moved["tumor_mask"].astype(float))[0]
        print(f"commanded {cmd:+.1f} mm   measured {(row - ref_row) * dx_mm:+.2f} mm")

    # ---- test 3: B0 map ----
    print("\n--- B0 map test ---")
    f = b0_map(valid, dxm, amp_hz=20.0, seed=0)
    steps = np.abs(np.diff(f, axis=0))[valid[1:] & valid[:-1]]
    print(f"largest error in brain : {np.abs(f[valid]).max():.1f} Hz (asked for 20)")
    print(f"mean error in brain    : {f[valid].mean():+.3f} Hz (should be about 0)")
    print(f"outside the brain      : {np.abs(f[~valid]).max():.1f} Hz (should be 0)")
    print(f"largest jump between neighbouring pixels: {steps.max():.2f} Hz (small = smooth)")

    # ---- test 4: does subtracting the first image remove the static B0 error? ----
    print("\n--- baseline subtraction test (3 T, TE = 20 ms, true heating = +10 degC at the tumor) ---")
    B0, TE = 3.0, 0.020
    heat = np.zeros(valid.shape)
    heat[d["tumor_mask"]] = 10.0                                     # degC above 37
    f_heat = hz_per_degC(B0) * heat                                  # Hz caused by heating
    S0 = np.exp(2j * np.pi * f * TE)                                 # frame 0: B0 error only
    S1 = np.exp(2j * np.pi * (f + f_heat) * TE)                      # later frame: B0 error + heating
    dphi = np.angle(S1 * np.conj(S0))                                # phase difference
    T_est = dphi / (2 * np.pi * hz_per_degC(B0) * TE)
    t = d["tumor_mask"]
    print(f"temperature rise in the tumor : true +10.00   measured {T_est[t].mean():+.2f}")
    print(f"error outside the tumor       : {np.abs(T_est[valid & ~t]).max():.2e} degC  (0 = B0 error removed)")

    # ---- test 5: drift ----
    print("\n--- drift test: a uniform drift looks like a fake temperature change ---")
    for B0 in (1.5, 3.0):
        fake = []
        for tt in (30, 60, 120):
            df = float(b0_drift(tt, rate_hz_per_s=0.02))
            fake.append(abs(df / hz_per_degC(B0)))
        print(f"B0 = {B0} T : {hz_per_degC(B0):+.2f} Hz per degC   fake error after 30/60/120 s = "
              f"{fake[0]:.2f} / {fake[1]:.2f} / {fake[2]:.2f} degC")