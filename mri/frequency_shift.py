"""Temperature-to-frequency shift calculation and scanner off-resonance mapping (Member 3).

Physical principles:
- Water proton resonance frequency drops with temperature due to hydrogen bond breaking:
    df_heat = hz_per_degC(B0, alpha_ppm) * (T - 37.0)
  where at 3 T with alpha = -0.01 ppm/°C:
    df_heat ≈ -1.28 Hz/°C.
- PRF shift is temperature-independent in adipose tissue (fat). Fat pixels MUST NOT
  receive the thermal frequency shift.
- Total scanner off-resonance includes:
    df_total = df_heat + df_static_b0 + df_drift + df_chem_shift
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from phantom.scan_conditions import b0_drift, b0_map, hz_per_degC


def compute_prf_shift(
    temp_map: np.ndarray,
    valid_mask: np.ndarray,
    fat_mask: np.ndarray,
    b0_t: float = 3.0,
    alpha_ppm: float = -0.01,
) -> np.ndarray:
    """Compute the heat-induced PRF frequency shift map (Hz).

    Parameters
    ----------
    temp_map : np.ndarray
        Temperature map in °C (shape: Ny, Nx).
    valid_mask : np.ndarray
        Boolean mask of pixels inside the head.
    fat_mask : np.ndarray
        Boolean mask of fat pixels.
    b0_t : float
        Main magnetic field strength in Tesla.
    alpha_ppm : float
        PRF temperature coefficient in ppm/°C (default: -0.01 ppm/°C).

    Returns
    -------
    df_heat : np.ndarray
        Heat-induced frequency shift in Hz. Water pixels receive the shift;
        fat pixels and background pixels are strictly zero.
    """
    scale = hz_per_degC(b0_t, alpha_ppm=alpha_ppm)
    delta_t = temp_map - 37.0

    # Apply PRF shift ONLY to water-containing pixels inside the valid head mask
    water_mask = valid_mask & (~fat_mask)
    df_heat = np.where(water_mask, scale * delta_t, 0.0)
    return df_heat


def compute_total_off_resonance(
    phantom: Dict[str, Any],
    temp_map: np.ndarray,
    t_s: float,
    cfg: Dict[str, Any],
    static_b0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute the composite off-resonance map (Hz) for an acquisition frame.

    Parameters
    ----------
    phantom : dict or np.lib.npyio.NpzFile
        Phantom dictionary containing 'valid', 'fat_mask', 'pixel_size_m', and optionally 'dw'.
    temp_map : np.ndarray
        Temperature map for this frame in °C (shape: Ny, Nx).
    t_s : float
        Frame acquisition timestamp in seconds.
    cfg : dict
        Experiment configuration dictionary.
    static_b0 : np.ndarray, optional
        Precomputed static B0 inhomogeneity map in Hz. If None, it is computed
        from configuration settings.

    Returns
    -------
    df_total : np.ndarray
        Total off-resonance map in Hz.
    df_heat : np.ndarray
        Thermal PRF frequency shift in Hz.
    static_b0 : np.ndarray
        Static B0 field error map in Hz.
    drift_val : float
        Uniform B0 drift value in Hz at time t_s.
    """
    valid = np.asarray(phantom["valid"], dtype=bool)
    fat_mask = np.asarray(phantom["fat_mask"], dtype=bool) if "fat_mask" in phantom else np.zeros_like(valid)
    pixel_size_m = float(phantom["pixel_size_m"])
    b0_t = float(cfg.get("field_T", 3.0))

    scan_cfg = cfg.get("scan", {})

    # 1. Heat-induced PRF shift (water only)
    df_heat = compute_prf_shift(temp_map, valid, fat_mask, b0_t=b0_t)

    # 2. Static B0 Inhomogeneity
    if static_b0 is None:
        if scan_cfg.get("b0_inhomogeneity", True):
            amp_hz = float(scan_cfg.get("b0_amp_hz", 20.0))
            seed = int(cfg.get("seed", 0))
            static_b0 = b0_map(valid, pixel_size_m=pixel_size_m, amp_hz=amp_hz, seed=seed)
        else:
            static_b0 = np.zeros_like(valid, dtype=float)

    # 3. Time-dependent B0 Drift
    drift_rate = float(scan_cfg.get("b0_drift_hz_per_s", 0.0))
    if abs(drift_rate) > 1e-9:
        drift_val = float(b0_drift(t_s, rate_hz_per_s=drift_rate))
    else:
        drift_val = 0.0

    # 4. Fat Chemical Shift (if stored in phantom dw in rad/s)
    if "dw" in phantom:
        # dw is off-resonance in rad/s from KomaMRI; convert to Hz
        chem_shift_hz = (np.asarray(phantom["dw"], dtype=float) / (2.0 * np.pi)) * valid
    else:
        chem_shift_hz = np.zeros_like(valid, dtype=float)

    # Composite off-resonance map
    df_total = (df_heat + static_b0 + drift_val + chem_shift_hz) * valid

    return df_total, df_heat, static_b0, drift_val


if __name__ == "__main__":
    ph = np.load("phantom/phantom.npz")
    valid = ph["valid"]
    fat = ph["fat_mask"]

    # Test +10 °C rise
    t_test = np.full(valid.shape, 37.0)
    t_test[valid] = 47.0

    df_heat = compute_prf_shift(t_test, valid, fat, b0_t=3.0)
    print(f"3 T, +10 °C rise:")
    print(f"  Water pixel shift: {df_heat[valid & ~fat].mean():.3f} Hz (expected ~ -12.77 Hz)")
    print(f"  Fat pixel shift:   {df_heat[fat].max():.3f} Hz (expected 0.0 Hz)")
