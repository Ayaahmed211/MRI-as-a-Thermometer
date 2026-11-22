"""Sequence-Based MRI Scanner Simulator and Pipeline Driver (Member 3).

Integrates:
- PyPulseq spoiled GRE sequence design.
- True temperature map to PRF frequency shift conversion.
- Static B0 inhomogeneity, time-dependent drift, and breathing motion.
- Discrete isochromat/Bloch signal acquisition modeling Ernst steady-state,
  T2* decay, and off-resonance phase evolution.
- Realistic complex k-space formation and post-simulation Gaussian noise.
- Output serialization to HDF5 (kspace.h5), NumPy arrays, and preview plots.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from phantom.scan_conditions import (
    add_noise,
    apply_motion,
    b0_drift,
    b0_map,
    breathing_shift,
    shift_map,
)
from mri.frequency_shift import compute_total_off_resonance
from mri.gre_sequence import design_gre_sequence


def simulate_frame_signal(
    phantom_dict: Dict[str, Any],
    df_total_hz: np.ndarray,
    seq_meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the MRI signal acquisition for a single frame using the designed GRE sequence.

    Signal physics:
    - Ernst steady-state transverse magnetization for spoiled GRE:
        M_xy = PD * (sin(alpha) * (1 - E1)) / (1 - cos(alpha) * E1)
      where E1 = exp(-TR / T1).
    - Echo time evolution:
        S(TE) = M_xy * exp(-TE / T2*) * exp(1j * 2 * pi * df_total * TE)
    - 2D Fourier spatial encoding corresponding to the Cartesian readout trajectory:
        kspace = fft2(S, norm="ortho")
    - Reconstructed complex image:
        img = ifft2(kspace, norm="ortho")

    Parameters
    ----------
    phantom_dict : dict
        Tissue parameter maps ('PD', 'T1', 'T2s', 'valid').
    df_total_hz : np.ndarray
        Total off-resonance map across the slice in Hz.
    seq_meta : dict
        Sequence acquisition parameters containing 'TE_s', 'TR_s', 'flip_rad'.

    Returns
    -------
    kspace : np.ndarray
        Clean complex k-space matrix (shape: Ny, Nx).
    img : np.ndarray
        Clean reconstructed complex image (shape: Ny, Nx).
    """
    valid = np.asarray(phantom_dict["valid"], dtype=bool)
    pd = np.asarray(phantom_dict["PD"], dtype=float)
    t1 = np.maximum(np.asarray(phantom_dict["T1"], dtype=float), 1e-4)
    t2s = np.maximum(np.asarray(phantom_dict["T2s"], dtype=float), 1e-4)

    te_s = float(seq_meta["TE_s"])
    tr_s = float(seq_meta["TR_s"])
    flip_rad = float(seq_meta["flip_rad"])

    # Ernst steady-state longitudinal magnetization tipping
    e1 = np.exp(-tr_s / t1)
    sin_a = np.sin(flip_rad)
    cos_a = np.cos(flip_rad)

    # Ernst steady-state transverse magnetization
    denom = 1.0 - cos_a * e1
    m_ss = np.where(valid, pd * (sin_a * (1.0 - e1)) / np.maximum(denom, 1e-6), 0.0)

    # Transverse relaxation T2* decay and phase accumulation over TE
    t2s_decay = np.exp(-te_s / t2s)
    phase_acc = np.exp(1j * (2.0 * np.pi * df_total_hz * te_s))

    # Complex transverse magnetization at echo time TE
    s_te = np.where(valid, m_ss * t2s_decay * phase_acc, 0.0)

    # Spatial Fourier encoding: scanner k-space trajectory
    kspace = np.fft.fft2(s_te, norm="ortho")
    img = np.fft.ifft2(kspace, norm="ortho")

    return kspace, img


def run_mri_simulation(
    cfg: Dict[str, Any],
    out_dir: Optional[str | Path] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """Execute the complete Stage 3 MRI sequence and scanner simulation.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary.
    out_dir : str or Path, optional
        Output folder (defaults to results/<cfg['name']>).
    show : bool
        Whether to display preview figures interactively.

    Returns
    -------
    dict
        Dictionary containing simulated k-space, reconstructed images, and metadata.
    """
    if out_dir is None:
        out_dir = Path("results") / cfg["name"]
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Virtual Patient Phantom
    phantom_path = Path(cfg["phantom"])
    if not phantom_path.exists():
        raise FileNotFoundError(f"Phantom file not found: {phantom_path}")
    phantom_base = np.load(phantom_path)

    valid = phantom_base["valid"]
    pixel_size_m = float(phantom_base["pixel_size_m"])
    tumor_mask = phantom_base["tumor_mask"]

    # 2. Load Ground-Truth Temperature and Timestamps from Block 2
    temp_true_path = out_dir / "temp_true.npy"
    t_true_path = out_dir / "t.npy"
    if not temp_true_path.exists() or not t_true_path.exists():
        raise FileNotFoundError(
            f"Ground-truth temperature files missing in {out_dir}. "
            "Please run Block 2 (heat model) first."
        )

    temp_true = np.load(temp_true_path)  # Shape: (N_frames, Ny, Nx)
    t_true = np.load(t_true_path)        # Shape: (N_frames,)
    num_frames = len(t_true)

    # 3. Design Spoiled GRE Sequence using PyPulseq
    seq_path_results = out_dir / "gre_sequence.seq"
    seq_path_mri = Path("mri/gre_sequence.seq")

    seq, seq_meta = design_gre_sequence(cfg, output_path=seq_path_results)
    # Also save a copy inside mri/ as specified
    try:
        seq_path_mri.parent.mkdir(parents=True, exist_ok=True)
        seq.write(str(seq_path_mri))
    except Exception:
        pass

    tr_s = seq_meta["TR_s"]
    te_s = seq_meta["TE_s"]
    matrix_y = seq_meta["matrix_y"]
    matrix_x = seq_meta["matrix_x"]
    b0_t = seq_meta["B0_T"]
    frame_duration_s = seq_meta["frame_duration_s"]

    print(f"  MRI sequence: {seq_meta['sequence_type']} | B0 = {b0_t:.1f} T")
    print(f"  TE = {seq_meta['TE_ms']:.2f} ms | TR = {seq_meta['TR_ms']:.2f} ms | Flip = {seq_meta['flip_deg']:.1f}°")
    print(f"  Matrix = {matrix_y} x {matrix_x} | Single frame duration = {frame_duration_s:.2f} s")

    # 4. Initialize Static B0 Inhomogeneity Map (Consistent across all frames)
    scan_cfg = cfg.get("scan", {})
    seed = int(cfg.get("seed", 0))

    if scan_cfg.get("b0_inhomogeneity", True):
        amp_hz = float(scan_cfg.get("b0_amp_hz", 20.0))
        static_b0 = b0_map(valid, pixel_size_m=pixel_size_m, amp_hz=amp_hz, seed=seed)
    else:
        static_b0 = np.zeros_like(valid, dtype=float)

    # Save static B0 field map so both Python and KomaMRI simulations share it
    np.save(out_dir / "static_b0.npy", static_b0)

    # 5. Timing & Temperature Interpolation Model
    # Nominal frame timestamps during acquisition
    # For baseline (frame 0), temperature is strictly 37.0 °C (unheated).
    # For subsequent frames, temperature is evaluated at each frame's acquisition center time.
    interp_temp = interpolate.interp1d(
        t_true,
        temp_true,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(temp_true[0], temp_true[-1]),
    )

    kspace_clean = np.zeros((num_frames, matrix_y, matrix_x), dtype=complex)
    images_clean = np.zeros((num_frames, matrix_y, matrix_x), dtype=complex)
    df_heat_all = np.zeros((num_frames, matrix_y, matrix_x), dtype=float)
    df_total_all = np.zeros((num_frames, matrix_y, matrix_x), dtype=float)

    motion_enabled = bool(scan_cfg.get("motion", False))
    motion_amp_mm = float(scan_cfg.get("motion_amp_mm", 2.0))
    breathing_period_s = float(scan_cfg.get("breathing_period_s", 4.0))

    print(f"  Simulating {num_frames} acquisition frames (1 baseline + {num_frames - 1} dynamic)...")

    for k in range(num_frames):
        t_k = float(t_true[k])

        # Temperature map for frame k
        if k == 0:
            # Baseline acquisition at 37 °C before heating
            t_frame = np.full((matrix_y, matrix_x), 37.0, dtype=float)
        else:
            # Evaluate ground truth at frame acquisition center time
            t_acq_center = (k - 1) * frame_duration_s + (matrix_y / 2.0) * tr_s
            t_eval = min(float(t_true[-1]), max(0.0, t_acq_center))
            t_frame = interp_temp(t_eval)

        # Apply breathing displacement if enabled
        if motion_enabled:
            dy_mm = float(breathing_shift(t_k, amp_mm=motion_amp_mm, period_s=breathing_period_s))
            phantom_current = apply_motion(phantom_base, dy_mm=dy_mm, dx_mm=0.0)
            t_frame_current, _ = shift_map(t_frame, valid, dy_mm=dy_mm, dx_mm=0.0, pixel_size_m=pixel_size_m)
            b0_current, _ = shift_map(static_b0, valid, dy_mm=dy_mm, dx_mm=0.0, pixel_size_m=pixel_size_m)
        else:
            phantom_current = phantom_base
            t_frame_current = t_frame
            b0_current = static_b0

        # Compute composite scanner off-resonance map
        df_total, df_heat, _, _ = compute_total_off_resonance(
            phantom=phantom_current,
            temp_map=t_frame_current,
            t_s=t_k,
            cfg=cfg,
            static_b0=b0_current,
        )

        # Simulate scanner signal acquisition
        k_clean, img_clean = simulate_frame_signal(
            phantom_dict=phantom_current,
            df_total_hz=df_total,
            seq_meta=seq_meta,
        )

        kspace_clean[k] = k_clean
        images_clean[k] = img_clean
        df_heat_all[k] = df_heat
        df_total_all[k] = df_total

    # 6. Noise Addition in K-Space (After Simulation)
    snr_req = scan_cfg.get("snr", None)
    if snr_req is not None and snr_req > 0:
        # Reference white matter region for SNR calibration
        wm_label = 0
        if "label" in phantom_base:
            lab = phantom_base["label"]
            valid_labels = lab[valid]
            wm_label = int(np.bincount(valid_labels[valid_labels >= 0]).argmax())
        ref_mask = (phantom_base["label"] == wm_label) & valid & (~phantom_base.get("fat_mask", np.zeros_like(valid)))

        kspace_noisy = add_noise(kspace_clean, snr=float(snr_req), seed=seed, ref_mask=ref_mask)
        mri_images = np.fft.ifft2(kspace_noisy, axes=(-2, -1), norm="ortho")
    else:
        kspace_noisy = kspace_clean.copy()
        mri_images = images_clean.copy()

    mri_magnitude = np.abs(mri_images)
    mri_phase = np.angle(mri_images)

    # 7. Save All Deliverables
    # 7A. kspace.h5 (HDF5 format with complete acquisition metadata)
    h5_path = out_dir / "kspace.h5"
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("kspace", data=kspace_noisy, compression="gzip")
        h5f.create_dataset("kspace_clean", data=kspace_clean, compression="gzip")
        h5f.create_dataset("t", data=t_true)

        # Datasets & attributes for easy loading by Member 4
        h5f.create_dataset("TE", data=te_s)
        h5f.create_dataset("TR", data=tr_s)
        h5f.create_dataset("flip_angle", data=seq_meta["flip_deg"])
        h5f.create_dataset("B0", data=b0_t)
        h5f.create_dataset("matrix", data=np.array([matrix_y, matrix_x], dtype=int))
        h5f.create_dataset("fov", data=np.array(seq_meta["fov_m"], dtype=float))

        # Metadata attributes
        h5f.attrs["simulator"] = "Python (analytic Bloch)"
        h5f.attrs["sequence"] = seq_meta["sequence_type"]
        h5f.attrs["TE_s"] = te_s
        h5f.attrs["TR_s"] = tr_s
        h5f.attrs["TE_ms"] = seq_meta["TE_ms"]
        h5f.attrs["TR_ms"] = seq_meta["TR_ms"]
        h5f.attrs["flip_deg"] = seq_meta["flip_deg"]
        h5f.attrs["B0_T"] = b0_t
        h5f.attrs["matrix_y"] = matrix_y
        h5f.attrs["matrix_x"] = matrix_x
        h5f.attrs["pixel_size_m"] = pixel_size_m
        h5f.attrs["slice_thickness_m"] = seq_meta["slice_thickness_m"]
        h5f.attrs["snr"] = float(snr_req) if snr_req else -1.0
        h5f.attrs["seed"] = seed
        h5f.attrs["kspace_convention"] = "unshifted (DC at [0,0]), reconstructed via np.fft.ifft2(..., norm='ortho')"

    # 7B. NumPy arrays
    np.save(out_dir / "mri_images.npy", mri_images)
    np.save(out_dir / "mri_magnitude.npy", mri_magnitude)
    np.save(out_dir / "mri_phase.npy", mri_phase)
    np.save(out_dir / "frequency_shift.npy", df_heat_all)

    print(f"  Saved outputs to {out_dir}:")
    print(f"    - kspace.h5 ({h5_path.stat().st_size / 1024:.1f} KB)")
    print(f"    - mri_images.npy (shape: {mri_images.shape}, complex)")
    print(f"    - mri_magnitude.npy (shape: {mri_magnitude.shape})")
    print(f"    - mri_phase.npy (shape: {mri_phase.shape})")
    print(f"    - frequency_shift.npy (shape: {df_heat_all.shape})")
    print(f"    - gre_sequence.seq")

    # 7C. Diagnostic Visualization (mri_preview.png)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Frame 0: Baseline (t = 0 s)
    im0 = axes[0, 0].imshow(np.where(valid, mri_magnitude[0], np.nan), origin="lower", cmap="gray")
    axes[0, 0].contour(tumor_mask, levels=[0.5], colors="r", linewidths=1.2)
    axes[0, 0].set_title(f"Baseline Magnitude (t = {t_true[0]:.0f} s)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.where(valid, mri_phase[0], np.nan), origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0, 1].contour(tumor_mask, levels=[0.5], colors="r", linewidths=1.2)
    axes[0, 1].set_title("Baseline Phase (rad)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Frame at end of heating (t ~ 60 s)
    i_heat = int(np.argmin(np.abs(t_true - cfg.get("heating", {}).get("duration_s", 60.0))))
    im2 = axes[0, 2].imshow(np.where(valid, mri_magnitude[i_heat], np.nan), origin="lower", cmap="gray")
    axes[0, 2].contour(tumor_mask, levels=[0.5], colors="r", linewidths=1.2)
    axes[0, 2].set_title(f"Peak Heating Magnitude (t = {t_true[i_heat]:.0f} s)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Phase difference map: arg(S_heat * conj(S_baseline))
    dphi_heat = np.angle(mri_images[i_heat] * np.conj(mri_images[0]))
    im3 = axes[1, 0].imshow(np.where(valid, dphi_heat, np.nan), origin="lower", cmap="coolwarm")
    axes[1, 0].contour(tumor_mask, levels=[0.5], colors="k", linewidths=1.2)
    axes[1, 0].set_title(f"Phase Difference Δφ (t = {t_true[i_heat]:.0f} s)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Heat-induced frequency shift map (Hz)
    im4 = axes[1, 1].imshow(np.where(valid, df_heat_all[i_heat], np.nan), origin="lower", cmap="inferno")
    axes[1, 1].contour(tumor_mask, levels=[0.5], colors="c", linewidths=1.2)
    axes[1, 1].set_title(f"PRF Frequency Shift Δf (Hz)")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Time series at tumor center
    ci, cj = phantom_base["tumor_center_idx"]
    dphi_series = np.angle(mri_images[:, ci, cj] * np.conj(mri_images[0, ci, cj]))
    temp_series = temp_true[:, ci, cj]

    ax_time = axes[1, 2]
    line1 = ax_time.plot(t_true, temp_series, "r-", label="True Temp (°C)")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("True Temperature (°C)", color="r")
    ax_time.tick_params(axis="y", labelcolor="r")

    ax_phi = ax_time.twinx()
    line2 = ax_phi.plot(t_true, dphi_series, "b--", label="Phase Diff Δφ (rad)")
    ax_phi.set_ylabel("Measured Phase Diff (rad)", color="b")
    ax_phi.tick_params(axis="y", labelcolor="b")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_time.legend(lines, labels, loc="upper right")
    ax_time.set_title("Tumor Center: Temp & Phase vs Time")

    plt.tight_layout()
    preview_path = out_dir / "mri_preview.png"
    plt.savefig(preview_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    print(f"    - mri_preview.png saved")

    return {
        "kspace": kspace_noisy,
        "kspace_clean": kspace_clean,
        "mri_images": mri_images,
        "mri_magnitude": mri_magnitude,
        "mri_phase": mri_phase,
        "frequency_shift": df_heat_all,
        "t": t_true,
        "meta": seq_meta,
    }


def find_julia_executable() -> Optional[str]:
    """Find local Julia binary path if installed."""
    import shutil
    cmd = shutil.which("julia")
    if cmd:
        return cmd
    candidates = [
        r"C:\Users\suhila\AppData\Local\Programs\julia-1.10.8\bin\julia.exe",
        str(Path.home() / "AppData/Local/Programs/julia-1.10.8/bin/julia.exe"),
    ]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c).resolve())
    return None


def run_mri_stage(cfg: Dict[str, Any], out: Path) -> None:
    """Pipeline Stage 3 entry point."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Pre-generate sequence and static B0 map in results directory
    seq_path_results = out / "gre_sequence.seq"
    seq, seq_meta = design_gre_sequence(cfg, output_path=seq_path_results)

    phantom_path = Path(cfg["phantom"])
    if phantom_path.exists():
        phantom_base = np.load(phantom_path)
        valid = phantom_base["valid"]
        pixel_size_m = float(phantom_base["pixel_size_m"])
        scan_cfg = cfg.get("scan", {})
        seed = int(cfg.get("seed", 0))
        if scan_cfg.get("b0_inhomogeneity", True):
            amp_hz = float(scan_cfg.get("b0_amp_hz", 20.0))
            static_b0 = b0_map(valid, pixel_size_m=pixel_size_m, amp_hz=amp_hz, seed=seed)
        else:
            static_b0 = np.zeros_like(valid, dtype=float)
        np.save(out / "static_b0.npy", static_b0)

    preferred_sim = str(cfg.get("simulator", "koma")).lower()
    julia_bin = find_julia_executable() if preferred_sim != "python" else None

    if julia_bin is not None:
        print(f"  Stage 3: Running KomaMRI.jl Bloch simulator via Julia ({julia_bin})...")
        import subprocess
        import yaml

        cfg_temp = out / "current_run_cfg.yaml"
        with open(cfg_temp, "w") as f:
            yaml.dump(cfg, f)

        script_koma = Path(__file__).resolve().parent / "simulate_koma.jl"
        try:
            res = subprocess.run([julia_bin, str(script_koma), str(cfg_temp)], capture_output=True, text=True)
            if cfg_temp.exists():
                cfg_temp.unlink()

            if res.returncode == 0:
                print("  Stage 3: KomaMRI simulation finished successfully.")
                h5_path = out / "kspace.h5"
                if h5_path.exists():
                    with h5py.File(h5_path, "r") as h5f:
                        kspace_clean = np.array(h5f["kspace_clean"])
                        t_true = np.array(h5f["t"])

                    snr_req = cfg.get("scan", {}).get("snr", None)
                    if snr_req is not None and float(snr_req) > 0:
                        wm_label = 0
                        if "label" in phantom_base:
                            lab = phantom_base["label"]
                            valid_labels = lab[valid]
                            wm_label = int(np.bincount(valid_labels[valid_labels >= 0]).argmax())
                        ref_mask = (phantom_base["label"] == wm_label) & valid & (~phantom_base.get("fat_mask", np.zeros_like(valid)))
                        kspace_noisy = add_noise(kspace_clean, snr=float(snr_req), seed=seed, ref_mask=ref_mask)
                    else:
                        kspace_noisy = kspace_clean.copy()

                    mri_images = np.fft.ifft2(kspace_noisy, axes=(-2, -1), norm="ortho")
                    mri_mag = np.abs(mri_images)
                    mri_phase = np.angle(mri_images)

                    with h5py.File(h5_path, "a") as h5f:
                        if "kspace" in h5f:
                            del h5f["kspace"]
                        h5f.create_dataset("kspace", data=kspace_noisy, compression="gzip")
                        h5f.attrs["simulator"] = "KomaMRI.jl (Bloch)"
                        h5f.attrs["snr"] = float(snr_req) if snr_req else -1.0

                    np.save(out / "mri_images.npy", mri_images)
                    np.save(out / "mri_magnitude.npy", mri_mag)
                    np.save(out / "mri_phase.npy", mri_phase)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(mri_mag[0], cmap="gray", origin="lower")
                    axes[0].set_title(f"Baseline Mag (t={t_true[0]:.0f}s)")
                    axes[1].imshow(mri_phase[0], cmap="twilight", origin="lower", vmin=-np.pi, vmax=np.pi)
                    axes[1].set_title("Baseline Phase")
                    i_mid = min(len(t_true)-1, 15)
                    dphi = np.angle(mri_images[i_mid] * np.conj(mri_images[0]))
                    axes[2].imshow(dphi, cmap="coolwarm", origin="lower")
                    axes[2].set_title(f"Phase Diff (t={t_true[i_mid]:.0f}s)")
                    plt.tight_layout()
                    plt.savefig(out / "mri_preview.png", dpi=120)
                    plt.close(fig)
                    return
            else:
                print("  [Warning] KomaMRI simulation failed. Falling back to Python simulation.")
                if res.stderr:
                    print("  Julia stderr:", res.stderr[:300])
        except Exception as e:
            print("  [Warning] Julia execution exception:", e)

    # Fallback to Python simulation
    print("  Stage 3: Running Python Ernst/Bloch scanner simulation...")
    run_mri_simulation(cfg, out_dir=out)


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Run Block 3: MRI Sequence & Scanner Simulation")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to experiment YAML configuration")
    parser.add_argument("--show", action="store_true", help="Display diagnostic preview window")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out = Path("results") / cfg["name"]
    run_mri_simulation(cfg, out_dir=out, show=args.show)
