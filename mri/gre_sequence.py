"""MRI Sequence Design using PyPulseq (Member 3).

Designs a 2D Spoiled Gradient Echo (GRE) sequence with:
- RF excitation with slice selection and slice refocusing.
- Phase-encoding gradient table with rewinder gradients.
- Frequency-encoding readout gradient with prewinder lobe.
- Spoiler (crusher) gradients along slice direction.
- RF spoiling (quadratic phase cycling).
- Configurable TE, TR, flip angle, FOV, matrix size, and slice thickness.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pypulseq as pp


def create_system_limits(
    max_grad: float = 32.0,
    max_slew: float = 130.0,
    rf_dead_time: float = 100e-6,
    rf_ringdown_time: float = 20e-6,
    adc_dead_time: float = 20e-6,
) -> pp.Opts:
    """Create scanner hardware limits specification for PyPulseq.

    Parameters
    ----------
    max_grad : float
        Maximum gradient amplitude in mT/m.
    max_slew : float
        Maximum gradient slew rate in T/m/s.
    rf_dead_time : float
        RF dead time in seconds.
    rf_ringdown_time : float
        RF ringdown time in seconds.
    adc_dead_time : float
        ADC dead time in seconds.

    Returns
    -------
    pp.Opts
        Hardware limits object.
    """
    return pp.Opts(
        max_grad=max_grad,
        grad_unit="mT/m",
        max_slew=max_slew,
        slew_unit="T/m/s",
        rf_dead_time=rf_dead_time,
        rf_ringdown_time=rf_ringdown_time,
        adc_dead_time=adc_dead_time,
    )


def design_gre_sequence(
    cfg: Dict[str, Any],
    output_path: Optional[str | Path] = None,
    system: Optional[pp.Opts] = None,
) -> Tuple[pp.Sequence, Dict[str, Any]]:
    """Design a 2D Spoiled Gradient Echo (GRE) sequence according to configuration.

    Parameters
    ----------
    cfg : dict
        Experiment configuration dictionary containing 'sequence' and 'field_T'.
    output_path : str or Path, optional
        Path to save the generated .seq file.
    system : pp.Opts, optional
        Scanner hardware specifications. If None, default scanner limits are used.

    Returns
    -------
    seq : pp.Sequence
        PyPulseq sequence object.
    meta : dict
        Acquisition parameters and timing metadata.
    """
    if system is None:
        system = create_system_limits()

    seq_cfg = cfg.get("sequence", {})
    b0_t = float(cfg.get("field_T", 3.0))

    # Read parameters with project defaults
    te_ms = float(seq_cfg.get("TE_ms", 20.0))
    te_s = te_ms * 1e-3

    # Default TR: max(30 ms, TE + 10 ms)
    default_tr_ms = max(30.0, te_ms + 10.0)
    tr_ms = float(seq_cfg.get("TR_ms", default_tr_ms))
    tr_s = tr_ms * 1e-3

    flip_deg = float(seq_cfg.get("flip_deg", 15.0))
    flip_rad = np.deg2rad(flip_deg)

    matrix_y = int(seq_cfg.get("matrix_y", 103))  # Phase encoding lines (rows)
    matrix_x = int(seq_cfg.get("matrix_x", 86))   # Readout samples (columns)

    fov_y = float(seq_cfg.get("fov_y_mm", 206.0)) * 1e-3  # FOV along y in meters
    fov_x = float(seq_cfg.get("fov_x_mm", 172.0)) * 1e-3  # FOV along x in meters
    slice_thickness = float(seq_cfg.get("slice_thickness_mm", 5.0)) * 1e-3

    seq = pp.Sequence(system=system)

    # 1. RF Excitation Pulse with slice-selection gradient
    rf_duration = 2.0e-3
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=flip_rad,
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4.0,
        system=system,
        delay=system.rf_dead_time,
        return_gz=True,
    )

    # 2. Readout Gradient & ADC Event
    # Choose dwell time (e.g. 40 us) aligned with both ADC raster (100 ns) and grad raster (10 us)
    dwell = 40.0e-6
    flat_time = matrix_x * dwell
    delta_kx = 1.0 / fov_x
    k_width_x = matrix_x * delta_kx

    gx = pp.make_trapezoid(
        channel="x",
        flat_area=k_width_x,
        flat_time=flat_time,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=matrix_x,
        duration=gx.flat_time,
        delay=gx.rise_time,
        system=system,
    )
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2.0, system=system)

    # 3. Phase-Encoding Gradients
    delta_ky = 1.0 / fov_y
    phase_areas = (np.arange(matrix_y) - matrix_y / 2.0) * delta_ky
    max_phase_area = max(abs(float(phase_areas.min())), abs(float(phase_areas.max())))
    gy_pre_max = pp.make_trapezoid(channel="y", area=max_phase_area, system=system)

    # Duration of prephasing block (common to gz_reph, gx_pre, gy_pre)
    t_pre = max(
        pp.calc_duration(gz_reph),
        pp.calc_duration(gx_pre),
        pp.calc_duration(gy_pre_max),
    )
    # Align to gradient raster
    t_pre = np.ceil(t_pre / system.grad_raster_time) * system.grad_raster_time
    gx_pre.delay = t_pre - pp.calc_duration(gx_pre)
    gz_reph.delay = t_pre - pp.calc_duration(gz_reph)

    # 4. TE Delay Calculation
    rf_center, _ = pp.calc_rf_center(rf)
    rf_time_to_end = pp.calc_duration(rf) - rf_center
    echo_center = gx.rise_time + gx.flat_time / 2.0
    te_delay_time = te_s - rf_time_to_end - t_pre - echo_center
    te_delay_time = round(te_delay_time / system.grad_raster_time) * system.grad_raster_time

    if te_delay_time < -1e-6:
        min_te = rf_time_to_end + t_pre + echo_center
        raise ValueError(
            f"Requested TE ({te_ms:.2f} ms) is too short. Minimum achievable TE is {min_te * 1e3:.2f} ms."
        )
    te_delay = pp.make_delay(te_delay_time) if te_delay_time > 1e-6 else None

    # 5. Spoilers & Phase Rewinder
    gz_spoil = pp.make_trapezoid(channel="z", area=4.0 / slice_thickness, system=system)
    t_post = max(pp.calc_duration(gy_pre_max), pp.calc_duration(gz_spoil))
    t_post = np.ceil(t_post / system.grad_raster_time) * system.grad_raster_time

    # 6. TR Delay Calculation
    te_delay_dur = te_delay_time if te_delay else 0.0
    dur_per_tr_base = pp.calc_duration(rf) + t_pre + te_delay_dur + pp.calc_duration(gx) + t_post
    tr_delay_time = tr_s - dur_per_tr_base
    tr_delay_time = round(tr_delay_time / system.grad_raster_time) * system.grad_raster_time

    if tr_delay_time < -1e-6:
        min_tr = dur_per_tr_base
        raise ValueError(
            f"Requested TR ({tr_ms:.2f} ms) is too short for the sequence timing. Minimum TR is {min_tr * 1e3:.2f} ms."
        )
    tr_delay = pp.make_delay(tr_delay_time) if tr_delay_time > 1e-6 else None

    # 7. Assemble Sequence Blocks with RF Spoiling
    rf_phase = 0.0
    rf_inc = 0.0
    rf_spoiling_inc = 117.0  # degrees

    for i in range(matrix_y):
        # RF spoiling phase cycling
        phase_rad = np.deg2rad(rf_phase)
        rf.phase_offset = phase_rad
        adc.phase_offset = phase_rad

        # Block 1: RF excitation + slice selection
        seq.add_block(rf, gz)

        # Block 2: Phase encoding + slice rephasing + readout prewinder
        gy_pre = pp.make_trapezoid(channel="y", area=phase_areas[i], duration=t_pre, system=system)
        seq.add_block(gx_pre, gy_pre, gz_reph)

        # Block 3: Delay to reach TE (if needed)
        if te_delay:
            seq.add_block(te_delay)

        # Block 4: Readout gradient + ADC sampling
        seq.add_block(gx, adc)

        # Block 5: Phase rewinder + slice spoiler gradient
        gy_rew = pp.make_trapezoid(channel="y", area=-phase_areas[i], duration=t_post, system=system)
        gz_spoil_curr = pp.make_trapezoid(channel="z", area=4.0 / slice_thickness, duration=t_post, system=system)
        seq.add_block(gy_rew, gz_spoil_curr)

        # Block 6: Delay to reach TR (if needed)
        if tr_delay:
            seq.add_block(tr_delay)

        # Update RF spoiling phase
        rf_inc = (rf_inc + rf_spoiling_inc) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

    # 8. Check Sequence Timing and Constraints
    ok, error_report = seq.check_timing()
    if not ok:
        raise RuntimeError(f"PyPulseq sequence timing check failed: {error_report}")

    # 9. Save .seq File If Requested
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        seq.write(str(out_p))

    actual_tr_s = dur_per_tr_base + (tr_delay_time if tr_delay else 0.0)
    frame_duration_s = matrix_y * actual_tr_s

    meta = {
        "sequence_type": "Spoiled GRE",
        "B0_T": b0_t,
        "TE_s": te_s,
        "TE_ms": te_ms,
        "TR_s": actual_tr_s,
        "TR_ms": actual_tr_s * 1e3,
        "flip_deg": flip_deg,
        "flip_rad": flip_rad,
        "matrix_y": matrix_y,
        "matrix_x": matrix_x,
        "matrix": [matrix_y, matrix_x],
        "fov_y_m": fov_y,
        "fov_x_m": fov_x,
        "fov_m": [fov_y, fov_x],
        "pixel_size_y_m": fov_y / matrix_y,
        "pixel_size_x_m": fov_x / matrix_x,
        "slice_thickness_m": slice_thickness,
        "frame_duration_s": frame_duration_s,
        "num_blocks": len(seq.block_events),
        "adc_samples": matrix_x,
        "adc_dwell_s": dwell,
        "rf_spoiling_inc_deg": rf_spoiling_inc,
    }

    return seq, meta


if __name__ == "__main__":
    import yaml

    cfg_path = Path("configs/baseline.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"field_T": 3.0, "sequence": {"TE_ms": 20, "TR_ms": 30, "flip_deg": 15}}

    seq, meta = design_gre_sequence(cfg, output_path="mri/gre_sequence.seq")
    print(f"Sequence designed successfully: {meta['sequence_type']}")
    print(f"TE: {meta['TE_ms']:.2f} ms | TR: {meta['TR_ms']:.2f} ms | Flip: {meta['flip_deg']:.1f} deg")
    print(f"Matrix: {meta['matrix']} | FOV: {np.array(meta['fov_m']) * 1000} mm")
    print(f"Frame acquisition duration: {meta['frame_duration_s']:.2f} s")
    print("Saved sequence to mri/gre_sequence.seq")
