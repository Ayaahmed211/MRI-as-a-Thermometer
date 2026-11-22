# ==============================================================================
# KomaMRI Scanner Simulation Engine for PRF Thermometry (Member 3)
# ==============================================================================
# Simulates the PyPulseq spoiled GRE sequence using KomaMRI.jl.
#
# Reads:
#   - gre_sequence.seq (from PyPulseq)
#   - phantom.npz (virtual brain phantom)
#   - temp_true.npy, t.npy (ground-truth thermal maps from Block 2)
#
# Computes:
#   - True PRF shift on water spins only (excluding fat)
#   - Static B0, drift, and chemical shift
#   - Bloch simulation with KomaMRI.simulate(obj, seq, sys)
#   - Raw complex k-space (103 x 86) and reconstructed complex images
#
# Usage:
#   julia --project=. mri/simulate_koma.jl configs/baseline.yaml
# ==============================================================================

using KomaMRI
using NPZ
using HDF5
using YAML
using FFTW

function hz_per_degC(B0::Float64, alpha_ppm::Float64=-0.01)
    return alpha_ppm * 1e-6 * 42.58e6 * B0
end

function resolve_path(base_dir::String, p::String)
    if isfile(p) || isdir(p)
        return p
    end
    candidate = joinpath(base_dir, p)
    if isfile(candidate) || isdir(candidate)
        return candidate
    end
    return p
end

function run_koma_simulation(config_path::String)
    println("======================================================================")
    println("  Running KomaMRI.jl Scanner Simulation")
    println("======================================================================")
    
    script_dir = @__DIR__
    project_dir = dirname(script_dir)
    resolved_cfg = resolve_path(project_dir, config_path)
    println("Configuration: ", resolved_cfg)
    cfg = YAML.load_file(resolved_cfg)

    exp_name = string(cfg["name"])
    results_dir = joinpath(project_dir, "results", exp_name)
    mkpath(results_dir)

    seq_path = joinpath(results_dir, "gre_sequence.seq")
    if !isfile(seq_path)
        seq_path = joinpath(script_dir, "gre_sequence.seq")
    end
    if !isfile(seq_path)
        error("Sequence file not found at $(seq_path). Run gre_sequence.py first.")
    end

    println("Loading PyPulseq sequence: ", seq_path)
    seq = read_seq(seq_path)
    println("Sequence blocks: ", length(seq))

    # Load phantom
    phantom_file = resolve_path(project_dir, string(cfg["phantom"]))
    println("Loading phantom: ", phantom_file)
    ph = npzread(phantom_file)

    valid = ph["valid"]
    PD = ph["PD"]
    T1 = ph["T1"]
    T2 = ph["T2"]
    T2s = ph["T2s"]
    xs = ph["x_axis"]
    ys = ph["y_axis"]
    fat_mask = haskey(ph, "fat_mask") ? ph["fat_mask"] : falses(size(valid))
    dw_base = haskey(ph, "dw") ? ph["dw"] : zeros(Float64, size(valid))

    Ny, Nx = size(valid)
    valid_idx = findall(valid .== true)
    Nspins = length(valid_idx)
    println("Grid: $(Ny) x $(Nx) | Simulated spins: $(Nspins)")

    # Spin spatial coordinates
    X = [Float64(xs[ci[2]]) for ci in valid_idx]
    Y = [Float64(ys[ci[1]]) for ci in valid_idx]
    Z = zeros(Float64, Nspins)

    rho = [Float64(PD[ci]) for ci in valid_idx]
    t1_vals = [Float64(T1[ci]) for ci in valid_idx]
    t2_vals = [Float64(T2[ci]) for ci in valid_idx]
    t2s_vals = [Float64(T2s[ci]) for ci in valid_idx]

    # Load ground-truth temperature
    temp_file = joinpath(results_dir, "temp_true.npy")
    t_file = joinpath(results_dir, "t.npy")
    if !isfile(temp_file) || !isfile(t_file)
        error("Ground truth temperature files missing in $(results_dir).")
    end

    temp_true = npzread(temp_file) # shape: (N_frames, Ny, Nx)
    t_true = npzread(t_file)       # shape: (N_frames,)
    N_frames = length(t_true)
    println("Total dynamic frames: ", N_frames)

    B0 = Float64(get(cfg, "field_T", 3.0))
    scale_hz = hz_per_degC(B0)
    sys = Scanner()

    scan_cfg = get(cfg, "scan", Dict())
    motion_enabled = Bool(get(scan_cfg, "motion", false))
    motion_amp_mm = Float64(get(scan_cfg, "motion_amp_mm", 2.0))
    breathing_period_s = Float64(get(scan_cfg, "breathing_period_s", 4.0))

    b0_inhom = Bool(get(scan_cfg, "b0_inhomogeneity", false))
    drift_rate = Float64(get(scan_cfg, "b0_drift_hz_per_s", 0.0))

    # Static B0 map
    static_b0_file = joinpath(results_dir, "static_b0.npy")
    static_b0 = (b0_inhom && isfile(static_b0_file)) ? npzread(static_b0_file) : zeros(Float64, Ny, Nx)

    # Pre-allocate k-space and reconstructed images
    kspace_all = zeros(ComplexF64, N_frames, Ny, Nx)
    images_all = zeros(ComplexF64, N_frames, Ny, Nx)
    df_heat_all = zeros(Float64, N_frames, Ny, Nx)

    println("\nSimulating frames using KomaMRI.simulate...")

    for k in 1:N_frames
        t_k = t_true[k]
        
        # Frame temperature
        T_frame = (k == 1) ? fill(37.0, Ny, Nx) : temp_true[k, :, :]
        delta_T = T_frame .- 37.0

        # PRF frequency shift (water only, fat excluded)
        df_heat = zeros(Float64, Ny, Nx)
        for ci in valid_idx
            if !fat_mask[ci]
                df_heat[ci] = scale_hz * delta_T[ci]
            end
        end
        df_heat_all[k, :, :] = df_heat

        # Drift at t_k
        drift_hz = drift_rate * t_k

        # Total off-resonance Δw in rad/s: dw_base + 2π * (df_heat + static_b0 + drift)
        dw_spins = [Float64(dw_base[ci] + 2π * (df_heat[ci] + static_b0[ci] + drift_hz)) for ci in valid_idx]

        # Breathing displacement (Y-axis shift)
        dy_m = motion_enabled ? (motion_amp_mm * sin(2π * t_k / breathing_period_s) * 1e-3) : 0.0
        Y_k = Y .+ dy_m

        # Construct frame Phantom
        obj = Phantom(
            x = X,
            y = Y_k,
            z = Z,
            ρ = rho,
            T1 = t1_vals,
            T2 = t2_vals,
            T2s = t2s_vals,
            Δw = dw_spins
        )

        # Simulate with KomaMRI Bloch engine
        print("  Simulating frame $(k)/$(N_frames) (t = $(t_k) s)... ")
        t0 = time()
        raw = simulate(obj, seq, sys)
        t_sim = time() - t0
        println("Done in $(round(t_sim, digits=1)) s")

        # Extract Cartesian k-space matrix
        kspace_frame = zeros(ComplexF64, Ny, Nx)
        for i in 1:Ny
            kspace_frame[i, :] = raw.profiles[i].data[:, 1]
        end
        kspace_all[k, :, :] = kspace_frame

        # Image reconstruction
        img_frame = ifft(kspace_frame) .* sqrt(Ny * Nx)
        images_all[k, :, :] = img_frame
    end

    # Save to HDF5 (kspace.h5) with Python/C-order (N_frames, Ny, Nx)
    out_h5 = joinpath(results_dir, "kspace.h5")
    println("\nSaving simulated k-space to: ", out_h5)
    kspace_h5 = permutedims(kspace_all, (3, 2, 1))
    h5open(out_h5, "w") do file
        file["kspace"] = kspace_h5
        file["kspace_clean"] = kspace_h5
        file["t"] = t_true
        file["B0"] = B0
        file["TE"] = Float64(get(cfg["sequence"], "TE_ms", 20.0)) * 1e-3
        file["TR"] = Float64(get(cfg["sequence"], "TR_ms", 30.0)) * 1e-3
        file["flip_angle"] = Float64(get(cfg["sequence"], "flip_deg", 20.0))
        file["matrix"] = [Ny, Nx]
        file["fov"] = [Float64(get(cfg["sequence"], "fov_y_mm", 206.0)) * 1e-3, Float64(get(cfg["sequence"], "fov_x_mm", 172.0)) * 1e-3]
        attrs(file)["simulator"] = "KomaMRI.jl (Bloch)"
        attrs(file)["B0_T"] = B0
        attrs(file)["TE_s"] = Float64(get(cfg["sequence"], "TE_ms", 20.0)) * 1e-3
        attrs(file)["TR_s"] = Float64(get(cfg["sequence"], "TR_ms", 30.0)) * 1e-3
        attrs(file)["flip_deg"] = Float64(get(cfg["sequence"], "flip_deg", 20.0))
        attrs(file)["kspace_convention"] = "unshifted (DC at [0,0]), reconstructed via np.fft.ifft2(..., norm='ortho')"
    end

    # Save NumPy deliverables
    npzwrite(joinpath(results_dir, "mri_images.npy"), images_all)
    npzwrite(joinpath(results_dir, "mri_magnitude.npy"), abs.(images_all))
    npzwrite(joinpath(results_dir, "mri_phase.npy"), angle.(images_all))
    npzwrite(joinpath(results_dir, "frequency_shift.npy"), df_heat_all)

    println("All KomaMRI outputs saved successfully in: ", results_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    cfg_file = length(ARGS) >= 1 ? ARGS[1] : "configs/baseline.yaml"
    run_koma_simulation(cfg_file)
end
