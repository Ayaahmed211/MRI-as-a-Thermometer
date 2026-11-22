using KomaMRI

# Resolve sequence path relative to this script
script_dir = @__DIR__
seq_path = joinpath(script_dir, "gre_sequence.seq")
if !isfile(seq_path)
    seq_path = joinpath(dirname(script_dir), "results", "baseline", "gre_sequence.seq")
end
if !isfile(seq_path)
    error("Sequence file not found at $(seq_path).")
end

seq = read_seq(seq_path)
sys = Scanner()

B0 = 3.0
TE = 0.020
dT = 10.0
alpha_ppm = -0.01
gamma_hz = 42.58e6
scale_hz = alpha_ppm * 1e-6 * gamma_hz * B0  # ≈ -1.2774 Hz/°C
df_heat = scale_hz * dT                       # ≈ -12.774 Hz
dphi_theory = 2π * df_heat * TE               # ≈ -1.605228 rad

# Define test spin
obj_base = Phantom(x=[0.0], y=[0.0], z=[0.0], ρ=[1.0], T1=[1.0], T2=[0.1], T2s=[0.06], Δw=[0.0])
obj_heat = Phantom(x=[0.0], y=[0.0], z=[0.0], ρ=[1.0], T1=[1.0], T2=[0.1], T2s=[0.06], Δw=[2π * df_heat])

raw_base = simulate(obj_base, seq, sys)
raw_heat = simulate(obj_heat, seq, sys)

# Check echo center of profile 1
s_base_1 = raw_base.profiles[1].data[43, 1]
s_heat_1 = raw_heat.profiles[1].data[43, 1]
dphi_koma = angle(s_heat_1 * conj(s_base_1))

err_rel = abs(abs(dphi_koma) - abs(dphi_theory)) / abs(dphi_theory)
pass_test = err_rel < 0.02

println("Theoretical dphi: ", round(dphi_theory, digits=6), " rad")
println("KomaMRI dphi:     ", round(dphi_koma, digits=6), " rad")
println("Relative error:   ", round(err_rel * 100, digits=4), "%")
println("RESULT: ", pass_test ? "PASS" : "FAIL")
