import numpy as np

RHO_B, C_B, TA = 1050.0, 3617.0, 37.0      # blood density, blood heat capacity, body temperature


def solve(props, Q=None, t_end=60.0, dt=0.25, frame_dt=3.0, T0=None):
    """Solve the Pennes bioheat equation on the 2D grid.

    props    : dict with rho, c, k, perf, valid, pixel_size_m
    Q        : function(t) -> array of added heat in W/m3 (None = no heating)
    t_end    : total time in seconds
    dt       : wanted time step (reduced automatically if it would be unstable)
    frame_dt : save one frame every frame_dt seconds
    T0       : starting temperature map (default: 37 C everywhere)
    Returns (times, frames) with frames of shape (Nt, rows, cols) in degC.
    """
    rho, c, k, perf, valid = (props[n] for n in ("rho", "c", "k", "perf", "valid"))
    dx = float(props["pixel_size_m"])
    rc = np.where(valid, rho * c, 1050.0 * 3600.0)
    kf = np.where(valid, k, 0.5)

    # conductivity on the faces between neighbours (harmonic mean)
    kp = np.pad(kf, 1, mode="edge")

    def face(nb):
        return 2 * kf * nb / (kf + nb)

    ky_p, ky_m = face(kp[2:, 1:-1]), face(kp[:-2, 1:-1])
    kx_p, kx_m = face(kp[1:-1, 2:]), face(kp[1:-1, :-2])

    # stability limit of the simple (explicit) scheme
    lim_diff = np.min(dx**2 * rc / (ky_p + ky_m + kx_p + kx_m))
    p = perf * RHO_B * C_B / rc
    lim_perf = 1.0 / max(p.max(), 1e-12)
    dt = min(dt, 0.5 * lim_diff, 0.5 * lim_perf)
    steps_per_frame = int(np.ceil(frame_dt / dt))
    dt = frame_dt / steps_per_frame               # make frames fall exactly on multiples of dt

    T = np.full(rho.shape, TA) if T0 is None else T0.astype(float).copy()
    n_frames = int(round(t_end / frame_dt)) + 1
    frames = np.zeros((n_frames,) + T.shape)
    times = np.arange(n_frames) * frame_dt
    frames[0] = T
    t = 0.0
    for f in range(1, n_frames):
        for _ in range(steps_per_frame):
            Tp = np.pad(T, 1, mode="constant", constant_values=TA)
            lap = (ky_p * (Tp[2:, 1:-1] - T) + ky_m * (Tp[:-2, 1:-1] - T)
                   + kx_p * (Tp[1:-1, 2:] - T) + kx_m * (Tp[1:-1, :-2] - T)) / dx**2
            q = Q(t) if Q is not None else 0.0
            T = T + dt * (lap - perf * RHO_B * C_B * (T - TA) + q) / rc
            T[~valid] = TA                         # outside the brain: stays at body temperature
            t += dt
        frames[f] = T
    return times, frames


# ======================================================================
# self-tests
# ======================================================================
if __name__ == "__main__":
    # ---- test 1: no heating -> nothing should change ----
    print("--- test 1: no heating, real brain ---")
    props = dict(np.load("heat/thermal_props.npz"))
    times, fr = solve(props, Q=None, t_end=60, frame_dt=3)
    print(f"largest deviation from 37 C after 60 s : {np.abs(fr[-1] - TA).max():.2e} degC  (should be 0)")

    # a uniform fake medium to compare with exact formulas
    shape = (103, 86)
    dx = 0.002
    uni = dict(rho=np.full(shape, 1050.0), c=np.full(shape, 3600.0), k=np.full(shape, 0.5),
               perf=np.zeros(shape), valid=np.ones(shape, bool), pixel_size_m=dx)
    alpha = 0.5 / (1050 * 3600)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    r2 = ((yy - 50) ** 2 + (xx - 43) ** 2) * dx**2

    # ---- test 2: a hot Gaussian spot spreading by diffusion (exact answer known) ----
    print("\n--- test 2: diffusion of a Gaussian spot (no blood flow) ---")
    s0, A, t_end = 0.006, 10.0, 30.0
    T0 = TA + A * np.exp(-r2 / (2 * s0**2))
    times, fr = solve(uni, T0=T0, t_end=t_end, frame_dt=t_end)
    s2 = s0**2 + 2 * alpha * t_end
    exact = TA + A * (s0**2 / s2) * np.exp(-r2 / (2 * s2))
    peak_err = abs(fr[-1][50, 43] - exact[50, 43]) / (exact[50, 43] - TA) * 100
    print(f"peak after {t_end:.0f} s : solver {fr[-1][50, 43] - TA:.3f}   exact {exact[50, 43] - TA:.3f}  (error {peak_err:.2f} %)")
    print(f"largest error anywhere : {np.abs(fr[-1] - exact).max():.3f} degC")

    # ---- test 3: blood flow cooling a uniformly warm tissue ----
    print("\n--- test 3: cooling by blood flow only ---")
    uni["perf"] = np.full(shape, 0.01)
    T0 = np.full(shape, TA + 10.0)
    times, fr = solve(uni, T0=T0, t_end=60, frame_dt=60)
    rate = 0.01 * RHO_B * C_B / (1050 * 3600)
    exact = 10.0 * np.exp(-rate * 60)
    print(f"excess temperature after 60 s : solver {fr[-1][50, 43] - TA:.3f}   exact {exact:.3f}")