"""ERA5 certified spectral reconstruction — H^1 rel-err analysis.

Same data as 18_era5_field.py (cached in 18_era5_raw.npz). This script
reports the metric that matches the synthetic OU experiments:

  rel_err_H1^2 = Σ_k (1+|k|^2) ∫|a_θ_k - a_emp_k|^2 + |b_θ_k - b_emp_k|^2 dt
                ───────────────────────────────────────────────────────
                Σ_k (1+|k|^2) ∫|a_emp_k|^2 + |b_emp_k|^2 dt

where a_emp_k(t_j) is the empirical Fourier mode trajectory (the "reference"
in this context), and a_θ_k(t) = c_0 + Σ_l α_{k,l} φ_l(t) is the
half-period reconstruction.

This isolates the certified bound (L-sweep) from the bandlimit
truncation (K_max choice).
"""
import os
import numpy as np
import importlib.util

ROOT = "/home/hubi/research/useful/papers/paper7_certified_spectral_pinn"
spec = importlib.util.spec_from_file_location("era5", os.path.join(ROOT, "src", "18_era5_field.py"))
era5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(era5)

RAW = os.path.join(ROOT, "results", "18_era5_raw.npz")
NPZ = os.path.join(ROOT, "results", "18b_era5_certified.npz")


def fit_traj(traj, t_norm, L):
    """LSQ fit of c_0 + Σ_{l=1}^L α_l φ_l(t)."""
    Phi = era5.halfperiod_basis(t_norm, L)
    A = np.concatenate([np.ones((len(t_norm), 1)), Phi], axis=1)
    coefs, *_ = np.linalg.lstsq(A, traj, rcond=None)
    pred = A @ coefs
    return coefs, pred


def H1_rel_err(a_emp, b_emp, a_fit, b_fit, modes, t_norm):
    """H^1-weighted relative error against empirical mode trajectories."""
    k2 = (modes ** 2).sum(axis=1)             # (J,)
    w = 1.0 + k2                              # (J,)
    dt = (t_norm[-1] - t_norm[0]) / (len(t_norm) - 1)
    err = ((a_emp - a_fit) ** 2 + (b_emp - b_fit) ** 2).sum(axis=0) * dt    # (J,)
    ref = (a_emp ** 2 + b_emp ** 2).sum(axis=0) * dt                        # (J,)
    return float(np.sqrt((w * err).sum() / (w * ref).sum()))


def run(K_max, T_field, t_norm):
    modes = era5.half_space_modes_2d(K_max)
    J = len(modes)
    a_emp, b_emp, T_mean_day, _, _ = era5.project_to_modes(T_field, modes)  # (n_days, J)

    out = []
    for L in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]:
        if L >= len(t_norm) - 1: break
        a_fit = np.zeros_like(a_emp); b_fit = np.zeros_like(b_emp)
        for j in range(J):
            _, a_fit[:, j] = fit_traj(a_emp[:, j], t_norm, L)
            _, b_fit[:, j] = fit_traj(b_emp[:, j], t_norm, L)
        rel = H1_rel_err(a_emp, b_emp, a_fit, b_fit, modes, t_norm)
        out.append((L, rel))
    return J, out


def main():
    z = np.load(RAW, allow_pickle=True)
    T_field = z["T_field"]
    n_days = T_field.shape[2]
    t_norm = np.linspace(0, 1, n_days)
    print(f"ERA5 data: {T_field.shape}, n_days={n_days}")
    print(f"Reporting H^1-weighted relative error against empirical mode trajectories.")
    print(f"(Apples-to-apples with the synthetic OU L-sweep table.)\n")

    results = {}
    for K_max in [3, 4, 5, 7]:
        print(f"--- K_max = {K_max} ---")
        J, sweep = run(K_max, T_field, t_norm)
        print(f"  J = {J}")
        print(f"  {'L':>4s} {'rel_H1':>9s}")
        for L, rel in sweep:
            print(f"  {L:4d}  {100*rel:7.3f}%")
        # Slope check
        if len(sweep) >= 4:
            Ls = np.array([s[0] for s in sweep])
            rels = np.array([s[1] for s in sweep])
            slope, _ = np.polyfit(np.log(Ls), np.log(rels), 1)
            print(f"  log-log slope = {slope:+.3f}")
        results[K_max] = sweep
        print()

    np.savez(NPZ,
             K_maxes=list(results.keys()),
             sweeps={k: np.array(v) for k, v in results.items()},
             T_field=T_field)
    print(f"saved: {NPZ}")


if __name__ == "__main__":
    main()
