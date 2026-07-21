"""ERA5 2m temperature field: certified spectral reconstruction via Open-Meteo.

Pipeline:
  1. Fetch daily-mean 2m temperature for a regular lat/lon grid over a region.
  2. At each day, project onto 2D Fourier modes (cos/sin in x and y).
  3. Per mode, fit half-period basis in time via L^2 projection.
  4. L-sweep + leave-day-out cross-validation.

Spatial domain: a lat/lon box, mapped to [0, 2π]^2 with periodic BCs (the
mode set is the same lex half-space as the synthetic experiments).

Temporal: [0, T_days], mapped to [0, 1] for the half-period basis at T=1.
"""

import os
import time
import json
import numpy as np
import requests

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
NPZ_RAW  = os.path.join(OUT_DIR, "18_era5_raw.npz")
NPZ_FIT  = os.path.join(OUT_DIR, "18_era5_field.npz")

# --- Region / grid / time ---
LAT_MIN, LAT_MAX = 45.0, 55.0     # central Europe
LON_MIN, LON_MAX = 0.0,  10.0
N_GRID           = 16             # → 16x16 = 256 spatial points
START_DATE       = "2024-12-01"
END_DATE         = "2025-02-28"   # 90 days of winter

# --- Spectral params ---
K_MAX = 3        # half-space modes in 2D → J = 14
L_TIME = 6       # temporal half-period basis size

API_URL = "https://archive-api.open-meteo.com/v1/archive"
BATCH_SIZE = 64  # points per request (URL length limit)


# ---------- Data fetch ----------

def fetch_era5_grid():
    """Fetch daily mean temperature on a (N_GRID × N_GRID) lat/lon grid."""
    lats = np.linspace(LAT_MIN, LAT_MAX, N_GRID)
    lons = np.linspace(LON_MIN, LON_MAX, N_GRID)
    LATs, LONs = np.meshgrid(lats, lons, indexing='ij')

    all_lats = LATs.flatten()
    all_lons = LONs.flatten()
    N = len(all_lats)

    print(f"Fetching {N} grid points × {START_DATE} to {END_DATE}...")

    T_per_point = []   # list of (n_days,) arrays
    dates_ref = None
    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        lat_str = ",".join(f"{x:.4f}" for x in all_lats[start:end])
        lon_str = ",".join(f"{x:.4f}" for x in all_lons[start:end])
        for attempt in range(3):
            try:
                r = requests.get(API_URL, params={
                    "latitude": lat_str,
                    "longitude": lon_str,
                    "start_date": START_DATE,
                    "end_date":   END_DATE,
                    "daily":      "temperature_2m_mean",
                    "timezone":   "UTC",
                }, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                print(f"  batch {start}-{end} attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"failed batch {start}-{end}")

        responses = data if isinstance(data, list) else [data]
        for resp in responses:
            d = resp["daily"]
            T_per_point.append(np.asarray(d["temperature_2m_mean"], dtype=np.float64))
            if dates_ref is None:
                dates_ref = d["time"]
        print(f"  batch {start:4d}–{end:4d}: ok, {len(responses)} points")

    T_arr = np.array(T_per_point)         # (N_points, n_days)
    n_days = T_arr.shape[1]
    print(f"Fetched: {T_arr.shape}, dates [{dates_ref[0]}, {dates_ref[-1]}]")

    # NaN check (Open-Meteo can return null for missing days)
    n_nan = np.isnan(T_arr).sum()
    if n_nan > 0:
        print(f"WARN: {n_nan} NaN values; filling with column mean.")
        col_mean = np.nanmean(T_arr, axis=0)
        for j in range(T_arr.shape[1]):
            mask = np.isnan(T_arr[:, j])
            T_arr[mask, j] = col_mean[j]

    T_field = T_arr.reshape(N_GRID, N_GRID, n_days)   # (Nlat, Nlon, n_days)
    return T_field, np.asarray(dates_ref), lats, lons


# ---------- Spectral projection ----------

def half_space_modes_2d(K_max):
    """Lex half-space of integer modes in 2D, k ≠ 0. One per cos/sin pair."""
    modes = []
    for kx in range(-K_max, K_max + 1):
        for ky in range(-K_max, K_max + 1):
            if kx*kx + ky*ky == 0:                continue
            if kx*kx + ky*ky > K_max * K_max:     continue
            # lex half-space: kx > 0, or (kx==0 and ky > 0)
            if kx > 0 or (kx == 0 and ky > 0):
                modes.append((kx, ky))
    return np.array(modes, dtype=np.int64)


def project_to_modes(T_field, modes):
    """For each day, project T(x,y) onto cos/sin(k·X) for k in modes.
    Returns (n_days, J), (n_days, J) for cos- and sin-coefficients.
    """
    Nlat, Nlon, n_days = T_field.shape
    # map lat/lon to [0, 2π] × [0, 2π]
    x = np.linspace(0, 2*np.pi, Nlat, endpoint=False)
    y = np.linspace(0, 2*np.pi, Nlon, endpoint=False)
    Xg, Yg = np.meshgrid(x, y, indexing='ij')

    J = len(modes)
    cos_basis = np.zeros((Nlat, Nlon, J))
    sin_basis = np.zeros((Nlat, Nlon, J))
    for j, (kx, ky) in enumerate(modes):
        arg = kx * Xg + ky * Yg
        cos_basis[..., j] = np.cos(arg)
        sin_basis[..., j] = np.sin(arg)

    # Orthonormal Fourier projection: a_k = (2/(Nlat*Nlon)) Σ T(x) cos(k·x)
    norm = 2.0 / (Nlat * Nlon)
    a = norm * np.einsum('xyt,xyk->tk', T_field, cos_basis)   # (n_days, J)
    b = norm * np.einsum('xyt,xyk->tk', T_field, sin_basis)

    # Also extract zero-mode (mean per day) for full reconstruction
    T_mean_day = T_field.mean(axis=(0,1))   # (n_days,)
    return a, b, T_mean_day, cos_basis, sin_basis


def halfperiod_basis(t, L):
    """φ_l(t) = sqrt(2) sin((2l-1)πt/2) on [0,1]  (T = 1)."""
    t = np.atleast_1d(t)
    l = np.arange(1, L + 1)
    return np.sqrt(2.0) * np.sin((2 * l[None, :] - 1) * np.pi * t[:, None] / 2)


def fit_per_mode(traj, t_norm, L_time):
    """Fit a(t) ≈ c_0 + Σ_l α_l φ_l(t)  via least squares."""
    Phi = halfperiod_basis(t_norm, L_time)               # (n, L)
    A = np.concatenate([np.ones((len(t_norm), 1)), Phi], axis=1)  # (n, L+1)
    coefs, *_ = np.linalg.lstsq(A, traj, rcond=None)
    return coefs


def reconstruct(coefs_a, coefs_b, coefs_mean, t_norm, cos_basis, sin_basis):
    """Reconstruct T(x, t) from temporal-fit coefficients.
    coefs_a, coefs_b : (J, 1+L)    coefs_mean : (1+L,)
    Returns (Nlat, Nlon, n_t)."""
    L = coefs_a.shape[1] - 1
    Phi = halfperiod_basis(t_norm, L)                    # (n_t, L)
    Phi_ext = np.concatenate([np.ones((len(t_norm), 1)), Phi], axis=1)  # (n_t, 1+L)
    a_t = Phi_ext @ coefs_a.T          # (n_t, J)
    b_t = Phi_ext @ coefs_b.T          # (n_t, J)
    mean_t = Phi_ext @ coefs_mean      # (n_t,)
    field = np.einsum('xyk,tk->xyt', cos_basis, a_t) \
          + np.einsum('xyk,tk->xyt', sin_basis, b_t)
    field += mean_t[None, None, :]
    return field


# ---------- Main ----------

def main():
    if os.path.exists(NPZ_RAW):
        print(f"loading cached {NPZ_RAW}")
        z = np.load(NPZ_RAW, allow_pickle=True)
        T_field = z["T_field"]; dates = z["dates"]
        lats = z["lats"]; lons = z["lons"]
    else:
        T_field, dates, lats, lons = fetch_era5_grid()
        np.savez(NPZ_RAW, T_field=T_field, dates=dates, lats=lats, lons=lons)
        print(f"cached → {NPZ_RAW}")

    Nlat, Nlon, n_days = T_field.shape
    print(f"\nField shape: {T_field.shape}, T range [{T_field.min():.1f}, {T_field.max():.1f}] °C")
    print(f"Per-day mean range: [{T_field.mean((0,1)).min():.1f}, {T_field.mean((0,1)).max():.1f}] °C")

    # Spectral basis: 2D half-space modes
    modes = half_space_modes_2d(K_MAX)
    J = len(modes)
    print(f"Spectral mode set: K_max={K_MAX}, J={J}")
    print(f"Modes: {modes.tolist()}")

    # Project T onto modes at each day
    a_traj, b_traj, T_mean_day, cos_basis, sin_basis = project_to_modes(T_field, modes)
    print(f"Mode trajectories: a {a_traj.shape}, b {b_traj.shape}")

    # τ = day index normalised to [0, 1]
    t_norm = np.linspace(0.0, 1.0, n_days)

    # --- In-sample fit ---
    print(f"\n--- In-sample fit, L_TIME={L_TIME} ---")
    coefs_a = np.array([fit_per_mode(a_traj[:, j], t_norm, L_TIME) for j in range(J)])  # (J, 1+L)
    coefs_b = np.array([fit_per_mode(b_traj[:, j], t_norm, L_TIME) for j in range(J)])
    coefs_mean = fit_per_mode(T_mean_day, t_norm, L_TIME)

    T_fit = reconstruct(coefs_a, coefs_b, coefs_mean, t_norm, cos_basis, sin_basis)
    err = T_fit - T_field
    rmse_in = float(np.sqrt(np.mean(err**2)))
    ref     = float(np.sqrt(np.mean(T_field**2)))
    ref_var = float(np.sqrt(np.mean((T_field - T_field.mean())**2)))
    print(f"In-sample RMSE = {rmse_in:.3f} °C")
    print(f"  rel RMSE vs |T|         = {100*rmse_in/ref:.2f}%")
    print(f"  rel RMSE vs |T - mean|  = {100*rmse_in/ref_var:.2f}%   (the meaningful one)")

    # --- L-sweep ---
    print(f"\n--- L-sweep: rel_err vs L_time ---")
    print(f"{'L':>4s} {'RMSE_C':>8s} {'rel_var':>9s}")
    sweep = []
    for L in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        if L >= n_days:
            break
        ca = np.array([fit_per_mode(a_traj[:, j], t_norm, L) for j in range(J)])
        cb = np.array([fit_per_mode(b_traj[:, j], t_norm, L) for j in range(J)])
        cm = fit_per_mode(T_mean_day, t_norm, L)
        # quick reconstruction
        Phi = halfperiod_basis(t_norm, L)
        Phi_ext = np.concatenate([np.ones((n_days, 1)), Phi], axis=1)
        a_t = Phi_ext @ ca.T
        b_t = Phi_ext @ cb.T
        mean_t = Phi_ext @ cm
        T_fit_L = np.einsum('xyk,tk->xyt', cos_basis, a_t) \
                + np.einsum('xyk,tk->xyt', sin_basis, b_t) + mean_t[None, None, :]
        rmse = float(np.sqrt(np.mean((T_fit_L - T_field)**2)))
        rel  = rmse / ref_var
        sweep.append((L, rmse, rel))
        print(f"{L:4d} {rmse:8.3f}  {100*rel:7.2f}%")

    # --- Leave-day-out CV ---
    print(f"\n--- Leave-day-out cross-validation at L={L_TIME} ---")
    cv_errs = []
    n_folds = min(20, n_days - 2)   # sample subset
    fold_days = np.linspace(1, n_days-2, n_folds).astype(int)
    for d_held in fold_days:
        mask = np.ones(n_days, dtype=bool); mask[d_held] = False
        t_tr = t_norm[mask]
        a_tr = a_traj[mask, :]; b_tr = b_traj[mask, :]; T_mean_tr = T_mean_day[mask]
        ca = np.array([fit_per_mode(a_tr[:, j], t_tr, L_TIME) for j in range(J)])
        cb = np.array([fit_per_mode(b_tr[:, j], t_tr, L_TIME) for j in range(J)])
        cm = fit_per_mode(T_mean_tr, t_tr, L_TIME)
        # predict at the held-out day
        Phi_he = halfperiod_basis(np.array([t_norm[d_held]]), L_TIME)
        Phi_ext = np.concatenate([np.ones((1,1)), Phi_he], axis=1)
        a_pred = (Phi_ext @ ca.T).flatten()
        b_pred = (Phi_ext @ cb.T).flatten()
        m_pred = (Phi_ext @ cm).item()
        T_pred = (cos_basis * a_pred[None, None, :]).sum(-1) \
               + (sin_basis * b_pred[None, None, :]).sum(-1) + m_pred
        T_true = T_field[..., d_held]
        rmse = float(np.sqrt(np.mean((T_pred - T_true)**2)))
        rel  = rmse / float(np.sqrt(np.mean((T_true - T_true.mean())**2)))
        cv_errs.append((d_held, rmse, rel))
    print(f"{'day':>5s} {'RMSE_C':>8s} {'rel':>8s}")
    for d, r, rl in cv_errs:
        print(f"{d:5d} {r:8.3f}  {100*rl:6.2f}%")
    print(f"\nCV mean rel error = {100*np.mean([r[2] for r in cv_errs]):.2f}%")
    print(f"CV mean RMSE      = {np.mean([r[1] for r in cv_errs]):.3f} °C")

    np.savez(NPZ_FIT,
             T_field=T_field, dates=dates,
             a_traj=a_traj, b_traj=b_traj, T_mean_day=T_mean_day,
             coefs_a=coefs_a, coefs_b=coefs_b, coefs_mean=coefs_mean,
             modes=modes, lats=lats, lons=lons, t_norm=t_norm,
             sweep=np.array(sweep), cv=np.array([(d, r, rl) for d, r, rl in cv_errs]),
             K_MAX=K_MAX, L_TIME=L_TIME)
    print(f"\nsaved: {NPZ_FIT}")


if __name__ == "__main__":
    main()
