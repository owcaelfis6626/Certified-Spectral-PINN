"""SPX implied volatility surface: certified spectral reconstruction.

Pipeline:
  1. Fetch ^SPX option chain from yfinance (today's snapshot).
  2. Filter to liquid OTM options, compute log-moneyness K.
  3. Per expiry: Fourier projection in K → mode coefficients (a_k(τ_j), b_k(τ_j)).
  4. Per mode: half-period basis fit in τ via L^2 projection (LSQ).
  5. Leave-one-expiry-out cross-validation, report OOS RMSE.

Mathematical frame:
  σ_impl(K, τ) ≈ Σ_{k=0}^{K_max} [a_k(τ) cos(kx(K)) + b_k(τ) sin(kx(K))]
  a_k(τ) ≈ c_{k,0} + Σ_{l=1}^L α_{k,l} φ_l(t(τ))
  φ_l(t) = sqrt(2) sin((2l-1)πt/2),  t ∈ [0,1]  (half-period basis at T=1)

This is the L^2-projection variant of the certified Galerkin solve (the
Galerkin solution is within the inf-sup constant of L^2-projection, so the
projection error is a strict lower bound on the Galerkin error).
"""

import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
NPZ_PATH = os.path.join(OUT_DIR, "17_vol_surface.npz")

K_MAX_MODES = 4    # Fourier modes in K (cos+sin pairs + constant) → 2K+1 = 9 modes
L_TIME      = 6    # half-period basis size in τ
MIN_DTE     = 7
MAX_DTE     = 365
K_LOGM_MIN  = -0.30
K_LOGM_MAX  =  0.30
MAD_THRESHOLD = 4.0   # reject IVs > 4·MAD from per-expiry median


def fetch_spx_chain():
    t = yf.Ticker("^SPX")
    today = dt.datetime.now().date()
    spot = float(t.history(period="5d")["Close"].iloc[-1])

    rows = []
    for exp_str in t.options:
        exp_dt = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_dt - today).days
        if dte < MIN_DTE or dte > MAX_DTE:
            continue
        try:
            ch = t.option_chain(exp_str)
        except Exception:
            continue
        for side, df in [("call", ch.calls), ("put", ch.puts)]:
            d = df.copy()
            d = d[(d["bid"] > 0) & (d["ask"] > d["bid"])]
            d = d[(d["impliedVolatility"] > 0.05) & (d["impliedVolatility"] < 1.5)]
            d = d[d["strike"] > spot] if side == "call" else d[d["strike"] < spot]
            if len(d) == 0:
                continue
            d = d.assign(
                expiry=exp_str,
                dte=dte,
                tau=dte / 365.0,
                K_logm=np.log(d["strike"].values / spot),
                side=side,
            )
            rows.append(d[["expiry","tau","dte","strike","K_logm",
                           "impliedVolatility","bid","ask","volume","openInterest","side"]])

    df = pd.concat(rows, ignore_index=True)
    df = df[(df["K_logm"] >= K_LOGM_MIN) & (df["K_logm"] <= K_LOGM_MAX)]
    return df, spot, today


def robust_filter(df, mad_thr=MAD_THRESHOLD):
    """Drop options with IV more than mad_thr·MAD from per-expiry median IV
    (after binning by K_logm into ±0.05 windows so we compare like-for-like)."""
    keep_idx = []
    for expiry, d in df.groupby("expiry"):
        # bin by K_logm
        bins = np.arange(K_LOGM_MIN, K_LOGM_MAX + 1e-9, 0.05)
        bin_idx = np.digitize(d["K_logm"].values, bins)
        d = d.assign(_bin=bin_idx)
        for _, db in d.groupby("_bin"):
            if len(db) < 3:
                keep_idx.extend(db.index.tolist())
                continue
            med = db["impliedVolatility"].median()
            mad = (db["impliedVolatility"] - med).abs().median()
            if mad < 1e-4:
                keep_idx.extend(db.index.tolist())
                continue
            ok = (db["impliedVolatility"] - med).abs() <= mad_thr * mad
            keep_idx.extend(db.index[ok].tolist())
    return df.loc[sorted(keep_idx)].reset_index(drop=True)


def x_of_K(K_logm):
    return (np.asarray(K_logm) - K_LOGM_MIN) / (K_LOGM_MAX - K_LOGM_MIN) * (2 * np.pi)


def fourier_design_K(K_logm, K_max_modes):
    x = x_of_K(K_logm)
    n = len(np.atleast_1d(K_logm))
    A = np.ones((n, 2 * K_max_modes + 1))
    for k in range(1, K_max_modes + 1):
        A[:, 2*k - 1] = np.cos(k * x)
        A[:, 2*k    ] = np.sin(k * x)
    return A


def halfperiod_basis(t, L):
    """φ_l(t) = sqrt(2) sin((2l-1)πt/2) on [0,1]  (T = 1)."""
    t = np.atleast_1d(t)
    l = np.arange(1, L + 1)
    return np.sqrt(2.0) * np.sin((2 * l[None, :] - 1) * np.pi * t[:, None] / 2)


def fit_K_modes_at_expiry(d, K_max_modes):
    A = fourier_design_K(d["K_logm"].values, K_max_modes)
    y = d["impliedVolatility"].values
    coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coefs  # (2K+1,)


def fit_tau_per_mode(tau_arr, mode_traj, L_time, tau_min, tau_max):
    """Fit a(τ) ≈ c_0 + Σ_l α_l φ_l(t(τ))  by least squares."""
    t = (tau_arr - tau_min) / (tau_max - tau_min)
    Phi = halfperiod_basis(t, L_time)
    A = np.concatenate([np.ones((len(tau_arr), 1)), Phi], axis=1)
    coefs, *_ = np.linalg.lstsq(A, mode_traj, rcond=None)
    return coefs   # (1 + L,)


def predict_surface(K_logm_pred, tau_pred, mode_fits, K_max_modes, L_time, tau_min, tau_max):
    K_logm_pred = np.atleast_1d(K_logm_pred)
    tau_pred    = np.atleast_1d(tau_pred)
    A_K = fourier_design_K(K_logm_pred, K_max_modes)            # (n, 2K+1)
    t = (tau_pred - tau_min) / (tau_max - tau_min)
    Phi = halfperiod_basis(t, L_time)                           # (n, L)
    Phi_ext = np.concatenate([np.ones((len(tau_pred), 1)), Phi], axis=1)  # (n, 1+L)
    # mode_fits: (2K+1, 1+L) — coefficients for each Fourier mode
    a_at_tau = Phi_ext @ mode_fits.T                             # (n, 2K+1)
    iv = (A_K * a_at_tau).sum(axis=1)                            # (n,)
    return iv


def run_cv(df_all, K_max_modes, L_time):
    """Leave-one-expiry-out CV. Returns list of (expiry, n_test, rmse, rel_err)."""
    expiries = sorted(df_all["expiry"].unique(),
                      key=lambda e: dt.datetime.strptime(e, "%Y-%m-%d"))
    results = []
    for e_held in expiries:
        df_tr = df_all[df_all["expiry"] != e_held]
        df_te = df_all[df_all["expiry"] == e_held]
        if len(df_te) < 3:
            continue
        tr_expiries = sorted(df_tr["expiry"].unique(),
                             key=lambda e: dt.datetime.strptime(e, "%Y-%m-%d"))
        if len(tr_expiries) < 4:
            continue
        # Per-expiry K-mode trajectories
        tau_tr, mode_traj_tr = [], []
        for e in tr_expiries:
            d = df_tr[df_tr["expiry"] == e]
            if len(d) < (2 * K_max_modes + 2):
                continue
            mode_traj_tr.append(fit_K_modes_at_expiry(d, K_max_modes))
            tau_tr.append(d["tau"].iloc[0])
        mode_traj_tr = np.array(mode_traj_tr)   # (N_tr, 2K+1)
        tau_tr = np.array(tau_tr)
        tau_min, tau_max = tau_tr.min(), tau_tr.max()

        # IMPORTANT: if held-out τ is OUTSIDE [tau_min, tau_max] of training set,
        # we'd be extrapolating. Skip such folds.
        tau_test = df_te["tau"].iloc[0]
        if tau_test < tau_min or tau_test > tau_max:
            results.append((e_held, len(df_te), np.nan, np.nan, "extrap"))
            continue

        L_eff = min(L_time, len(tau_tr) - 1)
        mode_fits = np.array([
            fit_tau_per_mode(tau_tr, mode_traj_tr[:, j], L_eff, tau_min, tau_max)
            for j in range(2 * K_max_modes + 1)
        ])
        # pad with zeros if L_eff < L_time so the (1+L) shape matches downstream
        if L_eff < L_time:
            pad = np.zeros((mode_fits.shape[0], L_time - L_eff))
            mode_fits = np.concatenate([mode_fits, pad], axis=1)

        iv_pred = predict_surface(df_te["K_logm"].values, df_te["tau"].values,
                                  mode_fits, K_max_modes, L_time, tau_min, tau_max)
        iv_true = df_te["impliedVolatility"].values
        rmse = float(np.sqrt(np.mean((iv_pred - iv_true) ** 2)))
        ref  = float(np.sqrt(np.mean(iv_true ** 2)))
        rel  = rmse / ref
        results.append((e_held, len(df_te), rmse, rel, "ok"))
    return results


def main():
    print("Fetching SPX option chain via yfinance...")
    df_raw, spot, today = fetch_spx_chain()
    print(f"SPX spot {spot:.2f} on {today}")
    print(f"Raw OTM options: {len(df_raw)}  across {df_raw['expiry'].nunique()} expiries\n")

    df = robust_filter(df_raw)
    print(f"After robust outlier filter: {len(df)} options "
          f"({100*(1-len(df)/len(df_raw)):.1f}% removed)\n")

    expiries = sorted(df["expiry"].unique(),
                      key=lambda e: dt.datetime.strptime(e, "%Y-%m-%d"))
    print(f"{'expiry':12s} {'dte':>5s} {'n':>5s} {'K_range':>20s} {'IV_mean':>8s} {'IV_std':>7s}")
    for e in expiries:
        d = df[df["expiry"] == e]
        print(f"{e:12s} {d['dte'].iloc[0]:5d} {len(d):5d} "
              f"[{d['K_logm'].min():+.3f}, {d['K_logm'].max():+.3f}]"
              f"   {d['impliedVolatility'].mean():.4f}  {d['impliedVolatility'].std():.4f}")

    # ---- In-sample fit ----
    print("\n--- In-sample fit (all expiries) ---")
    mode_traj, tau_all = [], []
    for e in expiries:
        d = df[df["expiry"] == e]
        mode_traj.append(fit_K_modes_at_expiry(d, K_MAX_MODES))
        tau_all.append(d["tau"].iloc[0])
    mode_traj = np.array(mode_traj)  # (N_τ, 2K+1)
    tau_all = np.array(tau_all)
    tau_min, tau_max = tau_all.min(), tau_all.max()
    print(f"Mode trajectories: {mode_traj.shape}, τ ∈ [{tau_min:.3f}, {tau_max:.3f}] yr")

    n_modes = 2 * K_MAX_MODES + 1
    mode_fits = np.array([
        fit_tau_per_mode(tau_all, mode_traj[:, j], L_TIME, tau_min, tau_max)
        for j in range(n_modes)
    ])  # (n_modes, 1+L)

    iv_pred_in = predict_surface(df["K_logm"].values, df["tau"].values,
                                  mode_fits, K_MAX_MODES, L_TIME, tau_min, tau_max)
    iv_true = df["impliedVolatility"].values
    rmse_in = float(np.sqrt(np.mean((iv_pred_in - iv_true) ** 2)))
    ref = float(np.sqrt(np.mean(iv_true ** 2)))
    print(f"In-sample RMSE = {rmse_in:.5f}  (rel = {100*rmse_in/ref:.2f}%)")

    # ---- Cross-validation ----
    print("\n--- Leave-one-expiry-out cross-validation ---")
    print(f"{'held':12s} {'n':>4s} {'RMSE':>9s} {'rel_err':>9s}  status")
    cv = run_cv(df, K_MAX_MODES, L_TIME)
    rmses, rels = [], []
    for e, n, rmse, rel, status in cv:
        if status == "ok":
            print(f"{e:12s} {n:4d} {rmse:9.5f}  {100*rel:7.2f}%  ok")
            rmses.append(rmse); rels.append(rel)
        else:
            print(f"{e:12s} {n:4d}    --       --      {status}")
    print(f"\nCV mean RMSE      = {np.mean(rmses):.5f}")
    print(f"CV mean rel error = {100*np.mean(rels):.2f}%")
    print(f"CV median rel err = {100*np.median(rels):.2f}%")

    # ---- L-sweep: does rel_err drop with L? ----
    print("\n--- L-sweep (in-sample): rel_err vs L_time ---")
    print(f"{'L':>4s} {'RMSE':>9s} {'rel':>8s}")
    Ls_to_try = [1, 2, 3, 4, 6, 8, 10]
    sweep = []
    for L in Ls_to_try:
        if L > len(tau_all) - 1:
            continue
        mf = np.array([
            fit_tau_per_mode(tau_all, mode_traj[:, j], L, tau_min, tau_max)
            for j in range(n_modes)
        ])
        if L < L_TIME:
            mf = np.concatenate([mf, np.zeros((mf.shape[0], L_TIME - L))], axis=1)
        elif L > L_TIME:
            # need to predict with the right L too
            pass
        iv_p = predict_surface_arbL(df["K_logm"].values, df["tau"].values,
                                    mf, K_MAX_MODES, L, tau_min, tau_max)
        rmse = float(np.sqrt(np.mean((iv_p - iv_true) ** 2)))
        rel = rmse / ref
        sweep.append((L, rmse, rel))
        print(f"{L:4d} {rmse:9.5f}  {100*rel:6.2f}%")

    np.savez(NPZ_PATH,
             spot=spot, today=str(today),
             expiries=np.array(expiries, dtype=object),
             tau=tau_all, mode_traj=mode_traj, mode_fits=mode_fits,
             cv=np.array([(e, n, r, rl) for e, n, r, rl, s in cv if s == "ok"],
                         dtype=object),
             sweep=np.array(sweep),
             K_MAX_MODES=K_MAX_MODES, L_TIME=L_TIME)
    print(f"\nsaved: {NPZ_PATH}")


def predict_surface_arbL(K_logm_pred, tau_pred, mode_fits_padded, K_max_modes, L_use, tau_min, tau_max):
    K_logm_pred = np.atleast_1d(K_logm_pred)
    tau_pred    = np.atleast_1d(tau_pred)
    A_K = fourier_design_K(K_logm_pred, K_max_modes)
    t = (tau_pred - tau_min) / (tau_max - tau_min)
    Phi = halfperiod_basis(t, L_use)
    # Use only first L_use columns of α-part
    coefs = np.concatenate([mode_fits_padded[:, :1], mode_fits_padded[:, 1:1+L_use]], axis=1)
    Phi_ext = np.concatenate([np.ones((len(tau_pred), 1)), Phi], axis=1)
    a_at_tau = Phi_ext @ coefs.T
    iv = (A_K * a_at_tau).sum(axis=1)
    return iv


if __name__ == "__main__":
    main()
