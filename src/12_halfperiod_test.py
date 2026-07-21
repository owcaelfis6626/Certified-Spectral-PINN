"""Test half-period sine basis: does the Galerkin error decay with L?

Half-period basis phi_l(t) = sqrt(2/T) sin((2l-1) pi t / (2T)) for l=1..L.
- phi_l(0) = 0 (preserves IC term in trial space)
- phi_l(T) = (-1)^(l-1) (FREE at terminal — captures terminal noise)

The Galerkin matrix M = <dt phi_l, phi_{l'}> is no longer antisymmetric
(it has diagonal entries 1/2 from the non-vanishing boundary at t=T).

Test:
  1. Solve (M^T + |k|^2 I) alpha = sigma I_k for several L.
  2. Compute ||u_th - u_ref||_Y / ||u_ref||_Y.
  3. Compare to the full-period Dirichlet sine result (plateau ~0.085).

If rel_err -> 0 as L -> infty, the half-period fix works.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle_halfperiod import WMContinuousOracleHalfPeriod


D, K_MAX = 2, 4
SIGMA, T = 1.0, 1.0
N_FINE = 8192
LS = [32, 64, 128, 256, 512]
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def solve_optimal_alpha(oracle):
    """Solve (M^T + |k|^2 I) alpha = sigma I_k per mode (column-vector form)."""
    M = oracle.M
    L = M.shape[0]
    I_eye = torch.eye(L, device=M.device, dtype=M.dtype)
    alpha = torch.zeros_like(oracle.I_a)
    beta  = torch.zeros_like(oracle.I_b)
    for j in range(oracle.J):
        A = M.T + oracle.k2[j] * I_eye
        alpha[j] = torch.linalg.solve(A, oracle.sigma * oracle.I_a[j])
        beta[j]  = torch.linalg.solve(A, oracle.sigma * oracle.I_b[j])
    return alpha, beta


@torch.no_grad()
def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=128, t_chunk=2):
    t_full = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    dt = oracle.T / (N_t_eval - 1)
    w = torch.ones(N_t_eval, device=oracle.device, dtype=oracle.dtype)
    w[0] = 0.5
    w[-1] = 0.5
    one_plus_k2 = (1.0 + oracle.k2)[None, :]

    err_Y_total = 0.0
    ref_Y_total = 0.0
    for c0 in range(0, N_t_eval, t_chunk):
        c1 = min(c0 + t_chunk, N_t_eval)
        t_eval = t_full[c0:c1]

        phase = oracle.omega_l[None, :] * t_eval[:, None]
        phi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)            # (N_t, L)
        ic_decay = torch.exp(-oracle.lambda_k[None, :] * t_eval[:, None])
        a_th = ic_decay * oracle.a0[None, :] + phi @ alpha.T
        b_th = ic_decay * oracle.b0[None, :] + phi @ beta.T
        a_ref, b_ref = oracle.a_ref(t_eval)
        e_a, e_b = a_th - a_ref, b_th - b_ref

        wc = w[c0:c1]
        err_Y_total += (wc * (one_plus_k2 * (e_a**2 + e_b**2)).sum(-1)).sum().item()
        ref_Y_total += (wc * (one_plus_k2 * (a_ref**2 + b_ref**2)).sum(-1)).sum().item()

        del a_th, b_th, a_ref, b_ref, e_a, e_b
        if oracle.device == "cuda":
            torch.cuda.empty_cache()

    return math.sqrt(err_Y_total / ref_Y_total)   # 2026-07-17 fix: was returning the SQUARED relative error


def main():
    print(f"P7 half-period sine basis test  d={D} K_max={K_MAX} N_fine={N_FINE}\n")
    print(f"{'L':>6} {'rel_err mean ± std':>22} {'wall':>8}")
    rows = []
    for L in LS:
        rel_errs = []
        t0 = time.time()
        for seed in SEEDS:
            oracle = WMContinuousOracleHalfPeriod(
                d=D, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L, N_fine=N_FINE,
                seed=42 + seed, device=DEVICE, dtype=DTYPE)
            alpha, beta = solve_optimal_alpha(oracle)
            rel_errs.append(rel_err_at_coeffs(alpha, beta, oracle))
            del oracle, alpha, beta
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        arr = np.array(rel_errs)
        print(f"{L:6d}  {arr.mean():.4f} ± {arr.std():.4f}  {time.time()-t0:7.1f}s",
              flush=True)
        rows.append((L, arr.mean(), arr.std()))

    print("\n=== Pairwise log-log rates ===")
    L_arr = np.array([r[0] for r in rows], dtype=float)
    re_arr = np.array([r[1] for r in rows])
    for i in range(1, len(rows)):
        slope = math.log(re_arr[i] / re_arr[i-1]) / math.log(L_arr[i] / L_arr[i-1])
        print(f"  L {int(L_arr[i-1]):4d}->{int(L_arr[i]):4d}: slope = {slope:+.3f}")
    overall_slope, _ = np.polyfit(np.log(L_arr), np.log(re_arr), 1)
    print(f"\nOverall log-log: rel_err ~ L^({overall_slope:.3f})")
    print(f"Target: -0.500 (theorem v2 prediction for 1/sqrt(L))")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "12_halfperiod_test.npz")
    np.savez(out, L=L_arr, rel_err_mean=re_arr,
             rel_err_std=np.array([r[2] for r in rows]))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
