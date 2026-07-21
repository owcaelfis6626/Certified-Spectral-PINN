"""Run the optimal Galerkin solve at L = 32, 64, 128, 256, 512, 1024 at d=2
to identify the asymptotic rate of the best-approximation error.

Theorem v2 predicts rel_err^2 ~ 1/L (so rel_err ~ L^{-1/2}).
Empirics at L<=128 suggest rel_err ~ L^{-0.4}, sub-asymptotic.

If the rate corrects to L^{-1/2} at large L, the theorem is right and we're
just pre-asymptotic. If it stays at L^{-0.4}, the theorem needs to account
for the Galerkin-vs-L^2-projection coupling explicitly.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle import WMContinuousOracle


D, K_MAX = 2, 4
SIGMA, T = 1.0, 1.0
N_FINE = 32768     # need N_fine >> max(L) for accurate Wiener integrals at high l
SEEDS = [0, 1, 2, 3, 4]
LS = [32, 64, 128, 256, 512, 1024]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32


def solve_optimal_alpha(oracle):
    M = oracle.M
    L = M.shape[0]
    I_eye = torch.eye(L, device=M.device, dtype=M.dtype)
    alpha = torch.zeros_like(oracle.I_a)
    beta  = torch.zeros_like(oracle.I_b)
    for j in range(oracle.J):
        A = -M + oracle.k2[j] * I_eye
        alpha[j] = torch.linalg.solve(A, oracle.sigma * oracle.I_a[j])
        beta[j]  = torch.linalg.solve(A, oracle.sigma * oracle.I_b[j])
    return alpha, beta


@torch.no_grad()
def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64, t_chunk=2):
    """rel_err = ||u_th - u_ref||_Y / ||u_ref||_Y, chunked over t."""
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

        phase = (math.pi * oracle.l_idx[None, :] / oracle.T) * t_eval[:, None]
        psi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)
        ic_decay = torch.exp(-oracle.lambda_k[None, :] * t_eval[:, None])
        a_th = ic_decay * oracle.a0[None, :] + psi @ alpha.T
        b_th = ic_decay * oracle.b0[None, :] + psi @ beta.T
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
    print(f"P7 high-L rate verification  d={D} K_max={K_MAX} N_fine={N_FINE}")
    print(f"Device: {DEVICE}\n")
    print(f"{'L':>6} {'rel_err mean ± std':>22} {'wall':>8}")

    rows = []
    for L in LS:
        rel_errs = []
        t0 = time.time()
        for seed in SEEDS:
            oracle = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                         L_test=L, N_fine=N_FINE,
                                         seed=42 + seed,
                                         device=DEVICE, dtype=DTYPE)
            alpha, beta = solve_optimal_alpha(oracle)
            re = rel_err_at_coeffs(alpha, beta, oracle)
            rel_errs.append(re)
            del oracle, alpha, beta
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        arr = np.array(rel_errs)
        print(f"{L:6d}  {arr.mean():.4f} ± {arr.std():.4f}  {time.time()-t0:7.1f}s",
              flush=True)
        rows.append((L, arr.mean(), arr.std()))

    print("\n=== log-log fits ===")
    L_arr = np.array([r[0] for r in rows], dtype=float)
    re_arr = np.array([r[1] for r in rows])

    # Pairwise log-log slopes
    print("Pairwise rates (rel_err ~ L^slope):")
    for i in range(1, len(rows)):
        slope = math.log(re_arr[i] / re_arr[i-1]) / math.log(L_arr[i] / L_arr[i-1])
        print(f"  L {int(L_arr[i-1]):4d}->{int(L_arr[i]):4d}: slope = {slope:+.3f}")

    # Linear regression for an overall rate
    log_L = np.log(L_arr)
    log_re = np.log(re_arr)
    slope, intercept = np.polyfit(log_L, log_re, 1)
    print(f"\nOverall log-log fit: rel_err = {math.exp(intercept):.3f} · L^({slope:.3f})")

    # Asymptotic-rate check: tail slope vs initial slope
    if len(rows) >= 4:
        tail_slope, _ = np.polyfit(log_L[-3:], log_re[-3:], 1)
        head_slope, _ = np.polyfit(log_L[:3], log_re[:3], 1)
        print(f"Head slope (L=32..128): {head_slope:.3f}")
        print(f"Tail slope (L=256..1024): {tail_slope:.3f}")
        print(f"Theory predicts: -0.500 (asymptotic)")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "10_high_L_rate.npz")
    np.savez(out, L=L_arr, rel_err_mean=re_arr,
             rel_err_std=np.array([r[2] for r in rows]))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
