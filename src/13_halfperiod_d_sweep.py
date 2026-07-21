"""Uniform-in-d check for half-period basis.

If the half-period basis truly fixes the architecture, the rate L^{-1}
should hold uniformly across d=2,3,5,7.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle_halfperiod import WMContinuousOracleHalfPeriod
from sampler import half_space_modes


K_MAX = 3
SIGMA, T = 1.0, 1.0
N_FINE = 4096
LS = [32, 64, 128, 256]
SEEDS = [0, 1, 2]
D_VALS = [2, 3, 5, 7]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def solve_optimal_alpha(oracle):
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
def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64, t_chunk=1):
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
        phi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)
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
    print(f"P7 half-period d-sweep  K_max={K_MAX}\n")
    print(f"{'d':>4} {'J':>6} {'L':>5} {'rel_err mean ± std':>22} {'wall':>8}")
    rows = []
    for d in D_VALS:
        J = len(half_space_modes(d, K_MAX, exclude_zero=True))
        for L in LS:
            re_seeds = []
            t0 = time.time()
            for seed in SEEDS:
                oracle = WMContinuousOracleHalfPeriod(
                    d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L, N_fine=N_FINE,
                    seed=42 + seed, device=DEVICE, dtype=DTYPE)
                alpha, beta = solve_optimal_alpha(oracle)
                re_seeds.append(rel_err_at_coeffs(alpha, beta, oracle))
                del oracle, alpha, beta
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            arr = np.array(re_seeds)
            print(f"{d:4d} {J:6d} {L:5d}  {arr.mean():.4f} ± {arr.std():.4f}"
                  f" {time.time()-t0:7.1f}s", flush=True)
            rows.append((d, J, L, arr.mean(), arr.std()))

    print("\n=== Slope check per d ===")
    arr = np.array(rows)
    for d in D_VALS:
        mask = arr[:, 0] == d
        Ls = arr[mask, 2]
        res = arr[mask, 3]
        if len(Ls) >= 2:
            slope, _ = np.polyfit(np.log(Ls), np.log(res), 1)
            print(f"  d={d}: rel_err ~ L^({slope:.3f})")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "13_halfperiod_d_sweep.npz")
    np.savez(out, rows=arr)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
