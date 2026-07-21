"""Sweep dimensions d=2,3,5,7 using the closed-form Galerkin solve.

Hypothesis from the theorem (corrected): inf-sup constant is uniform in d
because the smallest mode |k|=1 always gives beta_min = O(1) regardless
of d. Empirically, rel_err at fixed L should be d-independent at fixed
K_max.

At fixed K_max=3, J grows with d:
  d=2: J = ?
  d=3: J = ?
  d=5: J = ?
  d=7: J = 6218 (from P6 records)
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle import WMContinuousOracle


K_MAX  = 3
SIGMA  = 1.0
T      = 1.0
N_FINE = 2048      # cheaper for higher d
SEEDS  = [0, 1, 2]
D_VALS = [7]
LS     = [32, 64, 128]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32


def solve_optimal_alpha(oracle):
    """Solve (-M + |k|^2 I) alpha^T = sigma I per mode."""
    M = oracle.M
    L = M.shape[0]
    I_eye = torch.eye(L, device=M.device, dtype=M.dtype)
    alpha_star = torch.zeros_like(oracle.I_a)
    beta_star  = torch.zeros_like(oracle.I_b)
    for j in range(oracle.J):
        A = -M + oracle.k2[j] * I_eye
        alpha_star[j] = torch.linalg.solve(A, oracle.sigma * oracle.I_a[j])
        beta_star[j]  = torch.linalg.solve(A, oracle.sigma * oracle.I_b[j])
    return alpha_star, beta_star


@torch.no_grad()
def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64, t_chunk=1):
    """rel_err = ||u_th - u_ref||_Y / ||u_ref||_Y, chunked over t for large J."""
    t_full = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    dt = oracle.T / (N_t_eval - 1)
    w = torch.ones(N_t_eval, device=oracle.device, dtype=oracle.dtype)
    w[0] = 0.5
    w[-1] = 0.5
    one_plus_k2 = (1.0 + oracle.k2)[None, :]

    err_Y_total = 0.0
    ref_Y_total = 0.0
    for chunk_start in range(0, N_t_eval, t_chunk):
        chunk_end = min(chunk_start + t_chunk, N_t_eval)
        t_eval = t_full[chunk_start:chunk_end]

        phase = (math.pi * oracle.l_idx[None, :] / oracle.T) * t_eval[:, None]
        psi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)
        ic_decay = torch.exp(-oracle.lambda_k[None, :] * t_eval[:, None])
        a_th = ic_decay * oracle.a0[None, :] + psi @ alpha.T
        b_th = ic_decay * oracle.b0[None, :] + psi @ beta.T
        a_ref, b_ref = oracle.a_ref(t_eval)
        e_a, e_b = a_th - a_ref, b_th - b_ref

        w_chunk = w[chunk_start:chunk_end]
        err_Y_total += (w_chunk * (one_plus_k2 * (e_a**2 + e_b**2)).sum(-1)).sum().item()
        ref_Y_total += (w_chunk * (one_plus_k2 * (a_ref**2 + b_ref**2)).sum(-1)).sum().item()

        del a_th, b_th, a_ref, b_ref, e_a, e_b
        torch.cuda.empty_cache() if oracle.device == "cuda" else None

    return (err_Y_total * dt) / (ref_Y_total * dt)


def main():
    print(f"P7 d-sweep direct solve  K_max={K_MAX}  N_fine={N_FINE}\n")
    print(f"{'d':>4} {'J':>6} {'L':>5} {'rel_err':>14} {'wall':>8}")
    rows = []
    for d in D_VALS:
        for L in LS:
            re_seeds = []
            t0 = time.time()
            for seed in SEEDS:
                oracle = WMContinuousOracle(d=d, sigma=SIGMA, K_max=K_MAX, T=T,
                                             L_test=L, N_fine=N_FINE,
                                             seed=42 + seed,
                                             device=DEVICE, dtype=DTYPE)
                alpha, beta = solve_optimal_alpha(oracle)
                re = rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64)
                re_seeds.append(re)
                # Free memory for the next seed
                del oracle, alpha, beta
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
            arr = np.array(re_seeds)
            row = (d, oracle_J(d, K_MAX), L, arr.mean(), arr.std(), time.time() - t0)
            rows.append(row)
            print(f"{d:4d} {row[1]:6d} {L:5d} {arr.mean():9.4f} ±{arr.std():.4f}"
                  f" {row[5]:7.1f}s", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "07_d_sweep.npz")
    np.savez(out_path, rows=np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows]))
    print(f"\nsaved: {out_path}")


def oracle_J(d, K_max):
    """J for the half-space lex-ordered enumeration."""
    from sampler import half_space_modes
    return len(half_space_modes(d, K_max))


if __name__ == "__main__":
    main()
