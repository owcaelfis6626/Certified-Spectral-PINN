"""Diagnose the plateau: is alpha (Galerkin) equal to hat_x_{1:L} (L^2 projection)?

If they're equal (numerical precision): then the plateau is in the diagnostic.
If they differ: the difference is the "tail correction" and we have a real
                Galerkin-coefficient error that doesn't decay with L.

For each mode k and L value, compute:
  alpha_Galerkin   = (-M + |k|^2 I)^{-1} sigma I_k
  hat_x_L2_proj_l  = <x_k, psi_l>_{L^2(0,T)} for l=1..L
and compare.

x_k(t) is the OU stochastic integral computed from the fine Brownian path.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle import WMContinuousOracle


D, K_MAX = 2, 4
SIGMA, T = 1.0, 1.0
N_FINE = 32768
LS = [32, 64, 128, 256, 512]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float64


def hat_x_l2_projection(oracle, L_MAX, chunk_t=4096, chunk_l=128):
    """Compute hat_x_{k,l} = <x_k, psi_l>_{L^2(0,T)} for the OU process
    x_k(t) = sigma int_0^t e^{-|k|^2 (t-s)} dB(s).

    OU computed via forward Euler--Maruyama using the SAME dB path as the
    oracle, so x_k matches what `a_ref` uses.  Integration against psi_l
    is trapezoidal on the fine grid, chunked over l and over time to fit
    the 2 GB GPU.
    """
    t_fine = oracle.t_fine                  # (N_fine,)
    N_fine = t_fine.shape[0]
    dt_fine = oracle.dt_fine
    J = oracle.J

    hat_x_a = torch.zeros(J, L_MAX, device=DEVICE, dtype=DTYPE)
    hat_x_b = torch.zeros(J, L_MAX, device=DEVICE, dtype=DTYPE)

    # Vectorised OU integration via cumulative recursion (all modes in parallel).
    # x_a[j, n+1] = exp(-lambda_j dt) x_a[j, n] + sigma * dB_a[j, n]
    decay_step = torch.exp(-oracle.lambda_k.to(DTYPE) * dt_fine)   # (J,)
    x_a = torch.zeros(J, N_fine, device=DEVICE, dtype=DTYPE)
    x_b = torch.zeros(J, N_fine, device=DEVICE, dtype=DTYPE)
    dB_a_64 = oracle.dB_a.to(DTYPE)
    dB_b_64 = oracle.dB_b.to(DTYPE)
    sigma = oracle.sigma
    for n in range(N_fine - 1):
        x_a[:, n + 1] = decay_step * x_a[:, n] + sigma * dB_a_64[:, n]
        x_b[:, n + 1] = decay_step * x_b[:, n] + sigma * dB_b_64[:, n]

    # Trapezoidal weights
    w_trap = torch.ones(N_fine, device=DEVICE, dtype=DTYPE)
    w_trap[0] = 0.5
    w_trap[-1] = 0.5
    w_trap = w_trap * dt_fine

    # Project against psi_l in chunks of l to keep memory bounded.
    pi_over_T = math.pi / oracle.T
    sqrt2_T   = math.sqrt(2.0 / oracle.T)

    for l_start in range(1, L_MAX + 1, chunk_l):
        l_end = min(l_start + chunk_l, L_MAX + 1)
        l_idx = torch.arange(l_start, l_end, device=DEVICE, dtype=DTYPE)
        # psi at all fine times for this chunk of l: (N_fine, chunk_l)
        psi = sqrt2_T * torch.sin(pi_over_T * l_idx[None, :] * t_fine[:, None])

        # weighted psi  (N_fine, chunk_l)
        psi_w = psi * w_trap[:, None]

        # Projection: x (J, N_fine) @ psi_w (N_fine, chunk_l) -> (J, chunk_l)
        hat_x_a[:, l_start - 1:l_end - 1] = x_a @ psi_w
        hat_x_b[:, l_start - 1:l_end - 1] = x_b @ psi_w

        del psi, psi_w

    return hat_x_a, hat_x_b


def galerkin_alpha(oracle):
    """Galerkin solution restricted to the first L_test of oracle."""
    M = oracle.M
    L = M.shape[0]
    I_eye = torch.eye(L, device=M.device, dtype=DTYPE)
    M64 = M.to(DTYPE)
    k2_64 = oracle.k2.to(DTYPE)
    I_a_64 = oracle.I_a.to(DTYPE)
    I_b_64 = oracle.I_b.to(DTYPE)

    alpha = torch.zeros(oracle.J, L, device=M.device, dtype=DTYPE)
    beta  = torch.zeros(oracle.J, L, device=M.device, dtype=DTYPE)
    for j in range(oracle.J):
        A = -M64 + k2_64[j] * I_eye
        alpha[j] = torch.linalg.solve(A, oracle.sigma * I_a_64[j])
        beta[j]  = torch.linalg.solve(A, oracle.sigma * I_b_64[j])
    return alpha, beta


def main():
    print(f"P7 Galerkin vs L^2-projection d={D} K_max={K_MAX} N_fine={N_FINE}")
    print(f"Device: {DEVICE}\n")

    # Use one seed since we just want to see if alpha = hat_x_{1:L}.
    oracle_big = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                     L_test=max(LS), N_fine=N_FINE,
                                     seed=42, device=DEVICE, dtype=DTYPE)
    print(f"J = {oracle_big.J}")
    print("Computing hat_x via forward-Euler OU on fine grid (slow, one-off)...")

    import time
    t0 = time.time()
    hat_x_a_full, hat_x_b_full = hat_x_l2_projection(oracle_big, max(LS))
    print(f"  done in {time.time()-t0:.1f}s\n")

    # Free the big oracle's storage before iterating; we just need M and I_k per L.
    del oracle_big.dB_a, oracle_big.dB_b
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"{'L':>6} {'||alpha-hat_x||_2':>20} {'||hat_x_>L||_2':>20} {'ratio':>10}")
    for L in LS:
        oracle_L = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                       L_test=L, N_fine=N_FINE,
                                       seed=42, device=DEVICE, dtype=DTYPE)
        alpha_a, alpha_b = galerkin_alpha(oracle_L)
        hat_x_a_L = hat_x_a_full[:, :L]
        hat_x_b_L = hat_x_b_full[:, :L]
        hat_x_a_tail = hat_x_a_full[:, L:]
        hat_x_b_tail = hat_x_b_full[:, L:]

        norm_diff   = ((alpha_a - hat_x_a_L) ** 2 + (alpha_b - hat_x_b_L) ** 2).sum().sqrt()
        norm_tail   = ((hat_x_a_tail ** 2 + hat_x_b_tail ** 2).sum()).sqrt()

        print(f"{L:6d} {norm_diff.item():>20.4e} {norm_tail.item():>20.4e}"
              f" {(norm_diff/norm_tail).item():>10.4f}", flush=True)

        # Aggressive cleanup to keep memory under the 2 GB ceiling
        del oracle_L, alpha_a, alpha_b
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\nInterpretation:")
    print("- If ||alpha - hat_x_{1:L}|| << ||hat_x_>L||: Galerkin matches L^2 proj,")
    print("  plateau is in the diagnostic, not the architecture.")
    print("- If ||alpha - hat_x_{1:L}|| ~ ||hat_x_>L||: real Galerkin-vs-L^2 gap.")
    print("- If ||alpha - hat_x_{1:L}|| stays constant as L grows: plateau is REAL,")
    print("  Galerkin coefficient error doesn't decay.")


if __name__ == "__main__":
    main()
