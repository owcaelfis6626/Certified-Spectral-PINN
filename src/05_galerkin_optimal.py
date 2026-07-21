"""Compute the OPTIMAL Galerkin coefficients alpha^*, beta^* in closed form
and measure the resulting ||e||_Y^2.  This is the best the Galerkin
architecture can possibly achieve.

If rel_err at alpha^* is small (~1/sqrt(L)), the architecture is sound and
the MLP is just failing to optimize.  If rel_err at alpha^* is large,
Galerkin truncation is the fundamental limit.

Sweep L = {32, 64, 128, 256} at d=2 to see the convergence rate.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle import WMContinuousOracle


D       = 2
K_MAX   = 4
SIGMA   = 1.0
T       = 1.0
N_FINE  = 4096       # fine Brownian path for accurate Wiener integrals
SEEDS   = [0, 1, 2]
LS      = [32, 64, 128, 256]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32    # float64 OOM's the small GPU on the (Tb, J, N) tensor


def solve_optimal_alpha(oracle):
    """For each mode k, solve (M + |k|^2 I) alpha_k = sigma I_k^{a,b}.

    M is the same L x L matrix across modes; only the |k|^2 scalar differs.
    """
    M = oracle.M                                # (L, L)
    L = M.shape[0]
    I_eye = torch.eye(L, device=M.device, dtype=M.dtype)

    alpha_star = torch.zeros_like(oracle.I_a)   # (J, L)
    beta_star  = torch.zeros_like(oracle.I_b)

    # Residual is R = alpha @ M + |k|^2 alpha - sigma I (row equation).
    # Setting R = 0:  alpha (M + |k|^2 I) = sigma I
    # Transpose -> (M^T + |k|^2 I) alpha^T = sigma I^T.
    # M is antisymmetric so M^T = -M => solve with (|k|^2 I - M).
    for j in range(oracle.J):
        A = -M + oracle.k2[j] * I_eye            # (L, L)
        b_a = oracle.sigma * oracle.I_a[j]        # (L,)
        b_b = oracle.sigma * oracle.I_b[j]
        alpha_star[j] = torch.linalg.solve(A, b_a)
        beta_star[j]  = torch.linalg.solve(A, b_b)
    return alpha_star, beta_star


def compute_LRV_at_coeffs(alpha, beta, oracle):
    """L_RV at given (alpha, beta) -- same closed-form as loss_rv_galerkin."""
    k2 = oracle.k2[:, None]
    Ma = alpha @ oracle.M
    Mb = beta  @ oracle.M
    R_a = Ma + k2 * alpha - oracle.sigma * oracle.I_a
    R_b = Mb + k2 * beta  - oracle.sigma * oracle.I_b
    weight = 1.0 / (1.0 + oracle.k2[:, None] + oracle.mu_l[None, :])
    return ((R_a ** 2 + R_b ** 2) * weight).sum().item()


def compute_err_Y_at_coeffs(alpha, beta, oracle, N_t_eval=128):
    """||u_theta - u_ref||_Y^2 at given Galerkin coefficients."""
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)

    # Construct a_theta, b_theta at eval grid.
    phase = (math.pi * oracle.l_idx[None, :] / oracle.T) * t_eval[:, None]
    psi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)               # (N_t, L)
    ic_decay = torch.exp(-oracle.lambda_k[None, :] * t_eval[:, None])  # (N_t, J)
    a_th = ic_decay * oracle.a0[None, :] + psi @ alpha.T               # (N_t, J)
    b_th = ic_decay * oracle.b0[None, :] + psi @ beta.T

    a_ref, b_ref = oracle.a_ref(t_eval)
    e_a, e_b = a_th - a_ref, b_th - b_ref

    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    dt = oracle.T / (N_t_eval - 1)
    sq_per_t = (one_plus_k2 * (e_a ** 2 + e_b ** 2)).sum(-1)
    w = torch.ones_like(sq_per_t)
    w[0] = 0.5
    w[-1] = 0.5
    return (math.pi ** oracle.d) * (w * sq_per_t).sum().item() * dt


def ref_norm_y_sq(oracle, N_t_eval=128):
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    a_ref, b_ref = oracle.a_ref(t_eval)
    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    dt = oracle.T / (N_t_eval - 1)
    sq_per_t = (one_plus_k2 * (a_ref ** 2 + b_ref ** 2)).sum(-1)
    w = torch.ones_like(sq_per_t)
    w[0] = 0.5
    w[-1] = 0.5
    return (math.pi ** oracle.d) * (w * sq_per_t).sum().item() * dt


def main():
    print(f"P7 Galerkin OPTIMAL d={D} K_max={K_MAX}  Wiener path N_fine={N_FINE}\n")
    print(f"{'L':>6} {'rel_err_mean':>14} {'rel_err_std':>14} {'L_RV_opt':>14} {'wall':>8}")

    rows = []
    for L in LS:
        rel_errs, lrvs = [], []
        import time
        t0 = time.time()
        for seed in SEEDS:
            oracle = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                         L_test=L, N_fine=N_FINE,
                                         seed=42 + seed, device=DEVICE, dtype=DTYPE)
            alpha, beta = solve_optimal_alpha(oracle)
            err_Y = compute_err_Y_at_coeffs(alpha, beta, oracle)
            ref_Y = ref_norm_y_sq(oracle)
            L_RV  = compute_LRV_at_coeffs(alpha, beta, oracle)
            rel_errs.append(err_Y / ref_Y)
            lrvs.append(L_RV)
        rel_errs = np.array(rel_errs)
        lrvs = np.array(lrvs)
        print(f"{L:6d} {rel_errs.mean():14.4e} {rel_errs.std():14.4e}"
              f" {lrvs.mean():14.4e} {time.time()-t0:7.1f}s", flush=True)
        rows.append((L, rel_errs.mean(), rel_errs.std(), lrvs.mean()))

    print()
    print("Expected Galerkin scaling for Brownian regularity: rel_err ~ 1/sqrt(L)")
    L_vals = np.array([r[0] for r in rows])
    re_vals = np.array([r[1] for r in rows])
    for i in range(1, len(rows)):
        ratio = re_vals[i-1] / re_vals[i]
        L_ratio_sqrt = math.sqrt(L_vals[i] / L_vals[i-1])
        print(f"  L {L_vals[i-1]:3d}->{L_vals[i]:3d}: rel_err drops {ratio:.3f}x"
              f"  (theoretical for 1/sqrt(L) bound: {L_ratio_sqrt:.3f}x)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                             "05_galerkin_optimal.npz")
    np.savez(out_path, L_vals=L_vals, rel_err_mean=re_vals,
             rel_err_std=np.array([r[2] for r in rows]),
             L_RV_opt=np.array([r[3] for r in rows]))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
