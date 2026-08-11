"""Targeted 5-minute test: is N_q=16 (fixed time-quadrature points, a
memory-driven compromise per script 20's own comment) under-constraining the
d=5 lookup table (2*J*L ~ 172,000 params at L=128) relative to the reference
solution, rather than the model genuinely failing to represent it? If so,
rel_err should drop when N_q increases -- a single, falsifiable check before
committing to anything bigger.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle_ac import ACOracleHalfPeriod
from pinn_lookup import LookupGalerkinPINN
from losses_ac import loss_rv_ac

K_MAX, SIGMA, T, N_FINE, N_X = 3, 1.0, 1.0, 4096, 32
LBFGS_OUTER, LBFGS_MAXITER, LBFGS_TOL = 150, 20, 1e-12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64):
    import math
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval, device=oracle.device, dtype=oracle.dtype)
    dt = oracle.T / (N_t_eval - 1)
    w = torch.ones(N_t_eval, device=oracle.device, dtype=oracle.dtype)
    w[0] = 0.5; w[-1] = 0.5
    lam = (oracle.k2 - 1.0)[None, :]
    exp_decay = torch.exp(-lam * t_eval[:, None])
    phase = oracle.omega_l[None, :] * t_eval[:, None]
    phi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)
    a_th = exp_decay * oracle.a0[None, :] + phi @ alpha.T
    b_th = exp_decay * oracle.b0[None, :] + phi @ beta.T
    a_ref, b_ref = oracle.a_ref(t_eval)
    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    e_a, e_b = a_th - a_ref, b_th - b_ref
    err_Y = (w[:, None] * one_plus_k2 * (e_a**2 + e_b**2)).sum().item() * dt
    ref_Y = (w[:, None] * one_plus_k2 * (a_ref**2 + b_ref**2)).sum().item() * dt
    return (err_Y / ref_Y) ** 0.5


def train_and_eval(d, L, n_q, seed=0):
    oracle = ACOracleHalfPeriod(d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L,
                                 N_fine=N_FINE, N_x=N_X, seed=42 + seed,
                                 device=DEVICE, dtype=DTYPE)
    model = LookupGalerkinPINN(oracle)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_MAXITER,
                             history_size=50, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-16, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        loss = loss_rv_ac(model, oracle, N_q=n_q, N_x=N_X)
        loss.backward()
        return loss

    t0 = time.time()
    prev = None
    for _ in range(LBFGS_OUTER):
        loss = opt.step(closure).item()
        if prev is not None and abs(prev - loss) <= LBFGS_TOL * max(1.0, prev):
            break
        prev = loss

    alpha, beta = model.coefficients(oracle)
    re = rel_err_at_coeffs(alpha, beta, oracle)
    return re, loss, time.time() - t0


if __name__ == "__main__":
    d, L = 5, 128
    # N_q=16 already confirmed (rel_err=0.7873, matches script 20 exactly);
    # N_q=64 OOM'd (needs ~4x the memory, cubic_proj_fft's per-quadrature-point
    # N_x^5 FFT dominates). N_q=32 is the untested middle ground.
    print(f"N_q=32 test: d={d}, L={L} (N_q=16 -> 0.7873 confirmed; N_q=64 OOM'd)\n")
    re, lo, wall = train_and_eval(d, L, 32)
    print(f"  N_q=32  rel_err={re:.4f}  loss={lo:.3e}  wall={wall:.1f}s", flush=True)
