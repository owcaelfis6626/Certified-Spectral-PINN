"""Full d=5 sweep with the fixed N_q=32 (was 16, a stale d<=4 memory
compromise per script 20's own comment -- confirmed in script 22 to nearly
halve rel_err at L=128: 0.7873 -> 0.4341). This reruns L=32,64,128 properly
to see whether the negative (converging) slope seen at d=2,3,4 recovers, or
whether d=5 still degrades relative to them even with N_q fixed.

Single seed (matching script 20's own "--seeds 0, quick single-seed check"
mode) -- multi-seed error bars are a follow-up, not this run's scope.
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle_ac import ACOracleHalfPeriod
from pinn_lookup import LookupGalerkinPINN
from losses_ac import loss_rv_ac

K_MAX, SIGMA, T, N_FINE, N_X = 3, 1.0, 1.0, 4096, 32
N_Q = 32  # the fix: was 16
LBFGS_OUTER, LBFGS_MAXITER, LBFGS_TOL = 150, 20, 1e-12
LS = [32, 64, 128]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64):
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
    return math.sqrt(err_Y / ref_Y)


def train_lbfgs(d, L, seed):
    oracle = ACOracleHalfPeriod(d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L,
                                 N_fine=N_FINE, N_x=N_X, seed=42 + seed,
                                 device=DEVICE, dtype=DTYPE)
    model = LookupGalerkinPINN(oracle)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=LBFGS_MAXITER,
                             history_size=50, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-16, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        loss = loss_rv_ac(model, oracle, N_q=N_Q, N_x=N_X)
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


def main():
    d = 5
    print(f"P7 AC d=5 sweep, N_q={N_Q} (fixed from 16) K_max={K_MAX} N_x={N_X} "
          f"outer={LBFGS_OUTER}x{LBFGS_MAXITER}  device={DEVICE}\n")
    print(f"{'d':>4} {'J':>6} {'L':>5} {'rel_err':>10} {'loss(max)':>11} {'wall':>8}")
    rows = []
    for L in LS:
        t0 = time.time()
        re, lo, wall = train_lbfgs(d, L, seed=0)
        print(f"{d:4d} {'671':>6} {L:5d}  {re:.4f}    {lo:.3e}  {wall:7.1f}s", flush=True)
        rows.append((d, 671, L, re, lo))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    arr = np.array(rows)
    Ls = arr[:, 2]
    slope, _ = np.polyfit(np.log(Ls), np.log(arr[:, 3]), 1)
    print(f"\nd=5 (N_q=32): rel_err ~ L^({slope:+.3f})")
    print("compare: d=2 -0.36, d=3 -0.22, d=4 -0.08 (all N_q=16, established honest table)")
    print("         d=5 (N_q=16, this session): +0.015 (flat -- the artifact)")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "23_ac_d5_nq32_sweep.npz")
    np.savez(out, rows=arr, slope=slope)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
