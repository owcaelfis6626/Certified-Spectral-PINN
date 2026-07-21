"""P7.1 diagnostic: is the AC d>=3 divergence an OPTIMIZER problem?

The RVPINN AC loss (losses_ac.loss_rv_ac) is fully DETERMINISTIC: fixed
Gauss-Legendre nodes, exact FFT cubic, precomputed M / I_a / I_b. So training
the lookup table is full-batch deterministic minimisation of a smooth
(degree-6 polynomial) objective. For that, Adam with a fixed step budget and
fixed LR is the wrong tool as the parameter count (J*L) grows; L-BFGS is the
textbook choice.

This script compares, per (d, L), single seed:
  - Adam 5000 steps + cosine LR (the script-16 baseline), and
  - L-BFGS with strong-Wolfe line search (deterministic full-batch),
reporting BOTH final loss_rv and rel_err.

Decisive read-out:
  * L-BFGS loss << Adam loss AND rel_err decays with L  -> optimiser problem,
    P7.1 solved (swap the sweep optimiser).
  * L-BFGS drives loss -> ~0 but rel_err stays high/grows -> representation /
    discrete inf-sup problem (deeper; keep the honest open-problem framing).

d=2 is the control (Adam already converges there).
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
from sampler import half_space_modes


K_MAX  = 3
SIGMA  = 1.0
T      = 1.0
N_FINE = 4096
N_X    = 32
N_Q    = 16

# Adam baseline (identical to script 16)
ADAM_STEPS = 5000
ADAM_LR    = 1e-2
ADAM_LRMIN = 1e-4

# L-BFGS config
LBFGS_OUTER   = 120          # outer .step() calls
LBFGS_MAXITER = 20           # iterations per .step()
LBFGS_TOL     = 1e-12        # relative loss plateau early-stop

D_VALS = [2, 3]
LS     = [32, 64, 128]
SEED   = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32


@torch.no_grad()
def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64):
    """Y-norm relative error of the trained coeffs vs EM reference. (= script 16)"""
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                            device=oracle.device, dtype=oracle.dtype)
    dt = oracle.T / (N_t_eval - 1)
    w = torch.ones(N_t_eval, device=oracle.device, dtype=oracle.dtype)
    w[0] = 0.5; w[-1] = 0.5

    lam = (oracle.k2 - 1.0)[None, :]
    exp_decay = torch.exp(-lam * t_eval[:, None])              # (N_t, J)
    phase = oracle.omega_l[None, :] * t_eval[:, None]          # (N_t, L)
    phi   = math.sqrt(2.0 / oracle.T) * torch.sin(phase)       # (N_t, L)

    a_th = exp_decay * oracle.a0[None, :] + phi @ alpha.T      # (N_t, J)
    b_th = exp_decay * oracle.b0[None, :] + phi @ beta.T

    a_ref, b_ref = oracle.a_ref(t_eval)                        # (N_t, J)

    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    e_a, e_b = a_th - a_ref, b_th - b_ref
    err_Y = (w[:, None] * one_plus_k2 * (e_a**2 + e_b**2)).sum().item() * dt
    ref_Y = (w[:, None] * one_plus_k2 * (a_ref**2 + b_ref**2)).sum().item() * dt
    return math.sqrt(err_Y / ref_Y)   # 2026-07-17 fix: was returning the SQUARED relative error


def make_oracle(d, L, seed):
    return ACOracleHalfPeriod(
        d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L,
        N_fine=N_FINE, N_x=N_X, seed=42 + seed,
        device=DEVICE, dtype=DTYPE)


def train_adam(oracle):
    model = LookupGalerkinPINN(oracle)
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=ADAM_STEPS, eta_min=ADAM_LRMIN)
    t0 = time.time()
    for _ in range(ADAM_STEPS):
        opt.zero_grad()
        loss = loss_rv_ac(model, oracle, N_q=N_Q, N_x=N_X)
        loss.backward()
        opt.step()
        sched.step()
    alpha, beta = model.coefficients(oracle)
    re = rel_err_at_coeffs(alpha, beta, oracle)
    return re, loss.item(), time.time() - t0


def train_lbfgs(oracle):
    model = LookupGalerkinPINN(oracle)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=LBFGS_MAXITER,
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
    print(f"P7.1 optimiser diagnostic  K_max={K_MAX}  N_q={N_Q}  N_x={N_X}  "
          f"seed={SEED}  device={DEVICE}\n")
    header = (f"{'d':>3} {'J':>5} {'L':>5} | "
              f"{'Adam re':>9} {'Adam loss':>11} {'s':>6} | "
              f"{'LBFGS re':>9} {'LBFGS loss':>11} {'s':>6}")
    print(header)
    print("-" * len(header))

    rows = []
    for d in D_VALS:
        J = len(half_space_modes(d, K_MAX, exclude_zero=True))
        for L in LS:
            oracle = make_oracle(d, L, SEED)
            re_a, lo_a, ta = train_adam(oracle)
            re_b, lo_b, tb = train_lbfgs(oracle)
            print(f"{d:3d} {J:5d} {L:5d} | "
                  f"{re_a:9.4f} {lo_a:11.3e} {ta:6.1f} | "
                  f"{re_b:9.4f} {lo_b:11.3e} {tb:6.1f}", flush=True)
            rows.append((d, J, L, re_a, lo_a, re_b, lo_b))
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print("\n=== slope of rel_err vs L (log-log) ===")
    arr = np.array(rows)
    for d in D_VALS:
        m = arr[:, 0] == d
        Ls = arr[m, 2]
        if len(Ls) >= 2:
            sa, _ = np.polyfit(np.log(Ls), np.log(arr[m, 3]), 1)
            sb, _ = np.polyfit(np.log(Ls), np.log(arr[m, 5]), 1)
            print(f"  d={d}:  Adam slope {sa:+.3f}   LBFGS slope {sb:+.3f}")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "19_ac_lbfgs_diag.npz")
    np.savez(out, rows=arr)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
