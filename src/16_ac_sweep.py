"""Allen-Cahn d-sweep: d=2,3,4 with half-period basis.

Trains LookupGalerkinPINN against the AC weak-form loss and reports
rel_err vs L, analogous to the OU sweep in script 13.

Expected result: rel_err converges as L grows, uniformly in d,
even though J grows ~10x per 2 dimensions.
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


K_MAX  = 3
SIGMA  = 1.0
T      = 1.0
N_FINE = 4096
N_X    = 32       # spatial grid per dim; 32 > 2*3*K_max=18, no aliasing
N_Q    = 16       # quadrature points; fewer than OU to limit VRAM for d=4
N_STEPS = 5000
LR      = 1e-2
LR_MIN  = 1e-4   # cosine decay floor
D_VALS  = [2, 3, 4]
LS      = [32, 64, 128]
SEEDS   = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32


@torch.no_grad()
def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64):
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    dt = oracle.T / (N_t_eval - 1)
    w = torch.ones(N_t_eval, device=oracle.device, dtype=oracle.dtype)
    w[0] = 0.5; w[-1] = 0.5

    lam = (oracle.k2 - 1.0)[None, :]
    exp_decay = torch.exp(-lam * t_eval[:, None])          # (N_t, J)
    phase = oracle.omega_l[None, :] * t_eval[:, None]      # (N_t, L)
    phi   = math.sqrt(2.0 / oracle.T) * torch.sin(phase)  # (N_t, L)

    a_th = exp_decay * oracle.a0[None, :] + phi @ alpha.T  # (N_t, J)
    b_th = exp_decay * oracle.b0[None, :] + phi @ beta.T

    a_ref, b_ref = oracle.a_ref(t_eval)                    # (N_t, J)

    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    e_a, e_b = a_th - a_ref, b_th - b_ref
    err_Y = (w[:, None] * one_plus_k2 * (e_a**2 + e_b**2)).sum().item() * dt
    ref_Y = (w[:, None] * one_plus_k2 * (a_ref**2 + b_ref**2)).sum().item() * dt
    return math.sqrt(err_Y / ref_Y)   # 2026-07-17 fix: was returning the SQUARED relative error


def train_one(d, L, seed):
    oracle = ACOracleHalfPeriod(
        d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L,
        N_fine=N_FINE, N_x=N_X, seed=42 + seed,
        device=DEVICE, dtype=DTYPE)
    model = LookupGalerkinPINN(oracle)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR_MIN)

    t0 = time.time()
    for step in range(N_STEPS):
        opt.zero_grad()
        loss = loss_rv_ac(model, oracle, N_q=N_Q, N_x=N_X)
        loss.backward()
        opt.step()
        sched.step()

    alpha, beta = model.coefficients(oracle)
    re = rel_err_at_coeffs(alpha, beta, oracle)
    return re, loss.item(), time.time() - t0


def main():
    print(f"P7 Allen-Cahn d-sweep  K_max={K_MAX}  N_q={N_Q}  N_x={N_X}  "
          f"N_steps={N_STEPS}\n")
    print(f"{'d':>4} {'J':>6} {'L':>5} {'rel_err mean±std':>22} {'wall':>8}")

    rows = []
    for d in D_VALS:
        # Count J for this d (quick oracle-less check)
        from sampler import half_space_modes
        J = len(half_space_modes(d, K_MAX, exclude_zero=True))

        for L in LS:
            re_seeds, lrv_seeds = [], []
            t0 = time.time()
            for seed in SEEDS:
                re, lrv, _ = train_one(d, L, seed)
                re_seeds.append(re)
                lrv_seeds.append(lrv)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            arr = np.array(re_seeds)
            print(f"{d:4d} {J:6d} {L:5d}  {arr.mean():.4f} ± {arr.std():.4f}"
                  f"  {time.time()-t0:7.1f}s", flush=True)
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
                       "16_ac_sweep.npz")
    np.savez(out, rows=arr)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
