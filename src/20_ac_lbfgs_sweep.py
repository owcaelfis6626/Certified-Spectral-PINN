"""Allen-Cahn d-sweep with L-BFGS (replaces script 16's diverging Adam result).

P7.1 diagnostic (script 19) established that the AC RVPINN loss is deterministic
full-batch, and the d>=3 "divergence" was an Adam budget/conditioning failure,
not a representation limit: L-BFGS with strong-Wolfe line search drives loss_rv
to ~1e-9 and rel_err decays cleanly with L at d=3. This script produces the
publication numbers across d=2,3,4 with seeds, mirroring script 16's protocol.

Usage:
  python 20_ac_lbfgs_sweep.py                 # full: d=2,3,4  seeds 0,1,2
  python 20_ac_lbfgs_sweep.py --d 4 --seeds 0 # quick single-seed check
"""

import argparse
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

LBFGS_OUTER   = 150
LBFGS_MAXITER = 20
LBFGS_TOL     = 1e-12

LS = [32, 64, 128]

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


def train_lbfgs(d, L, seed):
    oracle = ACOracleHalfPeriod(
        d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L,
        N_fine=N_FINE, N_x=N_X, seed=42 + seed,
        device=DEVICE, dtype=DTYPE)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print(f"P7 Allen-Cahn L-BFGS d-sweep  K_max={K_MAX}  N_q={N_Q}  N_x={N_X}  "
          f"outer={LBFGS_OUTER}x{LBFGS_MAXITER}  device={DEVICE}\n")
    print(f"{'d':>4} {'J':>6} {'L':>5} {'rel_err mean±std':>22} "
          f"{'loss(max)':>11} {'wall':>8}")

    rows = []
    for d in args.d:
        J = len(half_space_modes(d, K_MAX, exclude_zero=True))
        for L in LS:
            re_seeds, lo_seeds = [], []
            t0 = time.time()
            for seed in args.seeds:
                re, lo, _ = train_lbfgs(d, L, seed)
                re_seeds.append(re)
                lo_seeds.append(lo)
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
            arr = np.array(re_seeds)
            print(f"{d:4d} {J:6d} {L:5d}  {arr.mean():.4f} ± {arr.std():.4f}"
                  f"  {max(lo_seeds):.3e}  {time.time()-t0:7.1f}s", flush=True)
            rows.append((d, J, L, arr.mean(), arr.std(), max(lo_seeds)))

    print("\n=== slope of rel_err vs L (log-log) ===")
    arr = np.array(rows)
    for d in args.d:
        m = arr[:, 0] == d
        Ls = arr[m, 2]
        if len(Ls) >= 2:
            slope, _ = np.polyfit(np.log(Ls), np.log(arr[m, 3]), 1)
            print(f"  d={d}: rel_err ~ L^({slope:+.3f})")

    tag = "full" if args.d == [2, 3, 4] and args.seeds == [0, 1, 2] else \
          f"d{''.join(map(str, args.d))}_s{''.join(map(str, args.seeds))}"
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"20_ac_lbfgs_sweep_{tag}.npz")
    np.savez(out, rows=arr)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
