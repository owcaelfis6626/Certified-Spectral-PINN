"""Dealiased d=5 AC sweep -- the point that is unreachable without checkpointing.

d=5 puts 32^5 = 33.5M grid points through cubic_proj_fft per quadrature step,
and loss_rv_ac retains every step for backward, so the N_q = 2L needed to
dealias the cubic (script 26) would want tens of GB. losses_ac_ckpt recomputes
those intermediates instead (gated exactly equal in value and gradient by
script 28), making N_q = 2L affordable at small chunk.

Streams one row per L, so the L=32/64 slope is available long before the
expensive L=128 point lands.
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from losses_ac_ckpt import loss_rv_ac_ckpt
from oracle_ac import ACOracleHalfPeriod
from pinn_lookup import LookupGalerkinPINN

spec = importlib.util.spec_from_file_location(
    "sweep20", os.path.join(HERE, "20_ac_lbfgs_sweep.py"))
sweep20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep20)

D = 5
LS = [32, 64, 128]
SEED = 0
DEALIAS_FACTOR = 2
CHUNK = 2  # ~1 GB retained per step at 32^5; keep the working set small

# d=5 reference points measured earlier this session (single seed, same code path)
PRIOR = {32: (0.7710, 0.5136), 64: (0.7869, 0.4740), 128: (0.7873, 0.4341)}
#        L  : (N_q=16,  N_q=32)


def train_ckpt(d, L, seed, n_q, chunk):
    oracle = ACOracleHalfPeriod(
        d=d, sigma=sweep20.SIGMA, K_max=sweep20.K_MAX, T=sweep20.T,
        L_test=L, N_fine=sweep20.N_FINE, N_x=sweep20.N_X, seed=42 + seed,
        device=sweep20.DEVICE, dtype=sweep20.DTYPE)
    model = LookupGalerkinPINN(oracle)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=sweep20.LBFGS_MAXITER,
        history_size=50, line_search_fn="strong_wolfe",
        tolerance_grad=1e-16, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        loss = loss_rv_ac_ckpt(model, oracle, N_q=n_q, N_x=sweep20.N_X,
                               chunk=chunk)
        loss.backward()
        return loss

    t0 = time.time()
    prev = None
    for _ in range(sweep20.LBFGS_OUTER):
        loss = opt.step(closure).item()
        if prev is not None and abs(prev - loss) <= sweep20.LBFGS_TOL * max(1.0, prev):
            break
        prev = loss

    alpha, beta = model.coefficients(oracle)
    re = sweep20.rel_err_at_coeffs(alpha, beta, oracle)
    return re, loss, time.time() - t0


def main():
    print(f"P7 AC DEALIASED d={D}  N_q={DEALIAS_FACTOR}*L  chunk={CHUNK}  "
          f"seed={SEED}  (checkpointed; gated by script 28)")
    print(f"K_max={sweep20.K_MAX}  N_x={sweep20.N_X}  device={sweep20.DEVICE}\n")
    print(f"{'L':>5} {'N_q':>5} {'rel_err':>9} {'N_q=16':>8} {'N_q=32':>8} "
          f"{'peak MiB':>9} {'wall':>10}")

    rows = []
    for L in LS:
        n_q = DEALIAS_FACTOR * L
        if sweep20.DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            re, lo, wall = train_ckpt(D, L, SEED, n_q, CHUNK)
        except torch.cuda.OutOfMemoryError:
            print(f"{L:5d} {n_q:5d}   OOM even with chunk={CHUNK}", flush=True)
            torch.cuda.empty_cache()
            continue
        peak = torch.cuda.max_memory_allocated() / 2**20 if sweep20.DEVICE == "cuda" else 0.0
        p16, p32 = PRIOR.get(L, (float("nan"), float("nan")))
        print(f"{L:5d} {n_q:5d}  {re:.4f}  {p16:8.4f} {p32:8.4f} "
              f"{peak:9.1f} {wall:9.1f}s", flush=True)
        rows.append((L, n_q, re, lo))
        if sweep20.DEVICE == "cuda":
            torch.cuda.empty_cache()

    if len(rows) >= 2:
        arr = np.array(rows)
        slope, _ = np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 2]), 1)
        print(f"\nd={D} dealiased slope: rel_err ~ L^({slope:+.3f})")
        print("  prior same-seed slopes: N_q=16 -> +0.015 (flat), N_q=32 -> -0.121")
        out = os.path.join(HERE, "..", "results", "29_ac_dealiased_d5.npz")
        np.savez(out, rows=arr, slope=slope)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
