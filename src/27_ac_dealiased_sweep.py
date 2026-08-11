"""THE definitive AC convergence sweep: N_q scaled with L to dealias the cubic.

Script 26 measured the temporal quadrature error of the cubic residual term
directly (no training) at L=128:

    N_q= 16 (published) -> 2.70   <- ~270% relative error
    N_q= 64             -> 1.10
    N_q=128 (= L)       -> 8.7e-3
    N_q=256 (= 2L)      -> 6.4e-6  (fp32 floor)

so the published AC table minimised a loss whose nonlinear term was wrong by
more than the term itself, and the error worsens as L grows -- i.e. precisely
along the axis the convergence study sweeps. losses_ac.py applies the cubic
dealiasing rule in SPACE (N_x > 6*K_max) but never the temporal analogue.

This sweep fixes that: N_q = DEALIAS_FACTOR * L per L, so every point is
resolved to the fp32 floor. The resulting slope is the true approximation
rate of the method, free of the aliasing artifact.

Single seed by default -- the question here is the RATE, and a properly
dealiased 3-seed run is ~3x this cost. Add seeds once the rate is known.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location(
    "sweep20", os.path.join(HERE, "20_ac_lbfgs_sweep.py"))
sweep20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep20)

LS = [32, 64, 128]
DS = [2, 3, 4]
SEED = 0
DEALIAS_FACTOR = 2  # N_q = 2L: the cubic dealiasing requirement, verified in script 26

# published N_q=16 baseline (3 seeds) for side-by-side reporting
PUBLISHED = {
    2: {32: 0.3132, 64: 0.2370, 128: 0.1898, "slope": -0.361},
    3: {32: 0.3733, 64: 0.3202, 128: 0.2736, "slope": -0.224},
    4: {32: 0.5132, 64: 0.4820, 128: 0.4587, "slope": -0.081},
}


def main():
    print("P7 Allen-Cahn DEALIASED sweep  N_q = "
          f"{DEALIAS_FACTOR}*L (was fixed 16)  seed={SEED}")
    print(f"K_max={sweep20.K_MAX}  N_x={sweep20.N_X}  device={sweep20.DEVICE}\n")
    print(f"{'d':>4} {'L':>5} {'N_q':>5} {'rel_err':>9} {'published':>10} "
          f"{'Δ':>8} {'loss':>11} {'wall':>9}")

    rows = []
    for d in DS:
        for L in LS:
            n_q = DEALIAS_FACTOR * L
            try:
                re, lo, wall = sweep20.train_lbfgs(d, L, SEED, n_q=n_q)
            except torch.cuda.OutOfMemoryError:
                print(f"{d:4d} {L:5d} {n_q:5d}   OOM -- dense {sweep20.N_X}^{d} "
                      f"grid x {n_q} steps retained for backward", flush=True)
                torch.cuda.empty_cache()
                continue
            pub = PUBLISHED.get(d, {}).get(L)
            pub_s = f"{pub:.4f}" if pub else "   -"
            dlt = f"{re - pub:+.4f}" if pub else "    -"
            print(f"{d:4d} {L:5d} {n_q:5d}  {re:.4f}  {pub_s:>10} {dlt:>8} "
                  f"{lo:.3e} {wall:8.1f}s", flush=True)
            rows.append((d, L, n_q, re, lo))
            if sweep20.DEVICE == "cuda":
                torch.cuda.empty_cache()

    arr = np.array(rows)
    print("\n=== slope of rel_err vs L (log-log), dealiased vs published ===")
    for d in DS:
        m = arr[arr[:, 0] == d] if len(arr) else np.empty((0, 5))
        if len(m) >= 2:
            slope, _ = np.polyfit(np.log(m[:, 1]), np.log(m[:, 3]), 1)
            pub = PUBLISHED.get(d, {}).get("slope")
            pub_s = f"{pub:+.3f}" if pub is not None else "  n/a"
            print(f"  d={d}: dealiased {slope:+.3f}   published(N_q=16) {pub_s}")

    out = os.path.join(HERE, "..", "results", "27_ac_dealiased_sweep.npz")
    np.savez(out, rows=arr)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
