"""N_q convergence check -- the diagnostic that distinguishes "under-resolved
quadrature" from "genuine approximation limit".

Script 22/23 showed N_q=16 -> 32 nearly halves rel_err at d=5. But a change
between two values only proves they DIFFER; it does not prove 32 is resolved.
If rel_err(N_q=32) ~= rel_err(N_q=64) at fixed (d, L), the quadrature is
converged there and the remaining error is approximation-theoretic. If it is
still moving at 64, everything measured at 16 or 32 is quadrature-limited.

Runs at L=128 (largest test space = most parameters = most demanding of the
quadrature). d<=4 only: the cubic projection uses a dense N_x^d grid FFT per
quadrature point, so N_q=64 at d=5 OOMs (33M grid points), while d=4 is 1M.

Imports script 20 directly (its filename is not a valid module name, hence
importlib) so the code path is EXACTLY the one that produced the published
table -- a reimplementation would not be a valid comparison.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

spec = importlib.util.spec_from_file_location(
    "sweep20", os.path.join(HERE, "20_ac_lbfgs_sweep.py"))
sweep20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep20)

L_FIXED = 128
# losses_ac.py's own docstring derives the cubic dealiasing rule in SPACE
# (N_x > 2*3*K_max) but never applies it in TIME. The temporal analogue needs
# N_q growing with L; at L=128 that is O(256), not the hardcoded 16. Push until
# rel_err stops moving or the run OOMs -- both outcomes are informative.
NQS = [16, 32, 64, 128, 256]
DS = [2, 3, 4]
SEED = 0


def main():
    print(f"P7 AC N_q convergence  L={L_FIXED}  seed={SEED}  "
          f"K_max={sweep20.K_MAX}  N_x={sweep20.N_X}  device={sweep20.DEVICE}")
    print(f"cubic dealiasing in time needs N_q = O(2L) = O({2 * L_FIXED}); "
          f"published runs used N_q=16\n")
    print(f"{'d':>4} {'N_q':>5} {'rel_err':>10} {'loss':>12} {'wall':>9}")

    rows = []
    for d in DS:
        prev = None
        for nq in NQS:
            try:
                re, lo, wall = sweep20.train_lbfgs(d, L_FIXED, SEED, n_q=nq)
            except torch.cuda.OutOfMemoryError:
                # a real capacity wall for the dense N_x^d-grid cubic projection,
                # not a failure of the experiment -- record it and move on
                print(f"{d:4d} {nq:5d}  OOM (dense {sweep20.N_X}^{d} grid x {nq} "
                      f"retained for backward)", flush=True)
                torch.cuda.empty_cache()
                break
            delta = "" if prev is None else f"   (Δ vs prev: {re - prev:+.4f})"
            print(f"{d:4d} {nq:5d}  {re:.4f}  {lo:.3e}  {wall:8.1f}s{delta}",
                  flush=True)
            rows.append((d, nq, re, lo))
            prev = re
            if sweep20.DEVICE == "cuda":
                torch.cuda.empty_cache()
        print()

    arr = np.array(rows)
    print("=== is the published N_q=16 number resolved? ===")
    for d in DS:
        m = arr[arr[:, 0] == d]
        if len(m) < 2:
            continue
        r_first, r_last = m[0, 2], m[-1, 2]
        nq_last = int(m[-1, 1])
        # movement over the LAST doubling vs total movement from N_q=16
        tail = abs(m[-1, 2] - m[-2, 2])
        total = abs(r_last - r_first)
        status = "converged" if tail < 0.1 * max(total, 1e-12) else "NOT converged"
        print(f"  d={d}: N_q=16 -> {r_first:.4f}, N_q={nq_last} -> {r_last:.4f} "
              f"(total Δ {r_last - r_first:+.4f}, last doubling {tail:.4f}) -> {status}")

    out = os.path.join(HERE, "..", "results", "25_nq_convergence.npz")
    np.savez(out, rows=arr)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
