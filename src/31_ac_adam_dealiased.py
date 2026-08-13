"""Re-measure the Adam divergence with the cubic term dealiased in time.

WHY THIS EXISTS
---------------
The paper's Adam claim ("converges at d=2, diverges for d>=3; fitted rel_err slope
+0.45 at d=3 and +0.64 at d=4") rests on results/16_ac_sweep.{log,npz}, and those
artifacts have two problems that are invisible in the paper:

  1. They were produced 2026-06-09, a month BEFORE the 2026-07-17 sqrt fix in
     16_ac_sweep.rel_err_at_coeffs. The stored column is therefore rel_err SQUARED,
     and the published slopes are the stored slopes HALVED by hand (verified:
     +0.8961/2 = +0.4480 -> "+0.45", +1.2923/2 = +0.6462 -> "+0.64", and for the
     N_q=16 remark -0.7102/2 = -0.3551 -> "-0.36"). The arithmetic is correct, but
     nothing in the repo records the correction, so no number in that paragraph can
     be reproduced from the artifacts without knowing to halve it. The script was
     fixed and never rerun.

  2. They were run at N_Q = 16 -- chosen, per its own comment, "to limit VRAM for
     d=4" -- which is exactly the aliased regime the paper's own dealiasing
     paragraph and Remark (rem:aliasing) identify as producing artifact rates. The
     cubic projection needs N_q >~ 2L in TIME, so L=128 at N_q=16 carries ~270%
     quadrature error. Leaving an aliased rate in the same document that warns
     about silent aliasing is the obvious referee target.

WHAT THIS CHANGES
-----------------
Nothing but the quadrature: same optimiser, same LR schedule, same steps, same
seeds, same oracle. N_q is set to 2L per cell, matching the dealiased convention
used by 30_campaign.py for every L-BFGS number in the paper. The comparison the
paper actually rests on -- Adam vs L-BFGS at fixed everything-else -- is preserved,
and both sides are now measured where the quadrature resolves the cubic term.

Writes to results/31_ac_adam_dealiased.{log,npz}. Does NOT overwrite the 16_*
artifacts: they are the provenance of published numbers and stay as they are.
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

spec = importlib.util.spec_from_file_location("ac_sweep", os.path.join(HERE, "16_ac_sweep.py"))
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

D_VALS = ac.D_VALS      # [2, 3, 4]
LS = ac.LS              # [32, 64, 128]
SEEDS = ac.SEEDS        # [0, 1, 2]
OUT = os.path.join(HERE, "..", "results", "31_ac_adam_dealiased")


def main():
    print(f"P7 Allen-Cahn Adam sweep, DEALIASED  K_max={ac.K_MAX}  N_q=2L  "
          f"N_x={ac.N_X}  N_steps={ac.N_STEPS}  device={ac.DEVICE}", flush=True)
    print("  (16_ac_sweep ran this at a fixed N_q=16; every other difference is held)\n",
          flush=True)
    print(f"{'d':>4} {'J':>6} {'L':>5} {'N_q':>5} {'rel_err mean±std':>22} {'wall':>9}",
          flush=True)

    from sampler import half_space_modes

    rows = []
    for d in D_VALS:
        J = len(half_space_modes(d, ac.K_MAX, exclude_zero=True))
        for L in LS:
            ac.N_Q = 2 * L          # the only knob this script moves
            re_seeds = []
            t0 = time.time()
            for seed in SEEDS:
                re, _, _ = ac.train_one(d, L, seed)
                re_seeds.append(re)
                if ac.DEVICE == "cuda":
                    torch.cuda.empty_cache()
            arr = np.array(re_seeds)
            print(f"{d:4d} {J:6d} {L:5d} {2*L:5d}  {arr.mean():.4f} ± {arr.std():.4f}"
                  f"  {time.time()-t0:8.1f}s", flush=True)
            rows.append((d, J, L, 2 * L, arr.mean(), arr.std()))
            np.savez(OUT + ".npz", rows=np.array(rows))   # checkpoint every cell

    arr = np.array(rows)
    print("\n=== rate  rel_err ~ L^s  (Adam, dealiased) ===", flush=True)
    print("  compare against the paper's aliased, hand-halved +0.45 (d=3) / +0.64 (d=4)",
          flush=True)
    for d in D_VALS:
        m = arr[:, 0] == d
        Ls, res = arr[m, 2], arr[m, 4]
        if len(Ls) >= 2:
            n = len(Ls)
            x, y = np.log(Ls), np.log(res)
            s, ic = np.polyfit(x, y, 1)
            resid = y - (s * x + ic)
            se = math.sqrt((resid ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum()) \
                if n > 2 else float("nan")
            print(f"  d={int(d)}: rel_err ~ L^({s:+.3f} ± {se:.3f})", flush=True)

    np.savez(OUT + ".npz", rows=arr)
    print(f"\nsaved: {OUT}.npz", flush=True)


if __name__ == "__main__":
    main()
