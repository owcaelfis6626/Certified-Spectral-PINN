"""Does Adam's seed-variance blowup track parameter count, or dimension?

THE QUESTION
------------
The dealiased d=4 sweep (31/32) found Adam's seed spread exploding with L at FIXED J=212:

    d=4 L=32   0.9 %      L=64  14.7 %      L=128  47.2 %

`discussion.tex:62` attributes Adam's failure to "the increasingly anisotropic cross-mode
Hessian as J GROWS". J is constant across those three cells, so whatever varies when Adam
breaks, it is not the mode count. The obvious alternative is the coefficient-table size
J*L -- LookupGalerkinPINN's actual parameter count.

THAT ALTERNATIVE IS ALREADY IN TROUBLE, and this script exists to settle it rather than
assume it. Sorting every completed cell by J*L does not order the spread:

     448 params  d=2 L=32    8.6 %          6,784 params  d=4 L=32    0.9 %
   1,792 params  d=2 L=128   9.2 %          7,808 params  d=3 L=128   8.1 %

d=4 L=32 has 15x the parameters of d=2 L=32 and one tenth the spread. So parameter count
alone does not explain it either, and the honest reading of the existing data is an
INTERACTION: spread grows with L within each d, but only becomes severe at d=4.

THE DISCRIMINATOR
-----------------
Push d=2 -- the dimension the paper calls safe -- to parameter counts that match and exceed
the d=4 cells where Adam broke:

    d=2 L=256   J*L =  3,584        d=2 L=1024  J*L = 14,336  ~ d=4 L=64  (13,568, 14.7 %)
    d=2 L=512   J*L =  7,168

  * If spread blows up at d=2 L=1024, the driver is parameter count and the paper's
    J-based mechanism sentence is simply wrong.
  * If d=2 stays tight at 14,336 parameters while d=4 fell apart at 13,568, then parameter
    count is NOT the driver, my J*L hypothesis is dead, and the mechanism is genuinely
    dimensional -- which would support the paper's instinct even though its numbers were
    aliasing artifacts.

Either outcome is publishable and either kills one of the two candidate explanations. The
ladder (256, 512, 1024) rather than a single point at 1024 because a lone cell cannot show
whether spread GROWS -- and growth with L is the entire phenomenon being tested.

COST
----
Cost is dominated by the N_q = 2L Python loop in loss_rv_ac (the run is dispatch-bound, not
FLOP-bound: 13 % GPU utilisation at 38 W), so it scales with L, roughly independent of d.
Expect ~1.5 h, ~3 h, ~6.5 h per seed. d=2 tensors are tiny (N_x^2 = 1024), so all three
seeds of a cell fit in VRAM together and run concurrently; the ceiling is CPU.

Seeds are split across processes, which is bit-identical to running them sequentially --
LookupGalerkinPINN uses init_scale=0.0 so parameters never touch the RNG, and the oracle
calls torch.manual_seed(42+seed) before its first draw. Verified empirically at d=2 L=32:
mean and std matched the sequential npz to 12 decimal places, delta exactly 0.0.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "results", "33_ac_adam_d2_ladder")
SCRATCH = OUT + "_seeds"
PY = sys.executable

D = 2
LS = [256, 512, 1024]
SEEDS = [0, 1, 2]
MAX_WORKERS = 3           # CPU-bound at ~2.4 cores each; a chess SPRT holds ~2 of 6


def load_sweep():
    spec = importlib.util.spec_from_file_location(
        "ac_sweep", os.path.join(HERE, "16_ac_sweep.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def worker(L, seed, out):
    ac = load_sweep()
    ac.N_Q = 2 * L
    import torch
    if ac.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    re, loss, wall = ac.train_one(D, L, seed)
    peak = (torch.cuda.max_memory_allocated() / 2**30) if ac.DEVICE == "cuda" else 0.0
    json.dump(dict(d=D, L=L, seed=seed, rel_err=float(re), loss=float(loss),
                   wall_s=float(wall), peak_gb=round(peak, 2)), open(out, "w"))


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    ac = load_sweep()
    from sampler import half_space_modes
    J = len(half_space_modes(D, ac.K_MAX, exclude_zero=True))
    print(f"d={D} ladder (J={J}), N_q=2L, {ac.N_STEPS} Adam steps, seeds {SEEDS}", flush=True)
    print(f"reference: d=4 L=64 has J*L=13,568 and 14.7% seed spread\n", flush=True)

    rows = []
    for L in LS:                       # cheapest first: a cheap cell that already blows up
        procs = []                     # would answer the question without paying for L=1024
        for s in SEEDS:
            path = os.path.join(SCRATCH, f"L{L}_s{s}.json")
            p = subprocess.Popen(
                [PY, os.path.abspath(__file__), "--worker", "--L", str(L),
                 "--seed", str(s), "--out", path],
                stdout=subprocess.DEVNULL, stderr=open(path + ".err", "w"))
            procs.append((p, s, path))
            print(f"[{time.strftime('%H:%M:%S')}]   launch L={L} seed={s} "
                  f"(J*L={J*L:,})", flush=True)
            while sum(1 for q, _, _ in procs if q.poll() is None) >= MAX_WORKERS:
                time.sleep(20)

        vals = []
        for p, s, path in procs:
            p.wait()
            if p.returncode == 0 and os.path.exists(path):
                vals.append(json.load(open(path)))
            else:
                tail = open(path + ".err").read()[-200:] if os.path.exists(path + ".err") else ""
                print(f"   FAILED L={L} seed={s} rc={p.returncode} :: {tail}", flush=True)

        if len(vals) != len(SEEDS):
            print(f"L={L}: {len(vals)}/{len(SEEDS)} seeds -- not recording, stopping ladder",
                  flush=True)
            break
        e = np.array([v["rel_err"] for v in vals])
        spread = 100 * e.std() / e.mean()
        rows.append((D, J, L, 2 * L, e.mean(), e.std()))
        np.savez(OUT + ".npz", rows=np.array(rows))
        print(f"[{time.strftime('%H:%M:%S')}] d={D} L={L}  J*L={J*L:,}  "
              f"{e.mean():.4f} +- {e.std():.4f}  ({spread:.1f}% spread)  "
              f"peak={max(v['peak_gb'] for v in vals):.1f}GB  "
              f"seeds {[round(v['rel_err'], 4) for v in vals]}", flush=True)

    print("\n=== verdict ===", flush=True)
    print("  d=4 L=64  (J*L=13,568): 14.7% spread   <- Adam breaking down", flush=True)
    for r in rows:
        print(f"  d=2 L={int(r[2]):<5}(J*L={int(r[1]*r[2]):>6,}): "
              f"{100*r[5]/r[4]:5.1f}% spread", flush=True)
    print("\n  spread blows up at matched J*L  -> parameter count is the driver", flush=True)
    print("  spread stays tight              -> parameter count is NOT; d is", flush=True)
    print(f"\nsaved: {OUT}.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--L", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        worker(a.L, a.seed, a.out)
    else:
        main()
