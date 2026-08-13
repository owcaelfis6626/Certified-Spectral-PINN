"""Finish the dealiased Adam sweep at d=4 by running seeds concurrently.

WHY THIS IS SAFE (bit-identical, not merely equivalent)
------------------------------------------------------
`16_ac_sweep.train_one(d, L, seed)` is a pure function of its arguments:

  * the model is `LookupGalerkinPINN(oracle)` with the default `init_scale=0.0`
    (pinn_lookup.py:26), so `init_scale * torch.randn(...)` is EXACTLY ZERO -- the
    parameters do not consume the RNG stream at all;
  * every random draw lives in the oracle, and `ACOracleHalfPeriod.__init__` calls
    `torch.manual_seed(seed)` immediately before the first `randn`
    (oracle_halfperiod.py:55), with `seed = 42 + seed`.

So there is no cross-seed RNG coupling: seed 2's result does not depend on seeds 0 and 1
having run first in the same process. Splitting the seeds across processes reproduces the
sequential numbers bit for bit. That is the whole justification -- if either fact were
false this script would silently change the paper's numbers.

WHY A POOL AND NOT JUST "RUN 3 AT ONCE"
---------------------------------------
The q-loop in `loss_rv_ac` retains ~N_q autograd intermediates of size N_x^d, so VRAM
scales with N_q = 2L. At d=4 (N_x^4 = 1,048,576) that is roughly

    L=32  (N_q=64)  ~2.6 GB      L=64 (N_q=128) ~4.2 GB      L=128 (N_q=256) ~7.4 GB

per worker. Three L=128 seeds at once would want ~22 GB on a 16 GB card. A flat width-3
policy would therefore OOM on exactly the cell that costs the most. Instead: a budgeted
pool, longest-job-first, so the 5.4 h L=128 seeds start immediately and the cheap cells
backfill the leftover VRAM alongside them.

CPU, NOT VRAM, IS LIKELY THE REAL CEILING. Each worker is dispatch-bound at ~2.4 cores
(13% GPU utilisation, 38 W of a ~300 W card -- the cost is kernel launches, not FLOPs), and
the box has 6 cores with a chess SPRT holding ~2. MAX_WORKERS is therefore 3, and the
honest expected speedup is ~2x, not 3x -- rising once the SPRT drains.

Appends d=4 rows to 31_ac_adam_dealiased.npz in its existing schema
(d, J, L, N_q, mean, std) so the two runs form one dataset.
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NPZ = os.path.join(HERE, "..", "results", "31_ac_adam_dealiased.npz")
SCRATCH = os.path.join(HERE, "..", "results", "_d4_seeds")
PY = sys.executable

D = 4
LS = [32, 64, 128]
SEEDS = [0, 1, 2]
EST_GB = {32: 2.6, 64: 4.2, 128: 7.4}
BUDGET_GB = 13.5          # of 16, leaving headroom for the desktop + fragmentation
MAX_WORKERS = 3           # CPU-bound at ~2.4 cores each on a 6-core box


def load_sweep():
    spec = importlib.util.spec_from_file_location(
        "ac_sweep", os.path.join(HERE, "16_ac_sweep.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def worker(d, L, seed, out):
    ac = load_sweep()
    ac.N_Q = 2 * L
    import torch
    if ac.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    re, loss, wall = ac.train_one(d, L, seed)
    # Report the real peak so EST_GB stops being a guess -- the L=128 budget decision
    # (can two 7.4 GB jobs share a 16 GB card?) turns on a number nobody has measured.
    peak = (torch.cuda.max_memory_allocated() / 2**30) if ac.DEVICE == "cuda" else 0.0
    json.dump(dict(d=d, L=L, seed=seed, rel_err=float(re), loss=float(loss),
                   wall_s=float(wall), peak_gb=round(peak, 2)), open(out, "w"))


def wait_for_d3(timeout_h=6.0):
    """Let the in-flight d=3 L=128 cell bank, then stop the sequential sweep."""
    t0 = time.time()
    while True:
        n = len(np.load(NPZ)["rows"]) if os.path.exists(NPZ) else 0
        alive = subprocess.run("pgrep -f 31_ac_adam_dealiased.py",
                               shell=True, capture_output=True).returncode == 0
        if n >= 6:
            print(f"[{time.strftime('%H:%M:%S')}] d=3 banked ({n} rows); stopping the "
                  f"sequential sweep before it starts d=4", flush=True)
            subprocess.run("pkill -f 31_ac_adam_dealiased.py", shell=True)
            time.sleep(5)
            return True
        if not alive:
            print(f"sequential sweep gone with only {n} rows -- proceeding anyway", flush=True)
            return n >= 6
        if time.time() - t0 > timeout_h * 3600:
            print("timed out waiting for d=3; leaving everything alone", flush=True)
            return False
        time.sleep(60)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    if not wait_for_d3():
        print("not starting the d=4 pool", flush=True)
        return

    max_workers = MAX_WORKERS   # local: the OOM path lowers it for the rest of the run
    ac = load_sweep()
    from sampler import half_space_modes
    J = len(half_space_modes(D, ac.K_MAX, exclude_zero=True))

    # longest first: the L=128 seeds are ~5.4 h each and must not start last
    todo = [(L, s) for L in sorted(LS, reverse=True) for s in SEEDS]
    running = []            # (proc, L, seed, path, t0)
    done = {}               # (L, seed) -> rel_err
    used = 0.0

    print(f"[{time.strftime('%H:%M:%S')}] d=4 pool: {len(todo)} jobs, "
          f"budget {BUDGET_GB} GB, max {max_workers} workers", flush=True)

    while todo or running:
        while todo and len(running) < max_workers:
            # Backfill, don't stall. Longest-first puts three 7.4 GB L=128 jobs at the head,
            # so testing only todo[0] would find it doesn't fit beside the running one and
            # give up -- leaving the pool at a single worker for the entire expensive phase,
            # i.e. no speedup at all. Scan for the first job that fits in the free budget.
            idx = next((i for i, (LL, _) in enumerate(todo)
                        if used + EST_GB[LL] <= BUDGET_GB), None)
            if idx is None:
                if running:
                    break               # wait for VRAM to free up
                idx = 0                 # nothing fits and nothing is running: run it alone
            L, s = todo.pop(idx)
            path = os.path.join(SCRATCH, f"d{D}_L{L}_s{s}.json")
            cmd = [PY, os.path.abspath(__file__), "--worker",
                   "--d", str(D), "--L", str(L), "--seed", str(s), "--out", path]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=open(path + ".err", "w"))
            running.append((p, L, s, path, time.time()))
            used += EST_GB[L]
            print(f"[{time.strftime('%H:%M:%S')}]   launch L={L} seed={s} "
                  f"(~{EST_GB[L]} GB, {used:.1f} GB in flight)", flush=True)

        time.sleep(20)

        for r in list(running):
            p, L, s, path, t0 = r
            if p.poll() is None:
                continue
            running.remove(r)
            used -= EST_GB[L]
            if p.returncode == 0 and os.path.exists(path):
                rec = json.load(open(path))
                done[(L, s)] = rec["rel_err"]
                print(f"[{time.strftime('%H:%M:%S')}]   done  L={L} seed={s}  "
                      f"rel_err={rec['rel_err']:.4f}  peak={rec.get('peak_gb',0):.1f}GB  ({(time.time()-t0)/3600:.2f} h)",
                      flush=True)
            else:
                tail = open(path + ".err").read()[-200:] if os.path.exists(path + ".err") else ""
                print(f"[{time.strftime('%H:%M:%S')}]   FAILED L={L} seed={s} "
                      f"rc={p.returncode} :: {tail}", flush=True)
                if "out of memory" in tail.lower():
                    # My VRAM estimate was wrong for this cell. Requeue it and drop to
                    # serial for the rest of the run rather than guessing a new number --
                    # a second OOM would cost hours, and correctness beats throughput here.
                    print("      -> OOM; requeueing and dropping the pool to 1 worker",
                          flush=True)
                    todo.insert(0, (L, s))
                    max_workers = 1

    # append complete cells to the shared npz
    rows = list(np.load(NPZ)["rows"]) if os.path.exists(NPZ) else []
    for L in LS:
        vals = [done[(L, s)] for s in SEEDS if (L, s) in done]
        if len(vals) != len(SEEDS):
            print(f"L={L}: only {len(vals)}/{len(SEEDS)} seeds -- NOT recording", flush=True)
            continue
        a = np.array(vals)
        rows.append((D, J, L, 2 * L, a.mean(), a.std()))
        print(f"d={D} L={L} N_q={2*L}  {a.mean():.4f} +- {a.std():.4f}", flush=True)
    arr = np.array(rows)
    np.savez(NPZ, rows=arr)

    print("\n=== rate  rel_err ~ L^s  (Adam, dealiased, N_q = 2L) ===", flush=True)
    print("  paper's aliased+hand-halved: d=3 +0.45, d=4 +0.64", flush=True)
    for d in sorted({int(v) for v in arr[:, 0]}):
        m = arr[:, 0] == d
        Ls, e = arr[m, 2], arr[m, 4]
        if len(Ls) < 2:
            continue
        x, y = np.log(Ls), np.log(e)
        s, ic = np.polyfit(x, y, 1)
        n = len(x)
        resid = y - (s * x + ic)
        se = (math.sqrt((resid ** 2).sum() / (n - 2) / ((x - x.mean()) ** 2).sum())
              if n > 2 else float("nan"))
        cells = "  ".join(f"L={int(v)}:{q:.4f}" for v, q in zip(Ls, e))
        print(f"  d={d}: {cells}   slope {s:+.3f} +/- {se:.3f}", flush=True)
    print(f"\nsaved: {NPZ}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--d", type=int)
    ap.add_argument("--L", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        worker(a.d, a.L, a.seed, a.out)
    else:
        main()
