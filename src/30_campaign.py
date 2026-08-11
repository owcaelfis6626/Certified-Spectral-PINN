"""14-day unattended dealiasing campaign: the definitive AC convergence table.

WHY THIS EXISTS
---------------
Script 26 measured that the published AC table minimised a loss whose cubic term
carried ~270% relative quadrature error at L=128, because N_q was pinned at 16
while L was swept to 128. losses_ac.py applies the cubic dealiasing rule in
SPACE (N_x > 6*K_max) but the temporal analogue (N_q ~ 2L) was never written
down. The error worsens with L -- i.e. along exactly the axis the convergence
study sweeps -- so the published AC rates, and the "the AC rate degrades with d"
limitation built on them, are measuring an artifact.

Partial dealiased evidence so far (seed 0): d=2 slope -0.544 (vs published
-0.361), and d=2/d=3 at L=32 land on top of each other (0.3327 vs 0.3309) where
the published numbers were clearly separated (0.3132 vs 0.3733). That is the
uniform-in-d signature showing up precisely where the paper claims d-dependence.
This campaign settles it.

The linear/uniform-in-d headline is NOT affected: the linear residual is exact
Galerkin (R_a = Ma + k2_1*alpha), no quadrature involved.

DESIGN FOR UNATTENDED OPERATION
-------------------------------
Anytime + resumable. Every completed cell is appended to a JSONL ledger the
moment it finishes; a restart skips whatever is already there. Cells are ordered
by information gain, not by a naive nested loop, so any prefix of the run is a
coherent result: phase 1 alone is the paper correction, phase 2 adds error bars,
phases 3-5 are extensions. Per-cell try/except plus a wall-clock deadline means
one OOM, NaN, or pathological cell cannot consume the window.

Usage:
  python 30_campaign.py --smoke     # cheapest cells only, validates plumbing
  python 30_campaign.py             # the campaign
  python 30_campaign.py --plan      # print the queue and exit, runs nothing
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "..", "results")

from losses_ac import loss_rv_ac
from losses_ac_ckpt import loss_rv_ac_ckpt
from oracle_ac import ACOracleHalfPeriod
from pinn_lookup import LookupGalerkinPINN
from sampler import half_space_modes

LEDGER = os.path.join(RESULTS, "30_campaign.jsonl")
STATUS = os.path.join(RESULTS, "30_campaign_status.txt")

K_MAX, SIGMA, T, N_FINE = 3, 1.0, 1.0, 4096
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LBFGS_OUTER, LBFGS_MAXITER, LBFGS_TOL = 150, 20, 1e-12

DEALIAS_FACTOR = 2          # N_q = 2L, the requirement verified in script 26
CKPT_BYTES_LIMIT = 4 << 30      # switch to checkpointing above ~4 GB retained
CKPT_CHUNK_BYTES = 5 << 29      # ~2.5 GB working set per checkpointed chunk.
# Sized from script 29's measurement: d=5, N_x=32, chunk=2 peaked at 1.79 GiB,
# so ~2.5 GB of retained chunk leaves ample headroom on a 16 GB card while
# keeping chunk > 1 wherever possible (chunk=1 is maximum recompute overhead).
DEFAULT_CELL_HOURS = 24.0

# published N_q=16 baseline (3 seeds), for side-by-side reporting only
PUBLISHED = {
    2: {32: 0.3132, 64: 0.2370, 128: 0.1898, "slope": -0.361},
    3: {32: 0.3733, 64: 0.3202, 128: 0.2736, "slope": -0.224},
    4: {32: 0.5132, 64: 0.4820, 128: 0.4587, "slope": -0.081},
    5: {32: 0.7710, 64: 0.7869, 128: 0.7873, "slope": +0.015},
}


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------

def cell_key(c):
    return (c["phase"], c["d"], c["L"], c["seed"], c["N_q"], c["N_x"])


def load_ledger():
    if not os.path.exists(LEDGER):
        return {}
    done = {}
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # tolerate a torn final line from a hard kill
            done[cell_key(r)] = r
    return done


def append_ledger(rec):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())      # survive a power cut, not just a crash


# --------------------------------------------------------------------------
# memory policy: pick the loss path from the retained-activation estimate
# --------------------------------------------------------------------------

def memory_plan(d, N_q, N_x):
    """cubic_proj_fft keeps ~3 arrays of N_x^d complex64 alive per quadrature
    step; loss_rv_ac retains all N_q of them for backward."""
    per_step = 3 * (N_x ** d) * 8
    if per_step * N_q <= CKPT_BYTES_LIMIT:
        return False, 0
    chunk = max(1, int(CKPT_CHUNK_BYTES // per_step))
    return True, min(chunk, N_q)


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

@torch.no_grad()
def rel_err_at_coeffs(alpha, beta, oracle, N_t_eval=64):
    """Relative error in the X-norm. Mirrors 20_ac_lbfgs_sweep.rel_err_at_coeffs
    exactly (including its 2026-07-17 sqrt fix) so numbers stay comparable."""
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                            device=oracle.device, dtype=oracle.dtype)
    dt = oracle.T / (N_t_eval - 1)
    w = torch.ones(N_t_eval, device=oracle.device, dtype=oracle.dtype)
    w[0] = 0.5
    w[-1] = 0.5

    lam = (oracle.k2 - 1.0)[None, :]
    exp_decay = torch.exp(-lam * t_eval[:, None])              # (N_t, J)
    phase = oracle.omega_l[None, :] * t_eval[:, None]          # (N_t, L)
    phi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)         # (N_t, L)

    a_th = exp_decay * oracle.a0[None, :] + phi @ alpha.T      # (N_t, J)
    b_th = exp_decay * oracle.b0[None, :] + phi @ beta.T
    a_ref, b_ref = oracle.a_ref(t_eval)

    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    e_a, e_b = a_th - a_ref, b_th - b_ref
    err_Y = (w[:, None] * one_plus_k2 * (e_a ** 2 + e_b ** 2)).sum().item() * dt
    ref_Y = (w[:, None] * one_plus_k2 * (a_ref ** 2 + b_ref ** 2)).sum().item() * dt
    # err_Y is the ABSOLUTE squared X-norm error, which is the quantity the
    # certificate bounds (||u-u_ref||^2_X <= C_1 L_RV + C_2/L); rel_err alone
    # cannot be fed back into the bound, so both are returned.
    return math.sqrt(err_Y / ref_Y), err_Y, ref_Y


def run_cell(c):
    """Train one (d, L, seed, N_q, N_x) cell. Returns a ledger record."""
    d, L, seed, n_q, N_x = c["d"], c["L"], c["seed"], c["N_q"], c["N_x"]
    deadline = time.time() + c["max_hours"] * 3600.0
    use_ckpt, chunk = memory_plan(d, n_q, N_x)

    rec = dict(c)
    rec.update(ckpt=use_ckpt, chunk=chunk, device=DEVICE,
               J=len(half_space_modes(d, K_MAX, exclude_zero=True)))

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    oracle = ACOracleHalfPeriod(
        d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L,
        N_fine=N_FINE, N_x=N_x, seed=42 + seed, device=DEVICE, dtype=DTYPE)
    rec["oracle_s"] = round(time.time() - t0, 1)

    model = LookupGalerkinPINN(oracle)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=LBFGS_MAXITER, history_size=50,
        line_search_fn="strong_wolfe", tolerance_grad=1e-16,
        tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        loss = (loss_rv_ac_ckpt(model, oracle, N_q=n_q, N_x=N_x, chunk=chunk)
                if use_ckpt else loss_rv_ac(model, oracle, N_q=n_q, N_x=N_x))
        loss.backward()
        return loss

    status, prev, n_outer, loss = "ok", None, 0, float("nan")
    for it in range(LBFGS_OUTER):
        loss = opt.step(closure).item()
        n_outer = it + 1
        if not math.isfinite(loss):
            status = "nan"
            break
        if prev is not None and abs(prev - loss) <= LBFGS_TOL * max(1.0, prev):
            status = "converged"
            break
        prev = loss
        if time.time() > deadline:
            # under-converged: recorded, but excluded from every rate fit
            status = "timeout"
            break

    alpha, beta = model.coefficients(oracle)
    re, err_Y, ref_Y = rel_err_at_coeffs(alpha, beta, oracle)
    rec.update(
        rel_err=re, err_X2=err_Y, ref_X2=ref_Y,
        loss=loss, n_outer=n_outer, status=status,
        wall_s=round(time.time() - t0, 1),
        peak_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        if DEVICE == "cuda" else 0.0,
        finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))

    del oracle, model, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------------
# the queue: ordered by information gain, so any prefix is a coherent result
# --------------------------------------------------------------------------

def build_queue(smoke=False):
    q = []

    def add(phase, d, L, seed, N_x=32, hours=DEFAULT_CELL_HOURS, note=""):
        q.append(dict(phase=phase, d=d, L=L, seed=seed,
                      N_q=DEALIAS_FACTOR * L, N_x=N_x,
                      max_hours=hours, note=note))

    if smoke:
        # cheapest possible cells that still exercise both loss paths and the
        # reduced-N_x path. Must complete in minutes.
        add("smoke", 2, 32, 0, note="non-ckpt path")
        add("smoke", 5, 32, 0, note="ckpt path")
        add("smoke", 4, 32, 0, N_x=20, note="reduced N_x path")
        return q

    # P1 -- the paper correction: dealiased table, seed 0. Highest value.
    #       d=2 and (d=3, L=32) are already in the ledger from the 2026-07-24
    #       run of script 27 and are skipped; they are listed so --plan shows
    #       the whole table.
    for d in (2, 3, 4, 5):
        for L in (32, 64, 128):
            add("P1", d, L, 0, note="dealiased table, seed 0")

    # P2 -- error bars on the same table.
    for seed in (1, 2):
        for d in (2, 3, 4, 5):
            for L in (32, 64, 128):
                add("P2", d, L, seed, note="error bars")

    # P3 -- a 4th L per d. Three points is a weak slope; four over a decade of
    #       L is what makes "the rate is L^(-1/2), uniformly in d" defensible.
    for d in (2, 3, 4, 5):
        add("P3", d, 256, 0, note="4th L point")

    # P4 -- N_x invariance. Once N_x clears the spatial dealiasing floor
    #       (N_x > 6*K_max = 18) the answer must not depend on it. Proving that
    #       at d=4 is what licenses using N_x=20 to reach d=6 in P5.
    for L in (32, 64, 128):
        add("P4", 4, L, 0, N_x=20, note="N_x invariance vs N_x=32")

    # P5 -- the frontier. d=6 is unreachable at N_x=32 (32^6 needs 8.6 GB per
    #       complex array); at N_x=20 it is 2.7 GB peak. J=2098.
    for L in (32, 64, 128):
        add("P5", 6, L, 0, N_x=20, hours=36.0, note="d=6 frontier")

    # P6 -- whatever the window still affords.
    for d in (2, 3, 4, 5):
        for seed in (1, 2):
            add("P6", d, 256, seed, note="4th point error bars")
    for d in (2, 3):
        add("P6", d, 512, 0, hours=36.0, note="5th L point")

    # P7/P8 -- overflow. The queue simply stops when it runs out, so
    # over-provisioning the tail costs nothing and keeps the GPU from idling if
    # the earlier phases come in under estimate.
    #   d=7 is deliberately absent: the nonlinear term needs an N_x^d grid and
    #   N_x > 6*K_max forces N_x >= 20, so 20^7 = 1.3e9 points (~10 GB per
    #   complex array) is out of reach. d=6 is the frontier for AC. (The LINEAR
    #   case has no grid at all -- exact Galerkin -- which is how it reached
    #   d=7, J=6217. Different constraint, not a contradiction.)
    for seed in (1, 2):
        for L in (32, 64, 128):
            add("P7", 6, L, seed, N_x=20, hours=36.0, note="d=6 error bars")
    for seed in (3, 4):
        for d in (2, 3, 4, 5):
            for L in (32, 64, 128):
                add("P8", d, L, seed, note="extra seeds")

    return q


# --------------------------------------------------------------------------
# status report
# --------------------------------------------------------------------------

def fit_slope(rows):
    """Log-log rate fit over DISTINCT L. Under-converged cells are excluded and
    never averaged in; smoke cells are plumbing checks, not measurements; and
    repeats of the same L (seeds, or a smoke/P1 duplicate) are averaged rather
    than entered as independent points, which would bias the fit toward
    whichever L happens to have been run twice."""
    good = [r for r in rows if r["status"] in ("ok", "converged")
            and r["phase"] != "smoke"
            and math.isfinite(r["rel_err"]) and r["rel_err"] > 0]
    by_L = defaultdict(list)
    for r in good:
        by_L[r["L"]].append(r["rel_err"])
    if len(by_L) < 2:
        return None, len(by_L)
    Ls = np.array(sorted(by_L), dtype=float)
    re = np.array([np.mean(by_L[L]) for L in sorted(by_L)], dtype=float)
    return float(np.polyfit(np.log(Ls), np.log(re), 1)[0]), len(by_L)


def write_status(done, queue, t_start):
    lines = []
    w = lines.append
    w("P7 AC DEALIASED CAMPAIGN  (N_q = 2L)")
    w(f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}   "
      f"uptime {(time.time() - t_start) / 3600:.1f} h")
    remaining = [c for c in queue if cell_key(c) not in done]
    w(f"cells done {len(done)}   remaining {len(remaining)}")

    spent = sum(r.get("wall_s", 0) for r in done.values()) / 3600.0
    w(f"GPU-hours logged {spent:.1f}")
    if remaining:
        w(f"next up: {remaining[0]['phase']} d={remaining[0]['d']} "
          f"L={remaining[0]['L']} seed={remaining[0]['seed']} "
          f"N_x={remaining[0]['N_x']}")
    else:
        w("QUEUE COMPLETE")

    for phase in ("smoke", "P1", "P2", "P3", "P4", "P5", "P6"):
        rows = [r for r in done.values() if r["phase"] == phase]
        if not rows:
            continue
        note = rows[0].get("note", "")
        w("")
        w(f"--- {phase}  ({note}) ---")
        w(f"{'d':>3} {'L':>5} {'sd':>3} {'N_q':>5} {'N_x':>4} {'rel_err':>9} "
          f"{'published':>10} {'status':>10} {'wall':>9}")
        for r in sorted(rows, key=lambda r: (r["d"], r["N_x"], r["seed"], r["L"])):
            pub = PUBLISHED.get(r["d"], {}).get(r["L"])
            pub_s = f"{pub:.4f}" if pub else "-"
            w(f"{r['d']:3d} {r['L']:5d} {r['seed']:3d} {r['N_q']:5d} "
              f"{r['N_x']:4d} {r['rel_err']:9.4f} {pub_s:>10} "
              f"{r['status']:>10} {r['wall_s'] / 60:8.1f}m")

    # rates, seed 0, N_x=32 -- the headline
    w("")
    w("=== rate  rel_err ~ L^s   (seed 0, N_x=32, converged cells only) ===")
    for d in (2, 3, 4, 5, 6):
        for N_x in (32, 20):
            rows = [r for r in done.values() if r["d"] == d and r["seed"] == 0
                    and r["N_x"] == N_x]
            if len(rows) < 2:
                continue
            s, n = fit_slope(rows)
            if s is None:
                continue
            pub = PUBLISHED.get(d, {}).get("slope")
            pub_s = f"{pub:+.3f}" if pub is not None else "  n/a"
            w(f"  d={d} N_x={N_x}: dealiased {s:+.3f}  ({n} pts)   "
              f"published(N_q=16) {pub_s}")
    w("")
    w("  reference: L^(-0.5) is the KL floor -- the optimal rate the linear")
    w("  case attains. A dealiased slope near -0.5 that does not degrade with d")
    w("  means the published d-dependence was the quadrature artifact.")

    with open(STATUS, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    # 2026-08-08: a driver update mid-session left the NVML userspace (610.57) ahead of the
    # running kernel module (610.43.03), so NEW processes get CUDA error 803 while already-running
    # ones keep their context. Without this guard a restart would silently fall back to
    # DEVICE="cpu", run cells orders of magnitude slower, and write them into the ledger as
    # perfectly valid results. Refuse instead: a missing GPU is an operator problem, not a
    # measurement to record.
    if not torch.cuda.is_available():
        sys.exit("REFUSING TO RUN: CUDA unavailable (driver/library mismatch?). "
                 "CPU cells would pollute the ledger. Fix the driver (reboot) and rerun.")

    queue = build_queue(smoke=args.smoke)
    done = load_ledger()
    t_start = time.time()

    if args.plan:
        print(f"{len(queue)} cells, {sum(1 for c in queue if cell_key(c) in done)}"
              f" already in ledger")
        for c in queue:
            mark = "DONE" if cell_key(c) in done else "    "
            ck, chunk = memory_plan(c["d"], c["N_q"], c["N_x"])
            print(f"  {mark} {c['phase']:>5} d={c['d']} L={c['L']:>3} "
                  f"seed={c['seed']} N_q={c['N_q']:>3} N_x={c['N_x']} "
                  f"{'ckpt(%d)' % chunk if ck else 'direct':>8}  {c['note']}")
        return

    print(f"campaign start {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"{len(queue)} cells, {len(done)} already done", flush=True)
    write_status(done, queue, t_start)

    for c in queue:
        k = cell_key(c)
        if k in done:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {c['phase']} d={c['d']} "
              f"L={c['L']} seed={c['seed']} N_q={c['N_q']} N_x={c['N_x']}",
              flush=True)
        try:
            rec = run_cell(c)
        except torch.cuda.OutOfMemoryError as e:
            rec = dict(c, status="oom", rel_err=float("nan"), loss=float("nan"),
                       wall_s=0.0, err=str(e)[:200],
                       finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            torch.cuda.empty_cache()
        except Exception as e:
            # one bad cell must not end a 14-day run
            traceback.print_exc()
            rec = dict(c, status="error", rel_err=float("nan"),
                       loss=float("nan"), wall_s=0.0,
                       err=f"{type(e).__name__}: {str(e)[:200]}",
                       finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        append_ledger(rec)
        done[k] = rec
        print(f"    -> rel_err={rec['rel_err']:.4f} status={rec['status']} "
              f"wall={rec['wall_s'] / 60:.1f}m", flush=True)
        write_status(done, queue, t_start)

    print("CAMPAIGN COMPLETE", flush=True)
    write_status(done, queue, t_start)


if __name__ == "__main__":
    main()
