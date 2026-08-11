# Dealiasing campaign — started 2026-07-25, running unattended

## When you get back, run this

```bash
/home/hubi/spde/venv/bin/python3 \
  /home/hubi/research/useful/papers/paper7_certified_spectral_pinn/src/31_analyze_campaign.py
```

Or just read `results/30_campaign_status.txt` — it's rewritten after every cell.

## What is running and why

`26_cubic_quadrature_error.py` established that the published AC table minimised
a loss whose cubic term carried **~270% relative quadrature error** at L=128.
`losses_ac.py` applies the cubic dealiasing rule in *space* (`N_x > 6·K_max`) but
the temporal analogue (`N_q ≳ 2L`) was never written down, so `N_q` sat at 16
while `L` was swept to 128 — the error grows along exactly the axis the
convergence study sweeps.

The campaign re-measures the whole AC table with `N_q = 2L`.

**The linear / uniform-in-d headline is not affected.** The linear residual is
exact Galerkin (`R_a = Ma + k2_1*alpha`), no quadrature. Only the nonlinear AC
extension and the "AC rate degrades with d" limitation are in question.

## What to look for

Partial evidence before the campaign (seed 0): dealiased d=2 slope **−0.544**
vs published −0.361, and d=2/d=3 at L=32 landing on top of each other (0.3327 vs
0.3309) where published had them clearly separated (0.3132 vs 0.3733).

Because the trained loss reaches `L_RV ~ 1e-9` while the error is O(1e-1), the
`C₂/L` term carries essentially the whole certificate, so:

> `rel_err ~ L^(−1/2)`  ⟺  `‖e‖²_X ~ C₂/L`  ⟺  **the theorem's rate is attained**

A slope near −1/2 that does *not* degrade with d means the published
d-dependence was the aliasing artifact, and the uniform-in-d claim extends from
the linear case to the nonlinear one. That is the outcome to check for — it is
not guaranteed, and a genuine residual d-dependence after dealiasing is an
equally publishable (if less pleasant) answer.

## Queue: 86 cells, priority-ordered

Any prefix is a coherent result — the run does not need to finish to be useful.

| phase | what | why it is at this priority |
|---|---|---|
| P1 | dealiased table, seed 0, d=2..5 | the paper correction |
| P2 | seeds 1,2 | error bars |
| P3 | L=256, seed 0 | 4th point — 3 points is a weak slope |
| P4 | d=4 at N_x=20 vs 32 | invariance check that licenses P5 |
| P5 | **d=6** at N_x=20 | frontier; unreachable at N_x=32 (8.6 GB/array) |
| P6 | L=256 error bars, L=512 | extra reach |
| P7/P8 | d=6 error bars, extra seeds | overflow so the GPU never idles |

d=7 is absent on purpose: the cubic needs an `N_x^d` grid and `N_x > 6·K_max`
forces `N_x ≥ 20`, so `20^7 ≈ 1.3e9` points is out of reach. **d=6 is the AC
frontier.** (The *linear* case has no grid — exact Galerkin — which is how it
reached d=7, J=6217. Different constraint, not a contradiction.)

Estimated ~7 days of GPU for the whole queue, with wide error bars: L-BFGS
iteration counts vary a lot per cell. The priority ordering is what makes the
estimate non-critical.

## Robustness

- **Resumable.** Every cell is appended to `results/30_campaign.jsonl` and
  `fsync`'d immediately. Restarting skips completed cells.
- **Crash-isolated.** OOM / NaN / exception in one cell is logged and the run
  continues. Each cell also has a wall-clock deadline (24 h; 36 h for d=6);
  a cell that hits it is recorded `status=timeout` and **excluded from every
  rate fit** rather than silently contributing an under-converged point.
- **Supervised.** `run_campaign.sh` restarts the driver on any crash, with
  backoff, and refuses to double-start (`flock`).
- **Reboot-proof.** `@reboot` crontab entry restarts it after a power cut.
- **Logout-proof.** `KillUserProcesses=no`, and the job is `setsid`-detached —
  it does not depend on this Claude session staying open.
- **Deterministic.** `LookupGalerkinPINN` inits at exactly zero (`init_scale=0`),
  so a cell depends only on its oracle seed, not on process RNG state.

Verified before launch: the smoke run of d=2/L=32 reproduced `rel_err = 0.3327`
exactly, matching the value script 27 produced on 2026-07-24 — the campaign's
code path is bit-identical to the one behind the existing dealiased numbers.

## Controls

```bash
tail -f  .../results/30_campaign.out                 # live cell log
cat      .../results/30_campaign_status.txt          # table + rates so far
cat      .../results/30_campaign_supervisor.log      # restarts, crashes
python3  .../src/30_campaign.py --plan               # queue + what is done

pkill -f 30_campaign.py                              # stop after current cell
rm .../results/30_campaign.lock                      # then release the lock
```

To drop a phase, edit `build_queue()` in `30_campaign.py` and restart — cells
already in the ledger are never recomputed.
