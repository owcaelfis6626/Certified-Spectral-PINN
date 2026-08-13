"""Is Adam's breakdown at d=2, L=256 a conditioning/precision artifact?

THE OBSERVATION
---------------
Dealiased Adam tracks L-BFGS to three decimals at d=2 for L = 32, 64, 128, then
stops converging at L = 256:

    L        Adam      L-BFGS    ratio
    128     0.1545     0.159     0.97x
    256     0.2645     0.117     2.26x     <- rel_err goes UP from L=128

Two candidate causes. (1) Optimiser: Adam's diagonal preconditioner cannot handle the
Galerkin operator's anisotropy. (2) Precision: the temporal test basis has eigenvalues
mu_l = omega_l^2 ~ l^2, so the operator's condition number grows like L^2; at L = 256
that is ~2.6e5 against float32 eps ~ 1.2e-7, and DTYPE is float32 (16_ac_sweep.py:38).

These are separable: rerun the identical cell in float64. If Adam recovers toward 0.117
the cause is precision; if it stays near 0.26 the cause is the optimiser proper.

THE CONFOUND THIS SCRIPT EXISTS TO AVOID
----------------------------------------
Setting DTYPE = float64 is NOT sufficient. losses_ac.cubic_proj_fft hardcodes
`torch.complex64` (lines 64-66) regardless of the working dtype, so the FFT cubic
projection -- the numerically heaviest step, and the one inside the N_q loop -- would
stay in single precision and the "float64" run would silently be mixed precision. That
would make a null result meaningless: we could not tell whether fp64 failed to help or
whether we never actually ran in fp64.

We therefore monkeypatch a dtype-consistent cubic_proj_fft into losses_ac for this run
only. The paper's source is not modified. The patch maps float32 -> complex64 (bit
identical to the current behaviour) and float64 -> complex128, so it is a strict
generalisation; the fp32 path is unchanged by construction.

Compares against the fp32 seed-0 value 0.2259 from 33_ac_adam_d2_ladder, same seed,
same everything else. One seed only: the effect being tested (0.2645 -> 0.117, a factor
2.3) is an order of magnitude larger than the 10.4% seed spread at this cell, so a
single seed discriminates. Runs niced -- the d=2 ladder has the machine.
"""
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import losses_ac

_orig_cubic = losses_ac.cubic_proj_fft


def cubic_proj_fft_dtype_consistent(a, b, modes_int, N_x, device, dtype):
    """cubic_proj_fft with the complex dtype derived from `dtype` instead of pinned."""
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    d = modes_int.shape[1]
    Nd = N_x ** d
    c = torch.complex(a, -b) / 2
    strides = torch.tensor([N_x ** (d - 1 - i) for i in range(d)],
                           device=device, dtype=torch.int64)
    k_pos = modes_int.to(torch.int64) % N_x
    k_neg = (-modes_int).to(torch.int64) % N_x
    lin_pos = (k_pos * strides).sum(-1)
    lin_neg = (k_neg * strides).sum(-1)
    C_flat = torch.zeros(Nd, dtype=cdtype, device=device)
    C_flat = C_flat.index_add(0, lin_pos, c.to(cdtype))
    C_flat = C_flat.index_add(0, lin_neg, c.conj().to(cdtype))
    u_grid = (torch.fft.ifftn(C_flat.view([N_x] * d)) * Nd).real
    U3 = torch.fft.fftn(u_grid ** 3).reshape(-1)
    u3_k = U3[lin_pos]
    return (2 * u3_k.real / Nd).to(dtype), (-2 * u3_k.imag / Nd).to(dtype)


def main():
    L, SEED, D = 256, 0, 2
    losses_ac.cubic_proj_fft = cubic_proj_fft_dtype_consistent

    spec = importlib.util.spec_from_file_location(
        "ac_sweep", os.path.join(HERE, "16_ac_sweep.py"))
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    ac.N_Q = 2 * L
    ac.DTYPE = torch.float64
    # 16_ac_sweep imported cubic_proj_fft's caller from losses_ac at module load; the
    # call is resolved through losses_ac's globals at runtime, so the patch above is live.
    # Assert it rather than assume:
    assert losses_ac.cubic_proj_fft is cubic_proj_fft_dtype_consistent

    print(f"d={D} L={L} N_q={ac.N_Q} seed={SEED} dtype={ac.DTYPE} "
          f"device={ac.DEVICE}", flush=True)
    print("fp32 reference (same seed, 33_ac_adam_d2_ladder): 0.2259", flush=True)
    print("L-BFGS reference at this cell:                    0.117\n", flush=True)

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if ac.DEVICE == "cuda" else None
    re, loss, wall = ac.train_one(D, L, SEED)
    peak = (torch.cuda.max_memory_allocated() / 2**30) if ac.DEVICE == "cuda" else 0.0

    verdict = ("PRECISION -- fp64 recovers convergence" if re < 0.17 else
               "OPTIMISER -- fp64 does not help" if re > 0.23 else
               "PARTIAL -- fp64 helps but does not close the gap")
    print(f"rel_err = {re:.4f}   (fp32 was 0.2259, L-BFGS 0.117)", flush=True)
    print(f"loss    = {loss:.3e}   wall {wall/3600:.2f} h   peak {peak:.2f} GB", flush=True)
    print(f"VERDICT: {verdict}", flush=True)

    out = os.path.join(HERE, "..", "results", "34_ac_d2_L256_fp64.json")
    json.dump(dict(d=D, L=L, N_q=ac.N_Q, seed=SEED, dtype="float64",
                   rel_err=float(re), loss=float(loss), wall_s=float(wall),
                   peak_gb=round(peak, 3), fp32_reference=0.2259,
                   lbfgs_reference=0.117, verdict=verdict,
                   note="cubic_proj_fft monkeypatched to complex128; paper source unmodified"),
              open(out, "w"), indent=2)
    print(f"saved: {out}   (total {(time.time()-t0)/3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
