"""Direct measurement of temporal-quadrature (aliasing) error in the cubic term.

The L-sweep infers quadrature trouble from end-to-end rel_err, which confounds
quadrature error with L-BFGS behaviour. This isolates it: for FIXED trial
coefficients, compute the cubic residual tensor

    cubic_{k,l} = \\int_0^T [u^3]_k(t) \\phi_l(t) dt

at several N_q and compare against a high-N_q reference. The difference is the
quadrature error alone -- no optimizer, no training.

losses_ac.py derives the cubic dealiasing rule in SPACE (N_x > 2*3*K_max) but
never applies the temporal analogue: N_q is fixed at 16 while L is swept to 128.
For a cubic nonlinearity the temporal grid must resolve the tripled bandwidth,
i.e. N_q = O(2L). This script measures whether that is what actually happens.

Runs under no_grad, so nothing is retained for backward and memory is O(1) in
N_q -- which is why this reaches d=5 and large N_q where the TRAINING runs OOM.
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

from losses_ac import cubic_proj_fft, gauss_legendre_01
from oracle_ac import ACOracleHalfPeriod

spec = importlib.util.spec_from_file_location(
    "sweep20", os.path.join(HERE, "20_ac_lbfgs_sweep.py"))
sweep20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep20)

K_MAX, SIGMA, T, N_FINE, N_X = 3, 1.0, 1.0, 4096, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# (d, N_q values to test, reference N_q). d=5 capped: no_grad keeps memory flat
# but each step is a 32^5 = 33.5M-point FFT, so the reference must stay sane.
CASES = [
    (2, [16, 32, 64, 128, 256, 512], 2048),
    (3, [16, 32, 64, 128, 256, 512], 2048),
    (4, [16, 32, 64, 128, 256], 1024),
    (5, [16, 32, 64, 128], 512),
]
L_FIXED = 128


@torch.no_grad()
def cubic_tensor(oracle, alpha, beta, n_q):
    """The cubic residual block of loss_rv_ac, isolated. Returns (J,L) x2."""
    device, dtype = oracle.device, oracle.dtype
    J, L = oracle.J, oracle.L_test
    k2_1 = (oracle.k2 - 1.0)[:, None]

    nodes_01, w_01 = gauss_legendre_01(n_q, device, dtype)
    t_q = oracle.T * nodes_01
    w_q = oracle.T * w_01

    phase = oracle.omega_l[None, :] * t_q[:, None]
    phi_q = math.sqrt(2.0 / oracle.T) * torch.sin(phase)          # (N_q, L)

    exp_decay = torch.exp(-k2_1.T * t_q[:, None])                 # (N_q, J)
    a_q = exp_decay * oracle.a0[None, :] + phi_q @ alpha.T        # (N_q, J)
    b_q = exp_decay * oracle.b0[None, :] + phi_q @ beta.T

    cubic_a = torch.zeros(J, L, device=device, dtype=dtype)
    cubic_b = torch.zeros(J, L, device=device, dtype=dtype)
    for q in range(n_q):
        ua3, ub3 = cubic_proj_fft(a_q[q], b_q[q], oracle.modes_int,
                                   N_X, device, dtype)
        cubic_a = cubic_a + w_q[q] * ua3[:, None] * phi_q[q].unsqueeze(0)
        cubic_b = cubic_b + w_q[q] * ub3[:, None] * phi_q[q].unsqueeze(0)
    return cubic_a, cubic_b


def main():
    print("P7 cubic-term temporal quadrature error (no training, no_grad)")
    print(f"L={L_FIXED}  K_max={K_MAX}  N_x={N_X}  device={DEVICE}")
    print(f"spatial dealias rule (from losses_ac docstring): N_x > 6*K_max = "
          f"{6 * K_MAX} -> N_x={N_X} OK")
    print(f"temporal analogue for a cubic: N_q = O(2L) = O({2 * L_FIXED}) "
          f"-> published runs used N_q=16\n")

    rows = []
    for d, nqs, nq_ref in CASES:
        oracle = ACOracleHalfPeriod(
            d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L_FIXED,
            N_fine=N_FINE, N_x=N_X, seed=42, device=DEVICE, dtype=DTYPE)
        J = oracle.J

        # fixed, seeded trial coefficients with 1/l decay (a smooth-in-time
        # solution profile); identical across N_q so only quadrature varies
        g = torch.Generator(device="cpu").manual_seed(1234)
        decay = 1.0 / torch.arange(1, L_FIXED + 1, dtype=torch.float64) ** 1.5
        alpha = (torch.randn(J, L_FIXED, generator=g, dtype=torch.float64)
                 * decay[None, :]).to(device=DEVICE, dtype=DTYPE) * 0.5
        beta = (torch.randn(J, L_FIXED, generator=g, dtype=torch.float64)
                * decay[None, :]).to(device=DEVICE, dtype=DTYPE) * 0.5

        t0 = time.time()
        ref_a, ref_b = cubic_tensor(oracle, alpha, beta, nq_ref)
        ref_norm = torch.sqrt((ref_a ** 2).sum() + (ref_b ** 2).sum()).item()
        t_ref = time.time() - t0

        print(f"--- d={d}  J={J}  reference N_q={nq_ref} ({t_ref:.1f}s) ---")
        print(f"{'N_q':>6} {'rel quad err':>14} {'N_q/2L':>9}")
        for nq in nqs:
            ca, cb = cubic_tensor(oracle, alpha, beta, nq)
            err = torch.sqrt(((ca - ref_a) ** 2).sum()
                             + ((cb - ref_b) ** 2).sum()).item() / ref_norm
            print(f"{nq:6d} {err:14.3e} {nq / (2 * L_FIXED):9.3f}", flush=True)
            rows.append((d, nq, err))
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        print()

    arr = np.array(rows)
    print("=== quadrature error at the PUBLISHED setting (N_q=16, L=128) ===")
    for d, _, _ in CASES:
        m = arr[(arr[:, 0] == d) & (arr[:, 1] == 16)]
        if len(m):
            print(f"  d={d}: {m[0, 2]:.3e}")
    out = os.path.join(HERE, "..", "results", "26_cubic_quadrature_error.npz")
    np.savez(out, rows=arr)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
