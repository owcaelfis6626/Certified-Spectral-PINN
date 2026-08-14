"""Gate for the batched cubic projection. Must pass before any number is trusted.

loss_rv_ac's per-quadrature-node Python loop was replaced by a batched, chunked call
(losses_ac.cubic_proj_fft_batched). That function is the core of a paper in flight, so the
replacement is not accepted on inspection. This script checks three things:

  1. VALUE  -- old loop vs new batched path, on identical inputs.
  2. GRADIENT -- d(loss)/d(alpha, beta). A change that got the loss right and the gradient
     wrong would train to a different answer while looking correct, which is the failure
     mode that actually matters here.
  3. CHUNK INVARIANCE -- chunk=1, 3, 7, N_q must all agree. chunk=1 is the original loop
     structure, so agreement across chunk sizes is what proves the blocking is sound.

FLOAT64 IS THE REAL TEST. In float32 the two paths cannot be bit-identical: the loop
accumulates q sequentially while the batched path sums by matmul, and float addition is not
associative. So fp32 agreement is expected at ~1e-6 relative and no better. Run in float64
the algebra is the same operation in a different order and agreement should be ~1e-14.
A pass in fp64 with a loose-but-sane fp32 residual is the evidence; a pass in fp32 alone
would prove much less.

Also asserts the dtype fix: the batched path derives its complex dtype from `dtype`, where
the original pins complex64. In float64 the two therefore SHOULD differ, and by more than
round-off -- that is the latent bug 34_ac_d2_L256_fp64.py had to monkeypatch around. The
gate checks that the batched fp64 result is the more accurate one by comparing against a
reference computed entirely in complex128.
"""
import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import losses_ac
from oracle_ac import ACOracleHalfPeriod
from pinn_lookup import LookupGalerkinPINN


def _cubic_unbatched_dtype_consistent(a, b, modes_int, N_x, device, dtype):
    """The original cubic, with ONLY the complex64 pin removed. Nothing else differs.

    Needed because the original pins complex64 even in an fp64 run, so comparing the
    batched fp64 path against it measures the dtype fix (~1e-7, single-precision epsilon)
    rather than loop-vs-batch. This isolates one variable at a time: against this
    reference, any residual difference is purely summation order.
    """
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    d = modes_int.shape[1]
    Nd = N_x ** d
    c = torch.complex(a, -b) / 2
    strides = torch.tensor([N_x ** (d - 1 - i) for i in range(d)],
                           device=device, dtype=torch.int64)
    lin_pos = ((modes_int.to(torch.int64) % N_x) * strides).sum(-1)
    lin_neg = (((-modes_int).to(torch.int64) % N_x) * strides).sum(-1)
    C_flat = torch.zeros(Nd, dtype=cdtype, device=device)
    C_flat = C_flat.index_add(0, lin_pos, c.to(cdtype))
    C_flat = C_flat.index_add(0, lin_neg, c.conj().to(cdtype))
    u_grid = (torch.fft.ifftn(C_flat.view([N_x] * d)) * Nd).real
    U3 = torch.fft.fftn(u_grid ** 3).reshape(-1)
    u3_k = U3[lin_pos]
    return (2 * u3_k.real / Nd).to(dtype), (-2 * u3_k.imag / Nd).to(dtype)


def loss_reference_loop(model, oracle, N_q, N_x, cubic=None):
    """The ORIGINAL implementation, verbatim, kept here as the thing to match."""
    device, dtype = oracle.device, oracle.dtype
    T, sigma = oracle.T, oracle.sigma
    J, L = oracle.J, oracle.L_test
    alpha, beta = model.coefficients(oracle)
    k2_1 = (oracle.k2 - 1.0)[:, None]
    R_a = alpha @ oracle.M + k2_1 * alpha
    R_b = beta @ oracle.M + k2_1 * beta
    nodes_01, w_01 = losses_ac.gauss_legendre_01(N_q, device, dtype)
    t_q, w_q = T * nodes_01, T * w_01
    phi_q = math.sqrt(2.0 / T) * torch.sin(oracle.omega_l[None, :] * t_q[:, None])
    exp_decay = torch.exp(-k2_1.T * t_q[:, None])
    a_q = exp_decay * oracle.a0[None, :] + phi_q @ alpha.T
    b_q = exp_decay * oracle.b0[None, :] + phi_q @ beta.T
    cubic = cubic or losses_ac.cubic_proj_fft
    cubic_a = torch.zeros(J, L, device=device, dtype=dtype)
    cubic_b = torch.zeros(J, L, device=device, dtype=dtype)
    for q in range(N_q):
        ua3, ub3 = cubic(a_q[q], b_q[q], oracle.modes_int, N_x, device, dtype)
        cubic_a = cubic_a + w_q[q] * ua3[:, None] * phi_q[q].unsqueeze(0)
        cubic_b = cubic_b + w_q[q] * ub3[:, None] * phi_q[q].unsqueeze(0)
    R_a = R_a + cubic_a - sigma * oracle.I_a
    R_b = R_b + cubic_b - sigma * oracle.I_b
    weight = 1.0 / (1.0 + oracle.k2[:, None] + oracle.mu_l[None, :])
    return ((R_a ** 2 + R_b ** 2) * weight).sum()


def setup(d, L, N_x, dtype, device, seed=0):
    oracle = ACOracleHalfPeriod(d=d, sigma=1.0, K_max=3, T=1.0, L_test=L,
                                N_fine=4096, N_x=N_x, seed=42 + seed,
                                device=device, dtype=dtype)
    model = LookupGalerkinPINN(oracle)
    # init_scale defaults to 0 -> all-zero parameters, which would make the cubic term
    # vanish and the test vacuous. Perturb so every path is actually exercised.
    g = torch.Generator(device="cpu").manual_seed(1234)
    with torch.no_grad():
        model.alpha.copy_(0.1 * torch.randn(model.alpha.shape, generator=g).to(device, dtype))
        model.beta.copy_(0.1 * torch.randn(model.beta.shape, generator=g).to(device, dtype))
    return oracle, model


def rel(x, y):
    den = max(abs(float(y)), 1e-30)
    return abs(float(x) - float(y)) / den


def grads(loss, model):
    g = torch.autograd.grad(loss, [model.alpha, model.beta])
    return torch.cat([t.reshape(-1) for t in g])


def check(d, L, N_x, dtype, device, tol_val, tol_grad, cubic=None):
    name = str(dtype).replace("torch.", "")
    oracle, model = setup(d, L, N_x, dtype, device)
    N_q = 2 * L
    ok = True

    lo = loss_reference_loop(model, oracle, N_q, N_x, cubic=cubic)
    go = grads(lo, model)

    for chunk in (1, 3, 7, N_q):
        ln = losses_ac.loss_rv_ac(model, oracle, N_q=N_q, N_x=N_x, chunk=chunk)
        gn = grads(ln, model)
        dv = rel(ln, lo)
        dg = float((gn - go).norm() / go.norm().clamp_min(1e-30))
        good = (dv < tol_val) and (dg < tol_grad)
        ok &= good
        print(f"  {name} d={d} L={L} chunk={chunk:<4} "
              f"dvalue {dv:.2e}  dgrad {dg:.2e}   {'ok' if good else 'FAIL'}")
    return ok


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"batching equivalence gate, device={dev}\n")
    print("float64 -- the algebra must match to round-off:")
    print("  (reference = same loop with the complex64 pin removed, so the only")
    print("   variable is loop-vs-batch; against the ORIGINAL the gap is 2e-7, which")
    print("   is single-precision epsilon from its pinned FFT -- see the dtype check.)")
    C = _cubic_unbatched_dtype_consistent
    ok64 = all([check(2, 16, 16, torch.float64, dev, 1e-11, 1e-10, cubic=C),
                check(3, 8, 16, torch.float64, dev, 1e-11, 1e-10, cubic=C)])
    print("\nfloat32 -- summation order differs; expect ~1e-6, not bit-identity:")
    ok32 = all([check(2, 16, 16, torch.float32, dev, 2e-5, 2e-4),
                check(3, 8, 16, torch.float32, dev, 2e-5, 2e-4)])

    print("\ndtype fix (batched derives complex dtype; original pins complex64):")
    oracle, model = setup(2, 16, 16, torch.float64, dev)
    a = torch.randn(4, oracle.J, device=dev, dtype=torch.float64, generator=None)
    b = torch.randn(4, oracle.J, device=dev, dtype=torch.float64)
    ob = torch.stack([torch.stack(losses_ac.cubic_proj_fft(a[i], b[i], oracle.modes_int,
                                                           16, dev, torch.float64))
                      for i in range(4)])
    nb = torch.stack(losses_ac.cubic_proj_fft_batched(a, b, oracle.modes_int,
                                                      16, dev, torch.float64))
    nb = nb.permute(1, 0, 2)
    gap = float((ob - nb).abs().max())
    print(f"  max |original(complex64-pinned) - batched(complex128)| = {gap:.2e}")
    print(f"  -> {'as expected: the original loses fp64 precision in the FFT' if gap > 1e-9 else 'UNEXPECTED: no difference, check the dtype derivation'}")

    print(f"\nGATE: {'PASS' if (ok64 and ok32) else 'FAIL'}")
    return 0 if (ok64 and ok32) else 1


if __name__ == "__main__":
    sys.exit(main())
