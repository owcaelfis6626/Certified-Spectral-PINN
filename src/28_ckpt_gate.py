"""GATE: loss_rv_ac_ckpt must reproduce loss_rv_ac exactly (value AND gradients).

Checkpointing is a memory optimisation, nothing else. If it changes the loss or
the gradient, every dealiased high-d number built on it is worthless -- so this
gate runs before any such number is produced.

Also reports peak CUDA memory for both paths, which is the whole point of the
exercise: the dealiased d=5 sweep is unreachable without the reduction.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from losses_ac import loss_rv_ac
from losses_ac_ckpt import loss_rv_ac_ckpt
from oracle_ac import ACOracleHalfPeriod
from pinn_lookup import LookupGalerkinPINN

K_MAX, SIGMA, T, N_FINE, N_X = 3, 1.0, 1.0, 4096, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

VAL_TOL = 1e-5   # relative, fp32
GRAD_TOL = 1e-4  # relative Frobenius


def build(d, L, seed=0):
    oracle = ACOracleHalfPeriod(
        d=d, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L,
        N_fine=N_FINE, N_x=N_X, seed=42 + seed, device=DEVICE, dtype=DTYPE)
    torch.manual_seed(7)
    model = LookupGalerkinPINN(oracle)
    # perturb off the initial state so gradients are non-trivial
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    return oracle, model


def grads_of(loss, model):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def check(d, L, n_q, chunk):
    oracle, model = build(d, L)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    l_ref = loss_rv_ac(model, oracle, N_q=n_q, N_x=N_X)
    g_ref = grads_of(l_ref, model)
    mem_ref = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else 0.0

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    l_ckpt = loss_rv_ac_ckpt(model, oracle, N_q=n_q, N_x=N_X, chunk=chunk)
    g_ckpt = grads_of(l_ckpt, model)
    mem_ckpt = torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else 0.0

    v_ref, v_ckpt = l_ref.item(), l_ckpt.item()
    v_rel = abs(v_ckpt - v_ref) / max(abs(v_ref), 1e-30)

    num = sum(((a - b) ** 2).sum().item() for a, b in zip(g_ref, g_ckpt))
    den = sum((a ** 2).sum().item() for a in g_ref)
    g_rel = (num ** 0.5) / max(den ** 0.5, 1e-30)

    ok = v_rel < VAL_TOL and g_rel < GRAD_TOL
    print(f"  d={d} L={L} N_q={n_q} chunk={chunk}")
    print(f"    loss  ref={v_ref:.9e}  ckpt={v_ckpt:.9e}  rel={v_rel:.2e}")
    print(f"    grad  rel Frobenius diff = {g_rel:.2e}")
    print(f"    peak mem  ref={mem_ref:8.1f} MiB   ckpt={mem_ckpt:8.1f} MiB"
          f"   ({mem_ref / max(mem_ckpt, 1e-9):.2f}x reduction)")
    print(f"    => {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    print("GATE: checkpointed cubic quadrature == reference (value + gradients)\n")
    results = []
    for d, L, n_q, chunk in [(2, 32, 32, 8), (2, 64, 64, 8), (3, 32, 64, 8),
                             (3, 64, 128, 16), (4, 32, 64, 8)]:
        results.append(check(d, L, n_q, chunk))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    print("=> GATE PASS" if all(results) else "=> GATE FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
