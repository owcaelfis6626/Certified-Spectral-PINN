"""Train the lookup-table Galerkin PINN at d=2.

Expected outcome: SGD on the (quadratic in alpha) loss reaches the
optimal Galerkin coefficients alpha^*. rel_err should match script 05's
direct-solve result (~0.13 at L=64).

If this works, the iterative training path is viable -- foundation for
the nonlinear AC extension.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle import WMContinuousOracle
from pinn_lookup import LookupGalerkinPINN
from losses import loss_rv_galerkin


D, K_MAX, SIGMA, T = 2, 4, 1.0, 1.0
L_TEST  = 64
N_FINE  = 4096
N_STEPS = 4000
LR      = 1e-2          # higher than MLP since the loss is purely quadratic
SEEDS   = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32


@torch.no_grad()
def eval_a_b_galerkin(model, oracle, t_eval):
    alpha, beta = model.coefficients(oracle)
    phase = (math.pi * oracle.l_idx[None, :] / oracle.T) * t_eval[:, None]
    psi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)
    ic_decay = torch.exp(-oracle.lambda_k[None, :] * t_eval[:, None])
    a = ic_decay * oracle.a0[None, :] + psi @ alpha.T
    b = ic_decay * oracle.b0[None, :] + psi @ beta.T
    return a, b


@torch.no_grad()
def diagnose(model, oracle, N_t_eval=256):
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    a_ref, b_ref = oracle.a_ref(t_eval)
    a_th,  b_th  = eval_a_b_galerkin(model, oracle, t_eval)
    e_a, e_b = a_th - a_ref, b_th - b_ref
    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    dt = oracle.T / (N_t_eval - 1)
    w = torch.ones(N_t_eval, device=oracle.device, dtype=oracle.dtype)
    w[0] = 0.5
    w[-1] = 0.5
    err_Y = (math.pi ** oracle.d) * (w * (one_plus_k2 * (e_a**2 + e_b**2)).sum(-1)).sum().item() * dt
    ref_Y = (math.pi ** oracle.d) * (w * (one_plus_k2 * (a_ref**2 + b_ref**2)).sum(-1)).sum().item() * dt
    return err_Y, ref_Y


def train_one(seed):
    torch.manual_seed(seed * 1000 + 1)
    oracle = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                 L_test=L_TEST, N_fine=N_FINE,
                                 seed=42 + seed, device=DEVICE, dtype=DTYPE)
    model = LookupGalerkinPINN(oracle).to(DEVICE).to(DTYPE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  J*L = {oracle.J * oracle.L_test * 2} params (model: {n_params})")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t_start = time.time()
    for step in range(N_STEPS):
        opt.zero_grad()
        L_RV = loss_rv_galerkin(model, oracle)
        L_RV.backward()
        opt.step()
        if step % 500 == 0 or step == N_STEPS - 1:
            err_Y, ref_Y = diagnose(model, oracle)
            print(f"  step {step:5d}  L_RV={L_RV.item():.4e}  rel_err={err_Y/ref_Y:.4f}"
                  f"  wall={time.time()-t_start:.1f}s", flush=True)

    L_RV_final = loss_rv_galerkin(model, oracle).item()
    err_Y, ref_Y = diagnose(model, oracle)
    print(f"  FINAL seed={seed}  L_RV={L_RV_final:.4e}  rel_err={err_Y/ref_Y:.4f}",
          flush=True)
    return dict(seed=seed, L_RV=L_RV_final, err_Y=err_Y, ref_Y=ref_Y,
                rel_err=err_Y / ref_Y, wall=time.time() - t_start)


def main():
    print(f"P7 LOOKUP-Galerkin d={D} K_max={K_MAX} L={L_TEST} N_steps={N_STEPS} lr={LR}")
    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        results.append(train_one(seed))

    print("\n=== summary ===")
    rel_es = np.array([r["rel_err"] for r in results])
    L_RVs  = np.array([r["L_RV"]    for r in results])
    print(f"  rel_err: mean={rel_es.mean():.4f}  std={rel_es.std():.4f}")
    print(f"  L_RV:    mean={L_RVs.mean():.4e}  std={L_RVs.std():.4e}")
    print(f"  (compare: optimal Galerkin at L=64 gave rel_err ~ 0.131)")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                             "06_train_lookup_d2.npz")
    np.savez(out_path, L_RV=L_RVs, rel_err=rel_es,
             err_Y=np.array([r["err_Y"] for r in results]),
             ref_Y=np.array([r["ref_Y"] for r in results]),
             seeds=np.array(SEEDS))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
