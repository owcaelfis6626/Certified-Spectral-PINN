"""Sweep (L_test, N_q) at d=2 to see if rel_err improves with richer
test space / quadrature.

Hypothesis: with L=32 the test space can't see network high-frequency
time content; increasing L should tighten the bound.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle import WMContinuousOracle
from pinn import ModeSpacePINN
from losses import loss_rv_continuous, loss_ic


D, K_MAX, SIGMA, T = 2, 4, 1.0, 1.0
N_FINE = 4096
N_STEPS = 4000   # shorter to fit the sweep budget
LR = 3e-3
LAMBDA_IC = 1.0
SEEDS = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Sweep configs: (L_test, N_q)
CONFIGS = [
    (32,  64),
    (32, 256),
    (64,  64),
    (64, 256),
    (128, 256),
]


@torch.no_grad()
def l2h1_error_sq(model, oracle, N_t_eval=256):
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    a_ref, b_ref = oracle.a_ref(t_eval)
    K_rep = oracle.modes_int.repeat_interleave(N_t_eval, 0)
    T_rep = t_eval.repeat(oracle.J)
    ab = model(K_rep, T_rep)
    a_th = ab[..., 0].view(oracle.J, N_t_eval).T
    b_th = ab[..., 1].view(oracle.J, N_t_eval).T
    e_a, e_b = a_th - a_ref, b_th - b_ref
    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    dt = oracle.T / (N_t_eval - 1)
    sq_per_t = (one_plus_k2 * (e_a ** 2 + e_b ** 2)).sum(-1)
    w = torch.ones_like(sq_per_t)
    w[0] = 0.5
    w[-1] = 0.5
    return (math.pi ** oracle.d) * (w * sq_per_t).sum().item() * dt


@torch.no_grad()
def ref_norm_y_sq(oracle, N_t_eval=256):
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    a_ref, b_ref = oracle.a_ref(t_eval)
    one_plus_k2 = (1.0 + oracle.k2)[None, :]
    dt = oracle.T / (N_t_eval - 1)
    sq_per_t = (one_plus_k2 * (a_ref ** 2 + b_ref ** 2)).sum(-1)
    w = torch.ones_like(sq_per_t)
    w[0] = 0.5
    w[-1] = 0.5
    return (math.pi ** oracle.d) * (w * sq_per_t).sum().item() * dt


def train_one(L_test, N_q, seed):
    torch.manual_seed(seed * 1000 + 1)
    oracle = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                 L_test=L_test, N_fine=N_FINE,
                                 seed=42 + seed, device=DEVICE, dtype=DTYPE)
    model = ModeSpacePINN(d=D, K_emb_t=8, hidden=256, depth=4, T_period=T).to(DEVICE).to(DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(N_STEPS):
        opt.zero_grad()
        L = loss_rv_continuous(model, oracle, N_q=N_q) + LAMBDA_IC * loss_ic(model, oracle)
        L.backward()
        opt.step()

    L_RV  = loss_rv_continuous(model, oracle, N_q=N_q).item()
    L_IC  = loss_ic(model, oracle).item()
    err_Y = l2h1_error_sq(model, oracle)
    ref_Y = ref_norm_y_sq(oracle)
    return dict(L_RV=L_RV, L_IC=L_IC, err_Y=err_Y, ref_Y=ref_Y,
                rel_err=err_Y / ref_Y, ratio=err_Y / max(L_RV, 1e-12),
                L_test=L_test, N_q=N_q, seed=seed)


def main():
    print(f"P7 sweep d={D} K_max={K_MAX} N_steps={N_STEPS}\n")
    all_results = []
    for L_test, N_q in CONFIGS:
        print(f"=== L_test={L_test}, N_q={N_q} ===", flush=True)
        per_cfg = []
        t0 = time.time()
        for seed in SEEDS:
            r = train_one(L_test, N_q, seed)
            print(f"  seed={seed}: L_RV={r['L_RV']:.3e}  err_Y={r['err_Y']:.3e}"
                  f"  rel_err={r['rel_err']:.3f}  ratio={r['ratio']:.0f}", flush=True)
            per_cfg.append(r)
        all_results.append(per_cfg)
        rel_errs = np.array([r['rel_err'] for r in per_cfg])
        ratios   = np.array([r['ratio']   for r in per_cfg])
        print(f"  summary: rel_err={rel_errs.mean():.3f}±{rel_errs.std():.3f}"
              f"  ratio={ratios.mean():.0f}±{ratios.std():.0f}  wall={time.time()-t0:.0f}s\n",
              flush=True)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "03_sweep_L_Nq.npz")
    payload = {f"cfg_{i}": np.array([(r['L_test'], r['N_q'], r['seed'], r['L_RV'],
                                       r['L_IC'], r['err_Y'], r['ref_Y'], r['rel_err'],
                                       r['ratio']) for r in per_cfg])
               for i, per_cfg in enumerate(all_results)}
    np.savez(out_path, **payload)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
