"""Training the Galerkin (architecture A) PINN at d=2.

Trial = test space (Dirichlet-sine in time) -> discrete inf-sup equals
continuous inf-sup, no L-dependent loss. IC enforced by construction
(IC term in the kernel of the operator), so L_IC drops out entirely.

Diagnostic at end: same Y-norm error and ratio as architecture B, but
now expected to be O(1) uniformly in L.
"""

import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from oracle import WMContinuousOracle
from pinn_galerkin import GalerkinModePINN
from losses import loss_rv_galerkin


D       = 2
K_MAX   = 4
SIGMA   = 1.0
T       = 1.0
L_TEST  = 64          # match the architecture-B sweep
N_FINE  = 4096
N_STEPS = 6000
LR      = 3e-3
SEEDS   = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@torch.no_grad()
def eval_a_b_galerkin(model, oracle, t_eval):
    """Construct a_{theta,k}(t), b_{theta,k}(t) on t_eval from the
    Galerkin coefficients.  Returns (N_t, J), (N_t, J)."""
    alpha, beta = model.coefficients(oracle)         # (J, L)
    N_t = t_eval.shape[0]
    # psi at eval points: (N_t, L) = sqrt(2/T) sin(pi l t / T).
    phase = (math.pi * oracle.l_idx[None, :] / oracle.T) * t_eval[:, None]
    psi = math.sqrt(2.0 / oracle.T) * torch.sin(phase)               # (N_t, L)
    # IC decay term: (N_t, J).
    ic_decay = torch.exp(-oracle.lambda_k[None, :] * t_eval[:, None])
    a = ic_decay * oracle.a0[None, :] + psi @ alpha.T                # (N_t, J)
    b = ic_decay * oracle.b0[None, :] + psi @ beta.T
    return a, b


@torch.no_grad()
def l2h1_error_sq(model, oracle, N_t_eval=256):
    t_eval = torch.linspace(0.0, oracle.T, N_t_eval,
                             device=oracle.device, dtype=oracle.dtype)
    a_ref, b_ref = oracle.a_ref(t_eval)
    a_th, b_th = eval_a_b_galerkin(model, oracle, t_eval)
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


def train_one(seed):
    torch.manual_seed(seed * 1000 + 1)
    oracle = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                 L_test=L_TEST, N_fine=N_FINE,
                                 seed=42 + seed, device=DEVICE, dtype=DTYPE)
    model = GalerkinModePINN(d=D, L_test=L_TEST, K_emb_l=4,
                              hidden=256, depth=4).to(DEVICE).to(DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t_start = time.time()
    for step in range(N_STEPS):
        opt.zero_grad()
        L_RV = loss_rv_galerkin(model, oracle)
        L_RV.backward()
        opt.step()

        if step % 500 == 0 or step == N_STEPS - 1:
            print(f"  step {step:5d}  L_RV={L_RV.item():.4e}  wall={time.time()-t_start:.1f}s",
                  flush=True)

    # Diagnostics
    L_RV_final = loss_rv_galerkin(model, oracle).item()
    err_Y  = l2h1_error_sq(model, oracle, N_t_eval=256)
    ref_Y  = ref_norm_y_sq(oracle, N_t_eval=256)
    rel_err = err_Y / ref_Y
    ratio   = err_Y / max(L_RV_final, 1e-12)
    print(f"  FINAL seed={seed}  L_RV={L_RV_final:.4e}  ||e||_Y^2={err_Y:.4e}"
          f"  ||u_ref||_Y^2={ref_Y:.4e}  rel_err={rel_err:.4f}  ratio={ratio:.2f}",
          flush=True)
    return dict(seed=seed, L_RV=L_RV_final, err_Y=err_Y, ref_Y=ref_Y,
                rel_err=rel_err, ratio=ratio, wall=time.time() - t_start)


def main():
    print(f"P7 Galerkin d={D} K_max={K_MAX} sigma={SIGMA} T={T} L={L_TEST} N_steps={N_STEPS}")
    print(f"Device: {DEVICE}")
    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        results.append(train_one(seed))

    print("\n=== summary ===")
    L_RVs  = np.array([r["L_RV"]    for r in results])
    errs   = np.array([r["err_Y"]   for r in results])
    refs   = np.array([r["ref_Y"]   for r in results])
    ratios = np.array([r["ratio"]   for r in results])
    rel_es = np.array([r["rel_err"] for r in results])
    print(f"  L_RV:        mean={L_RVs.mean():.4e}  std={L_RVs.std():.4e}")
    print(f"  ||e||_Y^2:   mean={errs.mean():.4e}  std={errs.std():.4e}")
    print(f"  ||u_ref||_Y^2: mean={refs.mean():.4e}")
    print(f"  rel error:   mean={rel_es.mean():.4f}  std={rel_es.std():.4f}")
    print(f"  ratio Y/RV:  mean={ratios.mean():.2f}  std={ratios.std():.2f}")

    out_path = os.path.join(RESULTS_DIR, "04_train_galerkin_d2.npz")
    np.savez(out_path, L_RV=L_RVs, err_Y=errs, ref_Y=refs, ratio=ratios,
             rel_err=rel_es, seeds=np.array(SEEDS),
             config=dict(d=D, K_max=K_MAX, L_test=L_TEST, N_steps=N_STEPS))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
