"""First end-to-end training run for the RVPINN x WM mode-space PINN.

Linear bandlimited-Laplacian SPDE on T^2 with K_max = 4.
Trains ModeSpacePINN to minimise L_RV + lambda_IC * L_IC.

Diagnostic at end:
- Final L_RV value.
- Actual Bochner-norm error ||u_theta - u_ref||_X^2 in mode space, computed
  by quadrature on the t_fine grid against the closed-form OU reference.
- Ratio L_RV / ||error||_X^2. If the bound is tight, this ratio approximates
  the empirical inverse inf-sup constant C_1.

If the ratio is bounded (say < 100) and stable across seeds, the certified
bound is meaningful empirically.
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


# ─── config ────────────────────────────────────────────────────────────────
D       = 2
K_MAX   = 4
SIGMA   = 1.0
T       = 1.0
L_TEST  = 32          # number of time test modes psi_l
N_FINE  = 4096        # Brownian path resolution
N_Q     = 64          # Gauss-Legendre quadrature nodes
N_STEPS = 6000
LR      = 3e-3
LAMBDA_IC = 1.0
SEEDS = [0, 1, 2]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── diagnostic: Bochner norm of error in mode space ───────────────────────

@torch.no_grad()
def l2h1_error_sq(model, oracle, N_t_eval: int = 256):
    """Compute the L^2(0,T; H^1(T^d)) part of the Bochner X-norm error:
        ||u_theta - u_ref||_Y^2 = pi^d sum_k int_0^T (1+|k|^2)
                                       (|a_th_k - a_ref_k|^2 + |b_th_k - b_ref_k|^2) dt.

    The full Bochner X-norm also has an H^1(0,T; H^{-1}) part, but that
    contains the white-noise singularity of dt(u_ref) and is ill-conditioned
    at FD diagnostics. The certified bound gives
        ||e||_Y^2 <= ||e||_X^2 <= C_1 L_RV + C_2 L_IC
    so checking ||e||_Y^2 / L_RV bounded still validates the theorem
    (it's a lower bound on the bound's tightness).
    """
    device = oracle.device
    dtype = oracle.dtype
    T_period = oracle.T
    J = oracle.J

    t_eval = torch.linspace(0.0, T_period, N_t_eval, device=device, dtype=dtype)
    a_ref, b_ref = oracle.a_ref(t_eval)              # (N_t_eval, J), (N_t_eval, J)

    K_rep = oracle.modes_int.repeat_interleave(N_t_eval, dim=0)
    T_rep = t_eval.repeat(J)
    ab = model(K_rep, T_rep)                          # (J*N_t_eval, 2)
    a_th = ab[..., 0].view(J, N_t_eval).T            # (N_t_eval, J)
    b_th = ab[..., 1].view(J, N_t_eval).T

    e_a = a_th - a_ref
    e_b = b_th - b_ref

    k2 = oracle.k2[None, :]                          # (1, J)
    one_plus_k2 = 1.0 + k2

    # Trapezoidal rule on [0, T].
    dt = T_period / (N_t_eval - 1)
    sq_per_t = (one_plus_k2 * (e_a ** 2 + e_b ** 2)).sum(-1)   # (N_t_eval,)
    w = torch.ones_like(sq_per_t)
    w[0] = 0.5
    w[-1] = 0.5
    integral = (w * sq_per_t).sum() * dt

    return (math.pi ** oracle.d) * integral.item()


@torch.no_grad()
def ref_norm_y_sq(oracle, N_t_eval: int = 256):
    """Baseline: ||u_ref||_Y^2 for context (the value the network must beat
    by approximating u_ref vs predicting 0)."""
    device = oracle.device
    dtype = oracle.dtype
    T_period = oracle.T

    t_eval = torch.linspace(0.0, T_period, N_t_eval, device=device, dtype=dtype)
    a_ref, b_ref = oracle.a_ref(t_eval)              # (N_t_eval, J), (N_t_eval, J)

    k2 = oracle.k2[None, :]
    one_plus_k2 = 1.0 + k2

    dt = T_period / (N_t_eval - 1)
    sq_per_t = (one_plus_k2 * (a_ref ** 2 + b_ref ** 2)).sum(-1)
    w = torch.ones_like(sq_per_t)
    w[0] = 0.5
    w[-1] = 0.5
    integral = (w * sq_per_t).sum() * dt
    return (math.pi ** oracle.d) * integral.item()


# ─── train one seed ────────────────────────────────────────────────────────

def train_one(seed: int):
    torch.manual_seed(seed * 1000 + 1)
    oracle = WMContinuousOracle(
        d=D, sigma=SIGMA, K_max=K_MAX, T=T, L_test=L_TEST, N_fine=N_FINE,
        seed=42 + seed, device=DEVICE, dtype=DTYPE,
    )
    model = ModeSpacePINN(d=D, K_emb_t=8, hidden=256, depth=4, T_period=T)
    model = model.to(DEVICE).to(DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    t_start = time.time()
    for step in range(N_STEPS):
        opt.zero_grad()
        L_RV = loss_rv_continuous(model, oracle, N_q=N_Q)
        L_IC = loss_ic(model, oracle)
        L = L_RV + LAMBDA_IC * L_IC
        L.backward()
        opt.step()

        if step % 500 == 0 or step == N_STEPS - 1:
            print(f"  step {step:5d}  L_RV={L_RV.item():.4e}  L_IC={L_IC.item():.4e}  wall={time.time()-t_start:.1f}s",
                  flush=True)

    # Final diagnostics
    L_RV_final = loss_rv_continuous(model, oracle, N_q=N_Q).item()
    L_IC_final = loss_ic(model, oracle).item()
    err_Y_sq   = l2h1_error_sq(model, oracle, N_t_eval=256)
    ref_Y_sq   = ref_norm_y_sq(oracle, N_t_eval=256)
    ratio      = err_Y_sq / max(L_RV_final, 1e-12)
    rel_err    = err_Y_sq / ref_Y_sq
    print(f"  FINAL seed={seed}  L_RV={L_RV_final:.4e}  ||e||_Y^2={err_Y_sq:.4e}"
          f"  ||u_ref||_Y^2={ref_Y_sq:.4e}  rel_err={rel_err:.4f}  ratio={ratio:.2f}",
          flush=True)
    return dict(
        seed=seed, L_RV=L_RV_final, L_IC=L_IC_final,
        err_Y_sq=err_Y_sq, ref_Y_sq=ref_Y_sq,
        ratio=ratio, rel_err=rel_err,
        wall=time.time() - t_start,
        J=oracle.J,
    )


def main():
    print(f"P7 d={D} K_max={K_MAX} sigma={SIGMA} T={T} L={L_TEST} N_q={N_Q} N_steps={N_STEPS}")
    print(f"Device: {DEVICE}")
    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        results.append(train_one(seed))

    print("\n=== summary ===")
    L_RVs   = np.array([r["L_RV"]      for r in results])
    errs    = np.array([r["err_Y_sq"]  for r in results])
    refs    = np.array([r["ref_Y_sq"]  for r in results])
    ratios  = np.array([r["ratio"]     for r in results])
    rel_es  = np.array([r["rel_err"]   for r in results])
    print(f"  L_RV:        mean={L_RVs.mean():.4e}  std={L_RVs.std():.4e}")
    print(f"  ||e||_Y^2:   mean={errs.mean():.4e}  std={errs.std():.4e}")
    print(f"  ||u_ref||_Y^2: mean={refs.mean():.4e}")
    print(f"  rel error:   mean={rel_es.mean():.4f}  std={rel_es.std():.4f}")
    print(f"  ratio Y/RV:  mean={ratios.mean():.2f}  std={ratios.std():.2f}")

    out_path = os.path.join(RESULTS_DIR, "01_train_d2.npz")
    np.savez(
        out_path,
        L_RV=L_RVs, err_Y_sq=errs, ref_Y_sq=refs, ratio=ratios, rel_err=rel_es,
        seeds=np.array(SEEDS),
        config=dict(d=D, K_max=K_MAX, sigma=SIGMA, T=T,
                    L_test=L_TEST, N_fine=N_FINE, N_q=N_Q, N_steps=N_STEPS),
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
