"""Sanity check: plug u_theta = u_ref into L_RV. Should give ~0.

If L_RV is large at the reference, the loss computation has a bug
(sign of I_a, mode ordering, scaling, etc.).
If L_RV is small at the reference, the loss is right and the issue is
elsewhere (theorem statement, training dynamics, function class, ...).

We do this by replacing the model's forward with a closed-form OU
evaluator that returns a_ref(t), b_ref(t) directly.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from oracle import WMContinuousOracle
from losses import gauss_legendre_01


D = 2
K_MAX = 4
SIGMA = 1.0
T = 1.0
L_TEST = 32
N_FINE = 4096
N_Q = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def lrv_at_reference(oracle, N_q=N_Q):
    """Compute L_RV directly with u_theta = u_ref (closed-form OU)."""
    device = oracle.device
    dtype = oracle.dtype
    T_period = oracle.T
    sigma = oracle.sigma
    J, L = oracle.J, oracle.L_test

    # Gauss-Legendre on (0, T).
    nodes_01, w_01 = gauss_legendre_01(N_q, device, dtype)
    t_q = T_period * nodes_01                      # (N_q,)
    w_q = T_period * w_01                          # (N_q,)

    # psi_l at quadrature nodes.
    phase = (math.pi * oracle.l_idx[None, :] / T_period) * t_q[:, None]
    psi_q = math.sqrt(2.0 / T_period) * torch.sin(phase)  # (N_q, L)

    # Reference values at quadrature nodes (closed-form OU).
    a_ref_q, b_ref_q = oracle.a_ref(t_q)            # (N_q, J), (N_q, J)
    # Transpose to (J, N_q).
    a_ref_q = a_ref_q.T
    b_ref_q = b_ref_q.T

    # We need dt a_ref(t_q) at the quadrature nodes. The OU SDE gives
    #   dt a_ref + |k|^2 a_ref = sigma * xi (white noise)
    # so the strong-form residual at u_ref is sigma * xi -- this is what
    # gets integrated against psi_l to give sigma * I_{k,l}.
    # Numerically: instead of computing dt a_ref(t_q) and feeding into
    # Lop, we should integrate the SDE in weak form:
    #
    #   int_0^T (dt a_ref + |k|^2 a_ref) psi_l dt
    # = -int_0^T a_ref(t) dpsi_l/dt dt + [a_ref(t) psi_l(t)]_0^T
    #   + |k|^2 int_0^T a_ref(t) psi_l(t) dt
    #
    # psi_l(0) = psi_l(T) = 0 so boundary terms vanish. The dpsi_l/dt is
    # an L^2 function on [0, T], so the integration is well-defined.
    #
    # Then subtract sigma * I_{k,l} and check the result is ~0.

    # dpsi_l/dt(t_q) = sqrt(2/T) * (pi l / T) cos(pi l t_q / T)
    l_idx = oracle.l_idx[None, :]                    # (1, L)
    dpsi_q = (math.sqrt(2.0 / T_period) * (math.pi * l_idx / T_period)
              * torch.cos(phase))                    # (N_q, L)

    # int a_ref dpsi_l dt via Gauss-Legendre:  (J, L) = a_ref @ (dpsi * w).
    int_kernel_d = dpsi_q * w_q[:, None]             # (N_q, L)
    int_a_dpsi   = a_ref_q @ int_kernel_d            # (J, L)
    int_b_dpsi   = b_ref_q @ int_kernel_d

    int_kernel_p = psi_q * w_q[:, None]              # (N_q, L)
    int_a_psi    = a_ref_q @ int_kernel_p            # (J, L)
    int_b_psi    = b_ref_q @ int_kernel_p

    k2 = oracle.k2[:, None]                          # (J, 1)
    proj_a = -int_a_dpsi + k2 * int_a_psi            # (J, L)
    proj_b = -int_b_dpsi + k2 * int_b_psi

    R_a = proj_a - sigma * oracle.I_a                # (J, L)
    R_b = proj_b - sigma * oracle.I_b

    weight = 1.0 / (1.0 + oracle.k2[:, None] + oracle.mu_l[None, :])
    L_RV = ((R_a ** 2 + R_b ** 2) * weight).sum()
    return L_RV.item(), R_a, R_b


def main():
    oracle = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                 L_test=L_TEST, N_fine=N_FINE,
                                 seed=42, device=DEVICE, dtype=DTYPE)
    print(f"J = {oracle.J}, L = {oracle.L_test}")
    print(f"I_a stats: mean = {oracle.I_a.mean():.4e}, std = {oracle.I_a.std():.4e}")
    print(f"I_a abs max = {oracle.I_a.abs().max():.4e}")
    print()

    # Check 1: L_RV at u_theta = u_ref using weak form.
    L_RV_ref, R_a, R_b = lrv_at_reference(oracle, N_q=N_Q)
    print(f"L_RV at u_theta = u_ref (weak form, N_q={N_Q}): {L_RV_ref:.4e}")
    print(f"  R_a abs max = {R_a.abs().max():.4e}")
    print(f"  R_a per-mode L^2 = {(R_a ** 2).sum(-1).sqrt()[:5].tolist()}")
    print()

    # Check 2: vary N_q (quadrature accuracy).
    for N_q in [8, 16, 32, 64, 128]:
        L_RV_ref, _, _ = lrv_at_reference(oracle, N_q=N_q)
        print(f"  N_q={N_q:4d}: L_RV at u_ref = {L_RV_ref:.4e}")
    print()

    # Check 3: vary N_fine (Brownian path resolution affecting I_a).
    for N_fine in [256, 1024, 4096, 16384]:
        oracle_f = WMContinuousOracle(d=D, sigma=SIGMA, K_max=K_MAX, T=T,
                                       L_test=L_TEST, N_fine=N_fine,
                                       seed=42, device=DEVICE, dtype=DTYPE)
        L_RV_ref, _, _ = lrv_at_reference(oracle_f, N_q=N_Q)
        print(f"  N_fine={N_fine:6d}: L_RV at u_ref = {L_RV_ref:.4e}")


if __name__ == "__main__":
    main()
