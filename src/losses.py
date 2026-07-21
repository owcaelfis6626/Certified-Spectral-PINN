"""RVPINN losses for the continuous-time linear SPDE.

L_RV(theta) = sum_{k in H, l=1..L} (|R^a_{k,l}|^2 + |R^b_{k,l}|^2) / (1 + |k|^2 + mu_l)
where R^a_{k,l} = <Lop a_{theta,k}, psi_l>_{[0,T]} - sigma I^a_{k,l}
       Lop a_k = dt a_k + |k|^2 a_k.
The temporal integral <Lop, psi_l> is approximated by Gauss-Legendre
quadrature on [0, T] with N_q nodes; psi_l(t) = sqrt(2/T) sin(pi l t / T).

L_IC(theta) = ||u_theta(., 0) - u_{ref}(., 0)||_{L^2(T^d)}^2
            = pi^d sum_k (|a_{theta,k}(0) - a_{k,0}|^2 + ...).

Conventions:
- Mode set H_{K_max} as enumerated by oracle.modes_int.
- Quadrature nodes t_q in (0, T) (strictly interior; psi_l vanishes at
  the endpoints anyway).
- All tensors live on oracle.device.
"""

import math

import numpy as np
import torch

from pinn import ModeSpacePINN
from oracle import WMContinuousOracle


def gauss_legendre_01(N_q: int, device, dtype):
    """Gauss-Legendre nodes/weights on (0, 1). Cached if N_q is small."""
    nodes, weights = np.polynomial.legendre.leggauss(N_q)  # on (-1, 1)
    nodes_01 = 0.5 * (nodes + 1.0)
    weights_01 = 0.5 * weights
    return (torch.tensor(nodes_01, device=device, dtype=dtype),
            torch.tensor(weights_01, device=device, dtype=dtype))


def loss_rv_continuous(
    model: ModeSpacePINN,
    oracle: WMContinuousOracle,
    N_q: int = 64,
):
    """RVPINN continuous-time loss in WEAK form (integration by parts in t).

    The strong residual R(t) = dt a + |k|^2 a - sigma xi(t) is ill-defined
    pointwise (white-noise xi). The weak residual against psi_l is
        <R, psi_l> = int_0^T (dt a + |k|^2 a) psi_l dt - sigma I_{k,l}
                   = -int_0^T a psi_l' dt + |k|^2 int_0^T a psi_l dt
                     + [a psi_l]_0^T - sigma I_{k,l}
    Boundary term vanishes since psi_l(0) = psi_l(T) = 0.
    The IC information enters through L_IC, not L_RV.

    Why weak form not strong: with autograd on the smooth network's dt a,
    the LHS is exactly the strong-form integral via Gauss-Legendre, but
    sigma I_{k,l} is a midpoint Riemann sum on the fine Brownian grid.
    These don't agree at finite N_q -- the network exploits the mismatch
    to drive L_RV below the discretization floor of the true reference.
    Weak form has no time derivative anywhere: psi_l' is analytical,
    sigma I_{k,l} is the genuinely defined Wiener integral.
    """
    device = oracle.device
    dtype = oracle.dtype
    T = oracle.T
    sigma = oracle.sigma
    J, L = oracle.J, oracle.L_test

    nodes_01, w_01 = gauss_legendre_01(N_q, device, dtype)
    t_q = T * nodes_01                                              # (N_q,)
    w_q = T * w_01                                                   # (N_q,)

    # psi_l(t_q) and psi_l'(t_q).
    phase = (math.pi * oracle.l_idx[None, :] / T) * t_q[:, None]    # (N_q, L)
    psi_q = math.sqrt(2.0 / T) * torch.sin(phase)                    # (N_q, L)
    coeff = math.sqrt(2.0 / T) * (math.pi * oracle.l_idx / T)        # (L,)
    dpsi_q = coeff[None, :] * torch.cos(phase)                       # (N_q, L)

    # Forward network at all (k, t_q) pairs -- no autograd through t.
    K_rep = oracle.modes_int.repeat_interleave(N_q, dim=0)          # (J*N_q, d)
    T_rep = t_q.repeat(J)                                            # (J*N_q,)
    ab = model(K_rep, T_rep)                                         # (J*N_q, 2)
    a = ab[..., 0].view(J, N_q)
    b = ab[..., 1].view(J, N_q)

    # Weak-form projection:
    #   proj = -int a psi_l' dt + |k|^2 int a psi_l dt
    int_kernel_p  = psi_q  * w_q[:, None]                            # (N_q, L)
    int_kernel_dp = dpsi_q * w_q[:, None]                            # (N_q, L)
    int_a_psi  = a @ int_kernel_p                                    # (J, L)
    int_a_dpsi = a @ int_kernel_dp
    int_b_psi  = b @ int_kernel_p
    int_b_dpsi = b @ int_kernel_dp

    k2 = oracle.k2[:, None]                                          # (J, 1)
    proj_a = -int_a_dpsi + k2 * int_a_psi                            # (J, L)
    proj_b = -int_b_dpsi + k2 * int_b_psi

    R_a = proj_a - sigma * oracle.I_a                                # (J, L)
    R_b = proj_b - sigma * oracle.I_b

    weight = 1.0 / (1.0 + oracle.k2[:, None] + oracle.mu_l[None, :])
    L_RV = ((R_a ** 2 + R_b ** 2) * weight).sum()
    return L_RV


def loss_rv_galerkin(model, oracle: WMContinuousOracle):
    """Closed-form L_RV for the Galerkin (architecture A) PINN.

    a_{theta,k}(t) = a_{0,k} exp(-|k|^2 t) + sum_l alpha_{k,l} psi_l(t).
    The IC term is in the kernel of dt + |k|^2 (exact homogeneous heat
    solution), so it contributes nothing to the weak residual. The
    residual against psi_{l'} is

        R^a_{k,l'} = sum_l alpha_{k,l} M_{l,l'} + |k|^2 alpha_{k,l'} - sigma I^a_{k,l'}

    where M_{l,l'} = <psi_l', psi_{l'}> is the Galerkin matrix (closed
    form in oracle.M). Same for the b/beta block.

    Loss:
        L_RV = sum_{k,l'} (R^a_{k,l'}^2 + R^b_{k,l'}^2) / (1 + |k|^2 + mu_{l'})
    """
    alpha, beta = model.coefficients(oracle)             # (J, L)
    k2 = oracle.k2[:, None]                              # (J, 1)

    # alpha @ M gives sum_l alpha[k, l] M[l, l'] -> (J, L).
    Ma = alpha @ oracle.M                                # (J, L)
    Mb = beta  @ oracle.M

    R_a = Ma + k2 * alpha - oracle.sigma * oracle.I_a    # (J, L)
    R_b = Mb + k2 * beta  - oracle.sigma * oracle.I_b

    weight = 1.0 / (1.0 + oracle.k2[:, None] + oracle.mu_l[None, :])
    return ((R_a ** 2 + R_b ** 2) * weight).sum()


def loss_ic(model: ModeSpacePINN, oracle: WMContinuousOracle):
    """IC loss in mode space: pi^d sum_k (|a_th_k(0) - a_{k,0}|^2 + ...).

    Independent of d in compute (J modes only), the pi^d scale just keeps
    the same normalisation as the spatiotemporal L^2 norm.
    """
    t0 = torch.zeros(oracle.J, device=oracle.device, dtype=oracle.dtype)
    ab = model(oracle.modes_int, t0)
    a, b = ab[..., 0], ab[..., 1]
    err = ((a - oracle.a0) ** 2 + (b - oracle.b0) ** 2).sum()
    return (math.pi ** oracle.d) * err
