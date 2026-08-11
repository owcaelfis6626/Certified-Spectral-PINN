"""Gradient-checkpointed variant of loss_rv_ac, for dealiased runs at high d.

loss_rv_ac accumulates the cubic term in a `for q in range(N_q)` loop; every
iteration's intermediates (u_grid, U3, C_flat -- each N_x^d) stay alive for
backward. At d=5 that is ~670 MB per quadrature step on a 32^5 grid, so the
N_q = 2L = 256 needed to dealias the cubic (script 26) would want tens of GB.

Checkpointing the loop in chunks recomputes those intermediates during backward
instead of storing them: memory becomes O(chunk) rather than O(N_q), at the cost
of one extra forward pass. This is what makes a *dealiased* d=5 point reachable
at all.

Separate module on purpose: losses_ac.py is the published code path and the
running experiments import it, so it is left untouched.

Gate: `python3 28_ckpt_gate.py` asserts this reproduces loss_rv_ac's value AND
its gradients. It is a memory optimisation only -- any numerical difference
beyond fp32 round-off is a bug.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint

from losses_ac import cubic_proj_fft, gauss_legendre_01


def _cubic_chunk(alpha, beta, exp_chunk, phi_chunk, w_chunk,
                 a0, b0, modes_int, N_x, J, L):
    """Cubic contribution of one block of quadrature points.

    Recomputed during backward under checkpointing, so it must be a pure
    function of its tensor arguments.
    """
    device, dtype = alpha.device, alpha.dtype
    a_q = exp_chunk * a0[None, :] + phi_chunk @ alpha.T     # (n_chunk, J)
    b_q = exp_chunk * b0[None, :] + phi_chunk @ beta.T      # (n_chunk, J)

    ca = torch.zeros(J, L, device=device, dtype=dtype)
    cb = torch.zeros(J, L, device=device, dtype=dtype)
    for i in range(phi_chunk.shape[0]):
        ua3, ub3 = cubic_proj_fft(a_q[i], b_q[i], modes_int, N_x, device, dtype)
        ca = ca + w_chunk[i] * ua3[:, None] * phi_chunk[i].unsqueeze(0)
        cb = cb + w_chunk[i] * ub3[:, None] * phi_chunk[i].unsqueeze(0)
    return ca, cb


def loss_rv_ac_ckpt(model, oracle, N_q: int = 16, N_x: int = 32, chunk: int = 8):
    """Same value/gradients as loss_rv_ac, O(chunk) instead of O(N_q) memory."""
    device, dtype = oracle.device, oracle.dtype
    T, sigma = oracle.T, oracle.sigma
    J, L = oracle.J, oracle.L_test

    alpha, beta = model.coefficients(oracle)
    k2_1 = (oracle.k2 - 1.0)[:, None]

    # linear Galerkin part: exact, no quadrature (unchanged from loss_rv_ac)
    R_a = alpha @ oracle.M + k2_1 * alpha
    R_b = beta @ oracle.M + k2_1 * beta

    nodes_01, w_01 = gauss_legendre_01(N_q, device, dtype)
    t_q = T * nodes_01
    w_q = T * w_01
    phase = oracle.omega_l[None, :] * t_q[:, None]
    phi_q = math.sqrt(2.0 / T) * torch.sin(phase)              # (N_q, L)
    exp_decay = torch.exp(-k2_1.T * t_q[:, None])              # (N_q, J)

    cubic_a = torch.zeros(J, L, device=device, dtype=dtype)
    cubic_b = torch.zeros(J, L, device=device, dtype=dtype)
    for s in range(0, N_q, chunk):
        e = min(s + chunk, N_q)
        ca, cb = checkpoint(
            _cubic_chunk, alpha, beta,
            exp_decay[s:e], phi_q[s:e], w_q[s:e],
            oracle.a0, oracle.b0, oracle.modes_int, N_x, J, L,
            use_reentrant=False)
        cubic_a = cubic_a + ca
        cubic_b = cubic_b + cb

    R_a = R_a + cubic_a - sigma * oracle.I_a
    R_b = R_b + cubic_b - sigma * oracle.I_b

    weight = 1.0 / (1.0 + oracle.k2[:, None] + oracle.mu_l[None, :])
    return ((R_a ** 2 + R_b ** 2) * weight).sum()
