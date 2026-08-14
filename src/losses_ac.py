"""RVPINN loss for the Allen-Cahn SPDE.

Residual split into linear (exact Galerkin) and cubic (quadrature) parts:

  R^a_{k,l} = [Ma]_{k,l} + (k²-1) α_{k,l}   ← linear Galerkin, no IBP
             + ∫ [u_θ³]^a_k φ_l dt            ← cubic, via FFT quadrature
             - σ I^a_{k,l}                     ← Wiener integral

Trial IC decay uses exp(-(k²-1)t) — in the kernel of the AC linear operator,
so IC contributions cancel in the Galerkin integral exactly (same as OU).

IBP is intentionally avoided: for the half-period basis φ_l(T) ≠ 0, so IBP
introduces a boundary term a_k(T)φ_l(T) that would corrupt the residual.

Cubic projection [u_θ³]^a_k via FFT on an N_x^d spatial grid:
  u(x) = Σ_k [a_k cos(k·x) + b_k sin(k·x)]
  [u³]^a_k = 2 Re(FFT(u³)[k]) / N_x^d
  [u³]^b_k = -2 Im(FFT(u³)[k]) / N_x^d

Requires N_x > 2 * 3 * K_max to avoid aliasing from the cubic.
With K_max=3: N_x ≥ 19; we use N_x=32 for all d.

The SAME dealiasing argument applies in TIME and was originally missed: the
quadrature evaluates u at N_q Gauss-Legendre nodes, cubes it pointwise, and
projects onto L half-period sine modes, so the temporal grid must resolve the
tripled bandwidth too. Measured directly in `26_cubic_quadrature_error.py`
(L=128, relative error of this cubic tensor vs a high-N_q reference):

    N_q= 16 -> 2.70      N_q=128 (=L)  -> 8.7e-3
    N_q= 32 -> 1.78      N_q=256 (=2L) -> 6.4e-6  (fp32 floor)

i.e. N_q must scale as ~2L, and the historical fixed N_q=16 made the cubic
term wrong by more than the term itself at large L -- worsening along exactly
the axis the L-convergence study sweeps. `_warn_temporal_aliasing` below makes
this loud instead of silent; it warns rather than asserts so the published
N_q=16 numbers stay reproducible.
"""

import math

import numpy as np
import torch


def cubic_proj_fft(a, b, modes_int, N_x, device, dtype):
    """Project u³ onto each mode in modes_int via FFT.

    a, b : (J,) Fourier coefficients of u (real, may carry grad)
    Returns (ua3, ub3) : (J,) cos/sin coefficients of u³
    """
    d = modes_int.shape[1]
    Nd = N_x ** d

    # c_k = (a_k - i b_k) / 2  so  u = Σ_k [c_k e^{ik·x} + c̄_k e^{-ik·x}]
    c = torch.complex(a, -b) / 2          # (J,) complex64

    strides = torch.tensor([N_x ** (d - 1 - i) for i in range(d)],
                           device=device, dtype=torch.int64)
    k_pos = modes_int.to(torch.int64) % N_x   # (J, d)
    k_neg = (-modes_int).to(torch.int64) % N_x
    lin_pos = (k_pos * strides).sum(-1)        # (J,)
    lin_neg = (k_neg * strides).sum(-1)

    C_flat = torch.zeros(Nd, dtype=torch.complex64, device=device)
    C_flat = C_flat.index_add(0, lin_pos, c.to(torch.complex64))
    C_flat = C_flat.index_add(0, lin_neg, c.conj().to(torch.complex64))

    # u on N_x^d grid; IFFT convention: ifftn(X)[n] = (1/N) Σ_k X[k] e^{2πi k n/N}
    u_grid = (torch.fft.ifftn(C_flat.view([N_x] * d)) * Nd).real
    U3 = torch.fft.fftn(u_grid ** 3).reshape(-1)  # (Nd,) complex

    u3_k = U3[lin_pos]                             # (J,) complex
    return (2 * u3_k.real / Nd).to(dtype), (-2 * u3_k.imag / Nd).to(dtype)


def gauss_legendre_01(N_q, device, dtype):
    nodes, weights = np.polynomial.legendre.leggauss(N_q)
    t = torch.tensor(0.5 * (nodes + 1.0), device=device, dtype=dtype)
    w = torch.tensor(0.5 * weights,        device=device, dtype=dtype)
    return t, w


_ALIAS_WARNED = set()


def _warn_temporal_aliasing(N_q, L):
    """Cubic dealiasing in time needs N_q ~ 2L (see module docstring). Warn once
    per (N_q, L) so a sweep does not spam, but never stay silent about it."""
    if N_q >= 2 * L or (N_q, L) in _ALIAS_WARNED:
        return
    _ALIAS_WARNED.add((N_q, L))
    import warnings
    warnings.warn(
        f"temporal aliasing: N_q={N_q} < 2*L={2 * L} for the cubic term; "
        f"the cubic residual is under-resolved (at L=128, N_q=16 gives ~270% "
        f"relative error -- see 26_cubic_quadrature_error.py). Convergence "
        f"rates measured by sweeping L at fixed N_q are not trustworthy.",
        RuntimeWarning, stacklevel=2)


def cubic_proj_fft_batched(a, b, modes_int, N_x, device, dtype):
    """cubic_proj_fft over a leading quadrature axis: (Q, J) in, (Q, J) out.

    Mathematically identical to looping cubic_proj_fft over q -- every operation is
    elementwise or an FFT over the trailing d axes, so a batch axis passes straight
    through. The point is dispatch cost, not arithmetic: the per-q loop issues ~15 tiny
    kernels per quadrature node, 5000 times per training run, which leaves the GPU idle
    ~89% of the time (measured: modal 1-2% utilisation, 38 W of a ~300 W card).

    Two differences from the unbatched function, both deliberate:
      * the complex dtype is DERIVED from `dtype` rather than pinned to complex64, so an
        fp64 run is actually fp64. The fp32 path is unchanged (float32 -> complex64).
      * results are summed by matmul rather than sequential accumulation, so in float32
        the last bits differ from the loop by summation order. In float64 the two agree
        to ~1e-15; see 35_batching_equivalence_gate.py.
    """
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    Q = a.shape[0]
    d = modes_int.shape[1]
    Nd = N_x ** d
    fft_dims = tuple(range(1, d + 1))

    c = torch.complex(a, -b) / 2                                    # (Q, J)
    strides = torch.tensor([N_x ** (d - 1 - i) for i in range(d)],
                           device=device, dtype=torch.int64)
    lin_pos = ((modes_int.to(torch.int64) % N_x) * strides).sum(-1)   # (J,)
    lin_neg = (((-modes_int).to(torch.int64) % N_x) * strides).sum(-1)

    C_flat = torch.zeros(Q, Nd, dtype=cdtype, device=device)
    C_flat = C_flat.index_add(1, lin_pos, c.to(cdtype))
    C_flat = C_flat.index_add(1, lin_neg, c.conj().to(cdtype))

    u_grid = (torch.fft.ifftn(C_flat.view(Q, *([N_x] * d)), dim=fft_dims) * Nd).real
    U3 = torch.fft.fftn(u_grid ** 3, dim=fft_dims).reshape(Q, Nd)
    u3_k = U3[:, lin_pos]                                            # (Q, J)
    return (2 * u3_k.real / Nd).to(dtype), (-2 * u3_k.imag / Nd).to(dtype)


def _default_chunk(N_q: int, N_x: int, d: int, dtype) -> int:
    """Largest q-block whose (chunk, N_x^d) complex buffer stays under ~256 MB.

    Batching trades launches for memory, and at d=4 the grid is N_x^4 = 1,048,576 points,
    so the full N_q=256 batch would want ~2 GB for one intermediate and several times that
    with the autograd graph. Chunking keeps the win while bounding the peak; chunk=1
    reproduces the original loop exactly.
    """
    itemsize = 16 if dtype == torch.float64 else 8
    per_q = (N_x ** d) * itemsize
    return max(1, min(N_q, (256 << 20) // max(per_q, 1)))


def loss_rv_ac(model, oracle, N_q: int = 16, N_x: int = 32, chunk: int = None):
    """RVPINN loss for Allen-Cahn."""
    _warn_temporal_aliasing(N_q, oracle.L_test)
    device, dtype = oracle.device, oracle.dtype
    T, sigma = oracle.T, oracle.sigma
    J, L = oracle.J, oracle.L_test

    alpha, beta = model.coefficients(oracle)    # (J, L)
    k2_1 = (oracle.k2 - 1.0)[:, None]          # (J, 1)

    # Linear Galerkin residual (exact — IC cancels, no boundary term issues)
    Ma = alpha @ oracle.M                       # (J, L)
    Mb = beta  @ oracle.M
    R_a = Ma + k2_1 * alpha                     # (J, L)
    R_b = Mb + k2_1 * beta

    # Cubic residual: ∫_0^T [u_θ(·,t)³]^{a,b}_k φ_l(t) dt via quadrature
    # u_θ at each t_q includes BOTH IC decay and α-part (IC matters for cubic)
    nodes_01, w_01 = gauss_legendre_01(N_q, device, dtype)
    t_q = T * nodes_01                          # (N_q,)
    w_q = T * w_01

    phase = oracle.omega_l[None, :] * t_q[:, None]              # (N_q, L)
    phi_q = math.sqrt(2.0 / T) * torch.sin(phase)               # (N_q, L)

    exp_decay = torch.exp(-k2_1.T * t_q[:, None])               # (N_q, J)
    a_q = exp_decay * oracle.a0[None, :] + phi_q @ alpha.T      # (N_q, J)
    b_q = exp_decay * oracle.b0[None, :] + phi_q @ beta.T

    # Batched over the quadrature axis in blocks of `chunk`. The per-q loop this replaces
    # was dispatch-bound: ~15 tiny kernel launches per node, N_q nodes, 5000 steps.
    if chunk is None:
        chunk = _default_chunk(N_q, N_x, oracle.modes_int.shape[1], dtype)
    cubic_a = torch.zeros(J, L, device=device, dtype=dtype)
    cubic_b = torch.zeros(J, L, device=device, dtype=dtype)
    for q0 in range(0, N_q, chunk):
        sl = slice(q0, min(q0 + chunk, N_q))
        ua3, ub3 = cubic_proj_fft_batched(a_q[sl], b_q[sl], oracle.modes_int,
                                          N_x, device, dtype)
        wq = w_q[sl][:, None]                                   # (q, 1)
        # cubic[j,l] = sum_q w_q * u3[q,j] * phi_q[q,l]  -- one matmul, not q rank-1 updates
        cubic_a = cubic_a + (ua3 * wq).T @ phi_q[sl]
        cubic_b = cubic_b + (ub3 * wq).T @ phi_q[sl]

    R_a = R_a + cubic_a - sigma * oracle.I_a   # (J, L)
    R_b = R_b + cubic_b - sigma * oracle.I_b

    weight = 1.0 / (1.0 + oracle.k2[:, None] + oracle.mu_l[None, :])
    return ((R_a ** 2 + R_b ** 2) * weight).sum()
