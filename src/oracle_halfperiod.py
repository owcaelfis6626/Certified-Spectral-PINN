"""WMContinuousOracleHalfPeriod: same WM oracle but with half-period sine
test basis phi_l(t) = sqrt(2/T) sin((2l-1) pi t / (2T)), l=1..L.

These satisfy:
- phi_l(0) = 0    (Dirichlet at t=0 — preserves IC term in the trial space)
- phi_l(T) = +-1  (FREE at t=T — captures terminal noise contribution)

This is the parabolic-natural basis (Dirichlet at 0, Neumann at T) for the
heat operator. The trial space's terminal value is free, removing the
artificial constraint that defeated the full-period Dirichlet sines.

Off-diagonal Galerkin matrix M is recomputed for this basis.
"""

import math

import torch

from sampler import half_space_modes


class WMContinuousOracleHalfPeriod:
    def __init__(
        self,
        d: int,
        sigma: float,
        K_max: int,
        T: float = 1.0,
        L_test: int = 32,
        N_fine: int = 4096,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.d, self.sigma, self.K_max = d, sigma, K_max
        self.T, self.L_test, self.N_fine = T, L_test, N_fine
        self.device, self.dtype = device, dtype

        modes_int = half_space_modes(d, K_max, exclude_zero=True)
        self.modes_int = modes_int.to(device)
        self.modes_f = modes_int.to(device=device, dtype=dtype)
        self.J = modes_int.shape[0]

        k2 = (self.modes_f ** 2).sum(-1)
        self.k2 = k2
        self.lambda_k = k2
        self.S_k = (sigma * sigma) / (2.0 * k2)
        self.ic_std = torch.sqrt(self.S_k)

        dt_fine = T / (N_fine - 1)
        self.dt_fine = dt_fine
        self.t_fine = torch.linspace(0.0, T, N_fine, device=device, dtype=dtype)
        self.t_mid = 0.5 * (self.t_fine[:-1] + self.t_fine[1:])

        torch.manual_seed(seed)
        bm_std = math.sqrt(dt_fine)
        self.dB_a = (torch.randn(self.J, N_fine - 1, device=device, dtype=dtype)
                     * bm_std).detach()
        self.dB_b = (torch.randn(self.J, N_fine - 1, device=device, dtype=dtype)
                     * bm_std).detach()

        self.a0 = (torch.randn(self.J, device=device, dtype=dtype)
                   * self.ic_std).detach()
        self.b0 = (torch.randn(self.J, device=device, dtype=dtype)
                   * self.ic_std).detach()

        # Half-period sine basis: phi_l(t) = sqrt(2/T) sin((2l-1) pi t / (2T))
        # Eigenvalue of -d^2/dt^2 with Dirichlet@0, Neumann@T is ((2l-1) pi / (2T))^2.
        l_idx = torch.arange(1, L_test + 1, device=device, dtype=dtype)
        self.l_idx = l_idx
        self.omega_l = (2 * l_idx - 1) * math.pi / (2.0 * T)         # (L,) angular frequencies
        self.mu_l = self.omega_l ** 2                                # (L,) sobolev weights

        # phi_l at fine-grid midpoints (for Wiener integral).
        phase = self.omega_l[None, :] * self.t_mid[:, None]
        self.phi_mid = math.sqrt(2.0 / T) * torch.sin(phase)         # (N_fine-1, L)

        # Wiener integrals I^{a/b}_{k,l} = int_0^T phi_l(t) dB^{a/b}_k(t).
        self.I_a = self.dB_a @ self.phi_mid                          # (J, L)
        self.I_b = self.dB_b @ self.phi_mid

        # Galerkin matrix: M_{l, l'} = <dt phi_l, phi_{l'}>_{L^2(0,T)}
        # = sqrt(2/T) omega_l int_0^T cos(omega_l t) sqrt(2/T) sin(omega_{l'} t) dt
        # Compute via Gauss-Legendre quadrature on (0, T) for accuracy.
        N_q = max(4 * L_test, 256)
        nodes, weights = self._gauss_legendre_01(N_q)
        t_q = (T * nodes).to(device=device, dtype=dtype)              # (N_q,)
        w_q = (T * weights).to(device=device, dtype=dtype)            # (N_q,)

        # dt phi_l(t_q) = sqrt(2/T) omega_l cos(omega_l t_q):  (L, N_q)
        cos_phase = self.omega_l[:, None] * t_q[None, :]
        dt_phi = math.sqrt(2.0 / T) * self.omega_l[:, None] * torch.cos(cos_phase)
        # phi_{l'}(t_q):  (L, N_q)
        sin_phase = self.omega_l[:, None] * t_q[None, :]
        phi_q = math.sqrt(2.0 / T) * torch.sin(sin_phase)

        # M_{l, l'} = sum_q w_q * dt_phi[l, q] * phi[l', q]
        # = (dt_phi * w_q) @ phi.T  -> (L, L)
        self.M = (dt_phi * w_q[None, :]) @ phi_q.T

    @staticmethod
    def _gauss_legendre_01(N_q):
        import numpy as np
        nodes, weights = np.polynomial.legendre.leggauss(N_q)
        return (torch.tensor(0.5 * (nodes + 1.0)),
                torch.tensor(0.5 * weights))

    @torch.no_grad()
    def a_ref(self, t: torch.Tensor):
        """OU reference at time t (1-D). Returns (T_batch, J) for both a and b."""
        if t.ndim == 0:
            t = t[None]
        mask = (self.t_mid[None, :] < t[:, None]).to(self.dtype)
        delta = t[:, None] - self.t_mid[None, :]
        delta_pos = torch.clamp(delta, min=0.0)
        exp_factor = torch.exp(-self.lambda_k[None, :, None] * delta_pos[:, None, :])
        weighted = exp_factor * mask[:, None, :]
        ic_decay = torch.exp(-self.lambda_k[None, :] * t[:, None])
        a = ic_decay * self.a0[None, :] + self.sigma * (weighted * self.dB_a[None, :, :]).sum(-1)
        b = ic_decay * self.b0[None, :] + self.sigma * (weighted * self.dB_b[None, :, :]).sum(-1)
        return a, b
