"""WMContinuousOracle: WM oracle for continuous-time RVPINN training.

Differences vs P6's WMNoiseOracle:
- Stores fine Brownian increments (not OU-consistent per-bin increments)
  so that arbitrary-t access and Wiener integrals against arbitrary test
  functions are well-defined.
- Precomputes scalar Wiener integrals I^{a/b}_{k,l} against the Dirichlet
  sine test basis psi_l(t) = sqrt(2/T) sin(pi l t / T), l=1..L.
- Exposes a closed-form reference solution a_k^ref(t), b_k^ref(t) for any
  t in [0, T] via the OU integral formula on the fine Brownian grid.

Pure linear SPDE: du = Delta u dt + sigma dW_{K_max} on T^d.
Mode-space: da_k = -|k|^2 a_k dt + sigma dB_k^a, ditto b_k.
Stationary IC: a_{k,0}, b_{k,0} ~ N(0, S(k)/2) with S(k) = sigma^2 / (2|k|^2).

DC mode (k=0) is excluded from the mode set (linear SPDE has no DC
contribution when the noise has zero mean).
"""

import math

import torch

from sampler import half_space_modes


class WMContinuousOracle:
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

        k2 = (self.modes_f ** 2).sum(-1)        # (J,) |k|^2 with k!=0 -> >= 1
        self.k2 = k2
        self.lambda_k = k2                       # decay rate of OU at each mode

        # Spectral density at non-DC modes: S(k) = sigma^2 / (2 |k|^2).
        # OU stationary variance per mode is sigma^2 / (2 lambda_k) = S(k).
        # In the cos/sin representation, Var(a_k) = Var(b_k) = S(k) so that
        # per-point Var(u(x)) = sum_k S(k) (P6 convention).
        self.S_k = (sigma * sigma) / (2.0 * k2)
        self.ic_std = torch.sqrt(self.S_k)

        # Fine Brownian increments. dt_fine = T / (N_fine - 1).
        dt_fine = T / (N_fine - 1)
        self.dt_fine = dt_fine
        self.t_fine = torch.linspace(0.0, T, N_fine, device=device, dtype=dtype)
        # Midpoints used for the deterministic-integrand Wiener sum.
        self.t_mid = 0.5 * (self.t_fine[:-1] + self.t_fine[1:])  # (N_fine-1,)

        torch.manual_seed(seed)
        bm_std = math.sqrt(dt_fine)
        # dB_a, dB_b have shape (J, N_fine-1); each entry ~ N(0, dt_fine).
        self.dB_a = (torch.randn(self.J, N_fine - 1, device=device, dtype=dtype)
                     * bm_std).detach()
        self.dB_b = (torch.randn(self.J, N_fine - 1, device=device, dtype=dtype)
                     * bm_std).detach()

        # Initial conditions sampled from stationary distribution.
        self.a0 = (torch.randn(self.J, device=device, dtype=dtype)
                   * self.ic_std).detach()
        self.b0 = (torch.randn(self.J, device=device, dtype=dtype)
                   * self.ic_std).detach()

        # Test basis psi_l(t) = sqrt(2/T) sin(pi l t / T), l=1..L_test.
        l_idx = torch.arange(1, L_test + 1, device=device, dtype=dtype)  # (L,)
        self.l_idx = l_idx
        self.mu_l = (math.pi * l_idx / T) ** 2                            # (L,)
        # psi at midpoints for the Wiener sum:
        # psi_l(t_mid_n) for shape (N_fine-1, L)
        phase = (math.pi * l_idx[None, :] / T) * self.t_mid[:, None]
        self.psi_mid = math.sqrt(2.0 / T) * torch.sin(phase)              # (N_fine-1, L)

        # Wiener integrals I^a_{k,l} = sum_n psi_l(t_mid_n) * dB^a[k, n].
        # dB_a: (J, N_fine-1); psi_mid: (N_fine-1, L). Result: (J, L).
        self.I_a = self.dB_a @ self.psi_mid     # (J, L)
        self.I_b = self.dB_b @ self.psi_mid     # (J, L)

        # Galerkin matrix M_{l,l'} = <psi_l', psi_{l'}> in L^2(0,T).
        # Closed form: -4ll'/(T(l^2-l'^2)) for l,l' different parity, else 0.
        # Used by the Galerkin (architecture A) L_RV.
        L = L_test
        l_grid = l_idx                          # (L,)
        # broadcast l (rows) and l' (cols)
        l_rows = l_grid[:, None]                # (L, 1)
        l_cols = l_grid[None, :]                # (1, L)
        denom  = (l_rows ** 2 - l_cols ** 2)     # (L, L), zero on diagonal
        parity_diff = ((l_rows + l_cols) % 2 == 1).to(dtype)  # 1 if different parity
        # Safe divide: replace 0s in denom by 1 (will be multiplied by parity_diff=0).
        denom_safe = torch.where(denom == 0,
                                  torch.ones_like(denom), denom.to(dtype))
        M = -4.0 * l_rows * l_cols * parity_diff / (T * denom_safe)
        self.M = M.to(dtype)                    # (L, L)

    # ------------------------------------------------------------
    # Reference solution access (diagnostics, not training)
    # ------------------------------------------------------------

    @torch.no_grad()
    def a_ref(self, t: torch.Tensor):
        """OU reference at time t (scalar or 1-D). Returns (T_batch, J) for a, b.

        Closed-form OU: a_k(t) = a_{k,0} e^{-lambda_k t}
                                + sigma sum_{n: t_mid_n < t} e^{-lambda_k (t - t_mid_n)} dB^a[k, n].
        """
        if t.ndim == 0:
            t = t[None]
        # Mask: which midpoints contribute to t? Those with t_mid_n < t.
        # (T_batch, N_fine-1)
        mask = (self.t_mid[None, :] < t[:, None]).to(self.dtype)
        # decay factor: e^{-lambda_k (t - t_mid_n)} for each (t, k, n).
        # Shape will be (T_batch, J, N_fine-1) — memory intensive.
        # For diagnostics at a small batch of t this is fine.
        lam = self.lambda_k                                            # (J,)
        # delta = t - t_mid_n   (T_batch, N_fine-1)
        delta = t[:, None] - self.t_mid[None, :]
        # decay factor exp(-lambda_k * delta) applied per (t, k, n)
        # Combine increments first to save memory:
        # contrib_k(t) = sum_n e^{-lambda_k delta(t, n)} mask(t, n) dB[k, n]
        # Compute by looping over batch dim of t if needed; here we vectorise.
        # delta_pos: relu(delta) so e^{-lambda * delta_pos} = 1 when delta < 0 (masked out)
        delta_pos = torch.clamp(delta, min=0.0)                        # (T_batch, N_fine-1)
        # Use einsum: for each t-row, contribution per mode is
        #   sum_n exp(-lambda_k delta_n) * mask_n * dB_kn
        # Tensor (T_batch, J): need exp(-lambda_k delta_n) shape (T_batch, J, N_fine-1).
        # That's (Tb, J, N) floats — at J=25, N=4096, Tb=16 it's 6.5M floats = 26 MB. OK.
        exp_factor = torch.exp(-lam[None, :, None] * delta_pos[:, None, :])  # (Tb, J, N)
        weighted = exp_factor * mask[:, None, :]                              # (Tb, J, N)

        # Decay of IC term: e^{-lambda_k t}
        ic_decay = torch.exp(-lam[None, :] * t[:, None])  # (Tb, J)
        a = ic_decay * self.a0[None, :] + self.sigma * (weighted * self.dB_a[None, :, :]).sum(-1)
        b = ic_decay * self.b0[None, :] + self.sigma * (weighted * self.dB_b[None, :, :]).sum(-1)
        return a, b   # (T_batch, J), (T_batch, J)

    @torch.no_grad()
    def u_init(self, points: torch.Tensor) -> torch.Tensor:
        """u_{ref}(x, 0) at (M, d) points -> (M,)."""
        kx = points @ self.modes_f.T
        return (self.a0 * torch.cos(kx) + self.b0 * torch.sin(kx)).sum(-1)
