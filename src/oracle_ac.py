"""Allen-Cahn oracle for P7.

∂_t u = Δu + u - u³ + σ Ẇ  on [0, 2π]^d

Mode-space form (cos/sin Fourier basis, k with |k|₂ ≤ K_max, k ≠ 0):
  d a_k = (1 - |k|²) a_k dt - [u³]^a_k dt + σ dB^a_k
  d b_k = (1 - |k|²) b_k dt - [u³]^b_k dt + σ dB^b_k

Reference computed by Euler-Maruyama on the same Brownian paths used for
the Wiener integrals. Test basis, M, I_a, I_b are identical to the OU oracle.
"""

import torch

from oracle_halfperiod import WMContinuousOracleHalfPeriod
from losses_ac import cubic_proj_fft


class ACOracleHalfPeriod(WMContinuousOracleHalfPeriod):
    """Allen-Cahn oracle. Inherits OU oracle; overrides reference with EM."""

    def __init__(self, N_x: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.N_x = N_x
        self._run_em_reference()

    def _run_em_reference(self):
        """Euler-Maruyama for AC in mode space. Stores (N_fine, J) arrays."""
        dt = self.dt_fine
        lam = 1.0 - self.k2          # (J,)  linear drift eigenvalue

        a = self.a0.clone()
        b = self.b0.clone()

        a_traj = torch.zeros(self.N_fine, self.J, device=self.device, dtype=self.dtype)
        b_traj = torch.zeros(self.N_fine, self.J, device=self.device, dtype=self.dtype)
        a_traj[0] = a
        b_traj[0] = b

        with torch.no_grad():
            for i in range(self.N_fine - 1):
                ua3, ub3 = cubic_proj_fft(a, b, self.modes_int, self.N_x,
                                          self.device, self.dtype)
                a = a + dt * (lam * a - ua3) + self.sigma * self.dB_a[:, i]
                b = b + dt * (lam * b - ub3) + self.sigma * self.dB_b[:, i]
                a_traj[i + 1] = a
                b_traj[i + 1] = b

        self.a_traj = a_traj   # (N_fine, J)
        self.b_traj = b_traj

    @torch.no_grad()
    def a_ref(self, t: torch.Tensor):
        """Linear-interpolated EM reference at time t. Returns (T_batch, J)."""
        if t.ndim == 0:
            t = t[None]
        t = t.clamp(0.0, self.T)
        i_lo = (t / self.dt_fine).long().clamp(0, self.N_fine - 2)
        frac = (t / self.dt_fine - i_lo.float()).clamp(0.0, 1.0)
        a = (1 - frac[:, None]) * self.a_traj[i_lo] + frac[:, None] * self.a_traj[i_lo + 1]
        b = (1 - frac[:, None]) * self.b_traj[i_lo] + frac[:, None] * self.b_traj[i_lo + 1]
        return a, b
