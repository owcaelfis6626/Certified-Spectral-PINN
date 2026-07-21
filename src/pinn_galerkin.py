"""GalerkinModePINN: trial = test space.

Network outputs the L-dim Dirichlet-sine coefficients alpha_{k,l}, beta_{k,l}
for each mode k. The full ansatz is

    a_{theta,k}(t) = a_{0,k} exp(-|k|^2 t) + sum_{l=1..L} alpha_{k,l} psi_l(t)
    b_{theta,k}(t) = b_{0,k} exp(-|k|^2 t) + sum_{l=1..L} beta_{k,l}  psi_l(t)
    psi_l(t) = sqrt(2/T) sin(pi l t / T)

Construction-time properties:
- a_{theta,k}(0) = a_{0,k} exactly (IC enforced; L_IC drops out).
- Trial subspace exactly equals the test space modulo the IC-kernel term
  -> discrete inf-sup equals continuous inf-sup, no L-dependent loss.

A single shared MLP maps (k_int, l_int) -> (alpha, beta), one forward per
(k, l) pair. Parameter count is d-independent except for the input layer.
"""

import math

import torch
import torch.nn as nn


class GalerkinModePINN(nn.Module):
    def __init__(
        self,
        d: int,
        L_test: int,
        K_emb_l: int = 4,
        hidden: int = 256,
        depth: int = 4,
    ):
        super().__init__()
        self.d = d
        self.L_test = L_test
        self.K_emb_l = K_emb_l

        # Input: k_int (d ints, cast to float)
        #      + |k|^2 (scalar)
        #      + l_scaled (l / L_test)
        #      + sinusoidal embed of l_scaled (2 K_emb_l)
        in_dim = d + 1 + 1 + 2 * K_emb_l
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.mlp = nn.Sequential(*layers)

    def _features(self, k_int: torch.Tensor, l_int: torch.Tensor) -> torch.Tensor:
        k_f = k_int.float()
        k_norm2 = (k_f * k_f).sum(-1, keepdim=True)
        l_f = l_int.float().unsqueeze(-1)
        l_scaled = l_f / self.L_test

        freqs = torch.arange(1, self.K_emb_l + 1,
                             dtype=k_f.dtype, device=k_f.device)
        phase = math.pi * freqs[None] * l_scaled
        l_emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        return torch.cat([k_f, k_norm2, l_scaled, l_emb], dim=-1)

    def forward(self, k_int: torch.Tensor, l_int: torch.Tensor) -> torch.Tensor:
        """k_int: (B, d), l_int: (B,). Returns (B, 2) = (alpha, beta)."""
        return self.mlp(self._features(k_int, l_int))

    def coefficients(self, oracle):
        """Return (alpha, beta) of shape (J, L) for all (k in H, l=1..L)."""
        J, L = oracle.J, oracle.L_test
        l_grid = torch.arange(1, L + 1, device=oracle.device)
        # Build (J*L, d) of mode repeats and (J*L,) of l repeats.
        k_rep = oracle.modes_int.repeat_interleave(L, dim=0)
        l_rep = l_grid.repeat(J)
        ab = self.forward(k_rep, l_rep)                  # (J*L, 2)
        alpha = ab[..., 0].view(J, L)
        beta  = ab[..., 1].view(J, L)
        return alpha, beta
