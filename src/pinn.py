"""ModeSpacePINN: maps (k_int, t) -> (a_k(t), b_k(t)).

Architecture: single shared MLP across modes. Input features:
- k_int (d ints, embedded as floats)
- |k|^2 (one scalar, prior on the PDE operator's spectrum)
- sinusoidal embedding of t with K_emb_t frequencies (2 K_emb_t scalars)

Input dim = d + 1 + 2 K_emb_t. Body fixed at hidden=256, depth=4.
The body is d-independent in parameter count except for the input layer.

Time derivatives are computed via autograd (continuous-time formulation),
matching option (a) of the theorem in notes/theorem_v1.md.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_time_embed(t: torch.Tensor, K_emb: int, T_period: float = 1.0):
    """t: (..., 1) -> (..., 2 K_emb) sinusoidal features."""
    freqs = torch.arange(1, K_emb + 1, device=t.device, dtype=t.dtype)
    phase = (2.0 * math.pi / T_period) * freqs[None] * t  # (..., K_emb)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class ModeSpacePINN(nn.Module):
    def __init__(
        self,
        d: int,
        K_emb_t: int = 8,
        hidden: int = 256,
        depth: int = 4,
        T_period: float = 1.0,
    ):
        super().__init__()
        self.d = d
        self.K_emb_t = K_emb_t
        self.T_period = T_period

        in_dim = d + 1 + 2 * K_emb_t
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.mlp = nn.Sequential(*layers)

    def _features(self, k_int: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """k_int: (B, d) integers (cast to float); t: (B,) floats. -> (B, in_dim)."""
        k_f = k_int.to(t.dtype)
        k_norm2 = (k_f * k_f).sum(-1, keepdim=True)
        t_emb = sinusoidal_time_embed(t.unsqueeze(-1), self.K_emb_t, self.T_period)
        return torch.cat([k_f, k_norm2, t_emb], dim=-1)

    def forward(self, k_int: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns (B, 2) — [a_k(t), b_k(t)]."""
        return self.mlp(self._features(k_int, t))

    def with_dt(self, k_int: torch.Tensor, t: torch.Tensor):
        """Returns (a, b, dt_a, dt_b), each shape (B,).

        Time derivative via autograd. Pass t with requires_grad already set
        if you want the dt outputs to participate in backprop later.
        """
        if not t.requires_grad:
            t = t.detach().clone().requires_grad_(True)
        ab = self.forward(k_int, t)
        a, b = ab[..., 0], ab[..., 1]
        ones = torch.ones_like(a)
        dt_a = torch.autograd.grad(a, t, grad_outputs=ones, create_graph=True)[0]
        dt_b = torch.autograd.grad(b, t, grad_outputs=ones, create_graph=True)[0]
        return a, b, dt_a, dt_b
