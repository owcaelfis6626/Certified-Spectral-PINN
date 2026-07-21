"""Mode enumeration utility, lifted from P6 sampler.

The mode set H_{K_max} = lex-half-space integer vectors with |k|^2 <= K_max^2.
One representative per cos/sin conjugate pair; k=0 excluded for our purposes
(DC unforced, but kept available for downstream code if needed).
"""

import itertools

import torch


def half_space_modes(d: int, K_max: int, exclude_zero: bool = True) -> torch.Tensor:
    """Integer modes k in Z^d with |k|^2 <= K_max^2 in lex-half-space."""
    K = K_max
    out = []
    for k in itertools.product(range(-K, K + 1), repeat=d):
        if sum(ki * ki for ki in k) > K * K:
            continue
        if all(ki == 0 for ki in k):
            if not exclude_zero:
                out.append(k)
            continue
        for i in range(d - 1, -1, -1):
            if k[i] != 0:
                if k[i] > 0:
                    out.append(k)
                break
    return torch.tensor(out, dtype=torch.int64)
