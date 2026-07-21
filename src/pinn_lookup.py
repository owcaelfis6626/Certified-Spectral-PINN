"""LookupGalerkinPINN: Galerkin coefficients as a direct parameter table.

Each (k, l) mode-test pair gets its own learnable scalar alpha_{k,l}, beta_{k,l}.
No MLP, no input embedding -- the parameters ARE the Galerkin coefficients.

Justification: for the linear SPDE, the optimal alpha^*_{k,l} =
(-M + |k|^2 I)^{-1} sigma I^a_{k,l} depends randomly on (k, l) through the
Wiener integrals I^a. An MLP can't smoothly interpolate random outputs.
Direct parameterization is the right architecture.

Cost: 2 J L parameters total. At d=7, K_max=3, L=64: 2 * 6218 * 64 ~ 800k
parameters. Still bounded by mode-set size, not by d.

For the linear case this is equivalent to the closed-form solve (SGD
converges to the same alpha^*). The reason to train iteratively rather
than solve directly is that the nonlinear AC case has no closed form --
the lookup-table parameterization is the foundation for the nonlinear
extension.
"""

import torch
import torch.nn as nn


class LookupGalerkinPINN(nn.Module):
    def __init__(self, oracle, init_scale: float = 0.0):
        super().__init__()
        self.d = oracle.d
        self.J = oracle.J
        self.L_test = oracle.L_test
        # Parameter tables: alpha[k, l], beta[k, l].
        self.alpha = nn.Parameter(init_scale * torch.randn(self.J, self.L_test,
                                                            device=oracle.device,
                                                            dtype=oracle.dtype))
        self.beta  = nn.Parameter(init_scale * torch.randn(self.J, self.L_test,
                                                            device=oracle.device,
                                                            dtype=oracle.dtype))

    def coefficients(self, oracle):
        """Match the MLP-Galerkin interface for loss_rv_galerkin."""
        return self.alpha, self.beta
