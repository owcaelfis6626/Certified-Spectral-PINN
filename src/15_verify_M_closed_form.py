"""Verify Lemma 3.1: closed form for the half-period Galerkin matrix M.

We claim:
  M_{l,l}    = 1/T                                  (diagonal)
  M_{l,l'}   = (2l-1) / (T(l+l'-1))                 (l != l', l+l' even)
  M_{l,l'}   = -(2l-1) / (T(l-l'))                  (l != l', l+l' odd)

Compare to the oracle's M (computed by Gauss-Legendre quadrature on the
analytical dt phi_l and phi_{l'}).

Also verify the symmetric/antisymmetric decomposition:
  M_{l,l'} + M_{l',l} = (2/T) (-1)^{l+l'}          (off-diagonal)
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle_halfperiod import WMContinuousOracleHalfPeriod


def M_closed_form(l, lp, T):
    """Closed-form M_{l,l'} per Lemma 3.1."""
    if l == lp:
        return 1.0 / T
    if (l + lp) % 2 == 0:
        return (2 * l - 1) / (T * (l + lp - 1))
    return -(2 * l - 1) / (T * (l - lp))


def main():
    T = 1.0
    L = 12

    # Build oracle (we only need M; it's computed via Gauss-Legendre)
    oracle = WMContinuousOracleHalfPeriod(
        d=2, sigma=1.0, K_max=2, T=T, L_test=L, N_fine=512, seed=0,
        device="cpu", dtype=torch.float64,
    )
    M_oracle = oracle.M.cpu().numpy()              # (L, L)

    # Build closed-form M
    M_closed = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            M_closed[i, j] = M_closed_form(i + 1, j + 1, T)

    diff = M_oracle - M_closed
    print(f"L = {L}, T = {T}")
    print(f"max |M_oracle - M_closed| = {np.abs(diff).max():.3e}")
    print(f"||diff||_F                = {np.linalg.norm(diff):.3e}")
    print(f"||M_oracle||_F            = {np.linalg.norm(M_oracle):.3e}")
    print(f"rel error                 = {np.linalg.norm(diff)/np.linalg.norm(M_oracle):.3e}")
    print()

    # Print small block for inspection
    print("M_oracle[:6, :6] =")
    print(np.array2string(M_oracle[:6, :6], precision=4, suppress_small=True))
    print("\nM_closed[:6, :6] =")
    print(np.array2string(M_closed[:6, :6], precision=4, suppress_small=True))
    print()

    # Verify diagonal: M_{l,l} = 1/T
    print(f"Diagonal entries (should all be 1/T = {1/T:.3f}):")
    print(f"  oracle:  {[f'{M_oracle[i,i]:.4f}' for i in range(6)]}")
    print(f"  closed:  {[f'{M_closed[i,i]:.4f}' for i in range(6)]}")
    print()

    # Verify symmetric/antisymmetric decomposition
    M_sym = M_oracle + M_oracle.T
    print(f"Symmetric component M + M^T (off-diagonal, should equal 2/T * (-1)^(l+l')):")
    print(f"  diagonal (should be 2/T = {2/T:.3f}):")
    print(f"    {[f'{M_sym[i,i]:.4f}' for i in range(6)]}")
    print(f"  M+M^T[0,1] (l=1, l'=2 odd, should be -2/T):  {M_sym[0,1]:.4f}  expected {-2/T:.4f}")
    print(f"  M+M^T[0,2] (l=1, l'=3 even, should be +2/T): {M_sym[0,2]:.4f}  expected {2/T:.4f}")
    print(f"  M+M^T[0,3] (l=1, l'=4 odd, should be -2/T):  {M_sym[0,3]:.4f}  expected {-2/T:.4f}")
    print()

    # Also check T != 1 to confirm scaling
    T2 = 2.5
    oracle2 = WMContinuousOracleHalfPeriod(
        d=2, sigma=1.0, K_max=2, T=T2, L_test=L, N_fine=512, seed=0,
        device="cpu", dtype=torch.float64,
    )
    M_oracle2 = oracle2.M.cpu().numpy()
    M_closed2 = np.array([[M_closed_form(i+1, j+1, T2) for j in range(L)] for i in range(L)])
    diff2 = M_oracle2 - M_closed2
    print(f"T = {T2}:")
    print(f"  max |M_oracle - M_closed| = {np.abs(diff2).max():.3e}")
    print(f"  rel error                 = {np.linalg.norm(diff2)/np.linalg.norm(M_oracle2):.3e}")


if __name__ == "__main__":
    main()
