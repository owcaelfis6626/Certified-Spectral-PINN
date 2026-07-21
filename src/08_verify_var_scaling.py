"""Verify the theorem v2 best-approximation bound.

For the OU stochastic integral
    x_k(t) = sigma int_0^t e^{-|k|^2 (t-s)} dB_k(s),
the L^2 projection onto the Dirichlet sine basis gives coefficients
    hat_x_{k,l} = int_0^T psi_l(t) x_k(t) dt,
with variance (by Ito isometry)
    Var(hat_x_{k,l}) = sigma^2 int_0^T [int_s^T psi_l(t) e^{-|k|^2 (t-s)} dt]^2 ds.

Theorem v2 claims Var ~ C_0 sigma^2 T^2 / l^2 for large l.
Verify by computing the inner integral analytically.
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIGMA = 1.0
T = 1.0
KS_TO_TEST = [1.0, 2.0, 3.0, 4.0, 9.0]   # |k|^2 values
L_VALUES = np.arange(1, 257)


def var_hat_x_kl_exact(k2, l, T=T, sigma=SIGMA):
    """Exact closed form for Var(hat_x_{k,l}).

    Inner integral I_l(s) = int_s^T psi_l(t) e^{-k2 (t-s)} dt.
    With psi_l(t) = sqrt(2/T) sin(pi l t / T):
        I_l(s) = sqrt(2/T) int_0^{T-s} sin(pi l (u+s)/T) e^{-k2 u} du
    Using sin(a + b) expansion and standard integrals
        int_0^X sin(omega u) e^{-lambda u} du = (omega - e^{-lambda X}(omega cos(omega X) + lambda sin(omega X))) / (omega^2 + lambda^2)
        int_0^X cos(omega u) e^{-lambda u} du = (lambda + e^{-lambda X}(omega sin(omega X) - lambda cos(omega X))) / (omega^2 + lambda^2)
    with omega = pi l / T, lambda = k2, X = T - s.

    Then Var = sigma^2 int_0^T I_l(s)^2 ds.

    We integrate the I_l(s)^2 numerically (smooth integrand).
    """
    omega = math.pi * l / T
    lam = k2
    pref = math.sqrt(2.0 / T)
    denom = omega ** 2 + lam ** 2

    # Quadrature in s.
    N_q = 256
    s, w = np.polynomial.legendre.leggauss(N_q)
    s = 0.5 * (s + 1.0) * T          # [0, T]
    w = 0.5 * w * T

    X = T - s                          # (N_q,)
    s_phase = omega * s
    X_phase = omega * X

    e_lam_X = np.exp(-lam * X)
    sin_int = (omega - e_lam_X * (omega * np.cos(X_phase) + lam * np.sin(X_phase))) / denom
    cos_int = (lam + e_lam_X * (omega * np.sin(X_phase) - lam * np.cos(X_phase))) / denom

    # sin(omega(u+s)) = sin(omega u)cos(omega s) + cos(omega u)sin(omega s)
    I = pref * (sin_int * np.cos(s_phase) + cos_int * np.sin(s_phase))

    var = sigma ** 2 * (w * I ** 2).sum()
    return var


def tail_sum_predicted(k2, L_cutoff, T=T, sigma=SIGMA):
    """E[||x_k - P_L x_k||^2_{L^2}] = sum_{l>L} Var(hat_x_{k,l})."""
    s = 0.0
    for l in range(L_cutoff + 1, 1024):
        s += var_hat_x_kl_exact(k2, l, T, sigma)
    return s


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # ─── Plot 1: Var(hat_x_{k,l}) vs l for several |k|^2 ───────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for k2 in KS_TO_TEST:
        vars_l = np.array([var_hat_x_kl_exact(k2, l) for l in L_VALUES])
        ax1.loglog(L_VALUES, vars_l, marker='o', markersize=2,
                   label=f'|k|² = {k2:.0f}')
    # Reference 1/l^2 line
    l_ref = L_VALUES.astype(float)
    ax1.loglog(l_ref, (l_ref / 10) ** -2 * 0.01, 'k--', alpha=0.5,
               label='∝ 1/l²')
    ax1.set_xlabel('l')
    ax1.set_ylabel(r'$\mathrm{Var}(\hat x_{k,l})$')
    ax1.set_title('Mode-l variance, several |k|²')
    ax1.legend(fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)

    # ─── Plot 2: l^2 · Var(hat_x_{k,l}) -- should plateau ──────────────
    for k2 in KS_TO_TEST:
        vars_l = np.array([var_hat_x_kl_exact(k2, l) for l in L_VALUES])
        ax2.semilogx(L_VALUES, L_VALUES ** 2 * vars_l, marker='o', markersize=2,
                     label=f'|k|² = {k2:.0f}')
    ax2.set_xlabel('l')
    ax2.set_ylabel(r'$l^2 \cdot \mathrm{Var}(\hat x_{k,l})$')
    ax2.set_title('Scaled variance — plateau = $C_0\,\sigma^2 T^2$')
    ax2.axhline(2 * SIGMA ** 2 * T ** 2 / math.pi ** 2, color='k', linestyle='--',
                alpha=0.5, label=r'$2\sigma^2 T^2/\pi^2$')
    ax2.legend(fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_ylim(0, None)

    fig.suptitle('Theorem v2: Verify Var(̂x_{k,l}) ~ $C_0\,\sigma^2 T^2 / l^2$', fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "var_scaling_v2.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"saved: {out_path}")

    # ─── Print the asymptotic plateau ──────────────────────────────────
    print("\nAsymptotic constant l^2 · Var(hat_x_{k,l}) at l=256 (the plateau):")
    for k2 in KS_TO_TEST:
        v = var_hat_x_kl_exact(k2, 256)
        print(f"  |k|² = {k2:5.1f}:  l² Var = {256**2 * v:.6f}  "
              f"(2σ²T²/π² = {2 * SIGMA**2 * T**2 / math.pi**2:.6f})")

    # ─── Compare to total tail and best-approximation error ────────────
    print("\nE[||x_k - P_L x_k||^2_{L^2}] = sum_{l>L} Var(hat_x_{k,l}):")
    print(f"{'|k|²':>6} {'L=32':>10} {'L=64':>10} {'L=128':>10}")
    for k2 in KS_TO_TEST:
        tails = [tail_sum_predicted(k2, L) for L in [32, 64, 128]]
        print(f"{k2:6.1f} {tails[0]:10.4e} {tails[1]:10.4e} {tails[2]:10.4e}")


if __name__ == "__main__":
    main()
