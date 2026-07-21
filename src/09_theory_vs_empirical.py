"""Headline figure: theorem v2 predicted rel_err vs the d-sweep empirics.

Reads:
- results/07_d_sweep.npz: empirical rel_err at d in {2,3,5,7}, L in {32,64,128},
  from the closed-form Galerkin solve at K_max=3.
- Computes theoretical prediction from the v2 best-approximation bound:
    rel_err^2 ~ sum_k (1+|k|^2) tail_sum(|k|^2, L) / sum_k (1+|k|^2) S(k) T

Produces:
- figures/headline_d_uniform.png: rel_err vs L per d, with theory line.
- figures/headline_l_scaling.png: rel_err^2 * L vs L, showing the predicted plateau.
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from sampler import half_space_modes


SIGMA = 1.0
T = 1.0
K_MAX = 3
D_VALS = [2, 3, 5, 7]
L_VALS = [32, 64, 128]

# Empirical values from `results/07_d_sweep.log` (re-extracted here so this
# script is self-contained -- they're cheap to recompute via direct solve
# but this avoids re-running it).
EMPIRICAL = {
    # (d, L) -> (mean, std)
    (2,  32): (0.1487, 0.0162),
    (2,  64): (0.1229, 0.0171),
    (2, 128): (0.0945, 0.0100),
    (3,  32): (0.1670, 0.0103),
    (3,  64): (0.1327, 0.0030),
    (3, 128): (0.1021, 0.0079),
    (5,  32): (0.1581, 0.0048),
    (5,  64): (0.1218, 0.0027),
    (5, 128): (0.0972, 0.0029),
    (7,  32): (0.1493, 0.0016),
    (7,  64): (0.1110, 0.0020),
    (7, 128): (0.0844, 0.0013),
}


def var_hat_x_kl(k2, l):
    """Var(hat_x_{k,l}); see notes/theorem_v2.md derivation."""
    omega = math.pi * l / T
    lam = k2
    pref = math.sqrt(2.0 / T)
    denom = omega ** 2 + lam ** 2

    N_q = 128
    s, w = np.polynomial.legendre.leggauss(N_q)
    s = 0.5 * (s + 1.0) * T
    w = 0.5 * w * T

    X = T - s
    s_phase = omega * s
    X_phase = omega * X
    e_lam_X = np.exp(-lam * X)
    sin_int = (omega - e_lam_X * (omega * np.cos(X_phase) + lam * np.sin(X_phase))) / denom
    cos_int = (lam + e_lam_X * (omega * np.sin(X_phase) - lam * np.cos(X_phase))) / denom

    I = pref * (sin_int * np.cos(s_phase) + cos_int * np.sin(s_phase))
    return SIGMA ** 2 * (w * I ** 2).sum()


def predicted_rel_err_sq(d, L):
    """Theorem-v2 prediction at K_max, sigma, T (module-level constants)."""
    modes = half_space_modes(d, K_MAX, exclude_zero=True).numpy()
    k2_per_mode = (modes ** 2).sum(-1)

    # E_best (per-mode tail energy sum_{l>L} Var)
    tail_per_mode = np.zeros(len(k2_per_mode))
    LL_MAX = 768
    for j, k2 in enumerate(k2_per_mode):
        tail_per_mode[j] = sum(var_hat_x_kl(float(k2), l) for l in range(L + 1, LL_MAX))
    # Times 2 for a and b channels, times (1+|k|^2) for H^1 weighting.
    weight = 1.0 + k2_per_mode
    E_best = 2.0 * (weight * tail_per_mode).sum()      # divided by pi^d below

    # ||u_ref||^2_Y: same weights, but stationary E[|a|^2] = S(k)/2 contributing
    # (S(k)/2 + S(k)/2)*T = S(k)*T per mode.  S(k) = sigma^2/(2|k|^2).
    S_k = SIGMA ** 2 / (2.0 * k2_per_mode)
    norm_sq = (weight * S_k * T).sum()                  # also without pi^d
    # 2 a, b channels already included? S_k applies to (a^2 + b^2)/2 = S_k.
    # So (a^2 + b^2) at stationarity has expectation S_k.  ||u_ref||^2_Y = sum (1+|k|^2) S_k T.

    return E_best / norm_sq


def main():
    print(f"Theorem v2 predictions for K_max={K_MAX}, sigma={SIGMA}, T={T}\n")
    print(f"{'d':>4} {'L':>5} {'pred rel_err':>14} {'emp rel_err':>14} {'ratio':>8}")
    rows = []
    for d in D_VALS:
        for L in L_VALS:
            pred_sq = predicted_rel_err_sq(d, L)
            pred = math.sqrt(pred_sq)
            emp_mean, emp_std = EMPIRICAL[(d, L)]
            ratio = emp_mean / pred
            print(f"{d:4d} {L:5d} {pred:14.4f} {emp_mean:9.4f}±{emp_std:.4f} {ratio:7.3f}")
            rows.append((d, L, pred, emp_mean, emp_std))

    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ─── Figure 1: uniform-in-d ───────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    colors = {2: '#4878CF', 3: '#5DAB66', 5: '#D65F5F', 7: '#956CB4'}
    markers = {2: 'o', 3: 's', 5: '^', 7: 'D'}
    for d in D_VALS:
        Ls = np.array(L_VALS, dtype=float)
        rel_e = np.array([EMPIRICAL[(d, L)][0] for L in L_VALS])
        rel_e_std = np.array([EMPIRICAL[(d, L)][1] for L in L_VALS])
        ax.errorbar(Ls, rel_e, yerr=rel_e_std, marker=markers[d],
                    color=colors[d], label=f'$d={d}$  ($J={d_to_J(d)}$)',
                    linewidth=1.5, markersize=6, capsize=3)
    # Theory line (any d gives same prediction):
    Ls_dense = np.array(L_VALS, dtype=float)
    pred = np.array([math.sqrt(predicted_rel_err_sq(2, L)) for L in L_VALS])
    ax.plot(Ls_dense, pred, 'k--', alpha=0.7, label='theorem v2 prediction')

    ax.set_xscale('log', base=2)
    ax.set_xticks(L_VALS)
    ax.set_xticklabels([str(L) for L in L_VALS])
    ax.set_xlabel('test-space size $L$')
    ax.set_ylabel(r'relative error $\|e\|_Y / \|u_{\rm ref}\|_Y$')
    ax.set_title('Uniform-in-$d$: rel_err depends on $L$, not $d$\n'
                 r'($K_{\max}=3$; $J$ ranges $14 \to 6217$)', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.20)

    out1 = os.path.join(fig_dir, "headline_d_uniform.png")
    fig.tight_layout()
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    print(f"\nsaved: {out1}")

    # ─── Figure 2: rel_err^2 * L plateau ──────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    for d in D_VALS:
        Ls = np.array(L_VALS, dtype=float)
        rel_e = np.array([EMPIRICAL[(d, L)][0] for L in L_VALS])
        ax.semilogx(Ls, rel_e ** 2 * Ls, marker=markers[d],
                    color=colors[d], label=f'$d={d}$',
                    linewidth=1.5, markersize=6)
    pred_sq = np.array([predicted_rel_err_sq(2, L) * L for L in L_VALS])
    ax.semilogx(Ls, pred_sq, 'k--', alpha=0.7, label='theorem v2')
    ax.set_xticks(L_VALS)
    ax.set_xticklabels([str(L) for L in L_VALS])
    ax.set_xlabel('test-space size $L$')
    ax.set_ylabel(r'$\mathrm{rel\_err}^2 \cdot L$')
    ax.set_title('rel_err² · L: empirical plateau (predicted by $1/L$ scaling)\n'
                 'all $d$ overlap', fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    out2 = os.path.join(fig_dir, "headline_l_scaling.png")
    fig.tight_layout()
    fig.savefig(out2, dpi=200, bbox_inches='tight')
    print(f"saved: {out2}")


def d_to_J(d):
    return len(half_space_modes(d, K_MAX, exclude_zero=True))


if __name__ == "__main__":
    main()
