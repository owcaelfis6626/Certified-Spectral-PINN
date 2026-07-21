"""Generate convergence figure for P7 paper.

Two-panel:
  Left  — half-period basis: rel_err vs L for d=2,3,5,7 (log-log)
  Right — Dirichlet-sine plateau vs half-period at d=2 (log-log)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "convergence.pdf")

# --- Data from experiments ---

# Half-period d-sweep (Tab. 2 in paper) — CORRECTED 2026-07-17 (honest rel_err,
# was rel_err^2; results/13_halfperiod_d_sweep_honest.log)
# d, J, L, mean, std
halfperiod = [
    (2,   14,  32,  0.2890, 0.0403),
    (2,   14,  64,  0.2053, 0.0068),
    (2,   14, 128,  0.1563, 0.0057),
    (2,   14, 256,  0.1075, 0.0142),
    (3,   61,  32,  0.2687, 0.0121),
    (3,   61,  64,  0.1948, 0.0051),
    (3,   61, 128,  0.1404, 0.0032),
    (3,   61, 256,  0.1032, 0.0041),
    (5,  671,  32,  0.2756, 0.0011),
    (5,  671,  64,  0.1978, 0.0010),
    (5,  671, 128,  0.1403, 0.0018),
    (5,  671, 256,  0.1034, 0.0003),
    (7, 6217,  32,  0.2828, 0.0010),
    (7, 6217,  64,  0.2028, 0.0002),
    (7, 6217, 128,  0.1430, 0.0004),
    (7, 6217, 256,  0.1057, 0.0005),
]

# Dirichlet-sine plateau (Tab. 1 in paper) — corrected
# (results/10_high_L_rate_honest.log; d=2, K_max=4)
dirichlet = [
    (32,  0.416, 0.031),
    (64,  0.368, 0.025),
    (128, 0.336, 0.029),
    (256, 0.336, 0.026),
    (512, 0.330, 0.028),
    (1024,0.325, 0.027),
]

# Half-period at d=2, K_max=4 for panel (b), same config as the Dirichlet
# sweep (results/12_halfperiod_test_honest.log)
halfperiod_b = [
    (32,  0.3152, 0.0093),
    (64,  0.2288, 0.0137),
    (128, 0.1681, 0.0086),
    (256, 0.1181, 0.0017),
    (512, 0.0844, 0.0026),
]

# ---

D_VALS = [2, 3, 5, 7]
J_VALS = {2: 14, 3: 61, 5: 671, 7: 6217}
colors = {2: "#1f77b4", 3: "#ff7f0e", 5: "#2ca02c", 7: "#d62728"}
markers = {2: "o", 3: "s", 5: "^", 7: "D"}

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# --- Panel A: half-period uniform-in-d ---
ax = axes[0]
for d in D_VALS:
    rows = [(r[2], r[3], r[4]) for r in halfperiod if r[0] == d]
    Ls   = np.array([r[0] for r in rows])
    means = np.array([r[1] for r in rows])
    stds  = np.array([r[2] for r in rows])
    label = rf"$d={d}$, $J={J_VALS[d]}$"
    ax.errorbar(Ls, means, yerr=stds, fmt=markers[d]+"-",
                color=colors[d], label=label,
                capsize=3, linewidth=1.4, markersize=5)

# Reference line 1.6/sqrt(L) (optimal KL rate)
Ls_ref = np.array([24, 300])
ax.plot(Ls_ref, 1.6 / np.sqrt(Ls_ref), "k--", linewidth=1.0,
        label=r"$1.6/\sqrt{L}$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$L$ (number of test functions)", fontsize=11)
ax.set_ylabel(r"$\mathrm{rel\_err}$", fontsize=11)
ax.set_title("(a) Half-period basis: uniform-in-$d$", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([32, 64, 128, 256])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, which="both", alpha=0.3)

# --- Panel B: Dirichlet plateau vs half-period at d=2 ---
ax = axes[1]

dir_L    = np.array([r[0] for r in dirichlet])
dir_mean = np.array([r[1] for r in dirichlet])
dir_std  = np.array([r[2] for r in dirichlet])
ax.errorbar(dir_L, dir_mean, yerr=dir_std,
            fmt="s--", color="#d62728", label=r"Dirichlet sine ($d=2$)",
            capsize=3, linewidth=1.4, markersize=5)

# half-period at d=2, K_max=4 (same config as the Dirichlet sweep)
hp_L    = np.array([r[0] for r in halfperiod_b])
hp_mean = np.array([r[1] for r in halfperiod_b])
hp_std  = np.array([r[2] for r in halfperiod_b])
ax.errorbar(hp_L, hp_mean, yerr=hp_std,
            fmt="o-", color="#1f77b4", label=r"Half-period ($d=2$)",
            capsize=3, linewidth=1.4, markersize=5)

ax.axhline(0.33, color="#d62728", linestyle=":", linewidth=0.8, alpha=0.7,
           label="plateau $\\approx 0.33$")
Ls_ref_b = np.array([24, 1200])
ax.plot(Ls_ref_b, 1.85 / np.sqrt(Ls_ref_b), "k--", linewidth=1.0,
        label=r"$\propto L^{-1/2}$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$L$ (number of test functions)", fontsize=11)
ax.set_ylabel(r"$\mathrm{rel\_err}$", fontsize=11)
ax.set_title("(b) Dirichlet sine plateau vs half-period", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.set_xticks([32, 64, 128, 256, 512, 1024])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved: {OUT}")
