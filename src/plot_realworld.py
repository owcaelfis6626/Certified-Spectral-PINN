"""Two-panel real-world figure for P7 paper.

(a) ERA5 H^1 rel-err vs L at K_max ∈ {3,4,5,7} — uniform across K_max
(b) SPX vol surface rel-err vs L with noise-floor fit
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = "/home/hubi/research/useful/papers/paper7_certified_spectral_pinn"
OUT  = os.path.join(ROOT, "figures", "real_world.pdf")

# --- Load data ---
era = np.load(os.path.join(ROOT, "results", "18b_era5_certified.npz"), allow_pickle=True)
era_sweeps = era["sweeps"].item()    # dict K_max → (N, 2)
vol = np.load(os.path.join(ROOT, "results", "17_vol_surface.npz"), allow_pickle=True)
vol_sweep = vol["sweep"]              # (N, 3): L, RMSE, rel

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

# (a) ERA5
ax = axes[0]
colors = {3:"#1f77b4", 4:"#ff7f0e", 5:"#2ca02c", 7:"#d62728"}
markers = {3:"o", 4:"s", 5:"^", 7:"D"}
J_lookup = {3:14, 4:24, 5:40, 7:74}
for K_max, sw in era_sweeps.items():
    Ls = sw[:, 0]
    rels = sw[:, 1]
    ax.plot(Ls, rels, markers[K_max]+"-", color=colors[K_max],
            label=rf"$K_{{\max}}={K_max}$, $J={J_lookup[K_max]}$",
            markersize=5, linewidth=1.4)

# Reference 1/L line and L^{-1/4} fit
L_fine = np.linspace(0.8, 60, 100)
ax.plot(L_fine, 0.55 * L_fine ** (-0.25), 'k--', linewidth=1.0,
        label=r"$0.55\,L^{-1/4}$ (fit)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([0.2, 0.3, 0.4, 0.5])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel(r"$L$ (half-period test functions in time)", fontsize=11)
ax.set_ylabel(r"$H^1$ relative error", fontsize=11)
ax.set_title(r"(a) ERA5 2m temperature, 90 d × $16{\times}16$ grid: uniform in $K_{\max}$", fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="lower left")

# (b) SPX vol surface
ax = axes[1]
Ls_vol = vol_sweep[:, 0].astype(float)
rels_vol = vol_sweep[:, 2].astype(float)
ax.plot(Ls_vol, rels_vol, "o-", color="#1f77b4", markersize=6, linewidth=1.5,
        label="SPX vol surface (41 expiries)")

def floor_model(L, a, b):
    return np.sqrt(a / L**2 + b)
popt, _ = curve_fit(floor_model, Ls_vol, rels_vol, p0=[0.02, 0.005])
a, b = popt
L_fine_v = np.linspace(0.5, 16, 100)
ax.plot(L_fine_v, floor_model(L_fine_v, a, b), 'k--', linewidth=1.0,
        label=fr"fit: $\sqrt{{{a:.3f}/L^2 + {b:.4f}}}$")
ax.axhline(np.sqrt(b), color='gray', linestyle=':', linewidth=0.8,
           label=fr"noise floor $\approx {100*np.sqrt(b):.1f}\%$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([0.05, 0.07, 0.10, 0.15])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel(r"$L$ (half-period test functions in $\tau$)", fontsize=11)
ax.set_ylabel("in-sample rel.\\ RMSE", fontsize=11)
ax.set_title(r"(b) SPX implied vol surface, $K_{\max}=4$: noise-floor saturation", fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved: {OUT}")
print(f"Vol surface: noise floor = {100*np.sqrt(b):.2f}%, signal coeff a = {a:.4f}")

# Pretty-print ERA5 slope check
print(f"\nERA5 L-sweep slopes (rel ∝ L^slope):")
for K_max, sw in era_sweeps.items():
    Ls = sw[:, 0]; rels = sw[:, 1]
    slope, _ = np.polyfit(np.log(Ls), np.log(rels), 1)
    print(f"  K_max = {K_max}, J = {J_lookup[K_max]}: slope = {slope:+.3f}")
