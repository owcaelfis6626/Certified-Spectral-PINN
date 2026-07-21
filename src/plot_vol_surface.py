"""Vol-surface figure for P7 paper.

Three panels:
  (a) Observed IV scatter (K_logm, τ) → IV  [validation visual]
  (b) Reconstructed surface contour, with observation points overlaid
  (c) L-sweep: rel_err² vs 1/L², fit a/L² + b (noise floor)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

SRC_DIR = os.path.dirname(__file__)
NPZ = os.path.join(SRC_DIR, "..", "results", "17_vol_surface.npz")
OUT = os.path.join(SRC_DIR, "..", "figures", "vol_surface.pdf")

# Reload state
d = np.load(NPZ, allow_pickle=True)
spot = float(d["spot"])
today = str(d["today"])
expiries = list(d["expiries"])
tau_arr = d["tau"]
mode_traj = d["mode_traj"]   # (N_τ, 2K+1)
mode_fits = d["mode_fits"]   # (2K+1, 1+L)
sweep = d["sweep"]           # (N_L, 3) : L, RMSE, rel
K_MAX_MODES = int(d["K_MAX_MODES"])
L_TIME      = int(d["L_TIME"])

# We need the raw observations again for panel (a). Re-fetch from yfinance OR cache.
# For now, reconstruct from mode_traj × the Fourier basis evaluated on a fine K grid,
# combined with the τ-direction fit.

K_LOGM_MIN, K_LOGM_MAX = -0.30, 0.30

def x_of_K(K):
    return (np.asarray(K) - K_LOGM_MIN) / (K_LOGM_MAX - K_LOGM_MIN) * (2 * np.pi)

def fourier_K(K, K_max):
    x = x_of_K(K); n = len(np.atleast_1d(K))
    A = np.ones((n, 2*K_max + 1))
    for k in range(1, K_max + 1):
        A[:, 2*k-1] = np.cos(k * x)
        A[:, 2*k  ] = np.sin(k * x)
    return A

def halfperiod(t, L):
    t = np.atleast_1d(t)
    l = np.arange(1, L+1)
    return np.sqrt(2.0) * np.sin((2*l[None,:]-1) * np.pi * t[:,None] / 2)

# Build a fine grid for the surface
n_grid = 60
K_grid = np.linspace(K_LOGM_MIN, K_LOGM_MAX, n_grid)
tau_min, tau_max = tau_arr.min(), tau_arr.max()
tau_grid = np.linspace(tau_min, tau_max, n_grid)
K_mesh, T_mesh = np.meshgrid(K_grid, tau_grid)

# Predict on the grid
A_K = fourier_K(K_mesh.flatten(), K_MAX_MODES)
t_norm = (T_mesh.flatten() - tau_min) / (tau_max - tau_min)
Phi = halfperiod(t_norm, L_TIME)
Phi_ext = np.concatenate([np.ones((len(t_norm), 1)), Phi], axis=1)
a_at_tau = Phi_ext @ mode_fits.T   # (n, 2K+1)
iv_grid = (A_K * a_at_tau).sum(axis=1).reshape(n_grid, n_grid)

# Build the figure
fig = plt.figure(figsize=(13.5, 4.2))

# (a) observed mode trajectories (vs τ, per mode coefficient)
ax = fig.add_subplot(1, 3, 1)
mode_labels = ['$a_0$']
for k in range(1, K_MAX_MODES+1):
    mode_labels += [f'$a_{k}$', f'$b_{k}$']
colors = cm.viridis(np.linspace(0, 1, mode_traj.shape[1]))
for j in range(mode_traj.shape[1]):
    ax.plot(tau_arr, mode_traj[:, j], 'o-',
            color=colors[j], markersize=3, linewidth=1.0, label=mode_labels[j])
ax.set_xlabel(r'$\tau$ (years to expiry)', fontsize=11)
ax.set_ylabel(r'mode coefficient', fontsize=11)
ax.set_title(r'(a) Per-mode trajectories from $K$-projection', fontsize=10)
ax.legend(ncol=3, fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# (b) reconstructed surface contour
ax = fig.add_subplot(1, 3, 2)
cs = ax.contourf(K_mesh, T_mesh, iv_grid, 25, cmap='viridis')
ax.set_xlabel(r'log-moneyness $K = \log(K/S)$', fontsize=11)
ax.set_ylabel(r'$\tau$ (years)', fontsize=11)
ax.set_title(rf'(b) Reconstructed surface, $K_{{\max}}={K_MAX_MODES}$, $L={L_TIME}$', fontsize=10)
plt.colorbar(cs, ax=ax, label='implied volatility')

# (c) L-sweep with noise-floor fit
ax = fig.add_subplot(1, 3, 3)
Ls = sweep[:, 0].astype(float)
rels = sweep[:, 2].astype(float)
ax.plot(Ls, rels, 'o-', color='C0', markersize=6, linewidth=1.5, label='SPX vol surface')

# Fit rel_err² = a/L² + b
from scipy.optimize import curve_fit
def model(L, a, b):
    return np.sqrt(a / L**2 + b)
popt, _ = curve_fit(model, Ls, rels, p0=[0.02, 0.005])
a, b = popt
L_fine = np.linspace(0.5, 16, 100)
ax.plot(L_fine, model(L_fine, a, b), 'k--', linewidth=1.0,
        label=fr'fit: $\sqrt{{{a:.3f}/L^2 + {b:.4f}}}$')
ax.axhline(np.sqrt(b), color='gray', linestyle=':', linewidth=0.8,
           label=fr'noise floor $\approx {100*np.sqrt(b):.1f}\%$')

# Reference 1/L line
ax.plot(L_fine, 0.14/L_fine, 'r:', linewidth=0.8, alpha=0.5, label=r'$0.14/L$ (reference)')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xticks([1, 2, 4, 8, 16])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([0.05, 0.07, 0.10, 0.15])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel(r'$L$ (half-period test functions in $\tau$)', fontsize=11)
ax.set_ylabel(r'in-sample rel.\ RMSE', fontsize=11)
ax.set_title('(c) $L$-sweep + noise-floor fit', fontsize=10)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f"saved: {OUT}")
print(f"Noise floor: {100*np.sqrt(b):.2f}%, signal coeff a = {a:.4f}")
print(f"At L = {Ls[0]:.0f}: model {100*model(Ls[0],a,b):.2f}%, observed {100*rels[0]:.2f}%")
print(f"At L = {Ls[-1]:.0f}: model {100*model(Ls[-1],a,b):.2f}%, observed {100*rels[-1]:.2f}%")
