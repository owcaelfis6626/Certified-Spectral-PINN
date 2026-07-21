# P7 — RVPINN×WM certified bound, theorem statement v1

*Goal: write down the cleanest statement of the bound, mark every constant
explicitly, identify which would worry a reviewer. No proofs yet — just
the statement.*

---

## 1. Setup

### 1.1 SPDE

Linear bandlimited-Laplacian SPDE on $\mathbb{T}^d = [0, 2\pi]^d / \sim$:
$$
\partial_t u = \Delta u + \sigma\,\dot W_{K_{\max}}, \qquad u(x, 0) = u_0(x).
$$
Bandlimited noise: $\dot W_{K_{\max}}(x, t) = \sum_{k \in \mathcal H} [\dot B_k^a(t)\,\cos(k\cdot x) + \dot B_k^b(t)\,\sin(k\cdot x)]$ with $\{B_k^{a/b}\}$ independent standard Brownian motions.

Mode set: $\mathcal H_{K_{\max}} = \{k \in \mathbb Z^d_{\ge 0,{\rm lex}} : 0 < |k| \le K_{\max}\}$ (lex-ordered half-lattice, $J = |\mathcal H|$ modes).

### 1.2 Reference realisation

Fix a noise sample path $\omega$. The reference $u_{\rm ref}(x, t; \omega) = \sum_k [a^{\rm ref}_k(t)\,\cos(k\cdot x) + b^{\rm ref}_k(t)\,\sin(k\cdot x)]$ solves the SDE per mode:
$$
da^{\rm ref}_k(t) = -|k|^2 a^{\rm ref}_k(t)\, dt + \sigma\, dB_k^a(t),
\qquad a^{\rm ref}_k(0) = a_{k,0}.
$$
(And same for $b_k^{\rm ref}$.)

### 1.3 Network

$u_\theta(x, t) = \sum_k [a_{\theta,k}(t)\,\cos(k\cdot x) + b_{\theta,k}(t)\,\sin(k\cdot x)]$ with $(a_{\theta,k}, b_{\theta,k}) = \psi_\theta(k, t)$ a smooth network output.

### 1.4 Test space

Tensor-product:
$$
\Psi_{k, l}^{a}(x, t) = \cos(k\cdot x)\,\psi_l(t), \qquad
\Psi_{k, l}^{b}(x, t) = \sin(k\cdot x)\,\psi_l(t),
$$
with $\psi_l(t) = \sqrt{2/T}\sin(\pi l t / T)$ for $l = 1, \ldots, L$.

The $\{\psi_l\}$ are the orthonormal $L^2(0,T)$ eigenbasis of $-\partial_t^2$ with Dirichlet BC; eigenvalues $\mu_l = \pi^2 l^2 / T^2$.

### 1.5 Weak residual

Per mode-test pair:
$$
R_{k,l}^{a}(\theta) := \int_0^T \bigl[\partial_t a_{\theta,k}(t) + |k|^2\,a_{\theta,k}(t)\bigr]\,\psi_l(t)\, dt \;-\; \sigma\,I_{k,l}^a,
$$
$$
I_{k,l}^a := \int_0^T \psi_l(t)\, dB_k^a(t) \quad\text{(scalar Wiener integral)}.
$$
$R_{k,l}^b$ is defined analogously.

### 1.6 Losses

RVPINN dual-norm loss:
$$
\mathcal L_{\rm RV}(\theta) := \sum_{k \in \mathcal H} \sum_{l=1}^{L} \frac{|R_{k,l}^a(\theta)|^2 + |R_{k,l}^b(\theta)|^2}{1 + |k|^2 + \mu_l}.
$$

IC loss:
$$
\mathcal L_{\rm IC}(\theta) := \|u_\theta(\cdot, 0) - u_0(\cdot)\|_{L^2(\mathbb T^d)}^2 = \pi^d \sum_{k} \bigl(|a_{\theta,k}(0) - a_{k,0}|^2 + |b_{\theta,k}(0) - b_{k,0}|^2\bigr).
$$

---

## 2. Theorem (v1, linear case, single realisation)

**Theorem (RVPINN certified bound for linear SPDE).**
*Assume $u_\theta \in C^1([0, T]; \ell^2(\mathcal H))$ in mode space.
There exist constants $C_1, C_2 > 0$ depending only on $T$ such that, for every realisation $\omega$ of the bandlimited Brownian noise,*
$$
\boxed{
\|u_\theta - u_{\rm ref}\|_{H^1(\mathbb T^d \times [0,T])}^2
\;\le\;
C_1\,\mathcal L_{\rm RV}(\theta)
\;+\;
C_2\,\mathcal L_{\rm IC}(\theta)
\;+\;
\epsilon_{\rm test}(L)
}
$$
*where the test-space truncation error is*
$$
\epsilon_{\rm test}(L) \le \frac{T^2}{\pi^2 L^2} \cdot \|\partial_t (u_\theta - u_{\rm ref})\|_{L^2(\mathbb T^d \times [0,T])}^2.
$$

**Constants (target form):**

| Constant | Expected value | Depends on |
|---|---|---|
| $C_1$ | $O(1)$, independent of $d, K_{\max}, J$ | $T$ only |
| $C_2$ | $\max(1, T)$ | $T$ only |
| $\epsilon_{\rm test}$ | $O(L^{-2})$, vanishes as $L \to \infty$ | $L, T$, error regularity |

**This is the goal shape.** The next subsection lists what would break it.

---

## 3. What could go wrong (and what to check)

### 3.1 The inf-sup constant on $\mathcal H_{K_{\max}}$

The bilinear form per mode is
$$
B_k(a, v) = \int_0^T \bigl[\partial_t a\cdot v + |k|^2\, a\cdot v\bigr]\, dt
$$
on $(a, v) \in X \times Y$ where:
- Trial: $X = \{a \in H^1(0, T) : a(0) = 0\}$ with norm $\|a\|_X^2 = \|a\|_{L^2}^2 + \|\partial_t a\|_{L^2}^2$.
- Test:  $Y = L^2(0, T)$ with norm $\|v\|_Y = \|v\|_{L^2}$.

For each $k$ with $|k| \ge 1$, the continuous inf-sup constant is
$$
\beta_k = \inf_{a \in X} \sup_{v \in Y} \frac{|B_k(a, v)|}{\|a\|_X \|v\|_Y} \;\geq\; \frac{|k|^2}{\sqrt{1 + |k|^4}}
\;\to\; 1 \text{ as } |k| \to \infty.
$$

**Crucial:** $\beta_k$ is uniformly bounded below over $\mathcal H$ (away from $k = 0$, which is excluded). The minimum $\beta_{\min}$ is attained at $|k| = 1$: $\beta_{\min} = 1/\sqrt 2$. **Independent of $d$.**

### 3.2 The discrete inf-sup constant for $\{\psi_l\}_{l=1..L}$

The test space is finite — we need a *discrete* inf-sup. The Fortin trick: construct an interpolant $\Pi_L : Y \to {\rm span}\{\psi_l\}$ with $\|\Pi_L v\|_{L^2} \leq \|v\|_{L^2}$. The discrete inf-sup constant is then $\beta_{k, L} \ge \beta_k \cdot (1 - {\rm gap}(L))$ with ${\rm gap}(L) \to 0$.

For the Dirichlet sine basis on $[0, T]$, $\Pi_L$ is the $L^2$-orthogonal projection, $\|\Pi_L\|_{L^2 \to L^2} = 1$, so ${\rm gap}(L) = 0$. **Discrete inf-sup = continuous inf-sup.**

### 3.3 Bandlimit truncation $\epsilon_{\rm band}$

Reference field $u_{\rm ref}$ already lives in $\mathcal H_{K_{\max}}$ (bandlimited noise). Network output $u_\theta$ lives in $\mathcal H_{K_{\max}}$ by construction. **No bandlimit truncation error.** This is the architectural reason mode-space is clean here: trial = ref = same finite-dim space.

### 3.4 Wiener integral subtlety

$I_{k,l}^a = \int_0^T \psi_l(t)\, dB_k^a(t)$ is defined per realisation as the limit of finite-sum approximations on a dense Brownian grid. Computing $I_{k,l}^a$ requires storing the Brownian path on a fine grid (say $N_{\rm fine} = 2^{12}$ points) and projecting via quadrature. This is a **one-time precomputation cost**, $O(J L N_{\rm fine})$ memory and arithmetic, done at oracle construction.

Storage at $d=7$, $K_{\max}=3$: $J = 6218$, $L = 32$, $N_{\rm fine} = 4096$ → $6218 \times 32 \times 4 = 800\,{\rm kB}$ for $I_{k,l}$, $6218 \times 4096 \times 4 = 100\,{\rm MB}$ for the fine Brownian path. Fine path can be discarded after precomputing $I_{k,l}$.

### 3.5 Test-space truncation $\epsilon_{\rm test}(L)$

Comes from the residual $R$ having $L^2$-energy outside $\text{span}\{\psi_l\}_{l=1..L}$. For smooth $R$, this is $O(L^{-2k})$ where $k$ is the regularity of $R$. The error term in §2 is the worst case ($k = 1$).

The relevant control: for the residual evaluated at the trained network $u_\theta$, the time-regularity of $\partial_t a_{\theta,k}$ depends on the network's smoothness. **Smooth network → fast convergence in $L$.**

---

## 4. Composition with P1's $W_2$ bound

P1 gives a Mishra–Molinaro-style bound on the law-distance:
$$
W_2\bigl(\mathcal L(u_\theta), \mathcal L(u_{\rm ref})\bigr) \le C_{W_2}\sqrt{\mathcal L_{\rm spec}(\theta)} + C_{\rm NTK}\,\text{(NTK term)}.
$$

P1's bound holds *in expectation over noise realisations* of the spectral loss $\mathcal L_{\rm spec}$. P7's RVPINN bound holds *path-wise*.

Composed:
$$
\mathbb E_\omega\bigl[\|u_\theta - u_{\rm ref}\|_{H^1} + W_2\bigl(\mathcal L(u_\theta), \mathcal L(u_{\rm ref})\bigr)\bigr]
\le C_1\,\mathbb E_\omega \mathcal L_{\rm RV} + C_2\,\mathcal L_{\rm IC} + C_{W_2}\sqrt{\mathbb E_\omega \mathcal L_{\rm spec}} + \epsilon_{\rm test}.
$$

Both legs certified, both vanishing in the right limits.

---

## 5. Verdict (honest)

**Structurally clean (high confidence):**

1. **No bandlimit truncation term.** Trial = ref = $\text{span}\{\cos/\sin(k\cdot x) : k \in \mathcal H\}$ by construction. Architectural win.
2. **Inf-sup constant uniform in $d$.** The smallest mode $|k| = 1$ gives the worst case; $\beta_{\min}$ depends on $K_{\min} = 1$ and $T$, not on $d$ or $K_{\max}$.
3. **Discrete inf-sup = continuous** (Dirichlet-sine $L^2$-projection is identity, ${\rm gap}(L) = 0$).
4. **Wiener integral precomputation is cheap** ($\sim 1\,{\rm MB}$ at $d=7$).

**One real subtlety I underestimated (medium confidence):**

The LHS is the spatiotemporal $H^1(\mathbb T^d \times [0, T])$ norm. In mode space:
$$
\|u\|_{H^1}^2 = \pi^d \sum_k \int_0^T \bigl[(1 + |k|^2)|a_k|^2 + |\partial_t a_k|^2\bigr]\, dt.
$$
The natural *parabolic* energy norm is
$$
\|u\|_X^2 = \pi^d \sum_k \int_0^T \bigl[(1 + |k|^2)|a_k|^2 + \tfrac{|\partial_t a_k|^2}{1 + |k|^2}\bigr]\, dt,
$$
which is what falls out of the parabolic Galerkin inf-sup naturally (the temporal-derivative term is weighted in $V^*$, not $V$).

These two norms differ by a factor of $(1 + |k|^2)$ in the $|\partial_t a|^2$ term. The Galerkin inf-sup gives a clean bound in $\|\cdot\|_X$, but a bound in the spatiotemporal $H^1$ requires an extra step (e.g., elliptic regularity in time, or restating the trial space).

**Two ways to handle it:**

- **(a) State the theorem in the Bochner norm $\|\cdot\|_X$** (= $L^2(0,T; H^1) \cap H^1(0,T; H^{-1})$). Clean constants, standard parabolic framework, but the norm is less reader-friendly.
- **(b) Stay in spatiotemporal $H^1$** and accept that $C_1$ depends on $K_{\max}$ — likely $C_1 = O(1 + K_{\max}^2)$. Still uniform in $d$, but the $K_{\max}$ scaling needs to be stated honestly.

Both are valid. **(a) gives the cleanest constants and is what I'd write for a theory venue (Math. Comp., SINUM).** (b) is more readable for an applied venue (JCP) but trades $K_{\max}$ tightness for prose simplicity.

**Decided 2026-06-08: go with (a).** Bochner norm $X = L^2(0,T; H^1) \cap H^1(0,T; H^{-1})$. Theory-venue framing. Constants stay $O(1)$ in $d, K_{\max}, L$.

**Estimated effort revision:** Filling in the proof from this statement is **4–8 hours**, not 1–2. The structure is clean but the per-mode parabolic Galerkin energy estimate needs to be written out carefully to land the constants. Still tractable in a 3-week sprint, just not a single afternoon.

**Status:** Statement is clean enough to start coding the architecture in parallel. The exact form of $C_1$ won't change the code.

**Tomorrow's task:** Decide (a) vs (b), and write out the per-mode energy estimate in detail (mode-by-mode duality argument + Grönwall in time). That fixes the constants once and for all.
