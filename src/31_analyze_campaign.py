"""Turn the campaign ledger into the corrected AC results: table, rates,
certificate check, and a LaTeX fragment ready for the paper.

Run this whenever; it reads whatever is in the ledger and reports on that.

THE CENTRAL IDENTITY
--------------------
The P7 certificate is

    ||u_theta - u_ref||^2_X  <=  C_1 * L_RV(theta)  +  C_2 / L

and the trained loss reaches L_RV ~ 1e-9 while the error term is O(1e-1), so
the C_2/L term carries essentially all of the bound. Therefore

    rel_err ~ L^(-1/2)   <=>   ||e||^2_X ~ C_2/L   <=>   the theorem's rate is
                                                          attained, tightly.

So the dealiased slope is not merely "a better number than -0.361": a slope of
-1/2 that does not degrade with d is the certificate being *saturated uniformly
in d*, which is the paper's headline claim extended from the linear case to the
nonlinear one. A slope flatter than -1/2 that worsens with d -- what the
published N_q=16 table shows -- is the opposite claim. Script 26 established
that the published table's cubic term carried ~270% quadrature error, so this
analysis is what decides which of the two the method actually does.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
LEDGER = os.path.join(RESULTS, "30_campaign.jsonl")

PUBLISHED = {
    2: {32: 0.3132, 64: 0.2370, 128: 0.1898, "slope": -0.361},
    3: {32: 0.3733, 64: 0.3202, 128: 0.2736, "slope": -0.224},
    4: {32: 0.5132, 64: 0.4820, 128: 0.4587, "slope": -0.081},
    5: {32: 0.7710, 64: 0.7869, 128: 0.7873, "slope": +0.015},
}
GOOD = ("ok", "converged")


def load():
    if not os.path.exists(LEDGER):
        sys.exit(f"no ledger at {LEDGER}")
    out = []
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("phase") == "smoke":
                continue          # plumbing checks, not measurements
            if r.get("status") in GOOD and math.isfinite(r.get("rel_err", float("nan"))):
                out.append(r)
    return out


def slope_with_se(Ls, res):
    """Log-log rate and its standard error. Needs >= 3 points for an SE."""
    x, y = np.log(np.asarray(Ls, float)), np.log(np.asarray(res, float))
    n = len(x)
    s, b = np.polyfit(x, y, 1)
    if n < 3:
        return s, float("nan")
    resid = y - (s * x + b)
    sxx = ((x - x.mean()) ** 2).sum()
    se = math.sqrt((resid ** 2).sum() / (n - 2) / sxx) if sxx > 0 else float("nan")
    return s, se


def main():
    rows = load()
    if not rows:
        sys.exit("ledger has no completed measurement cells yet")

    # group by (N_x, d, L) -> list over seeds
    cells = defaultdict(list)
    for r in rows:
        cells[(r["N_x"], r["d"], r["L"])].append(r)

    print("=" * 78)
    print("P7 ALLEN-CAHN, DEALIASED (N_q = 2L)   vs   PUBLISHED (N_q = 16)")
    print("=" * 78)
    print(f"{len(rows)} measurement cells   "
          f"{sum(r['wall_s'] for r in rows) / 3600:.1f} GPU-hours")

    for N_x in sorted({k[0] for k in cells}):
        print(f"\n--- N_x = {N_x} ---")
        print(f"{'d':>3} {'L':>5} {'seeds':>6} {'dealiased':>18} "
              f"{'published':>10} {'delta':>9}")
        for (nx, d, L) in sorted(k for k in cells if k[0] == N_x):
            g = cells[(nx, d, L)]
            re = np.array([r["rel_err"] for r in g])
            pub = PUBLISHED.get(d, {}).get(L)
            mean_s = (f"{re.mean():.4f} +- {re.std():.4f}" if len(re) > 1
                      else f"{re.mean():.4f}")
            pub_s = f"{pub:.4f}" if pub else "-"
            dlt = f"{re.mean() - pub:+.4f}" if pub else "-"
            print(f"{d:3d} {L:5d} {len(re):6d} {mean_s:>18} {pub_s:>10} {dlt:>9}")

    # ---------------- rates ----------------
    print("\n" + "=" * 78)
    print("RATE   rel_err ~ L^s      (-0.5 = KL floor = certificate saturated)")
    print("=" * 78)
    print(f"{'N_x':>4} {'d':>3} {'pts':>4} {'dealiased s':>16} "
          f"{'published s':>12} {'vs -1/2':>10}")
    rate_by_d = {}
    for N_x in sorted({k[0] for k in cells}):
        for d in sorted({k[1] for k in cells if k[0] == N_x}):
            pts = sorted((L, np.mean([r["rel_err"] for r in cells[(N_x, d, L)]]))
                         for (nx, dd, L) in cells if nx == N_x and dd == d)
            if len(pts) < 2:
                continue
            s, se = slope_with_se([p[0] for p in pts], [p[1] for p in pts])
            pub = PUBLISHED.get(d, {}).get("slope")
            s_str = f"{s:+.3f} +- {se:.3f}" if math.isfinite(se) else f"{s:+.3f}"
            pub_s = f"{pub:+.3f}" if pub is not None else "n/a"
            flag = "SATURATED" if abs(s + 0.5) < 0.08 else f"{s + 0.5:+.3f}"
            print(f"{N_x:4d} {d:3d} {len(pts):4d} {s_str:>16} {pub_s:>12} {flag:>10}")
            if N_x == 32:
                rate_by_d[d] = s

    if len(rate_by_d) >= 2:
        ds = sorted(rate_by_d)
        spread = max(rate_by_d.values()) - min(rate_by_d.values())
        pub_spread = (max(PUBLISHED[d]["slope"] for d in ds if d in PUBLISHED)
                      - min(PUBLISHED[d]["slope"] for d in ds if d in PUBLISHED))
        print(f"\n  slope spread over d={ds}:  dealiased {spread:.3f}   "
              f"published {pub_spread:.3f}")
        print("  (a dealiased spread far below the published one is the "
              "d-dependence\n   dissolving into what it always was: the "
              "temporal aliasing artifact)")

    # ---------------- N_x invariance ----------------
    shared = [(d, L) for (nx, d, L) in cells if nx == 20
              and (32, d, L) in cells]
    if shared:
        print("\n" + "=" * 78)
        print("N_x INVARIANCE   (both clear the spatial floor N_x > 6*K_max = 18)")
        print("=" * 78)
        for d, L in sorted(set(shared)):
            a = np.mean([r["rel_err"] for r in cells[(20, d, L)]])
            b = np.mean([r["rel_err"] for r in cells[(32, d, L)]])
            print(f"  d={d} L={L:>3}:  N_x=20 {a:.4f}   N_x=32 {b:.4f}   "
                  f"rel diff {abs(a - b) / b:.2%}")
        print("  -> licenses the reduced-N_x runs used to reach d=6")

    # ---------------- certificate ----------------
    cert = [r for r in rows if "err_X2" in r and r["N_x"] == 32]
    if cert:
        print("\n" + "=" * 78)
        print("CERTIFICATE   ||e||^2_X  vs  C_2/L      (C_2 = ||e||^2_X * L)")
        print("=" * 78)
        print("  C_2 flat in L  => the 1/L rate is attained")
        print("  C_2 flat in d  => the constant is uniform in d\n")
        print(f"{'d':>3} {'L':>5} {'||e||^2_X':>12} {'L_RV':>11} {'C_2':>11}")
        c2_by_d = defaultdict(list)
        for r in sorted(cert, key=lambda r: (r["d"], r["L"], r["seed"])):
            c2 = r["err_X2"] * r["L"]
            c2_by_d[r["d"]].append(c2)
            print(f"{r['d']:3d} {r['L']:5d} {r['err_X2']:12.4e} "
                  f"{r['loss']:11.3e} {c2:11.4e}")
        print()
        for d in sorted(c2_by_d):
            v = np.array(c2_by_d[d])
            print(f"  d={d}: C_2 = {v.mean():.4e} +- {v.std():.4e} "
                  f"({v.std() / v.mean():.1%} spread over L)")

    # ---------------- LaTeX ----------------
    tex = os.path.join(RESULTS, "31_ac_dealiased_table.tex")
    Ls = sorted({k[2] for k in cells if k[0] == 32})
    ds = sorted({k[1] for k in cells if k[0] == 32})
    with open(tex, "w") as f:
        f.write("% generated by 31_analyze_campaign.py -- dealiased AC table\n")
        f.write("\\begin{tabular}{r" + "r" * (2 * len(Ls)) + "}\n\\toprule\n")
        f.write(" & \\multicolumn{%d}{c}{published ($N_q=16$)}"
                " & \\multicolumn{%d}{c}{dealiased ($N_q=2L$)} \\\\\n"
                % (len(Ls), len(Ls)))
        f.write("$d$ & " + " & ".join(f"$L={L}$" for L in Ls) + " & "
                + " & ".join(f"$L={L}$" for L in Ls) + " \\\\\n\\midrule\n")
        for d in ds:
            pub = " & ".join(f"{PUBLISHED.get(d, {}).get(L, float('nan')):.4f}"
                             if PUBLISHED.get(d, {}).get(L) else "--" for L in Ls)
            new = " & ".join(
                f"{np.mean([r['rel_err'] for r in cells[(32, d, L)]]):.4f}"
                if (32, d, L) in cells else "--" for L in Ls)
            f.write(f"{d} & {pub} & {new} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nLaTeX table -> {tex}")


if __name__ == "__main__":
    main()
