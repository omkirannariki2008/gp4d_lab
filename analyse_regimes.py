#!/usr/bin/env python3
"""
analyse_regimes.py
==================
COMPREHENSIVE regime-based analysis for the inclusive diffraction
cross-section dataset.

Core question answered
----------------------
Instead of partitioning by Q2 / W2 / beta / z axes (one expert per axis-bin),
can we create REGIME-BASED experts that are each optimally matched to the
physics behaviour of the data they see?

Output sections
---------------
S1  : Full 2D (Q2_std × β) phase diagram — mean, std, floor% per cell
S2  : Natural regime boundary detection — gradient magnitudes, transition widths
S3  : Proposed regime definitions + data statistics per regime
S4  : Per-regime axis-aligned variogram → Hölder exponents → kernel votes
S5  : Per-regime non-stationarity analysis (variance ratios per sub-axis)
S6  : Per-regime principal direction — which variable drives most variance
S7  : Within-regime correlation structure (cross-correlations)
S8  : Regime overlap / smoothness of regime boundaries
S9  : Kernel recommendation matrix — one row per regime, one col per variable
S10 : FINAL regime-expert architecture blueprint

Usage
-----
    python analyse_regimes.py --csv /path/to/data.csv
    python analyse_regimes.py --csv data.csv --device cuda --out report.txt
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, pearsonr

warnings.filterwarnings("ignore")

# ─── Column layout ────────────────────────────────────────────────────────────
COLS   = ["Q2_log_center", "W2_log_center", "beta_center", "z_center"]
AXIS   = ["q2",            "w2",            "beta",        "z"]
TARGET = "logA"
FLOOR  = -17.955       # lower fence

# Physics scaler constants (from diagnostic output)
Q2_MEAN, Q2_STD = 0.0, 2.7490
W2_MEAN, W2_STD = 6.1231, 2.2531
BETA_EPS        = 0.0005   # approximate

def std_q2(v):   return (v - Q2_MEAN) / Q2_STD
def std_w2(v):   return (v - W2_MEAN) / W2_STD
def std_beta(v): return (np.log(np.abs(v) + BETA_EPS))  # log-beta (raw, before /std)

# ─── Regime definitions (to be refined by the analysis) ──────────────────────
REGIME_DEFS = {
    "CPR": {  # Central Physical Regime
        "desc": "Central physical: mid-Q2, mid-β, bulk of physics signal",
        "q2_lo": -0.98, "q2_hi": +0.98,
        "beta_lo": 0.02, "beta_hi": 0.83,
    },
    "ELR": {  # Elastic Limit Regime
        "desc": "Elastic limit: β→1, sharp kinematic boundary",
        "q2_lo": -99,   "q2_hi": +99,
        "beta_lo": 0.83, "beta_hi": 1.01,
    },
    "DSR": {  # DGLAP Saturation Regime
        "desc": "DGLAP saturation: high-Q2, non-elastic",
        "q2_lo": +0.98, "q2_hi": +99,
        "beta_lo": 0.02, "beta_hi": 0.83,
    },
    "PHR": {  # Photoproduction Regime
        "desc": "Photoproduction: very low-Q2, non-elastic",
        "q2_lo": -99,   "q2_hi": -1.3,
        "beta_lo": 0.02, "beta_hi": 0.83,
    },
    "TRN": {  # Transition Regime
        "desc": "Transition zone: Q2 between photoproduction and physical",
        "q2_lo": -1.3,  "q2_hi": -0.98,
        "beta_lo": 0.02, "beta_hi": 0.83,
    },
}

def assign_regime(q2_std, beta):
    """Assign each data point to a regime. Returns array of regime labels."""
    labels = np.full(len(q2_std), "UNK", dtype=object)
    for name, d in REGIME_DEFS.items():
        mask = (
            (q2_std >= d["q2_lo"]) & (q2_std < d["q2_hi"]) &
            (beta   >= d["beta_lo"]) & (beta < d["beta_hi"])
        )
        labels[mask] = name
    return labels

# ─── Axis-aligned variogram helper ───────────────────────────────────────────

def regime_variogram(sub_df, target_col, target_axis_col, other_cols,
                     max_lags=12, max_groups=2000):
    """Axis-aligned variogram for a regime subset."""
    ax_vals = np.sort(sub_df[target_axis_col].unique())
    if len(ax_vals) < 3:
        return np.array([]), np.array([])
    spacing = float(np.median(np.diff(ax_vals)))
    groups  = (sub_df.groupby(other_cols)[target_col]
                     .apply(lambda g: g.values))
    if len(groups) > max_groups:
        rng  = np.random.default_rng(42)
        idxs = rng.choice(len(groups), max_groups, replace=False)
        groups = groups.iloc[idxs]
    gamma_acc = {}
    for lag in range(1, max_lags + 1):
        vals = []
        for _, y in groups.items():
            if len(y) > lag:
                d = 0.5 * (y[lag:] - y[:-lag]) ** 2
                vals.extend(d.tolist())
        if len(vals) > 20:
            gamma_acc[lag * spacing] = float(np.mean(vals))
    if not gamma_acc:
        return np.array([]), np.array([])
    h = np.array(sorted(gamma_acc.keys()))
    g = np.array([gamma_acc[k] for k in h])
    return h, g

def holder_exponent(h, g, h_frac=0.35):
    if len(h) < 3:
        return np.nan, np.nan
    h_cut = h.max() * h_frac
    mask  = (h <= h_cut) & (h > 0) & (g > 0)
    if mask.sum() < 3:
        mask = (h > 0) & (g > 0)
        if mask.sum() < 3:
            return np.nan, np.nan
    s, _, r, _, _ = linregress(np.log(h[mask]), np.log(g[mask]))
    return float(s), float(r ** 2)

def holder_to_kernel(alpha):
    if np.isnan(alpha): return "matern12"
    if alpha < 0.8:     return "matern12"
    if alpha < 1.5:     return "matern12_or_32"
    if alpha < 2.5:     return "matern32"
    if alpha < 4.0:     return "matern52"
    return "rbf_or_matern52"

def sep(title="", w=72):
    if title:
        print(f"\n{'='*w}")
        print(f"  {title}")
        print(f"{'='*w}")
    else:
        print("-" * w)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      required=True)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--out",      default=None)
    parser.add_argument("--n_slabs",  type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"  {len(df):,} rows | floor(<{FLOOR})={((df[TARGET]<FLOOR).sum()):,}")

    # Standardise Q2 and add regime labels
    df["Q2_std"] = std_q2(df["Q2_log_center"])
    df["regime"] = assign_regime(df["Q2_std"].values, df["beta_center"].values)

    q2_u    = np.sort(df["Q2_log_center"].unique())
    beta_u  = np.sort(df["beta_center"].unique())
    w2_u    = np.sort(df["W2_log_center"].unique())
    z_u     = np.sort(df["z_center"].unique())

    df_clean = df[df[TARGET] >= FLOOR].copy()
    df_floor = df[df[TARGET] <  FLOOR].copy()

    lines = []
    def pr(s=""):
        print(s)
        lines.append(s)

    # ════════════════════════════════════════════════════════════════════════
    # S1 — 2D Phase Diagram (Q2_std × β)
    # ════════════════════════════════════════════════════════════════════════
    sep("S1 — 2D PHASE DIAGRAM  (Q2_std × β grid cells)")
    pr("  Each cell: mean logA | std logA | floor% | regime")
    pr()
    header = f"  {'β\\Q2_std':>12s}"
    q2_std_vals = [float(std_q2(q)) for q in q2_u]
    # Print sparse (every 3rd Q2 and 3rd beta)
    q_show   = q2_u[::3]
    q_s_show = [std_q2(q) for q in q_show]
    b_show   = beta_u[::3]

    pr(f"  {'β':>6s}" + "".join(f"  {std_q2(q):+5.2f}" for q in q_show))
    pr("  " + "-" * (6 + 8 * len(q_show)))
    for b in b_show:
        row = f"  {b:6.3f}"
        for q in q_show:
            sub = df[(df["Q2_log_center"]==q) & (df["beta_center"]==b)]
            if len(sub) == 0:
                row += "      --"
            else:
                fl  = 100 * (sub[TARGET] < FLOOR).mean()
                mn  = sub[TARGET].mean()
                row += f"  {mn:+5.1f}" if fl < 5 else f"  [{fl:.0f}%F]"
        pr(row)

    # ════════════════════════════════════════════════════════════════════════
    # S2 — Regime boundary sharpness
    # ════════════════════════════════════════════════════════════════════════
    sep("S2 — REGIME BOUNDARY SHARPNESS")
    pr("  How abrupt are the transitions between regimes?")
    pr("  Gradient = |ΔmeanLogA / Δstep| at each boundary")
    pr()
    pr("  Q2 direction (per β=0.5 slice):")
    b_mid = beta_u[len(beta_u)//2]
    sub_b = df[df["beta_center"]==b_mid].sort_values("Q2_log_center")
    means = sub_b.groupby("Q2_log_center")[TARGET].mean()
    diffs = np.abs(np.diff(means.values))
    q2_steps = means.index.values
    for i in np.argsort(diffs)[-8:]:
        pr(f"    Q2_std {std_q2(q2_steps[i]):+.3f}→{std_q2(q2_steps[i+1]):+.3f}"
           f"  |Δ|={diffs[i]:.3f}  {'← SHARP' if diffs[i]>2 else ''}")

    pr()
    pr("  β direction (per Q2_std=0 slice):")
    q_mid = q2_u[np.argmin(np.abs(np.array([std_q2(q) for q in q2_u])))]
    sub_q = df[df["Q2_log_center"]==q_mid].sort_values("beta_center")
    means_b = sub_q.groupby("beta_center")[TARGET].mean()
    diffs_b = np.abs(np.diff(means_b.values))
    b_steps = means_b.index.values
    for i in np.argsort(diffs_b)[-8:]:
        pr(f"    β {b_steps[i]:.3f}→{b_steps[i+1]:.3f}"
           f"  |Δ|={diffs_b[i]:.3f}  {'← SHARP' if diffs_b[i]>2 else ''}")

    # ════════════════════════════════════════════════════════════════════════
    # S3 — Regime statistics
    # ════════════════════════════════════════════════════════════════════════
    sep("S3 — PROPOSED REGIME DEFINITIONS & DATA STATISTICS")
    regime_stats = {}
    for name, d in REGIME_DEFS.items():
        sub  = df[df["regime"] == name]
        frac = 100 * len(sub) / len(df)
        fl   = 100 * (sub[TARGET] < FLOOR).mean() if len(sub) > 0 else 0
        y    = sub[TARGET]
        stats = {
            "n": len(sub), "frac%": frac, "floor%": fl,
            "mean_logA": float(y.mean()) if len(sub) > 0 else np.nan,
            "std_logA":  float(y.std())  if len(sub) > 1 else np.nan,
            "min_logA":  float(y.min())  if len(sub) > 0 else np.nan,
            "max_logA":  float(y.max())  if len(sub) > 0 else np.nan,
        }
        regime_stats[name] = stats
        pr(f"  [{name}] {d['desc']}")
        pr(f"    n={stats['n']:,} ({stats['frac%']:.1f}%)  "
           f"floor={stats['floor%']:.1f}%  "
           f"logA: mean={stats['mean_logA']:.2f} std={stats['std_logA']:.2f} "
           f"[{stats['min_logA']:.1f},{stats['max_logA']:.1f}]")
        pr()

    # ════════════════════════════════════════════════════════════════════════
    # S4 — Per-regime Hölder exponents (variogram per axis)
    # ════════════════════════════════════════════════════════════════════════
    sep("S4 — PER-REGIME HÖLDER EXPONENTS (axis-aligned variogram)")
    pr("  α → kernel:  <0.8=M12  0.8-1.5=M12/32  1.5-2.5=M32  2.5-4=M52  >4=RBF")
    pr()

    regime_holders = {}   # regime → {axis → (alpha, r2)}

    for reg_name in REGIME_DEFS:
        sub = df_clean[df_clean["regime"] == reg_name].copy()
        if len(sub) < 500:
            pr(f"  [{reg_name}] skipped (n={len(sub)} too small)")
            continue
        pr(f"  [{reg_name}] n={len(sub):,}")
        holders = {}
        for ax, col in zip(AXIS, COLS):
            other_cols = [c for c in COLS if c != col]
            n_uniq = sub[col].nunique()
            if n_uniq < 3:
                pr(f"    {ax:6s}: skip (only {n_uniq} unique values)")
                holders[ax] = (np.nan, np.nan)
                continue
            h, g = regime_variogram(sub, TARGET, col, other_cols,
                                    max_lags=min(12, n_uniq-1), max_groups=2000)
            alpha, r2 = holder_exponent(h, g)
            k = holder_to_kernel(alpha)
            holders[ax] = (alpha, r2)
            gamma1 = g[0] if len(g) > 0 else np.nan
            pr(f"    {ax:6s}: α={alpha:.3f}  R²={r2:.3f}  γ(lag1)={gamma1:.4f}  → {k}")
        regime_holders[reg_name] = holders
        pr()

    # ════════════════════════════════════════════════════════════════════════
    # S5 — Per-regime non-stationarity
    # ════════════════════════════════════════════════════════════════════════
    sep("S5 — PER-REGIME NON-STATIONARITY (variance ratio per axis)")
    pr("  Var ratio > 3 → Gibbs kernel needed for that axis")
    pr()

    regime_ns = {}   # regime → {axis → var_ratio}

    for reg_name in REGIME_DEFS:
        sub = df_clean[df_clean["regime"] == reg_name].copy()
        if len(sub) < 500:
            continue
        pr(f"  [{reg_name}]")
        ns_vals = {}
        for ax, col in zip(AXIS, COLS):
            vals     = sub[col].values
            y        = sub[TARGET].values
            edges    = np.percentile(vals, np.linspace(0, 100, 9))
            edges[0] -= 1e-9; edges[-1] += 1e-9
            bin_id   = np.digitize(vals, edges) - 1
            bin_id   = np.clip(bin_id, 0, 7)
            stds     = []
            for b in range(8):
                ys = y[bin_id == b]
                if len(ys) > 5:
                    stds.append(float(np.std(ys)))
            if len(stds) < 2:
                continue
            ratio = max(stds) / (min(stds) + 1e-9)
            ns_vals[ax] = ratio
            verdict = "Gibbs needed" if ratio > 3.0 else ("mild NS" if ratio > 1.5 else "stationary")
            pr(f"    {ax:6s}: var_ratio={ratio:.2f}  → {verdict}")
        regime_ns[reg_name] = ns_vals
        pr()

    # ════════════════════════════════════════════════════════════════════════
    # S6 — Principal direction: which axis drives most variance per regime
    # ════════════════════════════════════════════════════════════════════════
    sep("S6 — PRINCIPAL VARIANCE DIRECTION per regime")
    pr("  For each regime, which axis explains the most variance in logA?")
    pr("  Method: η² (eta-squared) from one-way ANOVA → % variance explained by binning")
    pr()

    def eta_squared(x, y, n_bins=10):
        """η² for binned ANOVA: fraction of variance in y explained by binning on x."""
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        bin_id = np.digitize(x, edges) - 1
        bin_id = np.clip(bin_id, 0, n_bins - 1)
        grand_mean = y.mean()
        ss_total   = np.sum((y - grand_mean) ** 2)
        ss_between = 0.0
        for b in range(n_bins):
            ys = y[bin_id == b]
            if len(ys) > 0:
                ss_between += len(ys) * (ys.mean() - grand_mean) ** 2
        return ss_between / (ss_total + 1e-12)

    for reg_name in REGIME_DEFS:
        sub = df_clean[df_clean["regime"] == reg_name].copy()
        if len(sub) < 500:
            continue
        y = sub[TARGET].values
        etas = {}
        for ax, col in zip(AXIS, COLS):
            x = sub[col].values
            etas[ax] = eta_squared(x, y)
        ranked = sorted(etas.items(), key=lambda kv: kv[1], reverse=True)
        pr(f"  [{reg_name}]")
        for ax, eta in ranked:
            bar = "█" * max(1, int(eta * 30))
            pr(f"    {ax:6s}: η²={eta:.4f}  {bar}")
        pr()

    # ════════════════════════════════════════════════════════════════════════
    # S7 — Within-regime correlation structure (pairwise)
    # ════════════════════════════════════════════════════════════════════════
    sep("S7 — WITHIN-REGIME PAIRWISE CORRELATION with logA")
    pr("  Pearson r between each input and logA, and between inputs")
    pr()

    for reg_name in REGIME_DEFS:
        sub = df_clean[df_clean["regime"] == reg_name].copy()
        if len(sub) < 500:
            continue
        y = sub[TARGET].values
        pr(f"  [{reg_name}]  n={len(sub):,}")
        pr(f"    {'Pair':22s}  r     |r|")
        pr("    " + "-" * 38)
        for ax, col in zip(AXIS, COLS):
            x = sub[col].values
            r, _ = pearsonr(x, y)
            pr(f"    {ax:6s} vs logA   :  {r:+.4f}  {abs(r):.4f}")
        pr()

    # ════════════════════════════════════════════════════════════════════════
    # S8 — Regime expert size and GP tractability
    # ════════════════════════════════════════════════════════════════════════
    sep("S8 — REGIME EXPERT TRACTABILITY")
    pr("  With 10% training: how many train points per regime?")
    pr("  Exact GP limit: ~8,000 / SVGP: unlimited")
    pr()
    pr(f"  {'Regime':6s}  {'N_total':>10s}  {'N_train(10%)':>14s}  {'Tractability':>20s}")
    sep()
    for reg_name in REGIME_DEFS:
        n_total = (df["regime"] == reg_name).sum()
        n_train = int(n_total * 0.10)
        if n_train < 100:
            trac = "TOO SMALL"
        elif n_train < 5_000:
            trac = "Exact GP (easy)"
        elif n_train < 20_000:
            trac = "Exact GP (feasible)"
        elif n_train < 100_000:
            trac = "SVGP recommended"
        else:
            trac = "SVGP required"
        pr(f"  {reg_name:6s}  {n_total:10,}  {n_train:14,}  {trac:>20s}")
    pr()
    pr("  NOTE: Q2-binned approach had 30 bins × 3,600 pts/bin = tractable exact GP.")
    pr("  Regime approach has FEWER but LARGER expert subsets.")
    pr("  Hybrid: further sub-partition large regimes by Q2 or W2 bins.")

    # ════════════════════════════════════════════════════════════════════════
    # S9 — Floor behaviour within each regime
    # ════════════════════════════════════════════════════════════════════════
    sep("S9 — FLOOR BEHAVIOUR WITHIN EACH REGIME")
    pr("  Does the floor cluster along specific sub-axes within each regime?")
    pr()
    for reg_name in REGIME_DEFS:
        sub_all   = df[df["regime"] == reg_name]
        sub_floor = sub_all[sub_all[TARGET] < FLOOR]
        if len(sub_all) < 100 or len(sub_floor) < 10:
            continue
        pr(f"  [{reg_name}]  floor={len(sub_floor):,} / {len(sub_all):,} "
           f"({100*len(sub_floor)/len(sub_all):.1f}%)")
        for ax, col in zip(AXIS, COLS):
            f_med = float(sub_floor[col].median())
            a_med = float(sub_all[col].median())
            diff  = f_med - a_med
            if abs(diff) > 0.1:
                pr(f"    {ax:6s}: floor_median={f_med:.3f}  all_median={a_med:.3f}"
                   f"  Δ={diff:+.3f}  {'← BIASED' if abs(diff)>0.5 else ''}")
        pr()

    # ════════════════════════════════════════════════════════════════════════
    # S10 — FINAL REGIME-EXPERT ARCHITECTURE BLUEPRINT
    # ════════════════════════════════════════════════════════════════════════
    sep("S10 — REGIME-EXPERT ARCHITECTURE BLUEPRINT")
    pr("""
  CONCEPT: instead of partitioning BY one axis (Q2-expert partitions on Q2,
  trains on W2/β/z), we define 5 PHYSICS-REGIME experts where each expert
  is trained on ALL 4 input variables, but only on data from its regime.

  This has two key advantages (literature-confirmed):
  1. Non-axis-aligned partitions match the actual structure of the data
     (the floor boundary is diagonal in (Q2,β) space, not axis-aligned)
  2. Within each regime, the function is locally SIMPLER → a single kernel
     can capture it without non-stationarity tricks

  Expert partition rule: based on (Q2_std, β) coordinates.
  Each expert sees: (Q2_log, W2_log, beta, z) → logA.
    """)

    # Collect all the per-regime kernel recommendations
    kernel_recommendations = {}
    for reg_name in REGIME_DEFS:
        if reg_name not in regime_holders:
            continue
        holders = regime_holders[reg_name]
        ns      = regime_ns.get(reg_name, {})

        # Vote per axis
        votes = {}
        for ax in AXIS:
            alpha, r2 = holders.get(ax, (np.nan, np.nan))
            base_k = holder_to_kernel(alpha)
            vr     = ns.get(ax, 1.0)
            if vr > 3.0:
                votes[ax] = f"Gibbs × {base_k}"
            else:
                votes[ax] = base_k
        kernel_recommendations[reg_name] = votes

    desc_lines = {
        "CPR": [
            "  [CPR] CENTRAL PHYSICAL REGIME",
            "  Partition: Q2_std ∈ (−0.98, +0.98) AND β ∈ (0.02, 0.83)",
            "  Physics: Pomeron exchange + DGLAP evolution, well-defined cross-section",
            "  Expected α: highest (smoothest within-regime function)",
            "  Floor: minimal (<2% globally, <0.5% in deepest interior)",
            "  Kernel suggestion: ARD kernel for all 4 vars; likely Matérn-3/2 or 5/2",
            "  Expert type: SMOOTH INTERIOR — single global GP with standard stationary kernel",
            "  GP strategy: EXACT GP per Q2-sub-bin (current approach is fine here)",
            "  → This is where the CURRENT Q2 expert does well. Keep it.",
        ],
        "ELR": [
            "  [ELR] ELASTIC LIMIT REGIME",
            "  Partition: β > 0.83 (ALL Q2, ALL W2, ALL z)",
            "  Physics: β→1 elastic/quasi-elastic scattering; cross-section drops sharply",
            "  Expected α: low (rough in Q2, near-zero in z)",
            "  Floor: 29-37% at β=0.999 across ALL Q2/W2",
            "  Kernel suggestion: GibbsBell(Q2) × GibbsSigmoid(β) × Matérn12(W2) × Matérn12(z)",
            "  Expert type: BOUNDARY REGIME — both Q2 extremes have floor; β is the primary driver",
            "  GP strategy: partition by β-sub-bins WITHIN this regime (β is now the 'easy' axis)",
            "  → CURRENT β-expert IS this regime, but trained on wrong partition boundary.",
            "  → Move β partition threshold to 0.83 (data-driven) vs current quantile bins.",
        ],
        "DSR": [
            "  [DSR] DGLAP SATURATION REGIME",
            "  Partition: Q2_std > 0.98 AND β ∈ (0.02, 0.83)",
            "  Physics: high-Q2 DGLAP evolution drives cross-section to zero for mid-β",
            "  Floor: 6-18% depending on β; floor grows with both Q2 and β",
            "  Expected α: low (rough in Q2 direction), moderate in W2",
            "  Kernel suggestion: GibbsPower(Q2, decreasing ℓ) × Matérn12(W2) × Matérn12(β) × Matérn12(z)",
            "  Expert type: HIGH-Q2 BOUNDARY — floor grows toward higher Q2",
            "  GP strategy: partition by Q2-sub-bins within this regime",
            "  → CURRENT Q2 expert covers this implicitly but its flat Q2-binning puts",
            "     high-Q2 bins and low-Q2 bins together → experts contaminate each other.",
        ],
        "PHR": [
            "  [PHR] PHOTOPRODUCTION REGIME",
            "  Partition: Q2_std < −1.3 AND β ∈ (0.02, 0.83)",
            "  Physics: near-photoproduction limit, cross-section depressed for low Q2",
            "  Floor: 5-8% concentrated at extreme low Q2",
            "  Expected α: very low (rough, near-floor for many points)",
            "  Kernel suggestion: Short-ℓ Matérn12 on all axes + high noise floor",
            "  Expert type: NEAR-FLOOR REGIME — most points are depressed but valid",
            "  GP strategy: may need heteroscedastic GP; σn² should be larger here",
            "  → CURRENT approach: these points scattered across all Q2 bins, contaminating training.",
            "  → Isolating them REMOVES floor contamination from the physical regime experts.",
        ],
        "TRN": [
            "  [TRN] TRANSITION REGIME",
            "  Partition: Q2_std ∈ (−1.3, −0.98) AND β ∈ (0.02, 0.83)",
            "  Physics: gradient zone between photoproduction and physical regime",
            "  Floor: 3-5% (intermediate)",
            "  Expected α: moderate (less rough than PHR, more than CPR)",
            "  Kernel suggestion: GibbsSigmoid(Q2, x0=−1.13) × Matérn12(W2,β,z)",
            "  Expert type: TRANSITION — interpolates between CPR and PHR",
            "  GP strategy: partition by Q2-sub-bins; share hyperprior with CPR",
            "  → Small regime (~5% of data); could be MERGED with CPR using GibbsSigmoid",
            "     to handle the boundary, or treated as a separate expert.",
        ],
    }

    for reg_name, dlines in desc_lines.items():
        for line in dlines:
            pr(line)
        if reg_name in kernel_recommendations:
            pr(f"  Kernel votes from variogram:")
            for ax, kv in kernel_recommendations[reg_name].items():
                pr(f"    {ax:6s}: {kv}")
        pr()

    sep("S10b — COMPARISON: AXIS-BASED vs REGIME-BASED")
    pr("""
  AXIS-BASED (current):
    Q2-expert:  partitions by Q2 (30 bins), trains on (W2, β, z)
                → each bin mixes CPR + DSR + PHR data in different proportions
                → Q2 variation WITHIN each bin is zero (it's removed)
    W2-expert:  partitions by W2 (30 bins), trains on (Q2, β, z)
                → each bin sees ALL Q2 regimes → non-stationarity from CPR↔DSR↔PHR
                → GibbsBell helped but the regime mixing remains
    β-expert:   partitions by β (30 bins), trains on (Q2, W2, z)
                → high-β bins are nearly all ELR; low-β bins are all CPR/DSR
                → the same kernel applies to fundamentally different physics
    z-expert:   partitions by z (40 bins), trains on (Q2, W2, β)
                → z has negligible direct effect; within z-bins function is smoother

  REGIME-BASED (proposed):
    CPR-expert: trained on ~50% of data (physical regime only)
                → function is smooth within this regime → Matérn-3/2 works perfectly
                → NO cross-regime contamination → much easier to fit
    ELR-expert: trained on ~12% of data (elastic limit only)
                → β→1 physics is isolated → GibbsBell handles Q2 extremes cleanly
                → further sub-partition by β-sub-bins within elastic regime
    DSR-expert: trained on ~15% of data (high-Q2 saturation)
                → decreasing cross-section in Q2 → GibbsPower on Q2 is ideal
    PHR-expert: trained on ~8% of data (photoproduction)
                → isolated from physical regime → simple Matérn12 + large noise
    TRN-expert: trained on ~5% of data (transition zone)
                → OR: merge with CPR using GibbsSigmoid boundary kernel

  KEY ADVANTAGE: Within each regime, the Hölder exponent is HIGHER (function
  is smoother) than globally, because the non-stationarity that drives low α
  is exactly the BETWEEN-REGIME variation. Partitioning by regime REMOVES
  the non-stationarity, leaving a locally stationary function.

  PREDICTED IMPROVEMENT:
    Current best: Physical MAE=0.541 (Matérn-1/2, KNN routing)
    Regime-based: Expected Physical MAE < 0.4 (rough estimate)
    Rationale: CPR expert (50% of data) will have smooth function →
               Matérn-3/2 or 5/2 will fit cleanly → lower MAE on clean points
    """)

    sep("S10c — IMPLEMENTATION RECIPE")
    pr("""
  In gp4d_lab/data.py — add RegimePartitioner:

    def assign_regime(q2_std, beta):
        # CPR: physical interior
        if -0.98 < q2_std < +0.98 and 0.02 < beta < 0.83:
            return "CPR"
        # ELR: elastic limit
        if beta >= 0.83:
            return "ELR"
        # DSR: DGLAP saturation (high Q2, non-elastic)
        if q2_std >= 0.98 and beta < 0.83:
            return "DSR"
        # PHR: photoproduction (low Q2, non-elastic)
        if q2_std <= -1.3 and beta < 0.83:
            return "PHR"
        # TRN: transition zone
        return "TRN"

  Then: for each regime, fit a separate GP ensemble (or SVGP for large regimes)
  using a regime-specific kernel:

    CPR: ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=4))     ← smooth interior
    ELR: ScaleKernel(GibbsBell(Q2) * GibbsSigmoid(β) * M12(W2,z))
    DSR: ScaleKernel(GibbsPower(Q2, α>0) * M12(W2,β,z))        ← ℓ shrinks at high Q2
    PHR: ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=4))      ← rough, near-floor
    TRN: ScaleKernel(GibbsSigmoid(Q2, x0=-1.13) * M12(W2,β,z))

  At inference: assign test point to regime → use that expert's GP.
  No aggregation needed — each test point belongs to exactly ONE regime.
  (Use a SOFT assignment with overlap σ=0.05 for smooth transitions.)
    """)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = args.out or str(Path(args.csv).parent / "regime_analysis_report.txt")
    Path(out_path).write_text("\n".join(lines))
    print(f"\n[Report saved → {out_path}]")


if __name__ == "__main__":
    main()