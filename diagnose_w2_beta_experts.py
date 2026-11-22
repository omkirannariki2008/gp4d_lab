#!/usr/bin/env python3
"""
diagnose_w2_beta_experts.py
===========================
Targeted analysis of W2-expert and beta-expert failure modes.

What this measures
------------------
1. Conditional mean + std of logA as function of Q2, marginalised over β/z,
   separately for each W2 slab and each β slab.
   → Reveals the exact shape and location of the Q2 transition.

2. Floor fraction as function of Q2 within each W2 / β slab.
   → Confirms where the photoproduction boundary sits in standardised Q2 space.

3. β-dimension conditional within W2 slabs.
   → Reveals whether β has a second boundary that the Gibbs kernel is missing.

4. Raw vs standardised coordinates of the floor boundary.
   → Gives us the exact gibbs_x0_init and gibbs_slope_init to use.

5. Joint (Q2, β) floor heatmap.
   → Shows whether both axes need a Gibbs component in the W2 expert.

Usage
-----
    python diagnose_w2_beta_experts.py --csv /path/to/data.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

COLS   = ["Q2_log_center", "W2_log_center", "beta_center", "z_center"]
TARGET = "logA"
FLOOR  = -17.955    # lower fence from S1 diagnostics

# ── Physics scaler constants (from S2 output — fill in from your run) ──────
# These are the mean / std used by PhysicsInputScaler4D.
# If you don't have them, set SCALER_KNOWN=False and the script will compute them.
SCALER_KNOWN = False
Q2_MEAN  = 0.0        # mean of logQ2 (S2: mean=-0.0000)
Q2_STD   = 2.7490     # std  of logQ2 (S2: std=2.7490)
BETA_EPS = 1e-3       # same as physics scaler default


def standardise_q2(q2_raw):
    return (q2_raw - Q2_MEAN) / Q2_STD


def standardise_beta(beta_raw):
    return (np.log(np.abs(beta_raw) + BETA_EPS))   # log-beta (before /std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--n_slabs",    type=int, default=6,
                        help="Number of slabs per axis")
    parser.add_argument("--out",        default=None,
                        help="Output text file (default: w2_beta_diagnosis.txt)")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"  {len(df):,} rows loaded")

    if SCALER_KNOWN is False:
        # Recompute from data
        global Q2_MEAN, Q2_STD
        Q2_MEAN = float(df["Q2_log_center"].mean())
        Q2_STD  = float(df["Q2_log_center"].std())
        print(f"  Q2 scaler: mean={Q2_MEAN:.4f}  std={Q2_STD:.4f}")

    df_clean = df[df[TARGET] > FLOOR].copy()
    df_floor = df[df[TARGET] <= FLOOR].copy()
    print(f"  Clean: {len(df_clean):,}  Floor: {len(df_floor):,} ({100*len(df_floor)/len(df):.2f}%)")

    # Standardised Q2 and log-beta
    df["Q2_std"]    = standardise_q2(df["Q2_log_center"])
    df_clean["Q2_std"] = standardise_q2(df_clean["Q2_log_center"])
    df_floor["Q2_std"] = standardise_q2(df_floor["Q2_log_center"])

    lines = []

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION A — Floor Q2 boundary (raw and standardised)
    # ═══════════════════════════════════════════════════════════════════════
    sep = "=" * 70
    lines.append(sep)
    lines.append("  A — FLOOR POINT Q2 BOUNDARY")
    lines.append(sep)

    q2_floor_raw = df_floor["Q2_log_center"]
    q2_floor_std = df_floor["Q2_std"]

    for pct, label in [(5,"p5"), (10,"p10"), (25,"p25"), (50,"p50"), (75,"p75"),
                        (90,"p90"), (95,"p95")]:
        raw_v = float(np.percentile(q2_floor_raw, pct))
        std_v = float(np.percentile(q2_floor_std, pct))
        lines.append(f"  {label:5s}: raw Q2_log={raw_v:7.3f}   standardised Q2={std_v:7.3f}")

    lines.append(f"\n  → Suggested gibbs_x0_init for W2/β experts:")
    x0_suggested = float(np.percentile(q2_floor_std, 75))  # 75th pct of floor → where 75% of floors have lower Q2
    lines.append(f"    Use the p75 standardised Q2 of floor points = {x0_suggested:.3f}")
    lines.append(f"    (This is where the transition from floor→normal occurs)")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION B — Floor fraction as function of Q2 (global)
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("\n" + sep)
    lines.append("  B — FLOOR FRACTION vs Q2 (global, all β/z)")
    lines.append(sep)

    q2_unique = np.sort(df["Q2_log_center"].unique())
    lines.append(f"  {'Q2_log':>10s}  {'Q2_std':>8s}  {'n_floor':>8s}  {'n_total':>8s}  {'floor%':>7s}")
    lines.append("  " + "-"*52)
    for q in q2_unique:
        sub   = df[df["Q2_log_center"] == q]
        nf    = (sub[TARGET] <= FLOOR).sum()
        nt    = len(sub)
        q_std = standardise_q2(q)
        lines.append(f"  {q:10.4f}  {q_std:8.3f}  {nf:8d}  {nt:8d}  {100*nf/nt:6.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION C — W2 expert: conditional logA vs Q2 per W2 slab
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("\n" + sep)
    lines.append("  C — W2-EXPERT: conditional logA vs Q2 per W2 slab")
    lines.append(sep)
    lines.append("  (Within each W2 slab, how does logA vary with Q2?)")
    lines.append("  Format: Q2_std | mean(logA) | std(logA) | floor%")

    w2_edges = np.percentile(df["W2_log_center"], np.linspace(0, 100, args.n_slabs + 1))
    for i in range(args.n_slabs):
        lo, hi = w2_edges[i], w2_edges[i + 1]
        slab = df[(df["W2_log_center"] >= lo) & (df["W2_log_center"] < hi)]
        lines.append(f"\n  W2 slab {i+1}/{args.n_slabs}: W2_log=[{lo:.3f},{hi:.3f}]  n={len(slab):,}")
        lines.append(f"  {'Q2_std':>8s}  {'mean_logA':>10s}  {'std_logA':>10s}  {'floor%':>7s}")
        lines.append("  " + "-" * 44)
        for q in q2_unique:
            sub = slab[slab["Q2_log_center"] == q]
            if len(sub) == 0:
                continue
            q_std = standardise_q2(q)
            mean_y = float(sub[TARGET].mean())
            std_y  = float(sub[TARGET].std())
            flr    = 100 * (sub[TARGET] <= FLOOR).mean()
            lines.append(f"  {q_std:8.3f}  {mean_y:10.3f}  {std_y:10.3f}  {flr:6.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION D — W2 expert: β boundary analysis
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("\n" + sep)
    lines.append("  D — W2-EXPERT: floor fraction vs β per W2 slab")
    lines.append(sep)
    lines.append("  (Does β→1 create a floor boundary inside W2 expert?)")

    beta_unique = np.sort(df["beta_center"].unique())
    for i in range(args.n_slabs):
        lo, hi = w2_edges[i], w2_edges[i + 1]
        slab = df[(df["W2_log_center"] >= lo) & (df["W2_log_center"] < hi)]
        lines.append(f"\n  W2 slab {i+1}/{args.n_slabs}: W2_log=[{lo:.3f},{hi:.3f}]")
        lines.append(f"  {'beta':>8s}  {'mean_logA':>10s}  {'floor%':>7s}")
        lines.append("  " + "-" * 32)
        for b in beta_unique:
            sub = slab[slab["beta_center"] == b]
            if len(sub) == 0:
                continue
            mean_y = float(sub[TARGET].mean())
            flr    = 100 * (sub[TARGET] <= FLOOR).mean()
            if flr > 0.1 or b > 0.8:   # show interesting ones
                lines.append(f"  {b:8.4f}  {mean_y:10.3f}  {flr:6.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION E — Beta expert: conditional logA vs Q2 per β slab
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("\n" + sep)
    lines.append("  E — BETA-EXPERT: conditional logA vs Q2 per β slab")
    lines.append(sep)
    lines.append("  (Within each β slab, how does logA vary with Q2?)")

    beta_edges = np.percentile(df["beta_center"], np.linspace(0, 100, args.n_slabs + 1))
    for i in range(args.n_slabs):
        lo, hi = beta_edges[i], beta_edges[i + 1]
        slab = df[(df["beta_center"] >= lo) & (df["beta_center"] < hi)]
        lines.append(f"\n  β slab {i+1}/{args.n_slabs}: β=[{lo:.4f},{hi:.4f}]  n={len(slab):,}")
        lines.append(f"  {'Q2_std':>8s}  {'mean_logA':>10s}  {'std_logA':>10s}  {'floor%':>7s}")
        lines.append("  " + "-" * 44)
        for q in q2_unique:
            sub = slab[slab["Q2_log_center"] == q]
            if len(sub) == 0:
                continue
            q_std  = standardise_q2(q)
            mean_y = float(sub[TARGET].mean())
            std_y  = float(sub[TARGET].std())
            flr    = 100 * (sub[TARGET] <= FLOOR).mean()
            lines.append(f"  {q_std:8.3f}  {mean_y:10.3f}  {std_y:10.3f}  {flr:6.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION F — Joint (Q2, β) floor heatmap for W2-expert decision
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("\n" + sep)
    lines.append("  F — JOINT (Q2_std, β) FLOOR FRACTION HEATMAP")
    lines.append(sep)
    lines.append("  Row=Q2_std bins, Col=β bins, Value=floor%")
    lines.append("  → Reveals if BOTH Q2 AND β need Gibbs in W2-expert")

    q2_bins   = pd.cut(df["Q2_std"],      bins=8, labels=False)
    beta_bins = pd.cut(df["beta_center"], bins=8, labels=False)
    df["Q2_bin"]   = q2_bins
    df["beta_bin"] = beta_bins

    pivot = df.groupby(["Q2_bin", "beta_bin"]).apply(
        lambda g: 100 * (g[TARGET] <= FLOOR).mean()
    ).unstack(fill_value=0)

    q2_labels   = [f"{v:.2f}" for v in np.linspace(df["Q2_std"].min(), df["Q2_std"].max(), 9)[:-1]]
    beta_labels = [f"{v:.2f}" for v in np.linspace(df["beta_center"].min(), df["beta_center"].max(), 9)[:-1]]

    header = "  Q2_std\\β |" + "".join(f"{b:>7s}" for b in beta_labels)
    lines.append(header)
    lines.append("  " + "-" * len(header))
    for qi, q_lbl in enumerate(q2_labels):
        if qi not in pivot.index:
            continue
        row_vals = [f"{pivot.loc[qi, bi]:6.1f}" if bi in pivot.columns else "   0.0"
                    for bi in range(8)]
        lines.append(f"  {q_lbl:>9s} |" + "".join(row_vals))

    # ═══════════════════════════════════════════════════════════════════════
    # SECTION G — Recommended kernel parameters
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("\n" + sep)
    lines.append("  G — RECOMMENDED KERNEL PARAMETERS (from this analysis)")
    lines.append(sep)

    # Find where floor fraction crosses 5%
    floor_by_q2 = {}
    for q in q2_unique:
        sub = df[df["Q2_log_center"] == q]
        floor_by_q2[q] = 100 * (sub[TARGET] <= FLOOR).mean()

    # Q2 where floor% first exceeds 5%
    boundary_q2_std = float(np.percentile(standardise_q2(
        np.array([q for q in q2_unique if floor_by_q2.get(q, 0) > 5.0] or [q2_unique[0]])
    ), 75))
    for q in sorted(q2_unique, reverse=True):
        if floor_by_q2[q] > 5.0:
            boundary_q2_std = float(standardise_q2(q))
            lines.append(f"\n  Q2 floor boundary: Q2_log={q:.3f}  Q2_std={boundary_q2_std:.3f}")
            lines.append(f"    → gibbs_x0_init = {boundary_q2_std:.2f}  (for W2 and β experts)")
            break
    else:
        boundary_q2_std = -1.33  # fallback
        lines.append(f"\n  No Q2 value with >5% floor — using physics default: Q2_std=-1.33")

    # Find β where floor% first exceeds 5%
    floor_by_beta = {}
    for b in beta_unique:
        sub = df[df["beta_center"] == b]
        floor_by_beta[b] = 100 * (sub[TARGET] <= FLOOR).mean()

    for b in sorted(beta_unique):
        if floor_by_beta[b] > 5.0:
            log_beta = float(np.log(b + BETA_EPS))
            lines.append(f"\n  β floor boundary: β={b:.4f}  log(β+eps)={log_beta:.3f}")
            lines.append(f"    → If β needs Gibbs in W2-expert, β_x0 ≈ {log_beta:.2f} (raw log space)")
            break

    lines.append("\n  Summary config changes:")
    lines.append(f"    cfg.kernels['w2'].gibbs_x0_init   = {boundary_q2_std:.2f}")
    lines.append(f"    cfg.kernels['beta'].gibbs_x0_init = {boundary_q2_std:.2f}")
    lines.append(f"    cfg.kernels['w2'].gibbs_slope_init  = 5.0  # steeper than default 3.0")
    lines.append(f"    cfg.kernels['beta'].gibbs_slope_init = 5.0")

    # Print and save
    out_path = args.out or str(Path(args.csv).parent / "w2_beta_diagnosis.txt")
    report   = "\n".join(lines)
    print(report)
    Path(out_path).write_text(report)
    print(f"\n[Report saved → {out_path}]")


if __name__ == "__main__":
    main()