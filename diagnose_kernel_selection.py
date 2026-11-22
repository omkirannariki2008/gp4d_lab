#!/usr/bin/env python3
"""
diagnose_kernel_selection.py
============================
Full-data diagnostic for GP kernel selection on the inclusive diffraction
cross-section dataset (1M rows: Q2_log_center, W2_log_center, beta_center,
z_center → logA).

What this script measures
-------------------------
1.  TARGET DISTRIBUTION   — logA stats, floor fraction, outlier bounds
2.  DIRECTIONAL VARIOGRAM — empirical semivariance γ(h) along each axis.
                            Log-log slope at small h → Hölder exponent α.
                            α maps directly to the correct Matérn ν.
3.  ROUGHNESS SUMMARY     — per-axis Hölder exponent → kernel recommendation
4.  NON-STATIONARITY      — variance ratio across bins per axis.
                            High ratio → Gibbs; low → stationary kernel OK.
5.  JUMP / DIP DETECTION  — sorted consecutive differences per axis.
                            Reports where |ΔlogA/Δx| exceeds the 99.9th pct.
6.  PER-EXPERT ANALYSIS   — each expert sees 3 remaining dims after partition.
                            Repeats variogram + roughness inside each axis-slab.
7.  KERNEL RECOMMENDATIONS — final lookup table for your gp4d_lab config.

GPU / batch details
-------------------
All pairwise operations are batched.  The variogram uses sorted consecutive
pairs (O(n) per axis, GPU-vectorised).  Full-data stats use torch.histc /
scatter_add with configurable batch_size.

Usage
-----
    # auto device, default batch
    python diagnose_kernel_selection.py --csv /path/to/data.csv

    # force GPU, large batch
    python diagnose_kernel_selection.py --csv data.csv --device cuda \
        --batch_size 200000 --n_lags 80

    # CPU-only, quick scan
    python diagnose_kernel_selection.py --csv data.csv --device cpu \
        --sample_rows 200000

Output
------
    console: rich table report with section headers
    file:    kernel_diagnosis_report.txt  (plain-text copy in same directory)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import warnings
import pandas as pd
import torch
from scipy.stats import linregress
from tqdm import tqdm

# ── optional rich console ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.rule import Rule
    from rich import print as rprint
    RICH = True
    console = Console(record=True)
except ImportError:
    RICH = False
    console = None

COLS   = ["Q2_log_center", "W2_log_center", "beta_center", "z_center"]
AXIS   = ["q2",            "w2",            "beta",        "z"]
TARGET = "logA"
FLOOR_VALUE = -50.0      # logA below this is treated as the unphysical floor

# ── Hölder exponent → Matérn ν mapping ──────────────────────────────────────
# For Matérn ν, γ(h) ~ h^min(2ν,2) near origin.
# We use the small-lag log-log slope α to recommend ν.
def holder_to_kernel(alpha: float) -> tuple[str, str]:
    """
    (composite_kernel_name, human_reason) from Hölder exponent.
    """
    if alpha < 0.8:
        return "matern12", f"α={alpha:.2f} < 0.8 → non-differentiable, very rough"
    elif alpha < 1.5:
        return "matern12_or_matern32", f"α={alpha:.2f} ∈ [0.8,1.5) → rough, borderline differentiable"
    elif alpha < 2.5:
        return "matern32", f"α={alpha:.2f} ∈ [1.5,2.5) → once differentiable"
    elif alpha < 4.0:
        return "matern52", f"α={alpha:.2f} ∈ [2.5,4.0) → twice differentiable"
    else:
        return "rbf_or_matern52", f"α={alpha:.2f} ≥ 4.0 → very smooth (nearly infinite deriv)"


# ── Variance ratio → stationarity verdict ───────────────────────────────────
def var_ratio_verdict(ratio: float) -> str:
    if ratio > 5.0:
        return "STRONGLY non-stationary → USE Gibbs kernel"
    elif ratio > 2.5:
        return "Moderately non-stationary → Gibbs preferred"
    elif ratio > 1.5:
        return "Mild non-stationarity → ARD Matérn acceptable"
    else:
        return "Approximately stationary → any standard kernel OK"


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1  —  data loading
# ════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str, sample_rows: int | None = None) -> pd.DataFrame:
    t0 = time.time()
    _pr(f"[bold]Loading[/bold] {csv_path} …")
    needed = COLS + [TARGET, "Q2_bin_idx"] if "Q2_bin_idx" in \
        pd.read_csv(csv_path, nrows=0).columns else COLS + [TARGET]
    df = pd.read_csv(csv_path, usecols=lambda c: c in needed)

    if sample_rows and len(df) > sample_rows:
        _pr(f"  Sampling {sample_rows:,} / {len(df):,} rows")
        df = df.sample(sample_rows, random_state=42).reset_index(drop=True)

    _pr(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2  —  target distribution
# ════════════════════════════════════════════════════════════════════════════

def analyse_target(df: pd.DataFrame) -> dict:
    y = df[TARGET].values
    n = len(y)

    floor_mask = y < FLOOR_VALUE
    normal_y   = y[~floor_mask]

    q25, q50, q75 = np.percentile(normal_y, [25, 50, 75])
    iqr = q75 - q25

    stats = {
        "n_total":        n,
        "n_floor":        int(floor_mask.sum()),
        "floor_pct":      100 * floor_mask.sum() / n,
        "floor_value":    float(y[floor_mask].mean()) if floor_mask.any() else np.nan,
        "mean_all":       float(y.mean()),
        "mean_normal":    float(normal_y.mean()),
        "median_normal":  float(q50),
        "std_all":        float(y.std()),
        "std_normal":     float(normal_y.std()),
        "iqr_normal":     float(iqr),
        "lower_fence":    float(q50 - 3 * iqr),
        "upper_fence":    float(q50 + 3 * iqr),
        "pct99":          float(np.percentile(normal_y, 99)),
        "pct01":          float(np.percentile(normal_y, 1)),
    }
    return stats


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3  —  directional variogram (GPU, sorted consecutive pairs)
# ════════════════════════════════════════════════════════════════════════════

def directional_variogram(
    x_vals:     torch.Tensor,   # (N,) axis values
    y_vals:     torch.Tensor,   # (N,) logA values
    n_lags:     int  = 60,
    h_pct_max:  float = 0.15,   # use lags up to this percentile of x-range
    device:     str  = "cpu",
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Empirical semivariogram along one axis using sorted consecutive pairs.
    Returns (h_centers, gamma, n_pairs_used).
    """
    x_vals = x_vals.to(device)
    y_vals = y_vals.to(device)

    idx     = torch.argsort(x_vals)
    x_s     = x_vals[idx]
    y_s     = y_vals[idx]

    dx  = (x_s[1:] - x_s[:-1]).abs()
    dy2 = 0.5 * (y_s[1:] - y_s[:-1]).pow(2)

    # Trim to physically meaningful lags only
    h_max  = float(torch.quantile(dx, 1.0 - 1e-4)) * h_pct_max
    valid  = dx < h_max
    dx_v   = dx[valid]
    dy2_v  = dy2[valid]

    if len(dx_v) < 20:
        return np.array([]), np.array([]), 0

    edges  = torch.linspace(0, float(dx_v.max()) * 1.001, n_lags + 1, device=device)
    bin_id = torch.bucketize(dx_v, edges) - 1
    bin_id = bin_id.clamp(0, n_lags - 1)

    gamma  = torch.zeros(n_lags, device=device)
    cnt    = torch.zeros(n_lags, device=device)
    gamma.scatter_add_(0, bin_id, dy2_v)
    cnt.scatter_add_(0, bin_id, torch.ones_like(dy2_v))

    ok     = cnt > 10
    h_mid  = 0.5 * (edges[:-1] + edges[1:])[ok]
    gamma  = (gamma / cnt.clamp(min=1))[ok]

    return h_mid.cpu().numpy(), gamma.cpu().numpy(), int(cnt[ok].sum().item())


def estimate_holder(
    h: np.ndarray,
    gamma: np.ndarray,
    h_frac: float = 0.20,   # use smallest h_frac of lag range for slope
) -> tuple[float, float]:
    """
    Log-log slope of variogram at small lags = Hölder exponent.
    Returns (alpha, R²).  alpha → Matérn ν.
    """
    if len(h) < 4:
        return np.nan, np.nan

    h_cut  = h.max() * h_frac
    mask   = (h < h_cut) & (h > 0) & (gamma > 0)

    if mask.sum() < 3:
        mask = (h > 0) & (gamma > 0)
        if mask.sum() < 3:
            return np.nan, np.nan

    lh = np.log(h[mask])
    lg = np.log(gamma[mask])
    slope, _, r, _, _ = linregress(lh, lg)
    return float(slope), float(r**2)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4  —  non-stationarity (variance ratio across bins)
# ════════════════════════════════════════════════════════════════════════════

def is_regular_grid(df: pd.DataFrame, col: str, tol: float = 0.05) -> bool:
    """Detect if axis is on a regular grid (uniform spacing)."""
    vals = np.sort(df[col].unique())
    if len(vals) < 3:
        return False
    gaps = np.diff(vals)
    cv   = float(gaps.std() / (gaps.mean() + 1e-12))
    return cv < tol


def axis_aligned_variogram(
    df:          pd.DataFrame,
    target_col:  str,         # axis we're measuring roughness along
    other_cols:  list,        # all other axis columns (held fixed)
    y_col:       str,
    max_lags:    int   = 14,
    max_groups:  int   = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Axis-aligned empirical variogram for REGULAR GRIDS.
    Fixes all other axes, varies the target axis → lag-k differences.
    
    Correct for 30×30×30×40 grids where consecutive-pair approach fails.
    Returns (h_array, gamma_array).
    """
    df_s = df.sort_values(target_col)
    ax_vals = np.sort(df_s[target_col].unique())
    if len(ax_vals) < 3:
        return np.array([]), np.array([])

    spacing = float(np.median(np.diff(ax_vals)))

    # Group by ALL other axes (each group is a 1D slice along target axis)
    groups = (
        df_s.groupby(other_cols)[y_col]
           .apply(lambda g: g.values)
    )

    # Random subsample of groups for speed
    if len(groups) > max_groups:
        rng   = np.random.default_rng(42)
        idxs  = rng.choice(len(groups), max_groups, replace=False)
        keys  = [groups.index[i] for i in idxs]
        groups = groups[keys]

    gamma_acc: dict = {}
    for lag in range(1, max_lags + 1):
        vals: list = []
        for _, y_arr in groups.items():
            if len(y_arr) > lag:
                d = 0.5 * (y_arr[lag:] - y_arr[:-lag]) ** 2
                vals.extend(d.tolist())
        if len(vals) > 20:
            gamma_acc[lag * spacing] = float(np.mean(vals))

    if not gamma_acc:
        return np.array([]), np.array([])

    h = np.array(sorted(gamma_acc.keys()))
    g = np.array([gamma_acc[k] for k in h])
    return h, g


def nonstationarity_test(

    x_vals: torch.Tensor,
    y_vals: torch.Tensor,
    n_bins: int = 12,
    device: str = "cpu",
) -> dict:
    """
    Divide x into n_bins equal-quantile slabs. Compute std(y) per slab.
    Returns variance ratio and per-bin stats.
    """
    x_vals = x_vals.to(device)
    y_vals = y_vals.to(device)

    edges  = torch.quantile(x_vals, torch.linspace(0, 1, n_bins + 1, device=device))
    edges[0]  -= 1e-6
    edges[-1] += 1e-6

    bin_id = torch.bucketize(x_vals, edges) - 1
    bin_id = bin_id.clamp(0, n_bins - 1)

    means  = []
    stds   = []
    counts = []
    for b in range(n_bins):
        mask = bin_id == b
        if mask.sum() < 5:
            continue
        yb = y_vals[mask]
        means.append(float(yb.mean()))
        stds.append(float(yb.std()))
        counts.append(int(mask.sum()))

    if not stds:
        return {"var_ratio": np.nan, "bin_means": [], "bin_stds": []}

    return {
        "var_ratio":  max(stds) / (min(stds) + 1e-9),
        "mean_range": max(means) - min(means),
        "bin_means":  means,
        "bin_stds":   stds,
        "bin_counts": counts,
        "std_cv":     np.std(stds) / (np.mean(stds) + 1e-9),  # coefficient of variation of per-bin stds
    }


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5  —  jump / dip detection
# ════════════════════════════════════════════════════════════════════════════

def detect_jumps(
    x_vals:    torch.Tensor,
    y_vals:    torch.Tensor,
    threshold: float = 5.0,    # sigma multiplier for calling a jump
    top_k:     int   = 10,
    device:    str   = "cpu",
) -> dict:
    """
    Sort by x. Compute |Δy| / median_abs(Δy) for consecutive pairs.
    Flag pairs where ratio > threshold·median as jumps.
    """
    x_vals = x_vals.to(device)
    y_vals = y_vals.to(device)

    idx    = torch.argsort(x_vals)
    x_s    = x_vals[idx]
    y_s    = y_vals[idx]

    dy     = (y_s[1:] - y_s[:-1]).abs()
    dx     = (x_s[1:] - x_s[:-1]).abs().clamp(min=1e-9)
    rate   = dy / dx    # |ΔlogA / Δx|

    med    = float(torch.median(rate))
    std_r  = float(rate.std())
    thresh = med + threshold * std_r

    jump_mask = rate > thresh
    n_jumps   = int(jump_mask.sum())

    # Top-k jump locations
    top_vals, top_idx = torch.topk(rate, min(top_k, len(rate)))
    jump_locs = x_s[top_idx].cpu().numpy()
    jump_mags = top_vals.cpu().numpy()

    return {
        "n_jumps":    n_jumps,
        "jump_frac":  n_jumps / max(len(dy), 1),
        "max_rate":   float(rate.max()),
        "median_rate": med,
        "p99_rate":   float(torch.quantile(rate, 0.99)),
        "top_locs":   jump_locs.tolist(),
        "top_mags":   jump_mags.tolist(),
        "threshold":  thresh,
    }


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6  —  per-expert analysis
# ════════════════════════════════════════════════════════════════════════════

def per_expert_roughness(
    df:        pd.DataFrame,
    partition_axis: str,        # e.g. "q2" — the axis being partitioned out
    n_slabs:   int   = 5,       # number of quantile slabs to analyse
    n_lags:    int   = 50,
    device:    str   = "cpu",
) -> dict:
    """
    For each quantile slab of partition_axis, compute Hölder exponents
    of logA along the 3 remaining axes.
    Returns {remaining_axis: [alpha_slab0, alpha_slab1, ...]}
    """
    ax_idx      = AXIS.index(partition_axis)
    col         = COLS[ax_idx]
    remain_axes = [a for a in AXIS if a != partition_axis]
    remain_cols = [COLS[AXIS.index(a)] for a in remain_axes]

    x_part = df[col].values
    edges  = np.percentile(x_part, np.linspace(0, 100, n_slabs + 1))
    edges[0]  -= 1e-9
    edges[-1] += 1e-9

    results = {ra: [] for ra in remain_axes}

    for s in range(n_slabs):
        mask = (x_part >= edges[s]) & (x_part < edges[s + 1])
        sub  = df[mask]
        if len(sub) < 200:
            for ra in remain_axes:
                results[ra].append(np.nan)
            continue

        for ra, rc in zip(remain_axes, remain_cols):
            x_t = torch.tensor(sub[rc].values, dtype=torch.float32)
            y_t = torch.tensor(sub[TARGET].values, dtype=torch.float32)
            other_ax_cols = [COLS[AXIS.index(a)]
                             for a in remain_axes if a != ra]
            if is_regular_grid(sub, rc):
                # axis-aligned approach for regular grids
                h, g = axis_aligned_variogram(
                    sub, rc, other_ax_cols, TARGET,
                    max_lags=min(12, sub[rc].nunique() - 1),
                    max_groups=1000,
                )
            else:
                x_t = torch.tensor(sub[rc].values, dtype=torch.float32)
                h, g, _ = directional_variogram(x_t, y_t, n_lags=n_lags, device=device)
            alpha, r2 = estimate_holder(h, g)
            results[ra].append(float(alpha) if r2 > 0.7 else np.nan)

    return results


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7  —  output helpers
# ════════════════════════════════════════════════════════════════════════════

def _pr(msg: str) -> None:
    if RICH:
        console.print(msg)
    else:
        print(msg.replace("[bold]", "").replace("[/bold]", "")
                  .replace("[red]", "").replace("[/red]", "")
                  .replace("[green]", "").replace("[/green]", "")
                  .replace("[yellow]", "").replace("[/yellow]", ""))


def _rule(title: str) -> None:
    if RICH:
        console.rule(f"[bold]{title}[/bold]")
    else:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")


def _table(headers: list[str], rows: list[list]) -> None:
    if RICH:
        t = Table(*headers, show_header=True, header_style="bold cyan")
        for r in rows:
            t.add_row(*[str(v) for v in r])
        console.print(t)
    else:
        # Simple plain-text table
        widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("-" * (sum(widths) + 2 * len(widths)))
        for r in rows:
            print(fmt.format(*r))


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser(description="GP kernel selection diagnostics")
    parser.add_argument("--csv",         required=True, help="Path to full CSV")
    parser.add_argument("--device",      default="auto",
                        help="cpu | cuda | auto")
    parser.add_argument("--batch_size",  type=int, default=500_000,
                        help="GPU batch size for variogram (default 500k)")
    parser.add_argument("--n_lags",      type=int, default=60,
                        help="Number of lag bins for variogram")
    parser.add_argument("--n_slabs",     type=int, default=5,
                        help="Number of slabs for per-expert analysis")
    parser.add_argument("--sample_rows", type=int, default=None,
                        help="Subsample CSV to this many rows (None = all)")
    parser.add_argument("--out",         default=None,
                        help="Output text file path (default: kernel_diagnosis_report.txt)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    _pr(f"[bold]Device:[/bold] {device}  (CUDA: {torch.cuda.is_available()})")
    if device == "cuda":
        _pr(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load ─────────────────────────────────────────────────────────────────
    df = load_data(args.csv, args.sample_rows)

    # Filter floor for variogram (keep for stats but not for roughness estimation)
    floor_mask    = df[TARGET] < FLOOR_VALUE
    df_clean      = df[~floor_mask].reset_index(drop=True)
    _pr(f"  Floor points removed for roughness analysis: "
        f"{floor_mask.sum():,} ({100*floor_mask.mean():.2f}%)")

    # Tensors (clean data for roughness, all data for other stats)
    tensors_all   = {c: torch.tensor(df[c].values,     dtype=torch.float32)
                     for c in COLS + [TARGET]}
    tensors_clean = {c: torch.tensor(df_clean[c].values, dtype=torch.float32)
                     for c in COLS + [TARGET]}
    y_all         = tensors_all[TARGET]
    y_clean       = tensors_clean[TARGET]

    report_lines: list[str] = []

    # ════════════════════════════════════════════════════════════════════════
    # S1 — Target Distribution
    # ════════════════════════════════════════════════════════════════════════
    _rule("S1 — TARGET DISTRIBUTION (logA)")
    ts = analyse_target(df)
    rows = [
        ["N total",           f"{ts['n_total']:,}"],
        ["Floor points",      f"{ts['n_floor']:,}  ({ts['floor_pct']:.2f}%)"],
        ["Floor value (mean)",f"{ts['floor_value']:.4f}"],
        ["Mean (normal)",     f"{ts['mean_normal']:.4f}"],
        ["Median (normal)",   f"{ts['median_normal']:.4f}"],
        ["Std (all)",         f"{ts['std_all']:.4f}"],
        ["Std (normal)",      f"{ts['std_normal']:.4f}"],
        ["IQR (normal)",      f"{ts['iqr_normal']:.4f}"],
        ["Lower fence (M-3*IQR)", f"{ts['lower_fence']:.4f}"],
        ["Upper fence (M+3*IQR)", f"{ts['upper_fence']:.4f}"],
        ["Std/IQR ratio",     f"{ts['std_all']/ts['iqr_normal']:.3f}  (>1.5 = heavy tail)"],
    ]
    _table(["Statistic", "Value"], rows)
    report_lines += [f"{r[0]:35s}: {r[1]}" for r in rows]

    # ════════════════════════════════════════════════════════════════════════
    # S2 — Input variable ranges & grid structure
    # ════════════════════════════════════════════════════════════════════════
    _rule("S2 — INPUT VARIABLE RANGES & GRID STRUCTURE")
    rows2 = []
    for ax, col in zip(AXIS, COLS):
        v  = df[col].values
        nu = len(np.unique(v))
        spacing_med = float(np.median(np.diff(np.sort(np.unique(v))))) if nu > 1 else np.nan
        rows2.append([
            ax, col,
            f"{v.min():.4f}", f"{v.max():.4f}",
            f"{v.mean():.4f}", f"{v.std():.4f}",
            f"{nu:,}", f"{spacing_med:.4f}" if not np.isnan(spacing_med) else "—",
        ])
    _table(["Axis","Column","Min","Max","Mean","Std","#Unique","Med spacing"], rows2)
    report_lines += ["\n--- INPUT VARIABLE RANGES ---"]
    report_lines += [str(r) for r in rows2]

    # ════════════════════════════════════════════════════════════════════════
    # S3 — Directional variogram + Hölder exponent (CLEAN data)
    # ════════════════════════════════════════════════════════════════════════
    _rule("S3 — DIRECTIONAL VARIOGRAM & HÖLDER EXPONENT")
    _pr("[yellow]Computing variogram along each axis (sorted consecutive pairs)…[/yellow]")

    vario_results: dict = {}
    holder_rows  = []

    for ax, col in zip(AXIS, COLS):
        t0 = time.time()
        other_cols = [c for c in COLS if c != col]

        # Auto-detect grid type: axis-aligned for regular grids
        if is_regular_grid(df_clean, col):
            _pr(f"  [{ax:4s}] Regular grid detected — using axis-aligned variogram")
            h, g = axis_aligned_variogram(
                df_clean, col, other_cols, TARGET,
                max_lags=min(14, df_clean[col].nunique() - 1),
                max_groups=3000,
            )
            n_pairs = len(g)
        else:
            x_t = tensors_clean[col].to(device)
            y_t = y_clean.to(device)
            h, g, n_pairs = directional_variogram(
                x_t, y_t, n_lags=args.n_lags, device=device
            )

        alpha, r2 = estimate_holder(h, g)
        k_rec, reason = holder_to_kernel(alpha)

        vario_results[ax] = {"h": h, "gamma": g, "alpha": alpha,
                             "r2": r2, "kernel": k_rec}

        elapsed = time.time() - t0
        holder_rows.append([
            ax,
            f"{alpha:.3f}" if not np.isnan(alpha) else "nan",
            f"{r2:.3f}"   if not np.isnan(r2)    else "nan",
            str(n_pairs),
            k_rec,
            f"{elapsed:.1f}s",
        ])
        _pr(f"  [{ax:4s}]  α={alpha:.3f}  R²={r2:.3f}  n_lags={n_pairs}  → {k_rec}  ({elapsed:.1f}s)")

    _rule("  VARIOGRAM SUMMARY TABLE")
    _pr("  [yellow]NOTE: nan α = too few points or too-coarse grid for slope fit.[/yellow]")
    _pr("  [yellow]On the real 1M-row dataset with continuous axes this will be populated.[/yellow]")
    _table(["Axis","Hölder α","R²","Pairs","Recommended kernel","Time"], holder_rows)
    report_lines += ["\n--- VARIOGRAM SUMMARY ---"]
    report_lines += [str(r) for r in holder_rows]

    # Variogram log-log slopes at small lags (extra detail)
    _pr("\n[bold]Small-lag variogram values (first 8 lags per axis):[/bold]")
    for ax in AXIS:
        h = vario_results[ax]["h"]
        g = vario_results[ax]["g"] if "g" in vario_results[ax] else vario_results[ax].get("gamma")
        if g is None or len(h) == 0:
            continue
        n_show = min(8, len(h))
        vals   = "  ".join(f"h={h[i]:.3f}→γ={g[i]:.4f}" for i in range(n_show))
        _pr(f"  [{ax:4s}] {vals}")

    # ════════════════════════════════════════════════════════════════════════
    # S4 — Non-stationarity test
    # ════════════════════════════════════════════════════════════════════════
    _rule("S4 — NON-STATIONARITY ANALYSIS (variance ratio across bins)")
    ns_rows = []
    ns_results: dict = {}

    for ax, col in zip(AXIS, COLS):
        x_t = tensors_clean[col].to(device)
        y_t = y_clean.to(device)
        ns  = nonstationarity_test(x_t, y_t, n_bins=12, device=device)
        ns_results[ax] = ns

        verdict = var_ratio_verdict(ns["var_ratio"])
        ns_rows.append([
            ax,
            f"{ns['var_ratio']:.2f}",
            f"{ns['mean_range']:.3f}",
            f"{ns['std_cv']:.3f}",
            verdict,
        ])

    _table(["Axis","Var ratio","Mean range","Std CV","Verdict"], ns_rows)
    report_lines += ["\n--- NON-STATIONARITY ---"]
    report_lines += [str(r) for r in ns_rows]

    # Per-bin std details
    _pr("\n[bold]Per-bin std(logA) detail:[/bold]")
    for ax in AXIS:
        stds = ns_results[ax]["bin_stds"]
        if not stds:
            continue
        bar = "".join("█" * max(1, int(s * 4)) for s in stds)
        _pr(f"  [{ax:4s}] {bar}  [{min(stds):.3f} → {max(stds):.3f}]")

    # ════════════════════════════════════════════════════════════════════════
    # S5 — Jump / dip detection
    # ════════════════════════════════════════════════════════════════════════
    _rule("S5 — JUMP / DIP DETECTION (consecutive sorted differences)")
    jump_rows = []
    jump_results: dict = {}

    for ax, col in zip(AXIS, COLS):
        # Physical dips: use CLEAN data (floor removed) so -87 jumps don't dominate
        x_t = tensors_clean[col].to(device)
        y_t = y_clean.to(device)
        jr  = detect_jumps(x_t, y_t, threshold=4.0, top_k=8, device=device)
        jump_results[ax] = jr

        locs_str = ", ".join(f"{v:.3f}" for v in jr["top_locs"][:5])
        jump_rows.append([
            ax,
            f"{jr['n_jumps']:,}  ({100*jr['jump_frac']:.2f}%)",
            f"{jr['max_rate']:.2f}",
            f"{jr['p99_rate']:.2f}",
            locs_str,
        ])

    _table(["Axis","#Jumps (frac)","Max |Δy/Δx|","p99 |Δy/Δx|","Top-5 jump locations"], jump_rows)
    report_lines += ["\n--- JUMP/DIP DETECTION ---"]
    report_lines += [str(r) for r in jump_rows]

    # Jump detail
    _pr("\n[bold]Top jump magnitudes per axis:[/bold]")
    for ax in AXIS:
        jr   = jump_results[ax]
        detail = "  ".join(
            f"loc={l:.3f}(Δ={m:.2f})"
            for l, m in zip(jr["top_locs"][:5], jr["top_mags"][:5])
        )
        _pr(f"  [{ax:4s}] {detail}")

    # ════════════════════════════════════════════════════════════════════════
    # S6 — Floor point distribution per axis
    # ════════════════════════════════════════════════════════════════════════
    _rule("S6 — FLOOR POINT DISTRIBUTION PER AXIS")
    _pr("[yellow]Showing where logA < -50 points cluster in each axis…[/yellow]")

    for ax, col in zip(AXIS, COLS):
        floor_vals = df.loc[floor_mask, col].values
        if len(floor_vals) == 0:
            continue
        lo, hi = floor_vals.min(), floor_vals.max()
        med    = np.median(floor_vals)
        q25f   = np.percentile(floor_vals, 25)
        q75f   = np.percentile(floor_vals, 75)
        _pr(
            f"  [{ax:4s}]  n={len(floor_vals):,}  "
            f"range=[{lo:.3f}, {hi:.3f}]  "
            f"median={med:.3f}  IQR=[{q25f:.3f},{q75f:.3f}]"
        )

    # ════════════════════════════════════════════════════════════════════════
    # S7 — Per-expert roughness analysis
    # ════════════════════════════════════════════════════════════════════════
    _rule("S7 — PER-EXPERT ROUGHNESS (within partition slabs)")
    _pr("[yellow]For each expert, analysing roughness of logA along the 3 remaining dims…[/yellow]")
    _pr("[yellow](This takes the longest — running with %d slabs per axis)[/yellow]" % args.n_slabs)

    expert_results: dict = {}
    for partition_ax in AXIS:
        _pr(f"\n  [bold]Expert: partition on {partition_ax.upper()}[/bold]")
        res = per_expert_roughness(
            df_clean, partition_ax, n_slabs=args.n_slabs,
            n_lags=args.n_lags, device=device
        )
        expert_results[partition_ax] = res

        remain_axes = [a for a in AXIS if a != partition_ax]
        for ra in remain_axes:
            alphas = [f"{a:.2f}" if not np.isnan(a) else "—" for a in res[ra]]
            mean_a = np.nanmean(res[ra]) if res[ra] else np.nan
            k_rec, _ = holder_to_kernel(mean_a)
            _pr(
                f"    {ra:4s}  α per slab: {alphas}  "
                f"→ mean α={mean_a:.3f}  → {k_rec}"
            )

    # ════════════════════════════════════════════════════════════════════════
    # S8 — KERNEL RECOMMENDATION SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    _rule("S8 — KERNEL RECOMMENDATION SUMMARY")
    _pr("[bold yellow]Reading instructions:[/bold yellow]")
    _pr("  α < 0.8  → matern12  (non-differentiable, sharp jumps)")
    _pr("  α 0.8-1.5 → matern12 or matern32")
    _pr("  α 1.5-2.5 → matern32")
    _pr("  α 2.5-4.0 → matern52")
    _pr("  α ≥ 4.0   → rbf or matern52\n")

    _pr("[bold]GLOBAL axis roughness (over all data):[/bold]")
    global_rows = []
    for ax in AXIS:
        vr   = vario_results[ax]
        ns   = ns_results[ax]
        jr   = jump_results[ax]
        k_rec = vr["kernel"]
        alpha = vr["alpha"]
        # If non-stationary AND jumpy → upgrade to Gibbs
        if ns["var_ratio"] > 3.0:
            k_rec = f"Gibbs × {k_rec}"
        global_rows.append([
            ax,
            f"{alpha:.3f}" if not np.isnan(alpha) else "nan",
            f"{ns['var_ratio']:.2f}",
            f"{jr['p99_rate']:.2f}",
            k_rec,
        ])
    _table(["Axis","α (Hölder)","Var ratio","p99 jump rate","Recommended for PRIMARY dim"],
           global_rows)

    _pr("\n[bold]PER-EXPERT KERNEL RECOMMENDATIONS:[/bold]")
    _pr("  (The primary dim is dim-0 of what the expert sees after partitioning)\n")
    expert_rec_rows = []
    for partition_ax in AXIS:
        remain_axes = [a for a in AXIS if a != partition_ax]
        rec_parts   = []
        for i, ra in enumerate(remain_axes):
            alphas = expert_results[partition_ax].get(ra, [])
            mean_a = np.nanmean(alphas) if alphas else np.nan
            k, _   = holder_to_kernel(mean_a)
            ns_val = ns_results[ra]["var_ratio"]
            if ns_val > 3.0:
                k = f"Gibbs×{k}" if "Gibbs" not in k else k
            if i == 0:
                rec_parts.append(f"DIM0({ra}):{k}")
            else:
                rec_parts.append(f"{ra}:{k}")
        expert_name  = f"{partition_ax.upper()}-expert"
        sees_dims    = "+".join(remain_axes)
        kernel_combo = "  |  ".join(rec_parts)
        expert_rec_rows.append([expert_name, sees_dims, kernel_combo])

    _table(["Expert","Sees dims","Kernel per dim"], expert_rec_rows)
    report_lines += ["\n--- KERNEL RECOMMENDATIONS ---"]
    report_lines += [str(r) for r in global_rows]
    report_lines += [str(r) for r in expert_rec_rows]

    # ════════════════════════════════════════════════════════════════════════
    # S9 — gp4d_lab composite kernel string suggestions
    # ════════════════════════════════════════════════════════════════════════
    _rule("S9 — gp4d_lab COMPOSITE KERNEL CONFIG STRINGS")
    _pr("  Copy-paste these into your must_try_config.py:\n")

    for partition_ax in AXIS:
        remain_axes = [a for a in AXIS if a != partition_ax]
        alphas_dim0 = expert_results[partition_ax].get(remain_axes[0], [])
        alpha0 = np.nanmean(alphas_dim0) if alphas_dim0 else np.nan
        k0, _  = holder_to_kernel(alpha0)

        ns0    = ns_results[remain_axes[0]]["var_ratio"]
        if ns0 > 3.0 and "Gibbs" not in k0:
            k0 = f"gibbs_sigmoid"

        # secondary dims → use rougher of the two remaining
        alphas_sec = []
        for ra in remain_axes[1:]:
            aa = expert_results[partition_ax].get(ra, [])
            alphas_sec.append(np.nanmean(aa) if aa else np.nan)
        alpha_sec = np.nanmin(alphas_sec) if alphas_sec else np.nan
        k_sec, _ = holder_to_kernel(alpha_sec)

        # Build composite string
        if k0 == k_sec:
            composite = k0.split("_or_")[0]  # just one kernel
        else:
            k0_base  = k0.split("_or_")[0].replace(" ", "").lower()
            k_sec_b  = k_sec.split("_or_")[0].replace(" ", "").lower()
            composite = f"{k0_base}_x_{k_sec_b}"

        _pr(f'  cfg.kernels["{partition_ax}"].composite_kernel = "{composite}"')

    # ════════════════════════════════════════════════════════════════════════
    # Save report
    # ════════════════════════════════════════════════════════════════════════
    out_path = args.out or str(Path(args.csv).parent / "kernel_diagnosis_report.txt")
    if RICH:
        plain = console.export_text()
        Path(out_path).write_text(plain)
    else:
        Path(out_path).write_text("\n".join(report_lines))
    print(f"\n[Report saved → {out_path}]")


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()