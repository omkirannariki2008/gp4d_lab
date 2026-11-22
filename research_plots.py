#!/usr/bin/env python3
"""
research_plots.py
=================
Research-grade diagnostic report for GP diffractive cross-section predictions.

Generates a multi-page PDF with two sections:
  Section A — Model Strengths (flex the best results)
  Section B — Areas for Improvement (honest diagnostics for future work)

Usage
-----
    python research_plots.py \
        --csv50  predictions_50ep.csv  \
        --csv300 predictions_300ep.csv \
        --csvmoe predictions_moe.csv   \
        --output diffraction_gp_report.pdf \
        [--sample 50000]   # rows to sample for scatter plots (default 40000)

Required columns in every CSV
------------------------------
    Q2_log_center, W2_log_center, beta_center, z_center,
    logA, pred_logA, pred_std_logA,
    pred_A_exp, pred_A_exp_low, pred_A_exp_high,
    pred_A_10, pred_A_10_low, pred_A_10_high,
    A_true_exp, A_true_10
"""

import argparse
import warnings
import textwrap
from pathlib import Path

import numpy             as np
import pandas            as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker   as ticker
from   matplotlib.backends.backend_pdf import PdfPages
from   matplotlib.colors               import LogNorm, Normalize
from   matplotlib.cm                   import ScalarMappable
import matplotlib.patches              as mpatches
from   scipy                           import stats
from   scipy.stats                     import gaussian_kde

warnings.filterwarnings("ignore")

# ─── Visual identity ──────────────────────────────────────────────────────────
FLOOR_THRESH  = -17.955
MODELS        = ("50ep", "300ep", "MoE")
M_LABEL       = {"50ep": "Q² M5/2 — 50 ep", "300ep": "Q² M5/2 — 300 ep", "MoE": "MoE KNN"}
M_COLOR       = {"50ep": "#90CAF9", "300ep": "#1565C0", "MoE": "#B71C1C"}
M_MARKER      = {"50ep": "s", "300ep": "D", "MoE": "o"}
PHYS_COLOR    = "#2E7D32"
FLOOR_COLOR   = "#E65100"
PERFECT_KW    = dict(color="#616161", ls="--", lw=1.2, zorder=0, label="Perfect")
ALPHA_SCATTER = 0.25
DPI           = 150

FEAT_LABEL = {
    "Q2_log_center": r"$\log Q^2$",
    "W2_log_center": r"$\log W^2$",
    "beta_center":   r"$\beta$",
    "z_center":      r"$z$",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "legend.fontsize":    8,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         DPI,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
})


# ─── Data loading & metrics ──────────────────────────────────────────────────

def load_data(csv_50: str, csv_300: str, csv_moe: str, sample: int) -> dict:
    """Load and validate the three prediction CSVs."""
    req_cols = [
        "Q2_log_center","W2_log_center","beta_center","z_center",
        "logA","pred_logA","pred_std_logA",
        "pred_A_exp","pred_A_exp_low","pred_A_exp_high",
        "pred_A_10","pred_A_10_low","pred_A_10_high",
        "A_true_exp","A_true_10",
    ]
    data = {}
    for key, path in zip(MODELS, [csv_50, csv_300, csv_moe]):
        print(f"  Loading {key}: {path} ...")
        df = pd.read_csv(path)
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{key}] Missing columns: {missing}")
        df["residual"]   = df["logA"] - df["pred_logA"]
        df["z_score"]    = df["residual"] / df["pred_std_logA"].clip(lower=1e-9)
        df["abs_error"]  = df["residual"].abs()
        df["is_floor"]   = df["logA"] < FLOOR_THRESH
        df["is_physical"]= ~df["is_floor"]
        # ── sample for scatter plots ──────────────────────────────────────────
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df), min(sample, len(df)), replace=False)
        df["_scatter"] = False
        df.loc[idx, "_scatter"] = True
        data[key] = df
        n_floor = df["is_floor"].sum()
        print(f"    {len(df):,} rows | floor {n_floor:,} ({100*n_floor/len(df):.1f}%)")
    return data


def compute_metrics(df: pd.DataFrame, mask=None) -> dict:
    d = df if mask is None else df[mask]
    res = d["residual"].values
    mae  = np.mean(np.abs(res))
    rmse = np.sqrt(np.mean(res**2))
    # NLPD
    sigma = d["pred_std_logA"].clip(lower=1e-9).values
    nlpd  = np.mean(0.5*np.log(2*np.pi*sigma**2) + (res**2)/(2*sigma**2))
    # coverage
    cov90 = np.mean(np.abs(d["z_score"]) < 1.6449)
    cov68 = np.mean(np.abs(d["z_score"]) < 1.0)
    r2    = 1 - np.var(res)/(np.var(d["logA"].values) + 1e-12)
    return dict(mae=mae, rmse=rmse, nlpd=nlpd, cov90=cov90, cov68=cov68, r2=r2,
                n=len(d))


def all_metrics(data: dict) -> pd.DataFrame:
    rows = []
    for key, df in data.items():
        m_all  = compute_metrics(df)
        m_phys = compute_metrics(df, df["is_physical"])
        m_fl   = compute_metrics(df, df["is_floor"])
        fl_det = (df.loc[df["is_floor"], "pred_logA"] < FLOOR_THRESH).mean()
        rows.append({
            "Model": M_LABEL[key],
            "Phys MAE":    m_phys["mae"],  "Phys RMSE":    m_phys["rmse"],
            "Phys NLPD":   m_phys["nlpd"], "Phys R²":      m_phys["r2"],
            "Cov 90%":     m_phys["cov90"],"Cov 68%":      m_phys["cov68"],
            "Floor MAE":   m_fl["mae"],    "Floor NLPD":   m_fl["nlpd"],
            "Floor Detect":fl_det,
            "All MAE":     m_all["mae"],   "All NLPD":     m_all["nlpd"],
        })
    return pd.DataFrame(rows).set_index("Model")


# ─── PDF helpers ─────────────────────────────────────────────────────────────

def _savefig(pdf: PdfPages, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _section_banner(pdf: PdfPages, title: str, subtitle: str, color: str):
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.set_facecolor(color)
    fig.patch.set_facecolor(color)
    ax.text(0.5, 0.60, title,   transform=ax.transAxes,
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white")
    ax.text(0.5, 0.28, subtitle, transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="white", alpha=0.9,
            wrap=True)
    ax.axis("off")
    _savefig(pdf, fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  COVER PAGE — metrics table
# ═══════════════════════════════════════════════════════════════════════════════

def page_cover(pdf: PdfPages, data: dict):
    met = all_metrics(data)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#0D1B2A")

    # Title block
    fig.text(0.5, 0.93, "Diffractive DIS Cross-Section  |  GP Prediction Diagnostic Report",
             ha="center", fontsize=16, fontweight="bold", color="white")
    fig.text(0.5, 0.88,
             r"Target: $\log\sigma^{D(3)}(Q^2,W^2,\beta,z)$   "
             r"·  Dataset: 1.08M grid points  ·  Train fraction: 10%",
             ha="center", fontsize=10, color="#90CAF9")

    ax = fig.add_axes([0.05, 0.12, 0.90, 0.70])
    ax.axis("off")
    ax.set_facecolor("#0D1B2A")

    cols = list(met.columns)
    rows = list(met.index)
    cell_text = []
    for idx in rows:
        row = []
        for c in cols:
            v = met.loc[idx, c]
            if "R²" in c:        row.append(f"{v:.4f}")
            elif "Detect" in c:  row.append(f"{v:.1%}")
            elif "Cov" in c:     row.append(f"{v:.1%}")
            else:                 row.append(f"{v:.4f}")
        cell_text.append(row)

    # Color cells by rank (green = best, red = worst per column)
    cell_colors = []
    for r_idx, row in enumerate(rows):
        r_colors = []
        for c in cols:
            vals = [met.loc[m, c] for m in rows]
            low_good = any(x in c for x in ["MAE","RMSE","NLPD"])
            rank = sorted(vals, reverse=not low_good).index(met.loc[row, c])
            alphas = [0.6, 0.3, 0.1]
            color_families = ["#1B5E20","#E65100","#B71C1C"]
            r_colors.append(color_families[rank] if rank < 3 else "#263238")
        cell_colors.append(r_colors)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=rows, colLabels=cols,
        cellColours=cell_colors,
        rowColours=["#1A237E"]*3,
        colColours=["#0D47A1"]*len(cols),
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(color="white")
        cell.set_edgecolor("#37474F")

    fig.text(0.5, 0.07,
             "Green = best · Orange = second · Red = worst per metric  "
             "·  Floor threshold: logA < −17.955",
             ha="center", fontsize=8, color="#78909C")
    _savefig(pdf, fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION A  —  GREATNESS
# ═══════════════════════════════════════════════════════════════════════════════

def page_a1_pred_vs_true(pdf: PdfPages, data: dict):
    """A1 — Predicted vs True in logA space (hexbin, all 3 models)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle("A1 · Predicted vs True Cross-Section  (logA scale)",
                 fontsize=13, fontweight="bold", y=1.01)

    for col, key in enumerate(MODELS):
        df    = data[key]
        sc    = df[df["_scatter"]]
        phys  = sc[sc["is_physical"]]
        floor = sc[sc["is_floor"]]

        for row, (subset, title_sfx) in enumerate(
                [(sc, "All points"), (phys, "Physical regime only")]):
            ax = axes[row, col]
            lo = min(subset["logA"].min(), subset["pred_logA"].min())
            hi = max(subset["logA"].max(), subset["pred_logA"].max())
            pad = (hi - lo) * 0.03

            hb = ax.hexbin(subset["logA"], subset["pred_logA"],
                           gridsize=60, mincnt=1,
                           cmap="Blues" if key != "MoE" else "Reds",
                           norm=LogNorm())
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], **PERFECT_KW)
            plt.colorbar(hb, ax=ax, shrink=0.8, label="count")

            m = compute_metrics(data[key], data[key]["is_physical"] if row else None)
            ax.set_title(f"{M_LABEL[key]}\n{title_sfx}", fontsize=9)
            ax.set_xlabel(r"True $\log A$", fontsize=8)
            ax.set_ylabel(r"Predicted $\log A$", fontsize=8)
            ax.annotate(f"MAE={m['mae']:.3f}\nRMSE={m['rmse']:.3f}\nR²={m['r2']:.4f}",
                        xy=(0.04, 0.92), xycoords="axes fraction",
                        va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)

    plt.tight_layout()
    _savefig(pdf, fig)


def page_a2_crosssection_scale(pdf: PdfPages, data: dict):
    """A2 — Predictions in A_exp and A_10 scales (physical regime)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle(r"A2 · Predicted vs True Cross-Section  "
                 r"($A_{\exp}=e^{\log A}$ and $A_{10}=10^{\log A/\ln10}$ scales, physical regime)",
                 fontsize=11, fontweight="bold", y=1.01)

    for col, key in enumerate(MODELS):
        df   = data[key]
        phys = df[df["is_physical"] & df["_scatter"]]

        for row, (y_true, y_pred, ylabel) in enumerate([
            ("A_true_exp", "pred_A_exp", r"$A_{\exp}$  (natural exp scale)"),
            ("A_true_10",  "pred_A_10",  r"$A_{10}$  (base-10 scale)"),
        ]):
            ax = axes[row, col]
            yv = phys[y_true].clip(lower=1e-30)
            yp = phys[y_pred].clip(lower=1e-30)
            lo = min(yv.min(), yp.min()); hi = max(yv.max(), yp.max())

            ax.scatter(yv, yp, s=3, alpha=0.15, color=M_COLOR[key],
                       rasterized=True)
            ax.plot([lo, hi], [lo, hi], **PERFECT_KW)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(f"True {ylabel}", fontsize=8)
            ax.set_ylabel(f"Predicted {ylabel}", fontsize=8)
            ax.set_title(f"{M_LABEL[key]}", fontsize=9)

            corr, _ = np.corrcoef(np.log(yv+1e-30), np.log(yp+1e-30))[0,1], None
            ax.annotate(f"log-corr={corr:.4f}",
                        xy=(0.04, 0.94), xycoords="axes fraction",
                        fontsize=7.5, va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    _savefig(pdf, fig)


def page_a3_residuals(pdf: PdfPages, data: dict):
    """A3 — Residual distributions and error comparison."""
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("A3 · Residual Analysis", fontsize=13, fontweight="bold")

    # Row 0, Col 0-2: KDE of residuals, per model, physical only
    ax_kde = fig.add_subplot(gs[0, :])
    for key in MODELS:
        phys_res = data[key].loc[data[key]["is_physical"], "residual"]
        xs = np.linspace(-10, 10, 400)
        kde = gaussian_kde(phys_res.sample(min(50000, len(phys_res)), random_state=0))
        ax_kde.plot(xs, kde(xs), color=M_COLOR[key], lw=2, label=M_LABEL[key])
    ax_kde.axvline(0, color="#616161", ls="--", lw=1.2)
    ax_kde.set_xlabel(r"Residual: $\log A - \widehat{\log A}$")
    ax_kde.set_ylabel("Density (physical regime)")
    ax_kde.set_title("Residual Distributions — Physical Regime")
    ax_kde.legend()
    ax_kde.set_xlim(-12, 12)

    # Row 1: per-model box-whisker across Q2 bins
    ax_box = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for i, key in enumerate(MODELS):
        df_p = data[key][data[key]["is_physical"]]
        q2_vals = sorted(df_p["Q2_log_center"].unique())
        positions = list(range(len(q2_vals)))
        bp_data   = [df_p.loc[df_p["Q2_log_center"]==q, "residual"].values
                     for q in q2_vals]
        bp = ax_box[i].boxplot(bp_data, positions=positions,
                                patch_artist=True, showfliers=False,
                                whiskerprops=dict(color=M_COLOR[key]),
                                boxprops=dict(facecolor=M_COLOR[key], alpha=0.5))
        ax_box[i].axhline(0, color="#616161", ls="--", lw=1)
        ax_box[i].set_xticks(positions[::5])
        ax_box[i].set_xticklabels([f"{q2_vals[j]:.1f}" for j in range(0,len(q2_vals),5)],
                                    rotation=45, fontsize=7)
        ax_box[i].set_title(f"{M_LABEL[key]}\nResidual per Q² bin", fontsize=9)
        ax_box[i].set_xlabel(r"$\log Q^2$"); ax_box[i].set_ylabel("Residual")
        ax_box[i].set_ylim(-15, 15)

    _savefig(pdf, fig)


def page_a4_calibration(pdf: PdfPages, data: dict):
    """A4 — Uncertainty calibration: coverage curves + z-score distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("A4 · Uncertainty Calibration — Physical Regime",
                 fontsize=13, fontweight="bold")

    # Left: calibration curve
    ax = axes[0]
    levels = np.linspace(0.01, 0.99, 80)
    ax.plot([0, 1], [0, 1], **PERFECT_KW)
    for key in MODELS:
        phys = data[key][data[key]["is_physical"]]
        z    = phys["z_score"].values
        emp  = [(np.abs(z) < stats.norm.ppf((1+lv)/2)).mean() for lv in levels]
        ax.plot(levels, emp, color=M_COLOR[key], lw=2, label=M_LABEL[key])
    ax.fill_between([0,1],[0,1],[0,1], alpha=0)
    ax.set_xlabel("Nominal Coverage"); ax.set_ylabel("Empirical Coverage")
    ax.set_title("Calibration Curve")
    ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.fill_between([0,1],[0,1],[1,1], alpha=0.05, color="gray",
                    label="Over-conservative")
    ax.fill_between([0,1],[0,0],[0,1], alpha=0.05, color="red",
                    label="Over-confident")

    # Middle: z-score distribution
    ax2 = axes[1]
    xs  = np.linspace(-5, 5, 200)
    ax2.plot(xs, stats.norm.pdf(xs), color="#616161", ls="--", lw=2,
             label=r"$\mathcal{N}(0,1)$ ideal")
    for key in MODELS:
        phys = data[key][data[key]["is_physical"]]
        z    = phys["z_score"].clip(-8, 8).sample(min(50000,len(phys)), random_state=0)
        kde  = gaussian_kde(z)
        ax2.plot(xs, kde(xs), color=M_COLOR[key], lw=2, label=M_LABEL[key])
    ax2.set_xlabel(r"$z$-score: $(y - \hat{y}) / \hat{\sigma}$")
    ax2.set_ylabel("Density")
    ax2.set_title(r"$z$-Score Distribution (ideal: $\mathcal{N}(0,1)$)")
    ax2.legend(); ax2.set_xlim(-5, 5)

    # Right: coverage bars at specific levels
    ax3 = axes[2]
    nom_levels = [0.50, 0.68, 0.90, 0.95]
    x_pos = np.arange(len(nom_levels))
    width = 0.25
    for i, key in enumerate(MODELS):
        phys = data[key][data[key]["is_physical"]]
        z    = phys["z_score"].values
        emp  = [(np.abs(z) < stats.norm.ppf((1+lv)/2)).mean() for lv in nom_levels]
        ax3.bar(x_pos + (i - 1)*width, emp, width=width*0.9,
                color=M_COLOR[key], label=M_LABEL[key], alpha=0.85)
    ax3.plot([-0.4, len(nom_levels)-0.6], [1, 1], "k--", lw=0.7)
    for j, lv in enumerate(nom_levels):
        ax3.axhline(lv, color="gray", ls=":", lw=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{int(lv*100)}%" for lv in nom_levels])
    ax3.set_ylabel("Empirical Coverage"); ax3.set_title("Coverage at Key Levels")
    ax3.legend(); ax3.set_ylim(0, 1.1)

    plt.tight_layout()
    _savefig(pdf, fig)


def page_a5_per_bin_accuracy(pdf: PdfPages, data: dict):
    """A5 — MAE per Q2 bin and per beta bin (physical regime)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("A5 · Per-Bin Accuracy — Physical Regime",
                 fontsize=13, fontweight="bold")

    feat_axes = [
        ("Q2_log_center", axes[0, 0], r"$\log Q^2$",  "MAE"),
        ("W2_log_center", axes[0, 1], r"$\log W^2$",  "MAE"),
        ("beta_center",   axes[1, 0], r"$\beta$",      "MAE"),
        ("z_center",      axes[1, 1], r"$z$",          "MAE"),
    ]

    for feat, ax, xlabel, ylabel in feat_axes:
        for key in MODELS:
            df_p  = data[key][data[key]["is_physical"]]
            bins  = sorted(df_p[feat].unique())
            maes  = [df_p.loc[df_p[feat]==b, "abs_error"].mean() for b in bins]
            ax.plot(bins, maes, color=M_COLOR[key], lw=1.8, marker=M_MARKER[key],
                    ms=4, label=M_LABEL[key])
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"MAE per {xlabel} bin")
        ax.legend(fontsize=7)

    plt.tight_layout()
    _savefig(pdf, fig)


def page_a6_phase_space_heatmap(pdf: PdfPages, data: dict):
    """A6 — 2D (Q2, beta) performance heatmap for each model."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle(r"A6 · Phase Space Performance  $(Q^2, \beta)$ — Physical Regime",
                 fontsize=13, fontweight="bold")

    q2_vals   = sorted(data[MODELS[0]]["Q2_log_center"].unique())
    beta_vals = sorted(data[MODELS[0]]["beta_center"].unique())

    for col, key in enumerate(MODELS):
        df_p = data[key][data[key]["is_physical"]]

        # MAE heatmap
        mae_grid = np.full((len(q2_vals), len(beta_vals)), np.nan)
        fl_grid  = np.full((len(q2_vals), len(beta_vals)), np.nan)
        q2_idx   = {v: i for i, v in enumerate(q2_vals)}
        b_idx    = {v: i for i, v in enumerate(beta_vals)}

        for (q2, beta), grp in df_p.groupby(["Q2_log_center", "beta_center"]):
            i, j = q2_idx[q2], b_idx[beta]
            mae_grid[i, j] = grp["abs_error"].mean()

        for (q2, beta), grp in data[key].groupby(["Q2_log_center", "beta_center"]):
            if q2 in q2_idx and beta in b_idx:
                i, j = q2_idx[q2], b_idx[beta]
                fl_grp = grp[data[key]["is_floor"].loc[grp.index]]
                fl_grid[i, j] = 1 - (fl_grp["pred_logA"] < FLOOR_THRESH).mean() if len(fl_grp)>0 else np.nan

        # Row 0: MAE
        im0 = axes[0, col].imshow(mae_grid, aspect="auto", origin="lower",
                                   cmap="RdYlGn_r", vmin=0, vmax=3)
        axes[0, col].set_xticks(range(0, len(beta_vals), 5))
        axes[0, col].set_xticklabels([f"{beta_vals[i]:.2f}"
                                       for i in range(0, len(beta_vals), 5)], rotation=45, fontsize=6)
        axes[0, col].set_yticks(range(0, len(q2_vals), 5))
        axes[0, col].set_yticklabels([f"{q2_vals[i]:.1f}"
                                       for i in range(0, len(q2_vals), 5)], fontsize=6)
        axes[0, col].set_xlabel(r"$\beta$"); axes[0, col].set_ylabel(r"$\log Q^2$")
        axes[0, col].set_title(f"{M_LABEL[key]}\nMAE (physical)")
        plt.colorbar(im0, ax=axes[0, col], shrink=0.9)

        # Row 1: floor detection miss rate
        im1 = axes[1, col].imshow(fl_grid, aspect="auto", origin="lower",
                                   cmap="RdYlGn", vmin=0, vmax=1)
        axes[1, col].set_xticks(range(0, len(beta_vals), 5))
        axes[1, col].set_xticklabels([f"{beta_vals[i]:.2f}"
                                       for i in range(0, len(beta_vals), 5)], rotation=45, fontsize=6)
        axes[1, col].set_yticks(range(0, len(q2_vals), 5))
        axes[1, col].set_yticklabels([f"{q2_vals[i]:.1f}"
                                       for i in range(0, len(q2_vals), 5)], fontsize=6)
        axes[1, col].set_xlabel(r"$\beta$"); axes[1, col].set_ylabel(r"$\log Q^2$")
        axes[1, col].set_title(f"{M_LABEL[key]}\nFloor miss rate (green=good)")
        plt.colorbar(im1, ax=axes[1, col], shrink=0.9)

    plt.tight_layout()
    _savefig(pdf, fig)


def page_a7_floor_detection(pdf: PdfPages, data: dict):
    """A7 — Floor detection deep-dive: the zero cross-section boundary."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle("A7 · Floor / Near-Zero Cross-Section Detection",
                 fontsize=13, fontweight="bold")

    for col, key in enumerate(MODELS):
        df_fl = data[key][data[key]["is_floor"] & data[key]["_scatter"]]

        # Row 0: pred vs true for floor points
        ax = axes[0, col]
        ymin = df_fl["logA"].min(); ymax = df_fl["logA"].max()
        pmin = df_fl["pred_logA"].min(); pmax = df_fl["pred_logA"].max()
        lo = min(ymin, pmin); hi = max(ymax, pmax)
        ax.scatter(df_fl["logA"], df_fl["pred_logA"],
                   s=3, alpha=0.3, color=FLOOR_COLOR, rasterized=True)
        ax.plot([lo, hi], [lo, hi], **PERFECT_KW)
        ax.axhline(FLOOR_THRESH, color="navy", ls=":", lw=1.2,
                   label=f"floor thresh {FLOOR_THRESH:.1f}")
        ax.axvline(FLOOR_THRESH, color="navy", ls=":", lw=1.2)
        m_fl = compute_metrics(data[key], data[key]["is_floor"])
        fl_det = (data[key].loc[data[key]["is_floor"], "pred_logA"] < FLOOR_THRESH).mean()
        ax.set_title(f"{M_LABEL[key]}\nFloor: pred vs true")
        ax.set_xlabel(r"True $\log A$"); ax.set_ylabel(r"Predicted $\log A$")
        ax.annotate(f"MAE={m_fl['mae']:.2f}\nDetected={fl_det:.1%}",
                    xy=(0.04, 0.94), xycoords="axes fraction",
                    va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.legend(fontsize=7)

        # Row 1: floor error histogram
        ax2 = axes[1, col]
        ax2.hist(df_fl["residual"], bins=60, color=FLOOR_COLOR, alpha=0.7,
                 density=True, label="floor residuals")
        ax2.axvline(0, color="k", ls="--", lw=1.2)
        ax2.axvline(df_fl["residual"].mean(), color="red", ls="-", lw=1.5,
                    label=f"mean={df_fl['residual'].mean():.1f}")
        ax2.set_xlabel(r"Residual ($\log A - \widehat{\log A}$)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{M_LABEL[key]}\nFloor residual distribution")
        ax2.legend(fontsize=7)

    plt.tight_layout()
    _savefig(pdf, fig)


def page_a8_model_progression(pdf: PdfPages, data: dict):
    """A8 — Model improvement progression: 50ep → 300ep → MoE."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("A8 · Model Improvement Progression — Physical Regime MAE by Q² Bin",
                 fontsize=12, fontweight="bold")

    q2_vals = sorted(data[MODELS[0]]["Q2_log_center"].unique())

    all_maes = {}
    for key in MODELS:
        df_p = data[key][data[key]["is_physical"]]
        all_maes[key] = [df_p.loc[df_p["Q2_log_center"]==q,"abs_error"].mean()
                         for q in q2_vals]

    # Absolute MAE per model
    ax = axes[0]
    for key in MODELS:
        ax.plot(q2_vals, all_maes[key], color=M_COLOR[key], lw=2,
                marker=M_MARKER[key], ms=4, label=M_LABEL[key])
    ax.set_xlabel(r"$\log Q^2$"); ax.set_ylabel("MAE")
    ax.set_title("Physical MAE per Q² bin")
    ax.legend()

    # Relative improvement 50ep → 300ep
    ax2 = axes[1]
    imp_300 = [(all_maes["50ep"][i] - all_maes["300ep"][i]) / (all_maes["50ep"][i]+1e-9)
               for i in range(len(q2_vals))]
    ax2.bar(q2_vals, imp_300, width=0.28, color="#1565C0", alpha=0.8,
            label="300ep vs 50ep")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel(r"$\log Q^2$"); ax2.set_ylabel("Relative MAE reduction")
    ax2.set_title("Improvement: 300ep over 50ep\n(positive = better)")
    ax2.legend()

    # Relative improvement 300ep → MoE
    ax3 = axes[2]
    imp_moe = [(all_maes["300ep"][i] - all_maes["MoE"][i]) / (all_maes["300ep"][i]+1e-9)
               for i in range(len(q2_vals))]
    colors_moe = [PHYS_COLOR if v >= 0 else FLOOR_COLOR for v in imp_moe]
    ax3.bar(q2_vals, imp_moe, width=0.28, color=colors_moe, alpha=0.8)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set_xlabel(r"$\log Q^2$"); ax3.set_ylabel("Relative MAE reduction")
    ax3.set_title("Improvement: MoE over 300ep\n(green=MoE better, red=300ep better)")

    plt.tight_layout()
    _savefig(pdf, fig)


def page_a9_summation_projections(pdf: PdfPages, data: dict):
    """A9 — Projection plots: Sum of logA projected on each kinematic axis."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(r"A9 · Projection Profile: Summation of $\log A$ on Kinematic Axes",
                 fontsize=13, fontweight="bold", y=0.98)

    subplot_mapping = [
        ("Q2_log_center", axes[0, 0]),
        ("W2_log_center", axes[0, 1]),
        ("beta_center",   axes[1, 0]),
        ("z_center",      axes[1, 1]),
    ]

    base_df = data[MODELS[0]]

    for feat, ax in subplot_mapping:
        # Group true logA
        true_grouped = base_df.groupby(feat)["logA"].sum()
        ax.plot(true_grouped.index, true_grouped.values, color="#212121", lw=2.2, label="True", zorder=1)

        # Group predicted logA for each model
        for key in MODELS:
            pred_grouped = data[key].groupby(feat)["pred_logA"].sum()
            ax.plot(pred_grouped.index, pred_grouped.values, color=M_COLOR[key], lw=1.5,
                    marker=M_MARKER[key], ms=4, alpha=0.9, label=M_LABEL[key], zorder=2)

        ax.set_xlabel(FEAT_LABEL[feat])
        ax.set_ylabel(r"$\sum \log A$")
        ax.set_title(f"Summation projection along {FEAT_LABEL[feat]}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    _savefig(pdf, fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION B  —  AREAS FOR IMPROVEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def page_b1_error_vs_inputs(pdf: PdfPages, data: dict):
    """B1 — Residuals vs each input variable (physical regime)."""
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(len(MODELS), 4, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle("B1 · Residuals vs Input Variables — Physical Regime",
                 fontsize=13, fontweight="bold")

    feats = ["Q2_log_center", "W2_log_center", "beta_center", "z_center"]
    for row, key in enumerate(MODELS):
        df_p = data[key][data[key]["is_physical"] & data[key]["_scatter"]]
        for col, feat in enumerate(feats):
            ax = fig.add_subplot(gs[row, col])
            ax.scatter(df_p[feat], df_p["residual"], s=2, alpha=0.15,
                       color=M_COLOR[key], rasterized=True)

            # Running median
            bins_q = pd.cut(df_p[feat], bins=20)
            grp    = df_p.groupby(bins_q)["residual"]
            mids   = [i.mid for i in grp.mean().index]
            ax.plot(mids, grp.median().values, color="red", lw=2, label="median")
            ax.plot(mids, grp.quantile(0.1).values, color="orange", lw=1,
                    ls="--", label="p10/p90")
            ax.plot(mids, grp.quantile(0.9).values, color="orange", lw=1, ls="--")
            ax.axhline(0, color="k", ls="--", lw=0.8)
            ax.set_xlabel(FEAT_LABEL[feat], fontsize=8)
            ax.set_ylabel("Residual" if col == 0 else "")
            if row == 0:
                ax.set_title(FEAT_LABEL[feat], fontsize=9)
            if col == 0:
                ax.annotate(M_LABEL[key], xy=(-0.35, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", fontsize=8, fontweight="bold")
            ax.set_ylim(-20, 20)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)

    _savefig(pdf, fig)


def page_b2_overconfidence_map(pdf: PdfPages, data: dict):
    """B2 — Overconfidence / underconfidence by kinematic region."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle(r"B2 · Calibration Failure Map  $(Q^2, \beta)$ — physical regime",
                 fontsize=12, fontweight="bold")

    q2_vals   = sorted(data[MODELS[0]]["Q2_log_center"].unique())
    beta_vals = sorted(data[MODELS[0]]["beta_center"].unique())

    for col, key in enumerate(MODELS):
        df_p = data[key][data[key]["is_physical"]]
        q2_idx = {v: i for i, v in enumerate(q2_vals)}
        b_idx  = {v: i for i, v in enumerate(beta_vals)}

        cov90_grid = np.full((len(q2_vals), len(beta_vals)), np.nan)
        std_grid   = np.full((len(q2_vals), len(beta_vals)), np.nan)

        for (q2, beta), grp in df_p.groupby(["Q2_log_center", "beta_center"]):
            if q2 not in q2_idx or beta not in b_idx:
                continue
            i, j = q2_idx[q2], b_idx[beta]
            z = grp["z_score"].values
            cov90_grid[i, j] = (np.abs(z) < 1.6449).mean()
            std_grid[i, j]   = grp["pred_std_logA"].mean()

        # Row 0: coverage deviation from 0.90
        dev_grid = cov90_grid - 0.90
        im0 = axes[0, col].imshow(dev_grid, aspect="auto", origin="lower",
                                   cmap="RdBu", vmin=-0.4, vmax=0.4)
        axes[0, col].set_title(f"{M_LABEL[key]}\n90% cov. deviation (0 = ideal)")
        axes[0, col].set_xlabel(r"$\beta$ index"); axes[0, col].set_ylabel(r"$\log Q^2$ index")
        plt.colorbar(im0, ax=axes[0, col], shrink=0.9)

        # Row 1: mean sigma
        im1 = axes[1, col].imshow(std_grid, aspect="auto", origin="lower",
                                   cmap="viridis")
        axes[1, col].set_title(f"{M_LABEL[key]}\nMean predictive σ")
        axes[1, col].set_xlabel(r"$\beta$ index"); axes[1, col].set_ylabel(r"$\log Q^2$ index")
        plt.colorbar(im1, ax=axes[1, col], shrink=0.9)

    plt.tight_layout()
    _savefig(pdf, fig)


def page_b3_high_error_regions(pdf: PdfPages, data: dict):
    """B3 — Where are the largest errors? Tail analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.suptitle("B3 · High-Error Region Analysis — Physical Regime",
                 fontsize=12, fontweight="bold")

    for col, key in enumerate(MODELS):
        df_p = data[key][data[key]["is_physical"]]

        # Row 0: 99th percentile errors in (Q2, beta) space
        q2_vals   = sorted(df_p["Q2_log_center"].unique())
        beta_vals = sorted(df_p["beta_center"].unique())
        q2_idx    = {v: i for i, v in enumerate(q2_vals)}
        b_idx     = {v: i for i, v in enumerate(beta_vals)}

        p99_grid = np.full((len(q2_vals), len(beta_vals)), np.nan)
        for (q2, beta), grp in df_p.groupby(["Q2_log_center","beta_center"]):
            if q2 in q2_idx and beta in b_idx:
                p99_grid[q2_idx[q2], b_idx[beta]] = grp["abs_error"].quantile(0.99)

        im0 = axes[0, col].imshow(p99_grid, aspect="auto", origin="lower",
                                   cmap="hot_r", vmin=0)
        axes[0, col].set_title(f"{M_LABEL[key]}\np99 absolute error")
        axes[0, col].set_xlabel(r"$\beta$ index"); axes[0, col].set_ylabel(r"$Q^2$ index")
        plt.colorbar(im0, ax=axes[0, col], shrink=0.9)

        # Row 1: scatter of error vs pred_std — Ideally points cluster on diagonal
        ax2 = axes[1, col]
        sc  = df_p[df_p["_scatter"]]
        ax2.scatter(sc["pred_std_logA"], sc["abs_error"], s=2, alpha=0.15,
                    color=M_COLOR[key], rasterized=True)
        xmax = min(sc["pred_std_logA"].quantile(0.99), 15)
        ax2.plot([0, xmax], [0, xmax], **PERFECT_KW)
        ax2.set_xlim(0, xmax); ax2.set_ylim(0, xmax)
        ax2.set_xlabel(r"Predicted $\hat{\sigma}$")
        ax2.set_ylabel(r"|Residual|")
        ax2.set_title(f"{M_LABEL[key]}\n|error| vs σ  (ideal: diagonal)")
        corr_val = np.corrcoef(sc["pred_std_logA"].clip(0, 20),
                                sc["abs_error"].clip(0, 20))[0, 1]
        ax2.annotate(f"r={corr_val:.3f}", xy=(0.70, 0.10), xycoords="axes fraction",
                     fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    _savefig(pdf, fig)


def page_b4_floor_vs_physical_errors(pdf: PdfPages, data: dict):
    """B4 — Side-by-side: how floor and physical regime errors compare."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("B4 · Physical vs Floor Error Summary by Model",
                 fontsize=12, fontweight="bold")

    met = all_metrics(data)
    for col, key in enumerate(MODELS):
        ax = axes[col]
        m_p = compute_metrics(data[key], data[key]["is_physical"])
        m_f = compute_metrics(data[key], data[key]["is_floor"])
        categories = ["MAE", "RMSE", "NLPD"]
        phys_vals  = [m_p["mae"], m_p["rmse"], m_p["nlpd"]]
        floor_vals = [m_f["mae"], m_f["rmse"], m_f["nlpd"]]
        x = np.arange(len(categories))
        ax.bar(x - 0.2, phys_vals,  0.38, color=PHYS_COLOR,  label="Physical", alpha=0.85)
        ax.bar(x + 0.2, floor_vals, 0.38, color=FLOOR_COLOR, label="Floor",    alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(categories)
        ax.set_yscale("log")
        ax.set_title(f"{M_LABEL[key]}")
        ax.set_ylabel("Metric value (log scale)")
        ax.legend()
        for i, (pv, fv) in enumerate(zip(phys_vals, floor_vals)):
            ax.text(i - 0.2, pv*1.1, f"{pv:.2f}", ha="center", fontsize=7, color=PHYS_COLOR)
            ax.text(i + 0.2, fv*1.1, f"{fv:.2f}", ha="center", fontsize=7, color=FLOOR_COLOR)

    plt.tight_layout()
    _savefig(pdf, fig)


def page_b5_where_models_differ(pdf: PdfPages, data: dict):
    """B5 — Where does each model win and lose vs the others?"""
    if len(data) < 3:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("B5 · Per-Q² Bin: Where Each Model Beats the Others — Physical Regime",
                 fontsize=11, fontweight="bold")

    q2_vals = sorted(data[MODELS[0]]["Q2_log_center"].unique())
    mae_by_q2 = {}
    for key in MODELS:
        df_p = data[key][data[key]["is_physical"]]
        mae_by_q2[key] = np.array([df_p.loc[df_p["Q2_log_center"]==q,"abs_error"].mean()
                                    for q in q2_vals])

    pairs = [("50ep","300ep"), ("300ep","MoE"), ("50ep","MoE")]
    for col, (k1, k2) in enumerate(pairs):
        ax = axes[col]
        diff = mae_by_q2[k1] - mae_by_q2[k2]   # positive = k2 is better
        colors = [PHYS_COLOR if d > 0 else FLOOR_COLOR for d in diff]
        ax.bar(q2_vals, diff, width=0.25, color=colors, alpha=0.8)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(r"$\log Q^2$"); ax.set_ylabel(f"MAE({k1}) − MAE({k2})")
        ax.set_title(f"{M_LABEL[k1]} vs {M_LABEL[k2]}\n"
                     f"(green = {k2} better, red = {k1} better)")

    plt.tight_layout()
    _savefig(pdf, fig)


def page_b6_prediction_intervals(pdf: PdfPages, data: dict):
    """B6 — Prediction interval width vs magnitude (do wide intervals predict well?)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("B6 · Prediction Interval Width Analysis  (A_exp scale)",
                 fontsize=12, fontweight="bold")

    for col, key in enumerate(MODELS):
        ax  = axes[col]
        df  = data[key][data[key]["is_physical"] & data[key]["_scatter"]]
        width = df["pred_A_exp_high"] - df["pred_A_exp_low"]
        truth = df["A_true_exp"]
        covered = (truth >= df["pred_A_exp_low"]) & (truth <= df["pred_A_exp_high"])

        # Scatter: width vs true A, coloured by coverage
        sc = ax.scatter(truth.clip(lower=1e-20), width.clip(lower=1e-20),
                        s=2, alpha=0.2, c=covered.astype(float),
                        cmap="RdYlGn", vmin=0, vmax=1, rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"True $A_{\exp}$")
        ax.set_ylabel(r"Interval width  $A_{\exp,hi} - A_{\exp,lo}$")
        ax.set_title(f"{M_LABEL[key]}\ncoverage={(covered.mean()):.1%}")
        plt.colorbar(sc, ax=ax, shrink=0.8, label="1=covered")

    plt.tight_layout()
    _savefig(pdf, fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv50",   required=True, help="Predictions CSV — 50 epoch run")
    parser.add_argument("--csv300",  required=True, help="Predictions CSV — 300 epoch run")
    parser.add_argument("--csvmoe",  required=True, help="Predictions CSV — MoE KNN run")
    parser.add_argument("--output",  default="gp_diffraction_report.pdf")
    parser.add_argument("--sample",  type=int, default=40_000,
                        help="Rows to subsample for scatter plots (default 40000)")
    args = parser.parse_args()

    print("Loading data ...")
    data = load_data(args.csv50, args.csv300, args.csvmoe, args.sample)

    out_path = Path(args.output)
    print(f"Writing report → {out_path} ...")

    with PdfPages(str(out_path)) as pdf:

        # Cover
        page_cover(pdf, data)

        # ── Section A banner ───────────────────────────────────────────────────
        _section_banner(pdf,
            "Section A — Model Strengths",
            "Accuracy · Calibration · Physical-regime excellence · Improvement trajectory",
            "#0D47A1")

        print("  A1  pred vs true scatter …")
        page_a1_pred_vs_true(pdf, data)
        print("  A2  cross-section scale …")
        page_a2_crosssection_scale(pdf, data)
        print("  A3  residuals …")
        page_a3_residuals(pdf, data)
        print("  A4  calibration …")
        page_a4_calibration(pdf, data)
        print("  A5  per-bin accuracy …")
        page_a5_per_bin_accuracy(pdf, data)
        print("  A6  phase-space heatmap …")
        page_a6_phase_space_heatmap(pdf, data)
        print("  A7  floor detection …")
        page_a7_floor_detection(pdf, data)
        print("  A8  model progression …")
        page_a8_model_progression(pdf, data)
        print("  A9  summation projections …")
        page_a9_summation_projections(pdf, data)

        # ── Section B banner ───────────────────────────────────────────────────
        _section_banner(pdf,
            "Section B — Areas for Improvement",
            "Systematic biases · Floor failures · Overconfidence · Where models diverge",
            "#B71C1C")

        print("  B1  error vs inputs …")
        page_b1_error_vs_inputs(pdf, data)
        print("  B2  overconfidence map …")
        page_b2_overconfidence_map(pdf, data)
        print("  B3  high-error regions …")
        page_b3_high_error_regions(pdf, data)
        print("  B4  floor vs physical …")
        page_b4_floor_vs_physical_errors(pdf, data)
        print("  B5  where models differ …")
        page_b5_where_models_differ(pdf, data)
        print("  B6  prediction intervals …")
        page_b6_prediction_intervals(pdf, data)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "GP Diffractive DIS — Diagnostic Report"
        d["Subject"] = "Inclusive diffraction cross-section prediction diagnostics"
        d["Keywords"] = "HERA, diffraction, GP, bSat, machine learning"

    print(f"\nDone → {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
