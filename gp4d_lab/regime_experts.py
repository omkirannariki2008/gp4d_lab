"""
gp4d_lab.regime_experts
=======================
Regime-based GP expert system for inclusive diffraction cross-sections.
"""

from __future__ import annotations

import json
import logging
import warnings
import dataclasses
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.kernels import (
    MaternKernel, ScaleKernel, ProductKernel,
    GridInterpolationKernel, RBFKernel,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import linregress

log = logging.getLogger("regime_experts")

# ─── Column names and default mappings ────────────────────────────────────────
COLS   = ["Q2_log_center", "W2_log_center", "beta_center", "z_center"]
AXIS_MAP = {
    "q2": 0,
    "w2": 1,
    "beta": 2,
    "z": 3
}
TARGET = "logA"
FLOOR  = -17.955

# Primary partition axis per regime
REGIME_PRIMARY_AXIS = {
    "CPR": "Q2_log_center",
    "ELR": "Q2_log_center",
    "DSR": "W2_log_center",
    "PHR": "beta_center",
    "TRN": "beta_center",
}

# ─── GP model ─────────────────────────────────────────────────────────────────

class _ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, covar):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module   = gpytorch.means.ConstantMean()
        self.covar_module  = covar

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

# ─── Kernel factory for regime experts ────────────────────────────────────────

def _make_regime_kernel(regime: str, use_ski: bool = False,
                         grid_sizes: list[int] | None = None) -> ScaleKernel:
    if regime == "CPR":
        # CPR already sub-partitioned on Q2, sees (W2, beta, z)
        inner = (
            MaternKernel(nu=1.5, active_dims=[0])
          * MaternKernel(nu=0.5, ard_num_dims=2, active_dims=[1, 2])
        )
    elif regime == "ELR":
        # Sees (Q2, W2, beta, z)
        try:
            from gp4d_lab.kernels import GibbsBellKernel1D
            gibbs = GibbsBellKernel1D(peak_init=0.0, active_dims=[0])
        except ImportError:
            from gp4d_lab.kernels import GibbsKernel1D
            gibbs = GibbsKernel1D(variant="sigmoid", x0_init=0.0,
                                   slope_init=3.0, active_dims=[0])
        inner = gibbs * MaternKernel(nu=1.5, active_dims=[1, 2, 3])
    elif regime == "DSR":
        inner = MaternKernel(nu=0.5, ard_num_dims=4)
    elif regime == "PHR":
        inner = MaternKernel(nu=1.5, ard_num_dims=4)
    elif regime == "TRN":
        try:
            from gp4d_lab.kernels import GibbsKernel1D
            gibbs = GibbsKernel1D(variant="sigmoid", x0_init=0.0, slope_init=3.0, active_dims=[0])
        except ImportError:
            gibbs = MaternKernel(nu=1.5, active_dims=[0])
        inner = gibbs * MaternKernel(nu=1.5, active_dims=[1, 2, 3])
    else:
        inner = MaternKernel(nu=0.5)

    # GridInterpolationKernel requires a stationary base kernel in GPyTorch.
    # Non-stationary kernels (like ELR/TRN containing Gibbs) cannot use SKI directly.
    if use_ski:
        if regime in ("ELR", "TRN"):
            log.warning(f"SKI (GridInterpolationKernel) is not supported for non-stationary regime {regime}. Falling back to exact kernel.")
            kernel = ScaleKernel(inner)
        else:
            n_dims = 3 if regime in ("CPR",) else 4
            g_sizes = grid_sizes or [30] * n_dims
            kernel = ScaleKernel(
                GridInterpolationKernel(inner, grid_size=max(g_sizes), num_dims=n_dims)
            )
    else:
        kernel = ScaleKernel(inner)

    return kernel

def _apply_lengthscale_constraints(kernel, lo=0.05, hi=10.0):
    c = gpytorch.constraints.Interval(lo, hi)
    if hasattr(kernel, "lengthscale_constraint"):
        kernel.lengthscale_constraint = c
        try:
            with torch.no_grad():
                ls = kernel.lengthscale
                kernel.lengthscale = torch.full_like(ls, (lo + hi) / 2.0)
        except Exception:
            pass
    if hasattr(kernel, "kernels"):
        for sub in kernel.kernels:
            _apply_lengthscale_constraints(sub, lo, hi)
    if hasattr(kernel, "base_kernel"):
        _apply_lengthscale_constraints(kernel.base_kernel, lo, hi)

# ─── RegimeExpertGP ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class RegimeExpertConfig:
    regime:            str
    n_epochs:          int   = 500
    lr:                float = 0.1
    early_stop:        int   = 30
    min_noise:         float = 0.02
    max_cholesky_size: int   = 2000
    cg_tol:            float = 1e-5
    max_cg_iter:       int   = 200
    preconditioner:    int   = 100
    use_ski:           bool  = False
    grid_sizes:        list  = dataclasses.field(default_factory=list)

class RegimeExpertGP:
    def __init__(self, cfg: RegimeExpertConfig):
        self.cfg   = cfg
        self.model = None
        self.lh    = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        device   = self._device
        tx = torch.tensor(X, dtype=torch.float32).to(device)
        ty = torch.tensor(y, dtype=torch.float32).to(device)

        log.info(f"  [{self.cfg.regime}] Started training with {len(tx)} points on {device}...")

        kernel = _make_regime_kernel(
            self.cfg.regime,
            use_ski=self.cfg.use_ski,
            grid_sizes=self.cfg.grid_sizes,
        )
        _apply_lengthscale_constraints(kernel)

        lh = GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(self.cfg.min_noise, 0.5)
        ).to(device)
        model = _ExactGPModel(tx, ty, lh, kernel).to(device)

        seen = set()
        params = [p for p in model.parameters()
                  if not (id(p) in seen or seen.add(id(p)))]
        optim = torch.optim.Adam(params, lr=self.cfg.lr)
        mll   = ExactMarginalLogLikelihood(lh, model)

        model.train(); lh.train()
        best_loss, no_improve = float("inf"), 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(self.cfg.n_epochs):
                optim.zero_grad()
                with gpytorch.settings.fast_computations(log_prob=True):
                    with gpytorch.settings.cg_tolerance(self.cfg.cg_tol), \
                          gpytorch.settings.max_cg_iterations(self.cfg.max_cg_iter), \
                          gpytorch.settings.max_cholesky_size(max(self.cfg.max_cholesky_size, len(tx)+1)), \
                          gpytorch.settings.cholesky_jitter(1e-3, 1e-6):
                        out  = model(tx)
                        loss = -mll(out, ty)
                loss.backward()
                optim.step()
                val = loss.item()
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    log.info(f"  [{self.cfg.regime}] Epoch {epoch+1:4d}/{self.cfg.n_epochs:4d} | Loss: {val:.4f}")
                if val < best_loss - 1e-4:
                    best_loss = val; no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.cfg.early_stop:
                    log.info(f"  [{self.cfg.regime}] Early stopping at epoch {epoch+1} (no improvement for {self.cfg.early_stop} epochs)")
                    break

        self.model = model; self.lh = lh
        log.info(f"  [{self.cfg.regime}] Training complete. Epochs: {epoch+1}, final loss={best_loss:.4f}")

    def predict(self, X: np.ndarray, chunk_size: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        if len(X) == 0:
            return np.empty(0), np.empty(0)
        self.model.eval(); self.lh.eval()
        device = self._device
        mu_all    = np.empty(len(X))
        sigma_all = np.empty(len(X))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, len(X), chunk_size):
                end  = min(start + chunk_size, len(X))
                tx   = torch.tensor(X[start:end], dtype=torch.float32).to(device)
                pred = self.lh(self.model(tx))
                mu_all[start:end]    = pred.mean.cpu().numpy()
                sigma_all[start:end] = pred.variance.sqrt().cpu().numpy()
        return mu_all, sigma_all

# ─── AutoRegimeDetector ───────────────────────────────────────────────────────

class AutoRegimeDetector:
    def __init__(
        self,
        feature_cols: list[str],
        target_col:   str,
        floor_thresh: float | None = None,
        phase_axes:   tuple[str, str] | None = ("Q2_log_center", "beta_center"),
        n_floor_bins: int = 8,
    ):
        self.fcols   = feature_cols
        self.tcol    = target_col
        self.floor   = floor_thresh
        self.p_axes  = phase_axes
        self.nbins   = n_floor_bins

        self._regions: dict = {}
        self._boundaries: dict = {}
        self._kernel_recs: dict = {}

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> None:
        if self.floor is None:
            y = df[self.tcol].values
            q25, q50, q75 = np.percentile(y, [25, 50, 75])
            iqr = q75 - q25
            self.floor = q50 - 3 * iqr
            if verbose:
                print(f"  Auto floor threshold: {self.floor:.3f}  "
                      f"(covers {100*(y<self.floor).mean():.2f}% of points)")

        if self.p_axes is None:
            self.p_axes = self._detect_phase_axes(df)
            if verbose:
                print(f"  Phase axes detected: {self.p_axes}")

        ff_map = self._floor_fraction_map(df)
        self._boundaries = self._detect_boundaries(ff_map, df)
        if verbose:
            for name, bnd in self._boundaries.items():
                print(f"  Boundary [{name}]: {bnd['desc']}")

        df_labeled = df.copy()
        df_labeled["_regime_"] = self._assign_labels(df)

        for reg_label in df_labeled["_regime_"].unique():
            sub = df_labeled[df_labeled["_regime_"] == reg_label]
            meta = self._analyse_regime(sub, reg_label)
            meta["frac"] = 100.0 * len(sub) / len(df)
            self._regions[reg_label] = meta
            if verbose:
                dom = meta["dominant_var"]
                print(f"  [{reg_label}] n={len(sub):,}  "
                      f"dom={dom}(eta2={meta['eta2'][dom]:.3f})  "
                      f"kernel={meta['kernel_rec']}")

        self._build_kernel_recs()

    def assign(self, df: pd.DataFrame) -> np.ndarray:
        return self._assign_labels(df)

    def report(self) -> str:
        lines = ["=" * 64, "  AUTO-DETECTED REGIME ARCHITECTURE", "=" * 64]
        for name, meta in self._regions.items():
            lines += [
                f"\n  [{name}]",
                f"    n = {meta['n']:,} ({meta['frac']:.1f}%)  floor = {meta['floor_pct']:.1f}%",
                f"    dominant variable: {meta['dominant_var']}  eta2 = {meta['eta2'][meta['dominant_var']]:.3f}",
                "    Hoelder alpha per variable:",
            ]
            for ax, alpha in meta["alpha"].items():
                k = self._holder_to_kernel(alpha)
                lines.append(f"      {ax:20s}: alpha={alpha:.3f}  -> {k}")
            lines.append(f"    Recommended kernel: {meta['kernel_rec']}")
        return "\n".join(lines)

    def _detect_phase_axes(self, df: pd.DataFrame) -> tuple[str, str]:
        floor_mask = df[self.tcol] < self.floor
        n_bins     = self.nbins
        best_score = -1.0
        best_pair  = (self.fcols[0], self.fcols[1])

        for i, c1 in enumerate(self.fcols):
            for c2 in self.fcols[i + 1:]:
                b1 = pd.cut(df[c1], bins=n_bins, labels=False).fillna(0).astype(int)
                b2 = pd.cut(df[c2], bins=n_bins, labels=False).fillna(0).astype(int)
                ff = np.zeros((n_bins, n_bins))
                ct = np.zeros((n_bins, n_bins))
                for r1, r2, is_f in zip(b1, b2, floor_mask):
                    ff[r1, r2] += int(is_f)
                    ct[r1, r2] += 1
                with np.errstate(invalid="ignore"):
                    ff_rate = np.where(ct > 0, ff / ct, 0)
                score = float(ff_rate.var())
                if score > best_score:
                    best_score = score
                    best_pair  = (c1, c2)
        return best_pair

    def _floor_fraction_map(self, df: pd.DataFrame) -> dict:
        c1, c2    = self.p_axes
        vals1     = np.sort(df[c1].unique())
        vals2     = np.sort(df[c2].unique())
        ff = {}
        for v1 in vals1:
            for v2 in vals2:
                cell = df[(df[c1] == v1) & (df[c2] == v2)]
                if len(cell) > 0:
                    ff[(v1, v2)] = float((cell[self.tcol] < self.floor).mean())
        return {"axes": (c1, c2), "vals1": vals1, "vals2": vals2, "ff": ff}

    def _detect_boundaries(self, ff_map: dict, df: pd.DataFrame) -> dict:
        # Boundaries confirmed by full-data diagnostic analysis (S2/S6 of analyse_regimes.py)
        # These are in RAW (non-standardised) Q2/β space
        c1, c2 = ff_map["axes"]
        return {
            "lo_c1": {"value": -3.652, "axis": c1, "side": "lo",
                      "desc": "PHR lower boundary (Q2_log=-3.652, Q2_std=-1.33)"},
            "hi_c1": {"value":  2.700, "axis": c1, "side": "hi",
                      "desc": "DSR upper boundary (Q2_log=+2.700, Q2_std=+0.98)"},
            "hi_c2": {"value":  0.827, "axis": c2, "side": "hi",
                      "desc": "ELR boundary (β=0.827, elastic limit)"},
        }

    def _assign_labels(self, df: pd.DataFrame) -> np.ndarray:
        c1, c2   = self.p_axes
        lo_c1    = self._boundaries["lo_c1"]["value"]
        hi_c1    = self._boundaries["hi_c1"]["value"]
        hi_c2    = self._boundaries["hi_c2"]["value"]

        v1 = df[c1].values
        v2 = df[c2].values

        labels = np.where(
            v2 >= hi_c2,     "ELR",
            np.where(
                v1 <= lo_c1, "PHR",
                np.where(
                    v1 >= hi_c1, "DSR",
                    np.where(
                        (v1 > lo_c1) & (v1 < lo_c1 * 0.75), "TRN",
                        "CPR"
                    )
                )
            )
        )
        return labels

    def _analyse_regime(self, sub: pd.DataFrame, label: str) -> dict:
        y         = sub[self.tcol].values
        n_total   = len(sub)
        floor_pct = 100.0 * (y < self.floor).mean()

        eta2 = {}
        for col in self.fcols:
            eta2[col] = self._eta_squared(sub[col].values, y)
        dom_var = max(eta2, key=eta2.get)

        alpha = {}
        for col in self.fcols:
            alpha[col] = self._holder_from_data(sub, col, y)

        var_ratio = {}
        for col in self.fcols:
            var_ratio[col] = self._var_ratio(sub[col].values, y)

        kernel_rec = self._auto_kernel(alpha, var_ratio, eta2)

        return {
            "n": n_total, "frac": 0.0, "floor_pct": floor_pct,
            "dominant_var": dom_var, "eta2": eta2,
            "alpha": alpha, "var_ratio": var_ratio,
            "kernel_rec": kernel_rec,
        }

    def _auto_kernel(self, alpha: dict, var_ratio: dict, eta2: dict) -> str:
        dom = max(eta2, key=eta2.get)
        dom_vr = var_ratio.get(dom, 1.0)
        dom_a  = alpha.get(dom, 0.5)

        sec_alphas = [a for k, a in alpha.items() if k != dom and not np.isnan(a)]
        sec_alpha  = np.nanmean(sec_alphas) if sec_alphas else 0.5
        sec_kernel = self._holder_to_kernel(sec_alpha)

        if dom_vr > 5.0:
            prefix = "gibbs_bell"
        elif dom_vr > 3.0:
            prefix = "gibbs_sigmoid"
        else:
            prefix = self._holder_to_kernel(dom_a)

        if prefix == sec_kernel:
            return f"{prefix}_ard"
        return f"{prefix}_x_{sec_kernel}"

    def _build_kernel_recs(self):
        for name, meta in self._regions.items():
            self._kernel_recs[name] = meta["kernel_rec"]

    @staticmethod
    def _eta_squared(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        bid    = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        gm     = y.mean()
        ss_t   = ((y - gm) ** 2).sum()
        ss_b   = sum(len(y[bid==b]) * (y[bid==b].mean() - gm)**2
                     for b in range(n_bins) if (bid==b).sum() > 0)
        return float(ss_b / (ss_t + 1e-12))

    @staticmethod
    def _var_ratio(x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> float:
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        bid  = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        stds = [y[bid==b].std() for b in range(n_bins) if (bid==b).sum() > 5]
        if len(stds) < 2:
            return 1.0
        return float(max(stds) / (min(stds) + 1e-9))

    def _holder_from_data(self, sub: pd.DataFrame, target_col: str,
                           y: np.ndarray, max_lags: int = 12) -> float:
        other_cols = [c for c in self.fcols if c != target_col]
        ax_vals    = np.sort(sub[target_col].unique())
        if len(ax_vals) < 3:
            return np.nan
        spacing = float(np.median(np.diff(ax_vals)))
        groups  = sub.groupby(other_cols)[self.tcol].apply(lambda g: g.values)
        if len(groups) > 1000:
            rng   = np.random.default_rng(42)
            idxs  = rng.choice(len(groups), 1000, replace=False)
            groups = groups.iloc[idxs]
        gamma_acc = {}
        for lag in range(1, min(max_lags + 1, len(ax_vals))):
            vals = []
            for _, ya in groups.items():
                if len(ya) > lag:
                    vals.extend((0.5 * (ya[lag:] - ya[:-lag])**2).tolist())
            if len(vals) > 20:
                gamma_acc[lag * spacing] = float(np.mean(vals))
        if len(gamma_acc) < 3:
            return np.nan
        h  = np.array(sorted(gamma_acc.keys()))
        g  = np.array([gamma_acc[k] for k in h])
        mask = (h > 0) & (g > 0)
        if mask.sum() < 3:
            return np.nan
        s, _, _, _, _ = linregress(np.log(h[mask]), np.log(g[mask]))
        return float(s)

    @staticmethod
    def _holder_to_kernel(alpha: float) -> str:
        if np.isnan(alpha) or alpha < 0.8:  return "matern12"
        if alpha < 1.5:                      return "matern12"
        if alpha < 2.5:                      return "matern32"
        if alpha < 4.0:                      return "matern52"
        return "matern52"

# ─── Ensemble Coordinator ─────────────────────────────────────────────────────

class RegimeGPEnsemble:
    """
    Ensemble managing regime detection, auto-selected strategies, and prediction.
    """
    def __init__(self, feature_cols: list[str], target_col: str, max_cholesky_size: int = 2000):
        self.fcols = feature_cols
        self.tcol = target_col
        self.max_cholesky_size = max_cholesky_size
        self.detector = None
        self.experts = {}       # {regime: single_expert or dict_of_binned_experts}
        self.metadata = {}      # {regime: {"strategy": str, "bin_edges": list, etc.}}

    def fit(self, df_train_raw: pd.DataFrame, df_train_scaled: pd.DataFrame, val_fraction: float = 0.10) -> None:
        # Step 1: Detect regimes from a sample of raw training data (using full if small)
        if len(df_train_raw) <= 1000:
            df_sample = df_train_raw
        else:
            n_sample = max(100, int(len(df_train_raw) * val_fraction))
            df_sample = df_train_raw.sample(n=n_sample, random_state=42)
        
        log.info("[Ensemble] Fitting AutoRegimeDetector on sample...")
        self.detector = AutoRegimeDetector(self.fcols, self.tcol)
        self.detector.fit(df_sample, verbose=False)
        log.info("[Ensemble] Detected boundaries and regimes successfully.")

        # Step 2: Assign regime labels in RAW space
        labels_train = self.detector.assign(df_train_raw)
        df_train_scaled_labeled = df_train_scaled.copy()
        df_train_scaled_labeled["_regime_"] = labels_train
        df_train_scaled_labeled["_raw_target_"] = df_train_raw[self.tcol].values

        # Step 3: Train each regime expert on standardized/scaled subsets
        for regime in ["CPR", "ELR", "DSR", "PHR", "TRN"]:
            sub = df_train_scaled_labeled[df_train_scaled_labeled["_regime_"] == regime]
            n_reg = len(sub)
            if n_reg == 0:
                log.warning(f"No training points in regime {regime}; skipping.")
                continue

            X_reg = sub[self.fcols].values
            y_reg = sub[self.tcol].values
            y_raw_reg = sub["_raw_target_"].values

            if regime in ("ELR", "TRN"):
                non_floor_mask = y_raw_reg >= FLOOR
                n_filtered = (~non_floor_mask).sum()
                if n_filtered > 0:
                    log.info(f"[Ensemble] [{regime}] Filtering {n_filtered} floor points from training "
                             f"({100*n_filtered/n_reg:.1f}%) to remove bimodal contamination")
                    X_reg = X_reg[non_floor_mask]
                    y_reg = y_reg[non_floor_mask]
                    n_reg = len(X_reg)
                    if n_reg == 0:
                        log.warning(f"All training points in regime {regime} were filtered out; skipping.")
                        continue

            # Choose strategy automatically: always use bin_exact (no SKI) for large regimes
            if n_reg <= self.max_cholesky_size:
                strategy = "exact"
            else:
                strategy = "bin_exact"

            self.metadata[regime] = {
                "strategy": strategy,
                "n_train": n_reg,
            }
            log.info(f"[Ensemble] Regime {regime}: strategy={strategy}, n_train={n_reg}")

            if strategy == "exact":
                cfg = RegimeExpertConfig(regime=regime, use_ski=False)
                expert = RegimeExpertGP(cfg)
                expert.fit(X_reg, y_reg)
                self.experts[regime] = expert

            elif strategy == "bin_exact":
                # Sub-partition along primary axis
                axis_name = REGIME_PRIMARY_AXIS[regime]
                col_idx = self.fcols.index(axis_name)
                
                # Check how many unique values of primary axis exist in scaled space
                unique_vals = np.sort(np.unique(X_reg[:, col_idx]))
                if len(unique_vals) <= 30:
                    # Discrete/grid-based binning by unique values
                    bin_centers = unique_vals
                    mids = 0.5 * (unique_vals[:-1] + unique_vals[1:])
                    bin_edges = np.concatenate([
                        [X_reg[:, col_idx].min() - 1e-5],
                        mids,
                        [X_reg[:, col_idx].max() + 1e-5],
                    ])
                    n_bins = len(unique_vals)
                else:
                    # Quantile-based bins
                    n_bins = int(np.ceil(n_reg / 1500))
                    percentiles = np.linspace(0, 100, n_bins + 1)
                    bin_edges = np.percentile(X_reg[:, col_idx], percentiles)
                    bin_edges[0] -= 1e-9
                    bin_edges[-1] += 1e-9
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                self.metadata[regime]["bin_edges"] = bin_edges.tolist()
                self.metadata[regime]["bin_centers"] = bin_centers.tolist()
                self.metadata[regime]["primary_axis_idx"] = col_idx
                self.metadata[regime]["remove_axis"] = (regime == "CPR")

                self.experts[regime] = {}
                log.info(f"[Ensemble] Regime {regime}: sub-partitioning into {n_bins} bins along {axis_name}...")

                for b in range(n_bins):
                    lo, hi = bin_edges[b], bin_edges[b+1]
                    mask = (X_reg[:, col_idx] >= lo) & (X_reg[:, col_idx] < hi)
                    if mask.sum() == 0:
                        continue
                    
                    X_sub = X_reg[mask]
                    if regime == "CPR": # Remove Q2 column
                        keep_cols = [j for j in range(X_reg.shape[1]) if j != col_idx]
                        X_sub = X_sub[:, keep_cols]

                    log.info(f"[Ensemble]   Training bin {b+1}/{n_bins} for regime {regime} with {len(X_sub)} points...")
                    cfg = RegimeExpertConfig(regime=regime, use_ski=False)
                    expert = RegimeExpertGP(cfg)
                    expert.fit(X_sub, y_reg[mask])
                    self.experts[regime][b] = expert

    def predict(self, df_test_raw: pd.DataFrame, df_test_scaled: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Assign test points to regimes in RAW space
        labels_test = self.detector.assign(df_test_raw)
        df_test_scaled_labeled = df_test_scaled.copy()
        df_test_scaled_labeled["_regime_"] = labels_test

        mu_all = np.zeros(len(df_test_scaled))
        sigma_all = np.zeros(len(df_test_scaled))

        for regime in ["CPR", "ELR", "DSR", "PHR", "TRN"]:
            mask = df_test_scaled_labeled["_regime_"] == regime
            if mask.sum() == 0:
                continue

            indices = np.where(mask)[0]
            X_test_reg = df_test_scaled_labeled.loc[mask, self.fcols].values
            
            if regime not in self.experts:
                log.warning(f"No expert trained for regime {regime}! Predicting zeros.")
                mu_all[indices] = 0.0
                sigma_all[indices] = 1.0
                continue

            strategy = self.metadata[regime]["strategy"]

            if strategy in ("exact", "ski"):
                expert = self.experts[regime]
                mu, sigma = expert.predict(X_test_reg)
                mu_all[indices] = mu
                sigma_all[indices] = sigma

            elif strategy == "bin_exact":
                # Find matching bins for each point
                bin_edges = np.array(self.metadata[regime]["bin_edges"])
                col_idx = self.metadata[regime]["primary_axis_idx"]
                vals = X_test_reg[:, col_idx]
                
                bin_ids = np.clip(np.searchsorted(bin_edges[1:], vals), 0, len(bin_edges) - 2)

                for b in np.unique(bin_ids):
                    b_mask = bin_ids == b
                    b_indices = indices[b_mask]
                    X_b_reg = X_test_reg[b_mask]

                    if b not in self.experts[regime]:
                        log.warning(f"No sub-expert trained for {regime} bin {b}!")
                        mu_all[b_indices] = 0.0
                        sigma_all[b_indices] = 1.0
                        continue

                    expert = self.experts[regime][b]
                    if self.metadata[regime]["remove_axis"]:
                        keep_cols = [j for j in range(X_test_reg.shape[1]) if j != col_idx]
                        X_b_reg = X_b_reg[:, keep_cols]

                    mu, sigma = expert.predict(X_b_reg)
                    mu_all[b_indices] = mu
                    sigma_all[b_indices] = sigma

        return mu_all, sigma_all

    def save_models(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        log.info(f"[Ensemble] Saving regime ensemble to {path}...")

        # 1. Save Detector boundaries
        det_state = {
            "floor": self.detector.floor,
            "p_axes": self.detector.p_axes,
            "_boundaries": self.detector._boundaries,
            "_regions": self.detector._regions,
            "_kernel_recs": self.detector._kernel_recs,
        }
        with open(path / "detector_state.json", "w") as f:
            json.dump(det_state, f, indent=2)

        # 2. Save ensemble metadata
        with open(path / "ensemble_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        # 3. Save PyTorch model state dicts
        for regime, val in self.experts.items():
            strategy = self.metadata[regime]["strategy"]
            if strategy in ("exact", "ski"):
                checkpoint = {
                    "model_state": val.model.state_dict(),
                    "likelihood_state": val.lh.state_dict(),
                }
                torch.save(checkpoint, path / f"{regime}.pt")
            elif strategy == "bin_exact":
                for b, expert in val.items():
                    checkpoint = {
                        "model_state": expert.model.state_dict(),
                        "likelihood_state": expert.lh.state_dict(),
                    }
                    torch.save(checkpoint, path / f"{regime}_bin_{b}.pt")

        log.info("[Ensemble] Ensemble saved successfully.")

    def load_models(self, path: Path | str, df_train_raw: pd.DataFrame, df_train_scaled: pd.DataFrame) -> bool:
        path = Path(path)
        if not path.exists():
            return False

        try:
            # 1. Load detector state
            with open(path / "detector_state.json", "r") as f:
                det_state = json.load(f)
            self.detector = AutoRegimeDetector(self.fcols, self.tcol)
            self.detector.floor = det_state["floor"]
            self.detector.p_axes = tuple(det_state["p_axes"])
            self.detector._boundaries = det_state["_boundaries"]
            self.detector._regions = det_state["_regions"]
            self.detector._kernel_recs = det_state["_kernel_recs"]

            # 2. Load ensemble metadata
            with open(path / "ensemble_metadata.json", "r") as f:
                self.metadata = json.load(f)

            # Assign labels in raw space
            labels_train = self.detector.assign(df_train_raw)
            df_train_scaled_labeled = df_train_scaled.copy()
            df_train_scaled_labeled["_regime_"] = labels_train
            df_train_scaled_labeled["_raw_target_"] = df_train_raw[self.tcol].values

            # 3. Reconstruct and load model parameters
            for regime, meta in self.metadata.items():
                strategy = meta["strategy"]
                sub = df_train_scaled_labeled[df_train_scaled_labeled["_regime_"] == regime]
                X_reg = sub[self.fcols].values
                y_reg = sub[self.tcol].values
                y_raw_reg = sub["_raw_target_"].values

                if regime in ("ELR", "TRN"):
                    non_floor_mask = y_raw_reg >= FLOOR
                    X_reg = X_reg[non_floor_mask]
                    y_reg = y_reg[non_floor_mask]

                if strategy == "exact":
                    cfg = RegimeExpertConfig(regime=regime, use_ski=False)
                    expert = RegimeExpertGP(cfg)
                    expert.fit(X_reg, y_reg)
                    checkpoint = torch.load(path / f"{regime}.pt")
                    expert.model.load_state_dict(checkpoint["model_state"])
                    expert.lh.load_state_dict(checkpoint["likelihood_state"])
                    self.experts[regime] = expert

                elif strategy == "ski":
                    n_unique = [max(2, len(np.unique(X_reg[:, d]))) for d in range(X_reg.shape[1])]
                    cfg = RegimeExpertConfig(regime=regime, use_ski=True, grid_sizes=n_unique)
                    expert = RegimeExpertGP(cfg)
                    expert.fit(X_reg, y_reg)
                    checkpoint = torch.load(path / f"{regime}.pt")
                    expert.model.load_state_dict(checkpoint["model_state"])
                    expert.lh.load_state_dict(checkpoint["likelihood_state"])
                    self.experts[regime] = expert

                elif strategy == "bin_exact":
                    bin_edges = np.array(meta["bin_edges"])
                    col_idx = meta["primary_axis_idx"]
                    n_bins = len(bin_edges) - 1

                    self.experts[regime] = {}
                    for b in range(n_bins):
                        lo, hi = bin_edges[b], bin_edges[b+1]
                        mask = (X_reg[:, col_idx] >= lo) & (X_reg[:, col_idx] < hi)
                        if mask.sum() == 0:
                            continue
                        
                        X_sub = X_reg[mask]
                        if meta["remove_axis"]:
                            keep_cols = [j for j in range(X_reg.shape[1]) if j != col_idx]
                            X_sub = X_sub[:, keep_cols]

                        cfg = RegimeExpertConfig(regime=regime, use_ski=False)
                        expert = RegimeExpertGP(cfg)
                        expert.fit(X_sub, y_reg[mask])
                        
                        checkpoint = torch.load(path / f"{regime}_bin_{b}.pt")
                        expert.model.load_state_dict(checkpoint["model_state"])
                        expert.lh.load_state_dict(checkpoint["likelihood_state"])
                        self.experts[regime][b] = expert

            log.info("[Ensemble] Reconstruct and load completed successfully.")
            return True
        except Exception as e:
            log.warning(f"Failed to load checkpoints: {e}. Reverting to training.")
            return False
