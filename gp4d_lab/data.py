"""
gp4d_lab.data
=============
Data loading, train/test splitting, and axis-based partitioning.

Public API
----------
  load_dataset(cfg)                 → X_train, y_train, X_test, y_test
  AxisPartitioner(axis, bins)       → partition train/test by any variable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DataConfig
from .scalers import get_scaler

log = logging.getLogger("gp4d_lab.data")


# ── Column layout ─────────────────────────────────────────────────────────────
# The 4 feature columns in canonical order
AXIS_NAMES   = ["q2", "w2", "beta", "z"]
AXIS_COL_IDX = {name: i for i, name in enumerate(AXIS_NAMES)}


@dataclass
class Dataset:
    """Container for a single train/test split plus metadata."""

    X_train:       np.ndarray
    y_train:       np.ndarray
    X_test:        np.ndarray
    y_test:        np.ndarray
    feature_cols:  list
    target_col:    str
    q2_bins_train: np.ndarray    # Q2 bin indices for training rows
    q2_bins_test:  np.ndarray
    scaler_X:      object        # fitted input scaler
    scaler_y:      object        # fitted target scaler
    raw_X_train:   np.ndarray    # unscaled (for axis-partitioner)
    raw_X_test:    np.ndarray


def load_dataset(cfg: DataConfig) -> Dataset:
    """
    Load CSV, sample train_fraction, scale, and return a Dataset.

    The train/test split uses stratified sampling over Q2 bins so that
    every Q2 bin has adequate training coverage even at 10%.
    """
    df = pd.read_csv(cfg.csv_path)

    required = cfg.feature_cols + [cfg.target_col]
    if cfg.q2_bin_col in df.columns:
        required.append(cfg.q2_bin_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # ── Stratified sampling by Q2 bin ────────────────────────────────────────
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.q2_bin_col in df.columns:
        # Stratify by Q2 bin
        train_idx, test_idx = [], []
        for _, grp in df.groupby(cfg.q2_bin_col):
            idx = grp.index.to_numpy().copy()
            if cfg.shuffle_before_split:
                rng.shuffle(idx)
            n_train = max(1, int(len(idx) * cfg.train_fraction))
            train_idx.extend(idx[:n_train].tolist())
            test_idx.extend(idx[n_train:].tolist())
    else:
        idx = df.index.to_numpy().copy()
        if cfg.shuffle_before_split:
            rng.shuffle(idx)
        n_train = max(1, int(len(idx) * cfg.train_fraction))
        train_idx = idx[:n_train].tolist()
        test_idx  = idx[n_train:].tolist()

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    raw_X_train = df_train[cfg.feature_cols].values.astype(np.float64)
    raw_X_test  = df_test[cfg.feature_cols].values.astype(np.float64)
    
    # Fold z-coordinate symmetry: z -> min(z, 1.0 - z)
    if "z_center" in cfg.feature_cols:
        z_idx = cfg.feature_cols.index("z_center")
        raw_X_train[:, z_idx] = np.minimum(raw_X_train[:, z_idx], 1.0 - raw_X_train[:, z_idx])
        raw_X_test[:, z_idx] = np.minimum(raw_X_test[:, z_idx], 1.0 - raw_X_test[:, z_idx])

    raw_y_train = df_train[cfg.target_col].values.astype(np.float64)
    raw_y_test  = df_test[cfg.target_col].values.astype(np.float64)

    q2_bins_train = (
        df_train[cfg.q2_bin_col].values if cfg.q2_bin_col in df_train.columns
        else np.zeros(len(df_train), dtype=int)
    )
    q2_bins_test = (
        df_test[cfg.q2_bin_col].values if cfg.q2_bin_col in df_test.columns
        else np.zeros(len(df_test), dtype=int)
    )

    # ── Outlier filtering on training set ────────────────────────────────────
    # Removes unphysical floor values from training only.
    # Test set is never touched — metrics remain honest.
    if getattr(cfg, "filter_target_outliers", False):
        lo = getattr(cfg, "target_clip_lo", -np.inf)
        hi = getattr(cfg, "target_clip_hi",  np.inf)
        mask        = (raw_y_train >= lo) & (raw_y_train <= hi)
        n_before    = len(raw_y_train)
        raw_X_train = raw_X_train[mask]
        raw_y_train = raw_y_train[mask]
        q2_bins_train = q2_bins_train[mask]
        n_removed   = n_before - len(raw_y_train)
        pct         = 100 * n_removed / n_before
        log.info(
            f"[outlier filter] Removed {n_removed:,} / {n_before:,} "
            f"training points ({pct:.2f}%) outside [{lo:.3f}, {hi:.3f}]"
        )
        if len(raw_y_train) == 0:
            raise ValueError(
                "Outlier filter removed ALL training points — "
                "check target_clip_lo / target_clip_hi values."
            )

    # ── Scalers ───────────────────────────────────────────────────────────────
    scaler_X = get_scaler("input", cfg.input_scaler)
    scaler_y = get_scaler("target", cfg.target_scaler)

    X_train = scaler_X.fit_transform(raw_X_train)
    X_test  = scaler_X.transform(raw_X_test)
    y_train = scaler_y.fit_transform(raw_y_train).ravel()
    y_test  = raw_y_test  # keep test targets in original scale for metrics

    log.info(
        f"Dataset loaded: {len(X_train)} train / {len(X_test)} test  |  "
        f"features={cfg.feature_cols}  target={cfg.target_col}"
    )

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_cols=cfg.feature_cols,
        target_col=cfg.target_col,
        q2_bins_train=q2_bins_train,
        q2_bins_test=q2_bins_test,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        raw_X_train=raw_X_train,
        raw_X_test=raw_X_test,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Axis Partitioner — splits data by any of the 4 variables
# ─────────────────────────────────────────────────────────────────────────────

class AxisPartitioner:
    """
    Partition training/test data along a given axis variable.

    Behaviour
    ---------
    - Splits the data into bins along the chosen axis.
    - For each bin, returns (X_sub, y_sub) where X_sub has the
      partition column removed (so the sub-GP sees D-1 dimensions).
    - At test time, maps each test point to its nearest bin.

    Parameters
    ----------
    axis : str
        One of "q2", "w2", "beta", "z".
    n_bins : int or None
        Number of equal-quantile bins. If None, uses unique values
        (appropriate for discrete columns like Q2_bin_idx).
    """

    def __init__(self, axis: str, n_bins: int | None = None):
        if axis not in AXIS_NAMES:
            raise ValueError(f"axis must be one of {AXIS_NAMES}, got '{axis}'")
        self.axis    = axis
        self.col_idx = AXIS_COL_IDX[axis]
        self.n_bins  = n_bins

        self._bin_edges:    np.ndarray | None = None
        self._bin_centers:  np.ndarray | None = None
        self._keep_cols:    list | None = None     # indices to keep (all except axis col)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "AxisPartitioner":
        """Determine bin boundaries from training data."""
        D = X.shape[1]
        self._keep_cols = [i for i in range(D) if i != self.col_idx]
        axis_vals = X[:, self.col_idx]

        if self.n_bins is None:
            # Treat unique values as bins (e.g., discrete Q2 indices)
            unique = np.sort(np.unique(axis_vals))
            self._bin_centers = unique
            # Boundaries at midpoints
            mids = 0.5 * (unique[:-1] + unique[1:])
            self._bin_edges = np.concatenate([
                [axis_vals.min() - 1],
                mids,
                [axis_vals.max() + 1],
            ])
        else:
            # Quantile-based bins
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self._bin_edges = np.percentile(axis_vals, percentiles)
            self._bin_edges[0]  -= 1e-9
            self._bin_edges[-1] += 1e-9
            self._bin_centers = 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])

        return self

    # ── Partition ─────────────────────────────────────────────────────────────

    def get_bin_index(self, axis_val: float) -> int:
        """Map a single axis value to its bin index (0-based)."""
        idx = int(np.searchsorted(self._bin_edges[1:], axis_val))
        return np.clip(idx, 0, len(self._bin_centers) - 1)

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """
        Split (X, y) into per-bin sub-arrays.

        Returns
        -------
        List of (bin_center, X_sub, y_sub) where X_sub has the
        partition axis column removed.
        """
        result = []
        axis_vals = X[:, self.col_idx]
        X_reduced = X[:, self._keep_cols]

        for i, center in enumerate(self._bin_centers):
            lo, hi = self._bin_edges[i], self._bin_edges[i + 1]
            mask = (axis_vals >= lo) & (axis_vals < hi)
            if mask.sum() == 0:
                continue
            result.append((center, X_reduced[mask], y[mask]))

        return result

    def reduce(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        For a test array, return (bin_indices, X_reduced).
        bin_indices[i] is the bin for test point i.
        X_reduced has the partition axis column removed.
        """
        axis_vals = X[:, self.col_idx]
        bin_ids = np.array([self.get_bin_index(v) for v in axis_vals])
        X_reduced = X[:, self._keep_cols]
        return bin_ids, X_reduced

    @property
    def n_effective_bins(self) -> int:
        return len(self._bin_centers) if self._bin_centers is not None else 0

    @property
    def remaining_dim(self) -> int:
        return len(self._keep_cols) if self._keep_cols else 3
