"""
gp4d_lab.experiment
====================
ExperimentRunner — orchestrates training, evaluation, and comparison.

Usage
-----
    from gp4d_lab.config import make_master_default, make_aggregation_sweep_configs
    from gp4d_lab.experiment import ExperimentRunner

    runner = ExperimentRunner(csv_path="data.csv")

    # Run a single config
    result = runner.run(make_master_default("data.csv"))

    # Sweep over aggregation strategies
    results = runner.sweep(make_aggregation_sweep_configs("data.csv"))

    # Print comparison table
    runner.print_leaderboard(results)
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, GridSpec, make_master_default
from .data import load_dataset, Dataset
from .ensemble import MultiAxisGPEnsemble
from .metrics import evaluate, format_metrics
from .aggregation import AggregationResult

log = logging.getLogger("gp4d_lab.experiment")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """All outputs from a single experiment run."""

    config:       ExperimentConfig
    metrics:      dict[str, float]
    agg_result:   AggregationResult
    y_pred:       np.ndarray        # inverse-scaled predictions
    y_std:        np.ndarray        # inverse-scaled uncertainties
    y_true:       np.ndarray        # test targets (original scale)
    train_time_s: float
    model_summary: str
    experiment_name: str = ""

    def to_row(self) -> dict:
        row = {"name": self.experiment_name, "train_time_s": self.train_time_s}
        row.update(self.metrics)
        return row


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Loads data once, then runs arbitrary configs against the same train/test split.
    """

    def __init__(self, csv_path: str, base_cfg: ExperimentConfig | None = None):
        self.csv_path = csv_path
        self.base_cfg = base_cfg or make_master_default(csv_path)
        self._dataset: Dataset | None = None

    def _ensure_dataset(self, cfg: ExperimentConfig) -> Dataset:
        """Cache dataset for reuse across experiments (same train/test split)."""
        if self._dataset is None:
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.data.csv_path = self.csv_path
            self._dataset = load_dataset(cfg_copy.data)
            log.info(
                f"[Runner] Dataset loaded: "
                f"{len(self._dataset.X_train)} train / "
                f"{len(self._dataset.X_test)} test"
            )
        return self._dataset

    # ── Single run ────────────────────────────────────────────────────────────

    def run(self, cfg: ExperimentConfig) -> RunResult:
        """
        Train + evaluate one config. Returns RunResult.
        Loads from disk if already completed to allow resumption.
        """
        cfg.data.csv_path = self.csv_path
        dataset = self._ensure_dataset(cfg)

        # Check if result already exists on disk
        out_dir = Path(cfg.results_dir) / cfg.name
        metrics_path = out_dir / "metrics.json"
        y_pred_path = out_dir / "y_pred.npy"
        y_std_path = out_dir / "y_std.npy"
        y_true_path = out_dir / "y_true.npy"
        summary_path = out_dir / "model_summary.txt"

        if metrics_path.exists() and y_pred_path.exists() and y_std_path.exists() and y_true_path.exists():
            log.info(f"[Runner] Found existing results for '{cfg.name}' at {out_dir}. Loading from disk...")
            try:
                with open(metrics_path) as f:
                    metrics_dict = json.load(f)
                y_pred = np.load(y_pred_path)
                y_std = np.load(y_std_path)
                y_true = np.load(y_true_path)
                summary_text = summary_path.read_text() if summary_path.exists() else ""
                
                # Reconstruct AggregationResult from predictions
                try:
                    mu_scaled = dataset.scaler_y.transform(y_pred.reshape(-1, 1)).ravel()
                    if hasattr(dataset.scaler_y, "iqr"):
                        sigma_scaled = y_std / dataset.scaler_y.iqr
                    elif hasattr(dataset.scaler_y, "std_"):
                        sigma_scaled = y_std / dataset.scaler_y.std_
                    elif hasattr(dataset.scaler_y, "range_"):
                        sigma_scaled = y_std / dataset.scaler_y.range_
                    else:
                        sigma_scaled = y_std
                except Exception:
                    mu_scaled = y_pred
                    sigma_scaled = y_std
                
                agg_result = AggregationResult(
                    mu=mu_scaled,
                    sigma=sigma_scaled,
                    weights=np.zeros((4, len(y_pred))),
                    expert_means=np.zeros((4, len(y_pred))),
                    expert_sigmas=np.zeros((4, len(y_pred))),
                    strategy=cfg.aggregation.strategy
                )
                
                return RunResult(
                    config=cfg,
                    metrics=metrics_dict,
                    agg_result=agg_result,
                    y_pred=y_pred,
                    y_std=y_std,
                    y_true=y_true,
                    train_time_s=0.0,  # Loaded from disk
                    model_summary=summary_text,
                    experiment_name=cfg.name,
                )
            except Exception as e:
                log.warning(f"[Runner] Failed to load saved results for '{cfg.name}': {e}. Re-running training...")

        log.info(f"\n{'#'*60}")
        log.info(f"  RUNNING: {cfg.name}")
        log.info(f"{'#'*60}")

        # ── Train ─────────────────────────────────────────────────────────────
        t0    = time.time()
        model = MultiAxisGPEnsemble(cfg)
        model.fit(dataset)
        elapsed = time.time() - t0

        # ── Predict ───────────────────────────────────────────────────────────
        agg_result = model.predict(dataset.X_test)

        # Inverse-scale predictions back to original target space
        mu_scaled    = agg_result.mu
        sigma_scaled = agg_result.sigma

        # Inverse-transform mean and uncertainty
        y_pred_orig = dataset.scaler_y.inverse_transform(
            mu_scaled.reshape(-1, 1)
        ).ravel()

        # Uncertainty: approximate by scaling std through target scaler
        y_std_orig = dataset.scaler_y.std_to_original(sigma_scaled)

        y_true = dataset.y_test   # already in original scale

        # ── Metrics ───────────────────────────────────────────────────────────
        metrics_dict = evaluate(
            y_true  = y_true,
            mu      = y_pred_orig,
            sigma   = y_std_orig,
            metrics = cfg.metrics.track,
        )

        log.info(f"\n[{cfg.name}] Metrics:")
        log.info(format_metrics(metrics_dict))
        log.info(f"[{cfg.name}] Train time: {elapsed:.1f}s")

        # ── Save ──────────────────────────────────────────────────────────────
        result = RunResult(
            config=cfg,
            metrics=metrics_dict,
            agg_result=agg_result,
            y_pred=y_pred_orig,
            y_std=y_std_orig,
            y_true=y_true,
            train_time_s=elapsed,
            model_summary=model.summary(),
            experiment_name=cfg.name,
        )

        if cfg.save_results:
            self._save_result(result, cfg)

        return result

    # ── Sweep ─────────────────────────────────────────────────────────────────

    def sweep(self, configs: list[ExperimentConfig]) -> list[RunResult]:
        """Run a list of configs sequentially and return all results."""
        import gc
        import torch
        results = []
        for cfg in configs:
            try:
                r = self.run(cfg)
                results.append(r)
            except Exception as e:
                log.error(f"[Runner] Config '{cfg.name}' failed: {e}", exc_info=True)
            finally:
                # Force cleanup of GPU memory between sweep configurations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    def grid_sweep(self, grid_spec: GridSpec) -> list[RunResult]:
        """
        Enumerate all combinations in grid_spec.flat_grid and run each.

        Example
        -------
            grid_spec = GridSpec(flat_grid={
                "aggregation.strategy":        ["rbcm", "gpoe", "hard"],
                "kernels.q2.composite_kernel": ["sm_x_matern52", "matern52"],
            })
            results = runner.grid_sweep(grid_spec)
        """
        base = grid_spec.base_config or self.base_cfg
        keys = list(grid_spec.flat_grid.keys())
        vals = list(grid_spec.flat_grid.values())

        configs = []
        for combo in itertools.product(*vals):
            cfg  = copy.deepcopy(base)
            name_parts = []
            for k, v in zip(keys, combo):
                _set_dotted(cfg, k, v)
                name_parts.append(f"{k.split('.')[-1]}={v}")
            cfg.name = "__".join(name_parts)
            configs.append(cfg)

        log.info(f"[Runner] Grid sweep: {len(configs)} configurations")
        return self.sweep(configs)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_leaderboard(
        self,
        results:         list[RunResult],
        primary_metric:  str = "mae",
        top_k:           int | None = None,
    ) -> pd.DataFrame:
        """
        Print a sorted leaderboard of all experiment results.
        Returns a DataFrame for further analysis.
        """
        rows = [r.to_row() for r in results]
        df   = pd.DataFrame(rows)

        if primary_metric in df.columns:
            df = df.sort_values(primary_metric)

        if top_k is not None:
            df = df.head(top_k)

        print("\n" + "=" * 80)
        print("  EXPERIMENT LEADERBOARD")
        print("=" * 80)
        print(df.to_string(index=False, float_format="{:.6f}".format))
        print("=" * 80 + "\n")

        return df

    def save_leaderboard(
        self, results: list[RunResult], path: str = "./results/leaderboard.csv"
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_row() for r in results]
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info(f"[Runner] Leaderboard saved → {path}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_result(self, result: RunResult, cfg: ExperimentConfig) -> None:
        out_dir = Path(cfg.results_dir) / cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Config
        cfg.save(out_dir / "config.json")

        # Metrics
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result.metrics, f, indent=2)

        # Predictions
        np.save(out_dir / "y_pred.npy",   result.y_pred)
        np.save(out_dir / "y_std.npy",    result.y_std)
        np.save(out_dir / "y_true.npy",   result.y_true)

        # Summary text
        (out_dir / "model_summary.txt").write_text(result.model_summary)

        log.info(f"[Runner] Results saved → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: set a dotted-path key in a nested dataclass/dict
# ─────────────────────────────────────────────────────────────────────────────

def _set_dotted(obj: Any, dotted_key: str, value: Any) -> None:
    """
    Set obj.a.b.c = value for dotted_key="a.b.c".
    Works with dataclasses and dicts.
    """
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)
