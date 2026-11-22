"""
run_q2_matern52_expert.py
=========================
Custom run script that trains/loads a partitioned GP ensemble on the Q2 axis only,
using the Matérn-5/2 covariance kernel.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_q2_matern52")

from gp4d_lab.config import (
    ExperimentConfig, DataConfig, AxisKernelConfig,
    AggregationConfig, TrainingConfig, MetricConfig,
)
from gp4d_lab.data import load_dataset
from gp4d_lab.ensemble import MultiAxisGPEnsemble
from gp4d_lab.metrics import evaluate, print_stratified_report


def make_q2_matern52_config(
    csv_path: str,
    train_frac: float,
    epochs: int,
    results_dir: str,
    name: str = "q2_partitioned_matern52"
) -> ExperimentConfig:
    """Create a config to train ONLY the Q2 partitioned GP expert using Matérn-5/2."""
    return ExperimentConfig(
        name=name,
        description="Partitioned GP on Q2 axis using Matérn-5/2 kernel",
        tags=["q2", "matern52", "partitioned"],
        
        data=DataConfig(
            csv_path=csv_path,
            train_fraction=train_frac,
            input_scaler="standard",
            target_scaler="robust",
            filter_target_outliers=False,  # Keep floor points in dataset
        ),
        
        # Configure Q2 with matern52 kernel
        kernels={
            "q2": AxisKernelConfig(
                axis="q2",
                composite_kernel="matern52",
                use_ard=True,
                use_scale_kernel=True,
            ),
        },
        
        # Enable ONLY the q2 axis
        aggregation=AggregationConfig(
            strategy="rbcm",
            learn_weights=False,
            rbcm_beta_floor=1e-4,
            prior_var_scale=1.0,
            axes_enabled=["q2"],
        ),
        
        training=TrainingConfig(
            n_epochs=epochs,
            lr=0.1,
            device="auto",
            cg_tol=1e-3,
            max_cg_iter=1000,
            oom_fallback="ski",
            early_stop_patience=25,
            early_stop_delta=1e-4,
            constrain_lengthscale=True,
            min_lengthscale=0.10,
            max_lengthscale=10.0,
            max_noise=None,            # Free noise
            predict_chunk_size=5000,
            verbose=True,              # Ensure GPyTorch epoch logs are printed
        ),
        
        metrics=MetricConfig(
            track=["mae", "rmse", "nlpd", "crps", "coverage_90", "coverage_68",
                   "mean_uncertainty", "uncertainty_calibration_error"],
            primary_metric="mae",
        ),
        
        save_results=True,
        results_dir=results_dir,
    )


def to_original(mu_scaled: np.ndarray, sigma_scaled: np.ndarray, scaler_y):
    mu = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
    sigma = scaler_y.std_to_original(sigma_scaled)
    return mu, sigma


def main():
    parser = argparse.ArgumentParser(description="Q2 Partitioned GP Matérn-5/2 Runner")
    parser.add_argument("--csv",          required=True,        help="Path to CSV data file")
    parser.add_argument("--train-frac",   type=float, default=0.10, help="Training fraction")
    parser.add_argument("--epochs",       type=int,   default=500,  help="Training epochs per partition")
    parser.add_argument("--results-dir",  type=str,   default="./results_q2_matern52", help="Directory to save/load models and metrics")
    parser.add_argument("--floor-thresh", type=float, default=-17.955, help="Physical floor threshold")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  Q2 PARTITIONED GP RUNNER (Matérn-5/2 Kernel)")
    log.info("=" * 70)
    log.info(f"  CSV         : {args.csv}")
    log.info(f"  Train frac  : {args.train_frac}")
    log.info(f"  Epochs      : {args.epochs}")
    log.info(f"  Results Dir : {args.results_dir}")
    log.info(f"  Floor Thresh: {args.floor_thresh}")
    log.info("=" * 70)

    # 1. Initialize configuration and load dataset
    cfg = make_q2_matern52_config(
        csv_path=args.csv,
        train_frac=args.train_frac,
        epochs=args.epochs,
        results_dir=args.results_dir
    )
    
    log.info("[Data] Loading dataset...")
    dataset = load_dataset(cfg.data)
    scaler_y = dataset.scaler_y
    y_test = dataset.y_test

    # 2. Build the MultiAxisGPEnsemble
    model = MultiAxisGPEnsemble(cfg)
    
    # 3. Fit (which automatically loads from checkpoint if it exists, otherwise trains and saves)
    log.info("[Fit] Fitting ensemble (attempts to load from checkpoint first)...")
    t0 = time.time()
    model.fit(dataset)
    log.info(f"[Fit] Fit operation completed in {time.time() - t0:.1f}s")

    # 4. Predict on the test set
    log.info("[Predict] Predicting on full test set...")
    t1 = time.time()
    result = model.predict(dataset.X_test)
    mu, sig = to_original(result.mu, result.sigma, scaler_y)
    log.info(f"[Predict] Prediction complete in {time.time() - t1:.1f}s")

    # 5. Evaluate and report results
    log.info("\n" + "=" * 60)
    log.info("  EVALUATION LEADERBOARD (All Points)")
    log.info("=" * 60)
    metrics_all = evaluate(y_test, mu, sig)
    for k, v in metrics_all.items():
        log.info(f"    {k:<40s}: {v:.6f}")

    log.info("\n[Evaluation] Stratified Report")
    print_stratified_report(y_test, mu, sig, floor_threshold=args.floor_thresh)
    
    log.info("\nSummary:")
    log.info(model.summary())
    
    # ── Save predictions ──────────────────────────────────────────────────────
    log.info("\n[Predictions] Creating predictions DataFrame...")
    df_preds = pd.DataFrame(dataset.raw_X_test, columns=cfg.data.feature_cols)
    df_preds[cfg.data.target_col] = y_test

    df_preds["pred_logA"] = mu
    df_preds["pred_std_logA"] = sig

    df_preds["pred_A_exp"] = np.exp(mu)
    df_preds["pred_A_exp_low"] = np.exp(mu - 1.96 * sig)
    df_preds["pred_A_exp_high"] = np.exp(mu + 1.96 * sig)

    df_preds["pred_A_10"] = 10**mu
    df_preds["pred_A_10_low"] = 10**(mu - 1.96 * sig)
    df_preds["pred_A_10_high"] = 10**(mu + 1.96 * sig)

    if cfg.data.target_col in df_preds.columns:
        df_preds["A_true_exp"] = np.exp(df_preds[cfg.data.target_col])
        df_preds["A_true_10"] = 10**df_preds[cfg.data.target_col]

    paths_to_save = [
        Path(args.results_dir) / cfg.name / "predictions.csv",
        Path(args.results_dir) / "predictions" / "predictions.csv",
        Path(args.results_dir) / "predictions.csv",
        Path("/home/omkiran/kishore/gp4d_lab/predictions") / f"predictions_{Path(args.results_dir).name}.csv",
    ]
    for p in paths_to_save:
        p.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"[Predictions] Saving predictions to {p}...")
        df_preds.to_csv(p, index=False)
        log.info(f"[Predictions] Saved predictions to {p} successfully.")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
