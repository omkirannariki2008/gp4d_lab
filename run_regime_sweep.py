"""
run_regime_sweep.py
===================
Runner pipeline for the regime-based GP expert system.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_regime_sweep")

# ─── Load library components ──────────────────────────────────────────────────
from gp4d_lab.config import DataConfig, MASTER_DEFAULT
from gp4d_lab.data import load_dataset
from gp4d_lab.regime_experts import RegimeGPEnsemble, FLOOR
from gp4d_lab.metrics import evaluate, print_stratified_report

def main():
    p = argparse.ArgumentParser(description="Regime-Based GP Sweep")
    p.add_argument("--csv",           required=True,          help="Path to CSV data")
    p.add_argument("--train-frac",    type=float, default=0.10)
    p.add_argument("--epochs",        type=int,   default=500)
    p.add_argument("--checkpoint-dir", type=str,   default="./results_regime/models")
    p.add_argument("--max-cholesky",  type=int,   default=2000, help="Max points for Exact GP")
    args = p.parse_args()

    log.info("=" * 72)
    log.info("  REGIME-BASED GP EXPERT SWEEP")
    log.info("=" * 72)
    log.info(f"  CSV           : {args.csv}")
    log.info(f"  Train frac    : {args.train_frac}")
    log.info(f"  Epochs        : {args.epochs}")
    log.info(f"  Max Cholesky  : {args.max_cholesky}")
    log.info("=" * 72)

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_cfg = DataConfig(
        csv_path=args.csv,
        train_fraction=args.train_frac,
        input_scaler="standard",
        target_scaler="robust",
        filter_target_outliers=False, # keepfloor points
    )
    dataset = load_dataset(data_cfg)
    scaler_y = dataset.scaler_y
    y_test   = dataset.y_test

    # 1. Create Raw Dataframes for detector and label assignment
    df_train_raw = pd.DataFrame(dataset.raw_X_train, columns=data_cfg.feature_cols)
    df_train_raw[data_cfg.target_col] = scaler_y.inverse_transform(
        dataset.y_train.reshape(-1, 1)
    ).ravel()

    df_test_raw = pd.DataFrame(dataset.raw_X_test, columns=data_cfg.feature_cols)
    df_test_raw[data_cfg.target_col] = y_test

    # 2. Create Scaled Dataframes for GP training and evaluation
    df_train_scaled = pd.DataFrame(dataset.X_train, columns=data_cfg.feature_cols)
    df_train_scaled[data_cfg.target_col] = dataset.y_train

    df_test_scaled = pd.DataFrame(dataset.X_test, columns=data_cfg.feature_cols)
    
    ensemble = RegimeGPEnsemble(
        feature_cols=data_cfg.feature_cols,
        target_col=data_cfg.target_col,
        max_cholesky_size=args.max_cholesky
    )

    checkpoint_path = Path(args.checkpoint_dir)
    loaded = False
    if checkpoint_path.exists():
        log.info(f"[Sweep] Checkpoint folder {checkpoint_path} found. Attempting to load models...")
        # Since loaders need to fit exact GPs structure, we pass raw and scaled df
        loaded = ensemble.load_models(checkpoint_path, df_train_raw, df_train_scaled)
        if loaded:
            log.info("[Sweep] Checkpoint loaded successfully. Skipping training.")

    if not loaded:
        log.info("[Sweep] No checkpoints loaded. Starting training...")
        # Override config epoch count for training experts
        from gp4d_lab.regime_experts import RegimeExpertConfig
        
        t0 = time.time()
        # Train ensemble using raw and scaled dataframes
        ensemble.fit(df_train_raw, df_train_scaled, val_fraction=0.10)
        log.info(f"[Sweep] Training complete in {time.time()-t0:.1f}s")
        
        # Save checkpoints
        ensemble.save_models(checkpoint_path)

    # ── Prediction ────────────────────────────────────────────────────────────
    log.info("\n[Sweep] Evaluating ensemble predictions on test set...")
    t1 = time.time()
    mu_scaled, sigma_scaled = ensemble.predict(df_test_raw, df_test_scaled)
    log.info(f"[Sweep] Prediction complete in {time.time()-t1:.1f}s")

    # Inverse scale predictions back to original physical target scale
    mu = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
    sigma = scaler_y.std_to_original(sigma_scaled)

    # ── Regime-by-Regime Metrics Report ────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  REGIME SWEEP — REGIME BREAKDOWN REPORT")
    log.info("="*60)
    
    labels_test = ensemble.detector.assign(df_test_raw)
    df_results = pd.DataFrame({
        "regime": labels_test,
        "y_true": y_test,
        "y_pred": mu,
        "sigma": sigma
    })

    REGIME_DESCS = {
        "CPR": "Central Physical — Pomeron + DGLAP interior",
        "ELR": "Elastic Limit — β→1 kinematic boundary",
        "DSR": "DGLAP Saturation — high-Q², W² primary",
        "PHR": "Photoproduction — very low Q², β+z primary",
        "TRN": "Transition — Q² boundary, smooth interior",
    }

    for regime in ["CPR", "ELR", "DSR", "PHR", "TRN"]:
        sub = df_results[df_results["regime"] == regime]
        n_reg = len(sub)
        if n_reg == 0:
            continue
        
        mae_reg = np.mean(np.abs(sub["y_true"] - sub["y_pred"]))
        rmse_reg = np.sqrt(np.mean((sub["y_true"] - sub["y_pred"])**2))
        
        # Coverage 90%
        from scipy.stats import norm
        z90 = norm.ppf(0.95)
        cov90 = np.mean(
            (sub["y_true"] >= sub["y_pred"] - z90 * sub["sigma"]) & 
            (sub["y_true"] <= sub["y_pred"] + z90 * sub["sigma"])
        )
        
        strategy = ensemble.metadata.get(regime, {}).get("strategy", "unknown")
        
        desc = REGIME_DESCS.get(regime, "Unknown Regime")
        log.info(f"  Regime [{regime}] ({desc}):")
        log.info(f"    n_test   : {n_reg} ({100*n_reg/len(df_results):.1f}%)")
        log.info(f"    strategy : {strategy}")
        log.info(f"    MAE      : {mae_reg:.4f}")
        log.info(f"    RMSE     : {rmse_reg:.4f}")
        log.info(f"    cov90    : {cov90:.4f}")
        log.info(f"    mean_sig : {sub['sigma'].mean():.4f}")
        log.info("-" * 40)

    # ── Leaderboard and Stratified Report ──────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  REGIME SWEEP — ALL-POINTS LEADERBOARD")
    log.info("="*60)
    m = evaluate(y_test, mu, sigma)
    for k, v in m.items():
        log.info(f"    {k:<40s}: {v:.6f}")

    log.info("\n[Sweep] Regime Sweep — Stratified report")
    print_stratified_report(y_test, mu, sigma, floor_threshold=FLOOR)
    
    log.info("\n[Sweep] Done.")

if __name__ == "__main__":
    main()
