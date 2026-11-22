"""
run_configurable_gp.py
======================
Configurable run script that trains/loads a partitioned GP ensemble on the Q2 axis,
allowing the user to switch between Matérn-5/2, 1D Gibbs MLP, and multi-dimensional Gibbs
kernels, with or without Hurdle classification gating.
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
log = logging.getLogger("run_configurable_gp")

from gp4d_lab.config import (
    ExperimentConfig, DataConfig, AxisKernelConfig,
    AggregationConfig, TrainingConfig, MetricConfig,
)
from gp4d_lab.data import load_dataset
from gp4d_lab.ensemble import MultiAxisGPEnsemble
from gp4d_lab.metrics import evaluate, print_stratified_report


def make_config(
    csv_path: str,
    train_frac: float,
    epochs: int,
    results_dir: str,
    kernel_name: str = "gibbs_multidim",
    use_hurdle: bool = True,
    classifier_type: str = "hist_gb",
    target_scaler: str = "conditional_robust",
    floor_val: float = -17.955,
    fixed_noise: float = 1e-4,
    use_physics_mean: bool = True,
    dkl_latent_dim: int = 2,
    dkl_hidden_dim: int = 16,
    name: str = "q2_configurable_gp"
) -> ExperimentConfig:
    """Create a configurable GP ensemble config based on command line arguments."""
    
    # Map the kernel name to the appropriate composite kernel config
    if kernel_name == "matern52":
        composite = "matern52"
    elif kernel_name == "gibbs_mlp":
        composite = "gibbs_mlp_x_matern52"
    elif kernel_name == "gibbs_multidim":
        composite = "gibbs_multidim"
    elif kernel_name == "dkl_matern52":
        composite = "dkl_matern52"
    elif kernel_name == "dkl_gibbs":
        composite = "dkl_gibbs"
    else:
        raise ValueError(f"Unknown kernel name: {kernel_name}")

    return ExperimentConfig(
        name=name,
        description=f"Configurable Partitioned GP on Q2 axis ({kernel_name}, hurdle={use_hurdle})",
        tags=["q2", kernel_name, "hurdle" if use_hurdle else "no_hurdle", "partitioned"],
        
        data=DataConfig(
            csv_path=csv_path,
            train_fraction=train_frac,
            input_scaler="standard",
            target_scaler=target_scaler,
            filter_target_outliers=False,  # Keep floor points to train Hurdle (if enabled)
            use_hurdle_model=use_hurdle,
            hurdle_floor_val=floor_val,
            hurdle_classifier_type=classifier_type,
        ),
        
        kernels={
            "q2": AxisKernelConfig(
                axis="q2",
                composite_kernel=composite,
                gibbs_mlp_hidden_dim=32 if "multidim" in kernel_name or "gibbs" in kernel_name else 16,
                dkl_latent_dim=dkl_latent_dim,
                dkl_hidden_dim=dkl_hidden_dim,
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
            fixed_noise=fixed_noise if fixed_noise > 0.0 else None,
            use_physics_mean=use_physics_mean,
            predict_chunk_size=5000,
            verbose=True,
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
    parser = argparse.ArgumentParser(description="Configurable Q2 Partitioned GP Runner")
    parser.add_argument("--csv",          required=True,        help="Path to CSV data file")
    parser.add_argument("--train-frac",   type=float, default=0.10, help="Training fraction")
    parser.add_argument("--epochs",       type=int,   default=200,  help="Training epochs per partition")
    parser.add_argument("--results-dir",  type=str,   default="./results_configurable", help="Directory to save/load models and metrics")
    parser.add_argument("--floor-thresh", type=float, default=-17.955, help="Physical floor threshold")
    parser.add_argument("--kernel",       type=str,   default="gibbs_multidim", choices=["matern52", "gibbs_mlp", "gibbs_multidim", "dkl_matern52", "dkl_gibbs"], help="GP covariance kernel")
    parser.add_argument("--use-hurdle",   action="store_true",  help="Enable Hurdle classifier gating")
    parser.add_argument("--no-hurdle",    action="store_false", dest="use_hurdle", help="Disable Hurdle classifier gating")
    parser.set_defaults(use_hurdle=True)
    parser.add_argument("--clf-type",     type=str,   default="hist_gb", choices=["hist_gb", "xgboost", "rf"], help="Hurdle classifier algorithm")
    parser.add_argument("--target-scaler", type=str,   default="conditional_robust", choices=["conditional_robust", "robust", "standard", "yeo_johnson"], help="Target scaling method")
    parser.add_argument("--fixed-noise",   type=float, default=1e-4, help="Fixed likelihood noise floor (0.0 to disable)")
    parser.add_argument("--no-physics-mean", action="store_true", help="Disable parametric physics-informed mean function")
    parser.add_argument("--dkl-latent-dim", type=int,   default=2,  help="DKL latent feature space dimension")
    parser.add_argument("--dkl-hidden-dim", type=int,   default=16, help="DKL network hidden layer size")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  CONFIGURABLE GPSurrogate RUNNER")
    log.info("=" * 70)
    log.info(f"  CSV         : {args.csv}")
    log.info(f"  Train fraction  : {args.train_frac}")
    log.info(f"  Epochs      : {args.epochs}")
    log.info(f"  Results Dir : {args.results_dir}")
    log.info(f"  Kernel      : {args.kernel}")
    log.info(f"  Hurdle Gating: {args.use_hurdle}")
    log.info(f"  Classifier  : {args.clf_type}")
    log.info(f"  TargetScale : {args.target_scaler}")
    log.info(f"  Fixed Noise : {args.fixed_noise}")
    log.info(f"  Physics Mean: {not args.no_physics_mean}")
    if "dkl" in args.kernel:
        log.info(f"  DKL Latent  : {args.dkl_latent_dim}")
        log.info(f"  DKL Hidden  : {args.dkl_hidden_dim}")
    log.info("=" * 70)

    # 1. Initialize configuration and load dataset
    name_suffix = f"{args.kernel}_{'hurdle' if args.use_hurdle else 'nohurdle'}"
    cfg = make_config(
        csv_path=args.csv,
        train_frac=args.train_frac,
        epochs=args.epochs,
        results_dir=args.results_dir,
        kernel_name=args.kernel,
        use_hurdle=args.use_hurdle,
        classifier_type=args.clf_type,
        target_scaler=args.target_scaler,
        floor_val=args.floor_thresh,
        fixed_noise=args.fixed_noise,
        use_physics_mean=not args.no_physics_mean,
        dkl_latent_dim=args.dkl_latent_dim,
        dkl_hidden_dim=args.dkl_hidden_dim,
        name=f"q2_config_{name_suffix}"
    )
    
    log.info("[Data] Loading dataset...")
    dataset = load_dataset(cfg.data)
    scaler_y = dataset.scaler_y
    y_test = dataset.y_test

    # 2. Build the MultiAxisGPEnsemble
    model = MultiAxisGPEnsemble(cfg)
    
    # 3. Fit
    log.info("[Fit] Fitting ensemble...")
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
    
    # Save metrics to JSON for automated experiment tracking
    from gp4d_lab.metrics import evaluate_stratified
    import json
    metrics_strat = evaluate_stratified(y_test, mu, sig, floor_threshold=args.floor_thresh)
    output_metrics = {
        "all": metrics_all,
        "physical": metrics_strat.get("physical", {}),
        "floor": metrics_strat.get("floor", {}),
        "floor_detected_frac": metrics_strat.get("floor_detected_frac", 0.0),
    }
    metrics_json_path = Path(args.results_dir) / cfg.name / "metrics.json"
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, "w") as f:
        json.dump(output_metrics, f, indent=4)
    log.info(f"[Metrics] Saved JSON metrics to {metrics_json_path}")
    
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

    prediction_filename = f"predictions_{name_suffix}.csv"
    paths_to_save = [
        Path(args.results_dir) / cfg.name / "predictions.csv",
        Path(args.results_dir) / "predictions" / prediction_filename,
        Path(args.results_dir) / prediction_filename,
    ]
    for p in paths_to_save:
        p.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"[Predictions] Saving predictions to {p}...")
        df_preds.to_csv(p, index=False)
        log.info(f"[Predictions] Saved predictions to {p} successfully.")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
