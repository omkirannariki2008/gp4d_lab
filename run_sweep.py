"""
run_sweep.py
============
Physics-informed GP sweep with:

  - NO outlier filtering  (floor points are kinematic boundary, not noise)
  - Per-axis physics kernels (Matérn-3/2 × Matérn-1/2, ChangePoint, Gibbs-sigmoid)
  - Option A : rBCM aggregation (learn_weights=False — pure entropy gating)
  - Option B : Nearest-Neighbour expert routing (KNN-competence)
  - Stratified evaluation : physical regime vs floor/zero-CS region
  - Competence lookup table : shows WHERE each expert is best
  - Per-region temperature calibration for post-hoc uncertainty tuning

Usage
-----
    python run_sweep.py --csv data/your_data.csv
    python run_sweep.py --csv data/your_data.csv --train-frac 0.10 --epochs 150
    python run_sweep.py --csv data/your_data.csv --skip-option-b  # faster
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("run_sweep")


# ─────────────────────────────────────────────────────────────────────────────
# Build config
# ─────────────────────────────────────────────────────────────────────────────

def make_sweep_config(csv_path: str, train_frac: float, epochs: int, max_noise: float | None):
    from gp4d_lab.config import (
        ExperimentConfig, DataConfig, AxisKernelConfig,
        AggregationConfig, TrainingConfig, MetricConfig,
    )

    return ExperimentConfig(
        name="physics_sweep_v2",
        description=(
            "Floor-inclusive sweep: Matérn-3/2×Matérn-1/2 (Q2), "
            "ChangePoint×Matérn-5/2 (W2, handles W²-dip), "
            "Gibbs-sigmoid×Matérn-1/2 (beta, smooth boundary), "
            "Matérn-3/2 (z). "
            "rBCM aggregation (no learned weights — pure entropy gating). "
            "Floor targets retained for physical correctness."
        ),
        tags=["sweep", "v2", "floor-inclusive", "changepoint", "rbcm"],

        data=DataConfig(
            csv_path=csv_path,
            train_fraction=train_frac,
            input_scaler="standard",
            target_scaler="robust",
            # ── KEY: do NOT filter floor values ──────────────────────────────
            filter_target_outliers=False,
        ),

        # ── Per-axis kernels (physics-motivated) ─────────────────────────────
        #
        #  Q2 expert sees (W2, β, z):
        #    Matérn-3/2 on W2 dim  — moderately rough for multi-Pomeron structure
        #    Matérn-1/2 on (β, z)  — roughest valid kernel; handles rapid CS drops
        #
        #  W2 expert sees (Q2, β, z):
        #    ChangePoint on Q2 dim — transitions between two Matérn-1/2 regimes
        #    at the dip location; physically motivated by the W²-dip structure
        #    Matérn-5/2 on (β, z)  — smooth marginal structure
        #
        #  beta expert sees (Q2, W2, z):
        #    Gibbs-sigmoid on Q2 — ℓ shrinks as Q2 increases (DGLAP threshold)
        #    Matérn-1/2 on (W2, z) — very rough; cross-section can jump sharply
        #
        #  z expert sees (Q2, W2, β):
        #    Full ARD Matérn-3/2 — rapidity gap distribution is smooth but not too smooth
        kernels={
            "q2": AxisKernelConfig(
                axis="q2",
                composite_kernel="matern32_x_matern12",
                use_ard=True,
                use_scale_kernel=True,
            ),
            "w2": AxisKernelConfig(
                axis="w2",
                composite_kernel="changepoint_x_matern52",
                use_ard=True,
                use_scale_kernel=True,
            ),
            "beta": AxisKernelConfig(
                axis="beta",
                composite_kernel="gibbs_sigmoid_x_matern12",
                gibbs_variant="sigmoid",
                gibbs_slope_init=2.0,
                gibbs_x0_init=0.0,
                use_ard=True,
                use_scale_kernel=True,
            ),
            "z": AxisKernelConfig(
                axis="z",
                composite_kernel="matern32",
                use_ard=True,
                use_scale_kernel=True,
            ),
        },

        aggregation=AggregationConfig(
            strategy="rbcm",
            # Option A: pure rBCM differential-entropy gating (no learned weights)
            learn_weights=False,
            rbcm_beta_floor=1e-4,
            prior_var_scale=1.0,
            axes_enabled=["q2", "w2", "beta", "z"],
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
            # Floor included → do NOT cap noise — let GP choose it freely
            max_noise=max_noise,
            predict_chunk_size=5000,
        ),

        metrics=MetricConfig(
            track=["mae", "rmse", "nlpd", "crps", "coverage_90", "coverage_68",
                   "mean_uncertainty", "uncertainty_calibration_error"],
            primary_metric="mae",
        ),

        save_results=True,
        results_dir="./results_sweep",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper — inverse-scale from robust scaler
# ─────────────────────────────────────────────────────────────────────────────

def to_original(mu_scaled: np.ndarray, sigma_scaled: np.ndarray, scaler_y):
    mu    = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
    sigma = scaler_y.std_to_original(sigma_scaled)
    return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Physics-informed GP sweep v2")
    p.add_argument("--csv",          required=True,        help="Path to CSV data file")
    p.add_argument("--train-frac",   type=float, default=0.10, help="Training fraction")
    p.add_argument("--epochs",       type=int,   default=500,  help="Training epochs per partition")
    p.add_argument("--max-noise",    type=float, default=None, help="Noise upper bound (None=free)")
    p.add_argument("--floor-thresh", type=float, default=-17.955,
                   help="logA threshold below which points are 'floor' (kinematic boundary)")
    p.add_argument("--n-bins",       type=int,   default=5,
                   help="Number of bins per axis for the competence lookup table")
    p.add_argument("--knn-k",        type=int,   default=5,
                   help="k neighbours for Option B (NN routing)")
    p.add_argument("--val-frac",     type=float, default=0.10,
                   help="Fraction of training set used as routing validation (Option B)")
    p.add_argument("--skip-option-b", action="store_true",
                   help="Skip Option B (NN routing) — faster, useful for debugging")
    p.add_argument("--skip-competence", action="store_true",
                   help="Skip the competence table (saves time on large datasets)")
    args = p.parse_args()

    log.info("=" * 70)
    log.info("  PHYSICS GP SWEEP  v2  (floor-inclusive, ChangePoint kernel)")
    log.info("=" * 70)
    log.info(f"  CSV         : {args.csv}")
    log.info(f"  Train frac  : {args.train_frac}")
    log.info(f"  Epochs      : {args.epochs}")
    log.info(f"  Max noise   : {args.max_noise} (None = unconstrained)")
    log.info(f"  Floor thresh: {args.floor_thresh}")
    log.info(f"  KNN k       : {args.knn_k}")
    log.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────────────
    from gp4d_lab.data import load_dataset
    from gp4d_lab.ensemble import MultiAxisGPEnsemble
    from gp4d_lab.metrics import (
        evaluate, print_stratified_report, uncertainty_decomposition
    )

    cfg     = make_sweep_config(args.csv, args.train_frac, args.epochs, args.max_noise)
    dataset = load_dataset(cfg.data)

    scaler_y = dataset.scaler_y
    y_test   = dataset.y_test           # original-scale test targets

    # ── Train / load Option A model ───────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  OPTION A: rBCM (pure entropy gating, no learned weights)")
    log.info("="*60)

    t0      = time.time()
    model_a = MultiAxisGPEnsemble(cfg)
    model_a.fit(dataset)
    log.info(f"\n[Sweep] Option A training complete ({time.time()-t0:.1f}s)")

    # ── Expert diagnostics ────────────────────────────────────────────────────
    log.info("\n[Sweep] Running expert diagnostics on 5000 test points...")
    diag_x = dataset.X_test[:5000]
    diag_y = dataset.y_test[:5000]
    model_a.print_expert_diagnostics(diag_x, dataset.y_train[:5000], scaler_y)

    # ── Option A: full test prediction ────────────────────────────────────────
    log.info("\n[Sweep] Option A — predicting on full test set...")
    result_a     = model_a.predict(dataset.X_test)
    mu_a, sig_a  = to_original(result_a.mu, result_a.sigma, scaler_y)

    # ── Option A: metrics ─────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  OPTION A — LEADERBOARD METRICS (all points)")
    log.info("="*60)
    m_a = evaluate(y_test, mu_a, sig_a)
    for k, v in m_a.items():
        log.info(f"    {k:<40s}: {v:.6f}")

    # Stratified report
    log.info("\n[Sweep] Option A — Stratified Report")
    print_stratified_report(y_test, mu_a, sig_a, floor_threshold=args.floor_thresh)

    # Uncertainty decomposition
    log.info("\n[Sweep] Option A — Uncertainty Decomposition")
    ud_a = uncertainty_decomposition(y_test, mu_a, sig_a)
    for k, v in ud_a.items():
        log.info(f"    {k:<40s}: {v}")

    # ── Competence lookup table (Option A predictions) ─────────────────────────
    if not args.skip_competence:
        log.info("\n[Sweep] Building regional competence lookup table...")
        model_a.print_competence_table(
            dataset.X_test[:5000], y_test[:5000], scaler_y,
            n_bins=args.n_bins
        )

    # ── Option B: NN routing ──────────────────────────────────────────────────
    if not args.skip_option_b:
        log.info("\n" + "="*60)
        log.info("  OPTION B: Nearest-Neighbour Expert Routing")
        log.info("="*60)

        # Use a slice of training set as routing validation
        N_train = len(dataset.X_train)
        n_val   = max(100, int(N_train * args.val_frac))
        rng     = np.random.default_rng(42)
        val_idx = rng.choice(N_train, size=n_val, replace=False)

        X_val  = dataset.X_train[val_idx]
        # Routing validation labels must be in original scale
        y_val_scaled = dataset.y_train[val_idx]
        y_val_orig   = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()

        log.info(f"[Sweep] Training KNN router on {n_val} val points (k={args.knn_k})...")
        t1           = time.time()
        mu_b_s, sig_b_s = model_a.nn_routing_predict(
            X_val, y_val_orig, dataset.X_test, scaler_y, k=args.knn_k
        )
        mu_b, sig_b  = to_original(mu_b_s, sig_b_s, scaler_y)
        log.info(f"[Sweep] Option B routing complete ({time.time()-t1:.1f}s)")

        # Option B metrics
        log.info("\n" + "="*60)
        log.info("  OPTION B — LEADERBOARD METRICS (all points)")
        log.info("="*60)
        m_b = evaluate(y_test, mu_b, sig_b)
        for k, v in m_b.items():
            log.info(f"    {k:<40s}: {v:.6f}")

        # Stratified report
        log.info("\n[Sweep] Option B — Stratified Report")
        print_stratified_report(y_test, mu_b, sig_b, floor_threshold=args.floor_thresh)

        # ── Side-by-side comparison ───────────────────────────────────────────
        log.info("\n" + "="*70)
        log.info("  OPTION A vs OPTION B — COMPARISON (physical regime only)")
        log.info("="*70)
        phys_mask = y_test >= args.floor_thresh
        from gp4d_lab.metrics import mae, rmse
        log.info(
            f"  {'':10s}  {'MAE':>10s}  {'RMSE':>10s}  {'Coverage_90':>12s}"
        )
        log.info("  " + "-"*48)
        for label, mu_v, sig_v in [
            ("Option A",  mu_a[phys_mask],  sig_a[phys_mask]),
            ("Option B",  mu_b[phys_mask],  sig_b[phys_mask]),
        ]:
            from scipy.stats import norm
            z90  = norm.ppf(0.95)
            cov90 = float(np.mean(
                (y_test[phys_mask] >= mu_v - z90*sig_v) &
                (y_test[phys_mask] <= mu_v + z90*sig_v)
            ))
            log.info(
                f"  {label:10s}  {mae(y_test[phys_mask], mu_v):10.4f}"
                f"  {rmse(y_test[phys_mask], mu_v):10.4f}"
                f"  {cov90:12.4f}"
            )
        log.info("  " + "-"*48)
    else:
        log.info("\n[Sweep] Skipping Option B (--skip-option-b set).")

    log.info("\n[Sweep] Done.")


if __name__ == "__main__":
    main()
