"""
run_sweep_data_driven.py
========================
Data-driven kernel sweep — kernels chosen from Hölder-exponent analysis.

Key findings that drive every kernel choice:
  - Cross-section is NOWHERE-DIFFERENTIABLE at grid scale on all axes (α≈0.5)
    → replace all Matérn-5/2 secondaries with Matérn-1/2
  - Q2 shows a 6× variance jump across its range (var_ratio=6.07)
    → Gibbs kernel (non-stationary ℓ) on the Q2 dimension of every expert
  - z is a weak signal carrier (40× less variance than others)
    → keep z-expert but use Matérn-3/2 secondary (within z-slabs function IS smoother)
  - No non-stationarity in W2, β, z dimensions themselves
    → Gibbs only on the Q2 dim (primary_dim=0 for W2/β/z experts)

Kernel config summary
---------------------
  Q2-expert  sees (W2, β, z):  ARD Matérn-1/2 on all 3  (matern12)
  W2-expert  sees (Q2, β, z):  Gibbs-sigmoid × Matérn-1/2  (gibbs_sigmoid_x_matern12)
  β-expert   sees (Q2, W2, z): Gibbs-sigmoid × Matérn-1/2  (gibbs_sigmoid_x_matern12)
  z-expert   sees (Q2, W2, β): Gibbs-sigmoid × Matérn-3/2  (gibbs_sigmoid_x_matern32)

All experts:  learn_weights=False, rBCM aggregation (Option A)
Option B:     KNN nearest-neighbour routing

Usage
-----
    python run_sweep_data_driven.py --csv data/your_data.csv
    python run_sweep_data_driven.py --csv data/your_data.csv --train-frac 0.05 --epochs 100
    python run_sweep_data_driven.py --csv data/your_data.csv --skip-option-b
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_sweep_data_driven")


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def make_data_driven_config(
    csv_path: str,
    train_frac: float,
    epochs: int,
    max_noise: float | None,
    gibbs_slope: float,
) -> "ExperimentConfig":
    from gp4d_lab.config import (
        ExperimentConfig, DataConfig, AxisKernelConfig,
        AggregationConfig, TrainingConfig, MetricConfig,
    )

    return ExperimentConfig(
        name="data_driven_matern12",
        description=(
            "Hölder-exponent-calibrated kernels. "
            "All secondaries Matérn-1/2 (α≈0.5, non-differentiable CS). "
            "Gibbs-sigmoid on Q2 dimension for every expert (6× variance ratio). "
            "z-expert uses Matérn-3/2 secondary (smoother within z-slabs). "
            "Floor targets fully retained — no outlier filtering."
        ),
        tags=["data-driven", "matern12", "gibbs-sigmoid", "hoelder"],

        data=DataConfig(
            csv_path=csv_path,
            train_fraction=train_frac,
            input_scaler="standard",  # Standardize features to zero mean and unit variance
            target_scaler="robust",
            filter_target_outliers=False,   # floor = kinematic boundary, not noise
        ),

        kernels={
            # ── Q2-expert sees (W2, β, z) ─────────────────────────────────────
            # Hölder exponents: W2 α=0.485, β α≈0.14, z α≈nugget
            # None of these 3 dims shows non-stationarity → no Gibbs needed
            # All are at α≈0 (nearly nugget) → Matérn-1/2 everywhere
            "q2": AxisKernelConfig(
                axis="q2",
                composite_kernel="matern12",   # full ARD Matérn-1/2 on all 3 dims
                use_ard=True,
                use_scale_kernel=True,
            ),

            # ── W2-expert sees (Q2, β, z) ─────────────────────────────────────
            # Q2: Hölder α=0.609 + var_ratio=6.07  → Gibbs-sigmoid (non-stationary)
            # β: α≈0.25 (within W2-slabs)          → Matérn-1/2 secondary
            # z: nugget behaviour                    → Matérn-1/2 secondary
            # Gibbs transition at Q2=x0, slope steeper than default (6× jump)
            "w2": AxisKernelConfig(
                axis="w2",
                composite_kernel="gibbs_sigmoid_x_matern12",
                gibbs_variant="sigmoid",
                gibbs_slope_init=gibbs_slope,   # 3.0 → steeper than default 2.0
                gibbs_x0_init=0.0,
                use_ard=True,
                use_scale_kernel=True,
            ),

            # ── β-expert sees (Q2, W2, z) ─────────────────────────────────────
            # Same Q2 non-stationarity → Gibbs-sigmoid on Q2 (primary_dim=0)
            # W2: α≈0.13 (within β-slabs) → Matérn-1/2
            # z: nugget → Matérn-1/2
            "beta": AxisKernelConfig(
                axis="beta",
                composite_kernel="gibbs_sigmoid_x_matern12",
                gibbs_variant="sigmoid",
                gibbs_slope_init=gibbs_slope,
                gibbs_x0_init=0.0,
                use_ard=True,
                use_scale_kernel=True,
            ),

            # ── z-expert sees (Q2, W2, β) ────────────────────────────────────
            # Q2: non-stationary → Gibbs-sigmoid (primary_dim=0)
            # W2: α=0.82 (within z-slabs) — smoother than other experts
            # β: α=0.93 (within z-slabs)  — nearly Matérn-3/2 territory
            # Use Matérn-3/2 secondary (not 1/2) because conditioning on z
            # substantially smooths out W2 and β
            "z": AxisKernelConfig(
                axis="z",
                composite_kernel="gibbs_sigmoid_x_matern32",
                gibbs_variant="sigmoid",
                gibbs_slope_init=gibbs_slope,
                gibbs_x0_init=0.0,
                use_ard=True,
                use_scale_kernel=True,
            ),
        },

        aggregation=AggregationConfig(
            strategy="rbcm",
            learn_weights=False,         # Option A: pure entropy gating, no learned weights
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
            # ── Matérn-1/2 conditioning fix ───────────────────────────────────
            # Matérn-1/2 (exponential kernel) is inherently rougher than Matérn-5/2.
            # With n=3600 and the old min_lengthscale=0.05, nearby points get K≈1
            # (near-duplicates) or distant points get K≈0 — both cause ill-conditioning.
            # 0.10 is the empirically safe lower bound at this grid spacing.
            min_lengthscale=0.10,    # do NOT tighten below 0.10 for Matérn-1/2
            max_lengthscale=10.0,
            # A 10× larger baseline jitter means Strategies 0 & 1 handle the
            # PSD issue internally, avoiding the slow float64 (Strategy 2) fallback.
            cholesky_jitter=1e-3,    # was 1e-4 — rougher kernel needs more regularisation
            # More noise floor is appropriate: Matérn-1/2 has less implicit smoothing
            # than Matérn-5/2, so the likelihood noise absorbs more of the roughness.
            min_noise=1e-3,          # was 1e-4
            max_noise=max_noise,
            predict_chunk_size=5000,
        ),

        metrics=MetricConfig(
            track=["mae", "rmse", "nlpd", "crps", "coverage_90", "coverage_68",
                   "mean_uncertainty", "uncertainty_calibration_error"],
            primary_metric="mae",
        ),

        save_results=True,
        results_dir="./results_data_driven",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_original(mu_scaled: np.ndarray, sigma_scaled: np.ndarray, scaler_y):
    mu    = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
    sigma = scaler_y.std_to_original(sigma_scaled)
    return mu, sigma


def _log_metrics(label: str, m: dict) -> None:
    log.info(f"\n  ── {label} ──")
    for k, v in m.items():
        log.info(f"    {k:<40s}: {v:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Data-driven Matérn-1/2 GP sweep (Hölder-calibrated kernels)"
    )
    p.add_argument("--csv",           required=True,          help="Path to CSV data file")
    p.add_argument("--train-frac",    type=float, default=0.10)
    p.add_argument("--epochs",        type=int,   default=1000)
    p.add_argument("--max-noise",     type=float, default=None,
                   help="Noise upper bound (None=unconstrained)")
    p.add_argument("--gibbs-slope",   type=float, default=3.0,
                   help="Gibbs-sigmoid initial slope (3.0 = steeper, matches 6× var_ratio)")
    p.add_argument("--floor-thresh",  type=float, default=-17.955,
                   help="logA threshold separating physical from floor points")
    p.add_argument("--n-bins",        type=int,   default=5,
                   help="Bins per axis for competence lookup table")
    p.add_argument("--knn-k",         type=int,   default=5,
                   help="k neighbours for Option B NN routing")
    p.add_argument("--val-frac",      type=float, default=0.10,
                   help="Fraction of training set for NN routing calibration")
    p.add_argument("--skip-option-b", action="store_true",
                   help="Skip Option B (NN routing) — useful when GPU is busy")
    p.add_argument("--skip-competence", action="store_true",
                   help="Skip competence table — saves time on large datasets")
    args = p.parse_args()

    log.info("=" * 72)
    log.info("  DATA-DRIVEN KERNEL SWEEP  (Hölder-exponent calibrated)")
    log.info("=" * 72)
    log.info(f"  CSV           : {args.csv}")
    log.info(f"  Train frac    : {args.train_frac}")
    log.info(f"  Epochs        : {args.epochs}")
    log.info(f"  Gibbs slope   : {args.gibbs_slope}  (steeper → sharper Q2 transition)")
    log.info(f"  Max noise     : {args.max_noise}  (None = unconstrained)")
    log.info(f"  Floor thresh  : {args.floor_thresh}")
    log.info("")
    log.info("  Kernel config:")
    log.info("    q2-expert  → matern12                 (ARD Matérn-1/2, all 3 dims)")
    log.info("    w2-expert  → gibbs_sigmoid_x_matern12 (Gibbs-Q2, Matérn-1/2 rest)")
    log.info("    β-expert   → gibbs_sigmoid_x_matern12 (Gibbs-Q2, Matérn-1/2 rest)")
    log.info("    z-expert   → gibbs_sigmoid_x_matern32 (Gibbs-Q2, Matérn-3/2 rest)")
    log.info("=" * 72)

    # ── Imports ───────────────────────────────────────────────────────────────
    from gp4d_lab.data import load_dataset
    from gp4d_lab.ensemble import MultiAxisGPEnsemble
    from gp4d_lab.metrics import (
        evaluate, print_stratified_report, uncertainty_decomposition,
    )

    # ── Load dataset ──────────────────────────────────────────────────────────
    cfg     = make_data_driven_config(
        args.csv, args.train_frac, args.epochs, args.max_noise, args.gibbs_slope
    )
    dataset = load_dataset(cfg.data)
    scaler_y = dataset.scaler_y
    y_test   = dataset.y_test   # original-scale

    # ── Train / load (Option A: rBCM) ─────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  TRAINING  (Option A: rBCM, pure entropy gating)")
    log.info("="*60)

    t0      = time.time()
    model   = MultiAxisGPEnsemble(cfg)
    model.fit(dataset)
    log.info(f"\n[Sweep] Training complete ({time.time()-t0:.1f}s)")

    # Print model summary
    log.info("\n" + model.summary())

    # ── Expert diagnostics on a small slice ───────────────────────────────────
    # Use y_test_scaled (correct ground truth) not y_train
    diag_n = min(2000, len(dataset.X_test))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    log.info(f"\n[Sweep] Expert diagnostics on {diag_n} test points...")
    model.print_expert_diagnostics(
        dataset.X_test[:diag_n],
        y_test_scaled[:diag_n],
        scaler_y,
    )

    # ── Option A: predict full test set ───────────────────────────────────────
    log.info("\n[Sweep] Option A — full test set prediction...")
    result_a    = model.predict(dataset.X_test)
    mu_a, sig_a = to_original(result_a.mu, result_a.sigma, scaler_y)

    # All-points leaderboard
    log.info("\n" + "="*60)
    log.info("  OPTION A — ALL-POINTS LEADERBOARD")
    log.info("="*60)
    m_a = evaluate(y_test, mu_a, sig_a)
    _log_metrics("Option A (all points)", m_a)

    # Stratified report (the number that counts for physics quality)
    log.info("\n[Sweep] Option A — Stratified report")
    print_stratified_report(y_test, mu_a, sig_a, floor_threshold=args.floor_thresh)

    # Uncertainty decomposition
    log.info("\n[Sweep] Option A — Uncertainty decomposition")
    ud_a = uncertainty_decomposition(y_test, mu_a, sig_a)
    for k, v in ud_a.items():
        log.info(f"    {k:<40s}: {v}")

    # ── Competence lookup table (reuses expert_means/sigmas from result_a) ─────
    if not args.skip_competence:
        comp_n = min(5000, len(dataset.X_test))
        log.info(f"\n[Sweep] Building regional competence table ({comp_n} points — reusing cached predictions)...")
        # regional_competence_map calls _predict_one_expert internally.
        # Pass the first comp_n rows of result_a.expert_means directly.
        # Since print_competence_table re-runs inference, we wrap it in a
        # short call on just comp_n points to keep it cheap.
        model.print_competence_table(
            dataset.X_test[:comp_n],
            mu_a[:comp_n],      # already original-scale — pass directly
            scaler_y,
            n_bins=args.n_bins,
        )

    # ── Option B: NN routing ──────────────────────────────────────────────────
    if not args.skip_option_b:
        log.info("\n" + "="*60)
        log.info("  OPTION B: Nearest-Neighbour Expert Routing")
        log.info("="*60)

        N_train = len(dataset.X_train)
        n_val   = max(100, int(N_train * args.val_frac))
        rng     = np.random.default_rng(42)
        val_idx = rng.choice(N_train, size=n_val, replace=False)

        X_val       = dataset.X_train[val_idx]
        y_val_scaled = dataset.y_train[val_idx]
        y_val_orig  = scaler_y.inverse_transform(
            y_val_scaled.reshape(-1, 1)
        ).ravel()

        log.info(f"[Sweep] KNN routing on {n_val} val points (k={args.knn_k})...")
        log.info("[Sweep] Reusing cached expert predictions from Option A for test set.")
        t1 = time.time()
        mu_b_s, sig_b_s = model.nn_routing_predict(
            X_val, y_val_orig, dataset.X_test, scaler_y, k=args.knn_k,
            cached_test_expert_preds=(result_a.expert_means, result_a.expert_sigmas),
        )
        mu_b, sig_b = to_original(mu_b_s, sig_b_s, scaler_y)
        log.info(f"[Sweep] Option B done ({time.time()-t1:.1f}s)")

        log.info("\n" + "="*60)
        log.info("  OPTION B — ALL-POINTS LEADERBOARD")
        log.info("="*60)
        m_b = evaluate(y_test, mu_b, sig_b)
        _log_metrics("Option B (all points)", m_b)

        log.info("\n[Sweep] Option B — Stratified report")
        print_stratified_report(y_test, mu_b, sig_b, floor_threshold=args.floor_thresh)

        # ── Side-by-side on physical regime only ─────────────────────────────
        phys   = y_test >= args.floor_thresh
        log.info("\n" + "="*72)
        log.info("  OPTION A vs OPTION B  — PHYSICAL REGIME ONLY (logA ≥ {:.3f})".format(
            args.floor_thresh
        ))
        log.info("="*72)

        from scipy.stats import norm as _norm
        from gp4d_lab.metrics import mae as _mae, rmse as _rmse

        log.info(f"  {'':12s}  {'MAE':>10s}  {'RMSE':>10s}  {'Coverage_90':>12s}  {'Mean σ':>10s}")
        log.info("  " + "-" * 58)
        for label, mu_v, sig_v in [
            ("Option A", mu_a[phys], sig_a[phys]),
            ("Option B", mu_b[phys], sig_b[phys]),
        ]:
            z90   = _norm.ppf(0.95)
            cov90 = float(np.mean(
                (y_test[phys] >= mu_v - z90*sig_v) &
                (y_test[phys] <= mu_v + z90*sig_v)
            ))
            log.info(
                f"  {label:12s}  {_mae(y_test[phys], mu_v):10.4f}"
                f"  {_rmse(y_test[phys], mu_v):10.4f}"
                f"  {cov90:12.4f}"
                f"  {sig_v.mean():10.4f}"
            )
        log.info("  " + "-" * 58)

    else:
        log.info("\n[Sweep] Skipping Option B (--skip-option-b).")

    log.info("\n[Sweep] Done.")


if __name__ == "__main__":
    main()
