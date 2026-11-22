#!/usr/bin/env python3
"""
run_must_try.py
================
Script to run the physics-bias custom configuration on the full dataset.
- Train split: 10%
- Validation/Test split: 90%
- Dataset: /home/omkiran/kishore/gp4d_lab/data/inclusive_A1__bSat_STU_q0_bin0-1079999_LQQ.csv
- Epochs: 200
- Composite kernels as justified by HERA physics:
  - q2: rbf_x_matern52 (RBF on W2: Pomeron slope near-constant inside q2 bin)
  - w2: gibbs_sigmoid_x_matern52 (GibbsSigmoid on Q2: abrupt soft->hard transition)
  - beta: gibbs_sigmoid_x_matern52 (GibbsSigmoid on Q2: same reasoning as w2)
  - z: matern52 (Matérn-5/2: twice differentiable robust default)
- Aggregation: Learned weights (NLL minimization over 20% validation split)
"""

import sys
from pathlib import Path
import time
import logging

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Initialize gp4d_lab which configures logging handlers
import gp4d_lab
from gp4d_lab.config import make_master_default
from gp4d_lab.experiment import ExperimentRunner

log = logging.getLogger("gp4d_lab.run_must_try")

def main():
    log.info("Starting must-try configuration experiment script...")
    
    csv_path = "/home/omkiran/kishore/gp4d_lab/data/inclusive_A1__bSat_STU_q0_bin0-1079999_LQQ.csv"
    if not Path(csv_path).exists():
        log.error(f"Dataset path {csv_path} does not exist!")
        sys.exit(1)
        
    # Initialize base config with default settings
    cfg = make_master_default(csv_path)
    
    # ── Experiment Metadata ──────────────────────────────────────────────────
    cfg.name = "must_try_physics_bias"
    cfg.description = (
        "HERA physics-motivated custom config: rbf_x_matern52 (q2), "
        "gibbs_sigmoid_x_matern52 (w2, beta), matern52 (z), with learned aggregation weights."
    )
    
    # ── Data Split ────────────────────────────────────────────────────────────
    # 10% train, 90% test (validation)
    cfg.data.train_fraction = 0.10
    cfg.data.shuffle_before_split = True
    cfg.data.input_scaler = "standard"  # Standardize features to zero mean and unit variance
    cfg.data.target_scaler = "robust"  # Standardize target to zero mean and unit variance

    # Outlier filtering on training set
    cfg.data.filter_target_outliers = True
    cfg.data.target_clip_lo         = -17.955   # median - 3*IQR
    cfg.data.target_clip_hi         = 28.626    # median + 3*IQR
    
    # ── Kernels ──────────────────────────────────────────────────────────────
    # Q2-expert sees (W2, β, z). RBF on W2: Pomeron slope ε is approx constant in Q2 bins.
    cfg.kernels["q2"].composite_kernel = "rbf_x_matern52"
    
    # W2-expert sees (Q2, β, z). GibbsSigmoid on Q2: encodes non-stationary soft->hard transition.
    cfg.kernels["w2"].composite_kernel = "gibbs_sigmoid_x_matern52"
    
    # β-expert sees (Q2, W2, z). GibbsSigmoid on Q2: same reasoning as W2.
    cfg.kernels["beta"].composite_kernel = "gibbs_sigmoid_x_matern52"
    
    # z-expert sees (Q2, W2, β). Matérn-5/2 ARD: robust default.
    cfg.kernels["z"].composite_kernel = "matern52"
    
    # ── Aggregation ──────────────────────────────────────────────────────────
    cfg.aggregation.strategy = "learned"
    cfg.aggregation.learn_weights = True
    cfg.aggregation.weight_epochs = 300       # 300 epochs cheap training on NLL
    cfg.aggregation.weight_val_frac = 0.20      # 20% validation split of training data
    cfg.aggregation.weight_lr = 0.02
    
    # ── Training ─────────────────────────────────────────────────────────────
    cfg.training.n_epochs            = 500
    cfg.training.early_stop_patience = 30
    cfg.training.early_stop_delta    = 1e-3
    cfg.training.device = "auto"
    cfg.training.predict_chunk_size = 5000     # chunking size for predictions

    # Lengthscale constraints
    cfg.training.constrain_lengthscale = True
    cfg.training.min_lengthscale       = 0.10
    cfg.training.max_lengthscale       = 10.0
    cfg.training.max_noise             = 0.50
    
    # Save results to ./results
    cfg.save_results = True
    cfg.results_dir = "./results_musttry"
    
    log.info("Configured Experiment Options:")
    log.info(f"  Name: {cfg.name}")
    log.info(f"  Train fraction: {cfg.data.train_fraction:.2f}")
    log.info(f"  Training epochs: {cfg.training.n_epochs}")
    log.info(f"  Device: {cfg.training.device}")
    log.info(f"  q2 composite kernel: {cfg.kernels['q2'].composite_kernel}")
    log.info(f"  w2 composite kernel: {cfg.kernels['w2'].composite_kernel}")
    log.info(f"  beta composite kernel: {cfg.kernels['beta'].composite_kernel}")
    log.info(f"  z composite kernel: {cfg.kernels['z'].composite_kernel}")
    log.info(f"  Aggregation strategy: {cfg.aggregation.strategy}")
    log.info(f"  Aggregation weight epochs: {cfg.aggregation.weight_epochs}")
    
    # ── Run Experiment ────────────────────────────────────────────────────────
    runner = ExperimentRunner(csv_path)
    
    log.info("Starting ExperimentRunner...")
    t0 = time.time()
    try:
        result = runner.run(cfg)
        elapsed = time.time() - t0
        log.info(f"Experiment completed successfully in {elapsed:.1f}s.")
        
        # Display primary metrics
        log.info("\n" + "=" * 50)
        log.info("EXPERIMENT COMPLETE: Leaderboard Metrics")
        log.info("=" * 50)
        for metric, val in result.metrics.items():
            log.info(f"  {metric:<30}: {val:.6f}")
        log.info("=" * 50)
        
        # ── Print Expert Diagnostics ──────────────────────────────────────────
        log.info("\nEvaluating per-expert diagnostics...")
        from gp4d_lab.ensemble import MultiAxisGPEnsemble
        ensemble = MultiAxisGPEnsemble(cfg)
        checkpoint_dir = Path(cfg.results_dir) / cfg.name / "models"
        dataset = runner._ensure_dataset(cfg)
        if ensemble.load_models(checkpoint_dir, dataset):
            y_val_scaled = dataset.scaler_y.transform(dataset.y_test.reshape(-1, 1)).ravel()
            ensemble.print_expert_diagnostics(dataset.X_test[:10000], y_val_scaled[:10000], dataset.scaler_y)
            
        # ── Print Uncertainty Decomposition and Calibration ───────────────────
        from gp4d_lab.metrics import uncertainty_decomposition, calibrated_metrics
        
        log.info("\n" + "=" * 50)
        log.info("UNCERTAINTY DECOMPOSITION & POST-HOC CALIBRATION")
        log.info("=" * 50)
        
        decomp = uncertainty_decomposition(result.y_true, result.y_pred, result.y_std)
        log.info(f"  Bias:                    {decomp['bias']:.6f}")
        log.info(f"  Residual Std:            {decomp['residual_std']:.6f}")
        log.info(f"  Mean Uncertainty (raw):  {decomp['mean_sigma']:.6f}")
        log.info(f"  Ratio σ / Residual:      {decomp['ratio_sigma_to_residual']:.6f}")
        log.info(f"  Calibration Temp T:      {decomp['calibration_T']:.6f}")
        log.info(f"  Interpretation:          {decomp['interpretation']}")
        
        cal_m, T, _ = calibrated_metrics(result.y_true, result.y_pred, result.y_std)
        log.info("-" * 50)
        log.info(f"  Calibrated Temp T:       {T:.6f}")
        log.info(f"  Calibrated Mean σ:       {result.y_std.mean() / T:.6f}")
        log.info(f"  Calibrated 90% Coverage: {cal_m['coverage_90']:.6f}")
        log.info(f"  Calibrated 68% Coverage: {cal_m['coverage_68']:.6f}")
        log.info(f"  Calibrated UCE:          {cal_m['uncertainty_calibration_error']:.6f}")
        log.info("=" * 50)
        
    except Exception as e:
        log.error(f"Experiment failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
