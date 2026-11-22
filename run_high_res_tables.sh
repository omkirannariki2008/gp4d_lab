#!/bin/bash
# ==============================================================================
# run_high_res_tables.sh
# ==============================================================================
# Run card script to train the champion MultiAxis DKL GP model.
# Defaults to 1 epoch for testing, run with `./run_high_res_tables.sh 400`
# for the full training run.
# ==============================================================================

set -e

# Number of epochs: default to 1 for test, or read from first argument
EPOCHS=${1:-1}

# Output parent directory
RESULTS_DIR="./results_high_res"
mkdir -p "$RESULTS_DIR"

echo "=== Sartre Inclusive-Diffraction Lookup Table Training Run Card ==="
echo "Model config: 4-Axis BCM DKL-GP Ensemble"
echo "Kernel: DKL + Matérn-5/2"
echo "Training Epochs: ${EPOCHS}"
echo "Train Fraction: 10% (stratified)"
echo "Floor Gating: Hurdle Classifier with Banded Routing (floor threshold: -87.336548)"
echo "=================================================================="

TABLES=(
  "inclusive_A1__bSat_STU_q0_bin0-1079999_LQQ"
  "inclusive_A1__bSat_STU_q0_bin0-1079999_TQQ"
  "inclusive_A1__bSat_STU_q0_bin0-1079999_TQQG"
  "inclusive_A1__bSat_STU_q1_bin0-1079999_LQQ"
  "inclusive_A1__bSat_STU_q1_bin0-1079999_TQQ"
  "inclusive_A1__bSat_STU_q1_bin0-1079999_TQQG"
  "inclusive_A1__bSat_STU_q2_bin0-1079999_LQQ"
  "inclusive_A1__bSat_STU_q2_bin0-1079999_TQQ"
  "inclusive_A1__bSat_STU_q2_bin0-1079999_TQQG"
  "inclusive_A1__bSat_STU_q3_bin0-1079999_LQQ"
  "inclusive_A1__bSat_STU_q3_bin0-1079999_TQQ"
  "inclusive_A1__bSat_STU_q3_bin0-1079999_TQQG"
)

for TABLE in "${TABLES[@]}"; do
  CSV_PATH="Tables/csv/high_res/${TABLE}.csv"
  OUT_PATH="${RESULTS_DIR}/${TABLE}"
  
  echo ""
  echo "------------------------------------------------------------"
  echo "Starting Training for: ${TABLE}"
  echo "------------------------------------------------------------"
  
  .venv/bin/python run_configurable_gp.py \
    --csv "${CSV_PATH}" \
    --train-frac 0.10 \
    --epochs "${EPOCHS}" \
    --results-dir "${OUT_PATH}" \
    --kernel dkl_matern52 \
    --use-hurdle \
    --clf-type hist_gb \
    --target-scaler conditional_robust \
    --fixed-noise 1e-4 \
    --dkl-latent-dim 3 \
    --dkl-hidden-dim 16 \
    --floor-thresh -87.336548
done

echo ""
echo "=== All Tables Completed Successfully! ==="
