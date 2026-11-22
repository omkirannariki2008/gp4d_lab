# gp4d_lab

**4-Axis Partitioned GP Ensemble for Inclusive Diffraction Cross-Section Prediction**

A modular, physics-informed Gaussian Process library built on [GPyTorch](https://gpytorch.ai),
designed for the high-energy physics problem of predicting log(σ) from (Q², W², β, ζ).

---

## Table of Contents

1. [Background](#background)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration Reference](#configuration-reference)
6. [Kernels](#kernels)
7. [Aggregation Strategies](#aggregation-strategies)
8. [Experiment Runner & Grid Sweep](#experiment-runner--grid-sweep)
9. [Metrics](#metrics)
10. [Numerical Stability (Grid Data)](#numerical-stability-grid-data)
11. [File Reference](#file-reference)
12. [Known Issues & Fixes](#known-issues--fixes)

---

## Background

This library targets **inclusive diffractive DIS** cross-section prediction where:

| Variable | Description | Range |
|---|---|---|
| `Q2_log_center` | log(Q²) — virtual photon virtuality | ~[-1, 4.6] |
| `W2_log_center` | log(W²) — γ*p centre-of-mass energy² | ~[0, 9.9] |
| `beta_center` | β — Bjorken-like momentum fraction | (0, 1) |
| `z_center` | ζ — rapidity gap variable | (0, 1) |
| `logA` | **target**: log(cross-section) | continuous |

The data is a **flattened 4D grid** (e.g., 100k rows = product of discrete bin values),
which creates numerical challenges (near-singular kernel matrices) that this library
handles explicitly.

---

## Architecture

```
MultiAxisGPEnsemble
│
├── Expert_Q2  : Q2-partitioned  → trains on (W2, β, ζ) per Q2 bin
│               Kernel: SpectralMixture(W2) × Matérn52(β, ζ)   ← multi-scale Pomeron
│
├── Expert_W2  : W2-partitioned  → trains on (Q2, β, ζ) per W2 bin
│               Kernel: GibbsPower(Q2) × Matérn52(β, ζ)         ← perturbative transition
│
├── Expert_β   : β-partitioned   → trains on (Q2, W2, ζ) per β bin
│               Kernel: GibbsSigmoid(Q2) × Matérn52(W2, ζ)      ← sharp structure at β→1
│
└── Expert_ζ   : z-partitioned   → trains on (Q2, W2, β) per ζ bin
                Kernel: Matérn52 ARD                              ← smooth rapidity axis

                         ↓  ↓  ↓  ↓
              rBCM / gPoE / Learned aggregation
                         ↓
                  (μ_agg, σ_agg)
```

Each expert partitions on its axis variable and exposes `(mean, std)` for every test point.
The **Aggregator** combines all 4 experts using precision-weighted combination.

---

## Installation

To install the library and its dependencies, run the following command from the root of the project directory:

```bash
pip install -e .
```

Alternatively, you can install the dependencies manually and add the package to your Python path:

```bash
pip install torch gpytorch numpy pandas scikit-learn scipy tqdm
export PYTHONPATH="/path/to/parent_of_gp4d_lab:$PYTHONPATH"
```

---

## Quick Start

### One-liner

```python
from gp4d_lab import run_experiment

result = run_experiment("your_data.csv")

print(f"MAE: {result.metrics['mae']:.4f}")
print(f"Coverage 90%: {result.metrics['coverage_90']:.3f}")
```

### Standard workflow

```python
from gp4d_lab.config import make_master_default
from gp4d_lab.data import load_dataset
from gp4d_lab.ensemble import MultiAxisGPEnsemble
from gp4d_lab.metrics import evaluate, format_metrics

# 1. Config
cfg = make_master_default("your_data.csv")
cfg.training.n_epochs = 150
cfg.aggregation.strategy = "rbcm"

# 2. Load data (10% train, stratified by Q2 bin)
dataset = load_dataset(cfg.data)

# 3. Train
model = MultiAxisGPEnsemble(cfg)
model.fit(dataset)

# 4. Predict
result = model.predict(dataset.X_test)

# 5. Inverse-scale and evaluate
y_pred = dataset.scaler_y.inverse_transform(result.mu.reshape(-1, 1)).ravel()
y_std  = dataset.scaler_y.std_to_original(result.sigma)

metrics = evaluate(dataset.y_test, y_pred, y_std)
print(format_metrics(metrics))
print(model.summary())
```

### Patching one setting

```python
cfg = make_master_default("data.csv")
cfg.aggregation.strategy = "learned"               # change strategy
cfg.kernels["w2"].composite_kernel = "matern52"    # override one kernel
cfg.training.n_epochs = 200
```

---

## Configuration Reference

All configuration lives in `config.py` as Python dataclasses — fully JSON-serialisable.

```python
from gp4d_lab.config import ExperimentConfig, DataConfig, TrainingConfig

cfg = ExperimentConfig(
    name="my_run",
    data=DataConfig(
        csv_path="data.csv",
        train_fraction=0.10,     # 10% of data used for training
        input_scaler="physics",  # "physics"|"standard"|"robust"|"quantile"|"none"
        target_scaler="robust",  # "robust"|"standard"|"none"
    ),
    training=TrainingConfig(
        n_epochs=150,
        lr=0.1,
        device="auto",            # "cpu"|"cuda"|"auto"
        # ── Grid-data stability ───────────────────
        max_cholesky_size=2000,  # use Cholesky for n ≤ this (not CG)
        cholesky_jitter=1e-4,    # diagonal jitter for numerical safety
        min_noise=1e-4,          # minimum likelihood noise (prevents K singularity)
        use_double=False,        # float64 for extreme ill-conditioning
    ),
)
```

### Save / load config

```python
cfg.save("run_config.json")
cfg2 = ExperimentConfig.load("run_config.json")
```

### Pre-built config factories

| Function | Description |
|---|---|
| `make_master_default(csv)` | Physics-informed kernels + rBCM (recommended) |
| `make_baseline_config(csv)` | All Matérn-5/2 + rBCM (comparison baseline) |
| `make_all_sm_config(csv)` | All SpectralMixture + rBCM |
| `make_aggregation_sweep_configs(csv)` | 6 configs × aggregation strategy |
| `make_kernel_sweep_configs(csv)` | 12 configs × per-axis kernel choice |

---

## Kernels

### Composite kernel names

| Name | Applied to dim 0 | Applied to dims 1,2 | Best axis |
|---|---|---|---|
| `sm_x_matern52` | SpectralMixture | Matérn-5/2 ARD | Q2 expert (W2 dim) |
| `gibbs_power_x_matern52` | Gibbs (power-law ℓ) | Matérn-5/2 ARD | W2 expert (Q2 dim) |
| `gibbs_sigmoid_x_matern52` | Gibbs (sigmoid ℓ) | Matérn-5/2 ARD | β expert (Q2 dim) |
| `matern52_x_sm` | Matérn-5/2 | SpectralMixture | any |
| `matern52` | Matérn-5/2 ARD | (same) | all — baseline |
| `spectral_mixture` | SpectralMixture ARD | (same) | all — alternative |
| `rbf` | RBF | (same) | reference only |

### Gibbs Kernel (custom)

The `GibbsKernel1D` is a **non-stationary** kernel where the correlation
length-scale is a learnable function of the input:

```
k(x, x') = sqrt(2ℓ(x)ℓ(x') / (ℓ(x)² + ℓ(x')²)) · exp(-(x-x')² / (ℓ(x)²+ℓ(x')²))
```

Three variants:

| Variant | ℓ(x) formula | Physics motivation |
|---|---|---|
| `power_law` | `ℓ₀ · exp(α·x)` | Q²: ℓ shrinks as Q² grows (perturbative transition) |
| `sigmoid` | `ℓ_min + Δ · σ(-slope·(x-x₀))` | β: ℓ shortens near β→1 (sharp elastic peak) |
| `linear` | `softplus(a + b·x)` | general monotone variation |

```python
from gp4d_lab.kernels import GibbsKernel1D

k = GibbsKernel1D(variant="power_law", alpha_init=-0.5)
k = GibbsKernel1D(variant="sigmoid", slope_init=2.0, x0_init=0.0)
```

---

## Aggregation Strategies

At prediction time, all 4 axis-experts produce `(μᵢ, σᵢ)`.
The Aggregator combines them via precision-weighted fusion:

```
σ_agg⁻² = Σᵢ βᵢ · σᵢ⁻²    (+ prior correction for rBCM)
μ_agg    = σ_agg² · Σᵢ βᵢ · σᵢ⁻² · μᵢ
```

| Strategy | βᵢ | Notes |
|---|---|---|
| `hard` | 1 for argmin σᵢ, 0 else | Winner-takes-all — fast but inconsistent |
| `poe` | 1 | Product-of-Experts — can be overconfident |
| `gpoe` | 1/M | Generalised PoE — more conservative |
| `precision` | σᵢ⁻² / Σσᵢ⁻² | Normalised precision — simple and robust |
| `rbcm` | 0.5·(log σ*² - log σᵢ²) | Robust BCM — **recommended** |
| `learned` | softmax(w), trained by NLL | Data-driven — adapts to which expert is best |

### Learned weights

```python
cfg.aggregation.strategy      = "rbcm"   # base strategy
cfg.aggregation.learn_weights = True     # override with learned on top
cfg.aggregation.weight_epochs = 200
cfg.aggregation.weight_lr     = 0.05
cfg.aggregation.weight_val_frac = 0.20  # 20% of train set used for weight training
```

After training, inspect with:

```python
print(model._aggregator.learned_weights)
# e.g. [0.61, 0.21, 0.12, 0.06] → Q2 expert is most trusted
```

---

## Experiment Runner & Grid Sweep

### Single run with saved results

```python
from gp4d_lab.experiment import ExperimentRunner

runner = ExperimentRunner("data.csv")
result = runner.run(cfg)
# Saved to cfg.results_dir/cfg.name/:
#   config.json, metrics.json, y_pred.npy, y_std.npy, model_summary.txt
```

### Grid sweep

```python
from gp4d_lab.config import GridSpec

grid = GridSpec(flat_grid={
    "aggregation.strategy":        ["rbcm", "gpoe", "precision", "learned"],
    "kernels.q2.composite_kernel": ["sm_x_matern52", "matern52"],
    "kernels.w2.composite_kernel": ["gibbs_power_x_matern52", "matern52"],
})
results = runner.grid_sweep(grid)
df = runner.print_leaderboard(results, primary_metric="mae")
runner.save_leaderboard(results, "results/leaderboard.csv")
```

### Using pre-built sweep configs

```python
from gp4d_lab.config import make_aggregation_sweep_configs, make_kernel_sweep_configs

# Compare all 6 aggregation strategies (same kernels)
results = runner.sweep(make_aggregation_sweep_configs("data.csv"))

# Test per-axis kernel choices (fixed rBCM aggregation)
results = runner.sweep(make_kernel_sweep_configs("data.csv"))
```

---

## Metrics

All metrics operate in the **original (unscaled)** target space.
Remember to inverse-transform predictions before calling `evaluate`.

| Metric | Description | Lower is better |
|---|---|---|
| `mae` | Mean Absolute Error | ✓ |
| `rmse` | Root Mean Squared Error | ✓ |
| `mape` | Mean Absolute % Error | ✓ |
| `nlpd` | Negative Log Predictive Density | ✓ |
| `crps` | Continuous Ranked Probability Score | ✓ |
| `coverage_90` | Empirical coverage at 90% CI | → should be ~0.90 |
| `coverage_68` | Empirical coverage at 68% CI | → should be ~0.68 |
| `mean_uncertainty` | Mean predictive std | informational |
| `uncertainty_calibration_error` | Mean \|empirical − nominal\| coverage | ✓ |

```python
from gp4d_lab.metrics import evaluate, format_metrics

m = evaluate(y_true, y_pred, y_std,
             metrics=["mae", "rmse", "nlpd", "crps", "coverage_90"])
print(format_metrics(m))
```

---

## Numerical Stability (Grid Data)

A flattened 4D physics table has many **near-duplicate points** within each GP bin
(e.g., partitioning on Q² leaves a 4×4×4 grid of identical (W², β, ζ) coordinates
across multiple bins). This causes two GPyTorch warnings:

### Warning 1: Duplicate parameters in optimizer

**Cause**: `ExactGP` registers `likelihood` as a submodule, so `model.parameters()`
already includes `likelihood.parameters()`. Adding them again doubles every parameter.

**Fix (already applied in `models.py`)**:
```python
# WRONG (old code):
optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), ...)

# CORRECT:
seen = set()
unique_params = [p for p in model.parameters()
                 if not (id(p) in seen or seen.add(id(p)))]
optimizer = torch.optim.Adam(unique_params, lr=cfg.lr)
```

### Warning 2: CG terminated with residual above tolerance

**Cause**: Near-identical input points make the kernel matrix ill-conditioned.
As noise → 0, the system becomes near-singular and Conjugate Gradient fails to converge.

**Fix (already applied)**:

```python
# In TrainingConfig — use these settings for grid/duplicate data:
cfg.training.max_cholesky_size = 2000   # switch to Cholesky instead of CG for small bins
cfg.training.cholesky_jitter   = 1e-4   # add diagonal jitter for numerical safety
cfg.training.min_noise         = 1e-4   # prevent noise collapsing to zero
cfg.training.use_double        = False  # set True for extreme ill-conditioning
```

The Cholesky solver is more numerically stable than CG for small-to-medium datasets.
For bins with n ≤ `max_cholesky_size`, GPyTorch will automatically use Cholesky.

---

## File Reference

```
gp4d_lab/
├── __init__.py        Public API + run_experiment() one-liner
├── config.py          All configuration as dataclasses (DataConfig, TrainingConfig, etc.)
├── kernels.py         GibbsKernel1D (custom) + build_axis_kernel() factory
├── scalers.py         PhysicsInputScaler4D + full scaler registry
├── data.py            CSV loading, stratified split, AxisPartitioner
├── models.py          ExactGPModel, SKIGPModel, training loop, OOM fallback
├── aggregation.py     All 6 aggregation strategies + LearnedWeightAggregator
├── ensemble.py        MultiAxisGPEnsemble — the main model class
├── metrics.py         11 evaluation metrics (MAE, NLPD, CRPS, coverage, etc.)
└── experiment.py      ExperimentRunner, grid sweep, leaderboard
```

---

## Data Format

The CSV must contain at minimum:

```
Q2_log_center, W2_log_center, beta_center, z_center, Q2_bin_idx, logA
```

- `Q2_log_center` and `W2_log_center` are already log-scaled.
- `beta_center` and `z_center` are in (0, 1) — the physics scaler applies `log(x + ε)`.
- `Q2_bin_idx` is an integer bin label (0, 1, 2, …); used for stratified train/test split.
- `logA` is the target (log cross-section).

---

## Reproducibility

All random operations respect `cfg.data.random_seed`:
```python
cfg.data.random_seed = 42   # controls train/test split and weight initialisation
```

---

## Citation / Physics Reference

This library is designed for Inclusive Diffraction at HERA (H1/ZEUS experiments).
The kinematic variables follow the standard notation of diffractive DIS:
F₂^D(3)(β, Q², x_P), where x_P ≈ Q²/(W² + Q² - M_p²).
