"""
gp4d_lab.config
===============
Master configuration system using Python dataclasses.
Every knob in the system lives here — one place to change anything.

Hierarchy
---------
  ExperimentConfig
    ├── DataConfig
    ├── AxisKernelConfig  (one per axis: q2 / w2 / beta / z)
    ├── AggregationConfig
    ├── TrainingConfig
    └── MetricConfig

Quick usage
-----------
    from gp4d_lab.config import ExperimentConfig, MASTER_DEFAULT

    cfg = MASTER_DEFAULT                        # use defaults
    cfg.aggregation.strategy = "rbcm"           # patch one field
    cfg.kernels["q2"].primary_kernel = "gibbs_power"

    # Serialize / deserialize
    import json
    json.dump(cfg.to_dict(), open("my_run.json", "w"), indent=2)
    cfg2 = ExperimentConfig.from_dict(json.load(open("my_run.json")))
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Kernel name registry (for validation & docs)
# ─────────────────────────────────────────────────────────────────────────────

KERNEL_NAMES = {
    # Standard
    "matern52",          # ARD Matérn-5/2 on all remaining dims
    "matern32",          # ARD Matérn-3/2
    "rbf",               # RBF / squared-exponential
    # Spectral
    "spectral_mixture",  # Wilson & Adams SM on all remaining dims
    # Gibbs (variable length-scale, non-stationary)
    "gibbs_power",       # ℓ(x) = ℓ₀ · exp(α·x)  — tightens with x
    "gibbs_sigmoid",     # ℓ(x) = ℓ_min + Δ·σ(-slope·(x-x₀)) — transitions at x₀
    "gibbs_linear",      # ℓ(x) = softplus(a + b·x)
    # Product combos (primary on dim-0, Matérn-5/2 on rest)
    "gibbs_power_x_matern52",
    "gibbs_sigmoid_x_matern52",
    "sm_x_matern52",     # SM on dim-0, Matérn on rest
    "matern52_x_sm",     # Matérn on dim-0, SM on rest
    "rbf_x_matern52",
    "sm_x_rbf",
    "gibbs_power_x_rbf",
    "gibbs_sigmoid_x_rbf",
    "matern32_x_matern12",
    "gibbs_sigmoid_x_matern12",
    "gibbs_power_x_matern12",    # Gibbs-power on dim-0, Matérn-1/2 on rest
    "gibbs_sigmoid_x_matern32",  # Gibbs-sigmoid on dim-0, Matérn-3/2 on rest
    "changepoint_x_matern52",
    "matern12",
    "gibbs_bell",
    "gibbs_bell_x_matern12",
    "gibbs_bell_x_gibbs_sigmoid_x_matern12",
    "gibbs_mlp",
    "gibbs_mlp_x_matern52",
    "gibbs_multidim",
    "dkl_matern52",
    "dkl_gibbs",
}

AGGREGATION_STRATEGIES = {
    "hard",         # argmin variance (winner-takes-all)
    "poe",          # Product-of-Experts (equal β=1)
    "gpoe",         # Generalised PoE (β = 1/M, variance-based)
    "rbcm",         # Robust BCM — differential entropy β  [RECOMMENDED]
    "precision",    # Precision-only weighting (simple 1/σ² normalised)
    "learned",      # Softmax weights trained on validation set
}


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Everything about where data lives and how to load it."""

    csv_path: str = ""

    # Column names — must match root_to_csv.py output
    feature_cols: list = field(default_factory=lambda: [
        "Q2_log_center",  # log(Q²) — already log-scaled
        "W2_log_center",  # log(W²) — already log-scaled
        "beta_center",    # β ∈ [0, 1]
        "z_center",       # ζ ∈ [0, 1]
    ])
    q2_bin_col:  str = "Q2_bin_idx"
    target_col:  str = "logA"

    # Sampling
    train_fraction:  float = 0.10
    random_seed:     int   = 42
    shuffle_before_split: bool = True

    # Input scaler name (matches ScalerRegistry)
    input_scaler:  str = "physics"   # "physics"|"standard"|"robust"|"quantile"|"none"
    target_scaler: str = "robust"    # "robust"|"standard"|"none"

    # ── Outlier filtering ─────────────────────────────────────────────────────
    # Removes unphysical floor values (e.g. -87.3365) from TRAINING only.
    # Test set is kept intact so metrics reflect real-world performance.
    # Set target_clip_lo = median - 3*IQR from your inspect_dataset.py output.
    filter_target_outliers: bool  = False
    target_clip_lo:         float = -17.955   # median - 3*IQR
    target_clip_hi:         float = 28.626    # median + 3*IQR

    # ── Hurdle Model Classifier ───────────────────────────────────────────────
    use_hurdle_model:       bool  = False
    hurdle_floor_val:       float = -17.955
    hurdle_classifier_type: str   = "hist_gb"  # "hist_gb" | "xgboost" | "rf"



@dataclass
class AxisKernelConfig:
    """
    Kernel configuration for ONE axis-partitioned GP expert.

    When the expert is trained on the 3 remaining dimensions after
    partitioning on its axis variable, these settings govern the kernel
    that expert uses internally.

    Example — Q2 expert (axis="q2") sees (W2, β, z):
      primary_kernel  = "spectral_mixture"  → applied to dim 0 (W2)
      secondary_kernel = "matern52"         → applied to dims 1, 2 (β, z)
      primary_dim     = 0                   → confirms dim 0 is the "special" one
    """

    axis:               str   = "q2"             # which variable is partitioned out
    primary_kernel:     str   = "matern52"        # kernel for primary_dim
    primary_dim:        int   = 0                 # which remaining dim gets primary kernel
    secondary_kernel:   str   = "matern52"        # kernel for remaining dims
    use_scale_kernel:   bool  = True              # wrap in ScaleKernel
    use_ard:            bool  = True              # ARD on secondary kernel

    # Spectral Mixture params
    sm_num_mixtures:    int   = 4                 # num components for SM kernel

    # Gibbs params
    gibbs_variant:      str   = "power_law"       # "power_law"|"sigmoid"|"linear"
    gibbs_alpha_init:   float = -0.5              # initial α for power-law ℓ(x) = ℓ₀·e^(αx)
    gibbs_slope_init:   float = 2.0               # initial slope for sigmoid variant
    gibbs_x0_init:      float = 0.0               # transition midpoint for sigmoid
    gibbs_bell_peak_init: float = 0.0             # x_peak for GibbsBell (0 = mid-Q2, minimum floor)
    gibbs_x0_beta_init:  float = 2.0              # x0 for the β Gibbs-sigmoid in W2-expert
    gibbs_mlp_hidden_dim: int   = 16              # hidden layer size for MLPLengthscale function

    # Composite shorthand — overrides primary/secondary if set
    # e.g. "gibbs_power_x_matern52" sets primary=gibbs_power, secondary=matern52
    composite_kernel:   str   = ""                # "" means use primary/secondary

    # DKL parameters
    dkl_latent_dim:     int   = 2                 # latent dimension for DKL mapping
    dkl_hidden_dim:     int   = 16                # hidden layer size for DKL neural network



    def resolve(self) -> "AxisKernelConfig":
        """Expand composite_kernel shorthand into primary/secondary."""
        cfg = copy.deepcopy(self)
        combos = {
            "gibbs_power_x_matern52":   ("gibbs_power",   "matern52"),
            "gibbs_sigmoid_x_matern52": ("gibbs_sigmoid",  "matern52"),
            "gibbs_mlp_x_matern52":     ("gibbs_mlp",      "matern52"),
            "gibbs_multidim":           ("gibbs_multidim", "gibbs_multidim"),
            "dkl_matern52":             ("dkl_matern52",   "dkl_matern52"),
            "dkl_gibbs":                ("dkl_gibbs",      "dkl_gibbs"),
            "sm_x_matern52":            ("spectral_mixture", "matern52"),
            "matern52_x_sm":            ("matern52", "spectral_mixture"),
            "matern52":                 ("matern52",        "matern52"),
            "rbf":                      ("rbf",             "rbf"),
            "spectral_mixture":         ("spectral_mixture", "spectral_mixture"),
            "rbf_x_matern52":           ("rbf",             "matern52"),
            "sm_x_rbf":                  ("spectral_mixture", "rbf"),
            "gibbs_power_x_rbf":         ("gibbs_power",      "rbf"),
            "gibbs_sigmoid_x_rbf":       ("gibbs_sigmoid",    "rbf"),
            "matern32":                  ("matern32",         "matern32"),
            "matern32_x_matern12":       ("matern32",         "matern12"),
            "gibbs_sigmoid_x_matern12":  ("gibbs_sigmoid",    "matern12"),
            "gibbs_power_x_matern12":    ("gibbs_power",      "matern12"),
            "gibbs_sigmoid_x_matern32":  ("gibbs_sigmoid",    "matern32"),
            "changepoint_x_matern52":    ("changepoint",      "matern52"),
            "matern12":                  ("matern12",         "matern12"),
            "gibbs_bell_x_matern12":     ("gibbs_bell",       "matern12"),
            "gibbs_bell_x_gibbs_sigmoid_x_matern12": None,
        }
        if cfg.composite_kernel in combos:
            res = combos[cfg.composite_kernel]
            if res is not None:
                cfg.primary_kernel, cfg.secondary_kernel = res
        return cfg


@dataclass
class AggregationConfig:
    """
    How to combine predictions from the 4 axis-experts at inference time.
    """

    strategy: str = "rbcm"    # See AGGREGATION_STRATEGIES

    # --- rBCM / gPoE specific ---
    rbcm_beta_floor: float = 1e-4   # clamp β_i to avoid negative weights
    prior_var_scale: float = 1.0    # scale applied to prior variance in rBCM

    # --- Learned weights ---
    learn_weights:    bool  = True
    weight_lr:        float = 0.05
    weight_epochs:    int   = 200
    weight_val_frac:  float = 0.20   # fraction of train set used as validation for weight learning
    weight_reg:       float = 1e-3   # L2 regularizer on raw weights
    weight_init:      str   = "uniform"   # "uniform" | "precision_based" | "entropy_based"

    # --- Axis selection ---
    axes_enabled: list = field(default_factory=lambda: ["q2", "w2", "beta", "z"])


@dataclass
class TrainingConfig:
    """Training loop settings (shared by all experts)."""

    n_epochs:       int   = 150
    lr:             float = 0.1
    device:         str   = "auto"    # "cpu"|"cuda"|"auto"
    cg_tol:         float = 1e-3
    max_cg_iter:    int   = 1000
    oom_fallback:   str   = "ski"     # "ski"|"skip"
    ski_grid_size:  int   = 30
    verbose:        bool  = True
    early_stop_patience: int = 20     # epochs without improvement to early-stop
    early_stop_delta: float = 1e-4    # minimum improvement threshold

    # Numerical stability for grid / duplicate-heavy physics data
    max_cholesky_size: int   = 2000   # force Cholesky (not CG) for n <= this
    cholesky_jitter:   float = 1e-4   # diagonal regulariser for Cholesky safety
    min_noise:         float = 0.02   # noise lower bound — prevents K being singular
    use_double:        bool  = False  # float64 for worst-case ill-conditioning
    fixed_noise:       float | None = None  # None to optimize noise; float (e.g. 1e-4) to freeze noise
    use_physics_mean:  bool  = False  # True to use Regge-like parametric mean function


    # Memory optimization
    predict_chunk_size: int  = 5000   # chunk size during prediction to prevent OOM

    # ── Lengthscale constraints ───────────────────────────────────────────────
    # Prevents collapse below grid spacing (~0.04-0.05 in scaled space).
    # Collapsed ℓ causes GP to revert to prior variance at every test point.
    constrain_lengthscale:  bool  = True
    min_lengthscale:        float = 0.10    # must be > grid spacing
    max_lengthscale:        float = 10.0

    # Noise upper bound — prevents outlier-driven noise explosion.
    # Set None to disable. In IQR-scaled units, 0.50 is generous.
    max_noise:              float = 0.50


@dataclass
class MetricConfig:
    """Which metrics to compute during experiments."""

    track: list = field(default_factory=lambda: [
        "mae", "rmse", "mape", "nlpd", "crps", "coverage_90", "coverage_68",
        "mean_uncertainty", "uncertainty_calibration_error"
    ])
    primary_metric:  str  = "mae"         # metric used for comparing experiments
    lower_is_better: bool = True          # for primary_metric


# ─────────────────────────────────────────────────────────────────────────────
# Top-level experiment config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Single experiment configuration. Holds ALL settings for one run.
    """

    name:        str = "experiment"
    description: str = ""
    tags:        list = field(default_factory=list)

    data:        DataConfig        = field(default_factory=DataConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    training:    TrainingConfig    = field(default_factory=TrainingConfig)
    metrics:     MetricConfig      = field(default_factory=MetricConfig)

    # Per-axis kernel configs — keyed by axis name
    # Default: all use Matérn-5/2 (swap per experiment)
    kernels: dict = field(default_factory=lambda: {
        "q2":   AxisKernelConfig(axis="q2",   composite_kernel="sm_x_matern52"),
        "w2":   AxisKernelConfig(axis="w2",   composite_kernel="gibbs_power_x_matern52"),
        "beta": AxisKernelConfig(axis="beta", composite_kernel="gibbs_sigmoid_x_matern52"),
        "z":    AxisKernelConfig(axis="z",    composite_kernel="matern52"),
    })

    # Output
    save_results: bool = True
    results_dir:  str  = "./results"

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        # kernels are dicts of dataclasses; asdict handles them recursively
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = copy.deepcopy(d)
        cfg = cls()
        cfg.name        = d.get("name", cfg.name)
        cfg.description = d.get("description", cfg.description)
        cfg.tags        = d.get("tags", cfg.tags)

        if "data" in d:
            cfg.data = DataConfig(**d["data"])
        if "aggregation" in d:
            cfg.aggregation = AggregationConfig(**d["aggregation"])
        if "training" in d:
            cfg.training = TrainingConfig(**d["training"])
        if "metrics" in d:
            cfg.metrics = MetricConfig(**d["metrics"])

        if "kernels" in d:
            cfg.kernels = {
                axis: AxisKernelConfig(**spec)
                for axis, spec in d["kernels"].items()
            }
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def clone(self, **overrides) -> "ExperimentConfig":
        """Deep-copy + apply top-level key overrides."""
        new = copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(new, k, v)
        return new


# ─────────────────────────────────────────────────────────────────────────────
# Grid experiment spec — describes a sweep
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GridSpec:
    """
    Describes a hyperparameter grid for ExperimentRunner.sweep().

    flat_grid: dict where each key is a dotted path into ExperimentConfig
    and value is a list of options.

    Example
    -------
        GridSpec(flat_grid={
            "aggregation.strategy":          ["rbcm", "gpoe", "precision", "learned"],
            "kernels.q2.composite_kernel":   ["sm_x_matern52", "matern52"],
            "kernels.w2.composite_kernel":   ["gibbs_power_x_matern52", "matern52"],
            "aggregation.learn_weights":     [True, False],
        })
    """
    flat_grid:  dict = field(default_factory=dict)
    base_config: Any = None   # ExperimentConfig; set at runtime


# ─────────────────────────────────────────────────────────────────────────────
# MASTER DEFAULT CONFIG — the recommended starting point
# ─────────────────────────────────────────────────────────────────────────────

def make_master_default(csv_path: str = "") -> ExperimentConfig:
    """
    Returns the recommended master default ExperimentConfig.
    Physical motivation for each kernel choice is documented inline.
    """
    return ExperimentConfig(
        name="master_default",
        description=(
            "4-axis partitioned GP ensemble with rBCM aggregation. "
            "Physics-informed kernels per axis. 10% training fraction."
        ),
        tags=["diffraction", "inclusive", "Q2", "W2", "beta", "z"],

        data=DataConfig(
            csv_path=csv_path,
            train_fraction=0.10,
            input_scaler="physics",
            target_scaler="robust",
        ),

        # ── Kernel choices ──────────────────────────────────────────────
        # Q2 expert sees (W2, β, z):
        #   SM on W2 — captures multi-scale Pomeron structure (soft+hard)
        #   Matérn-5/2 on (β, z) — smooth but flexible
        #
        # W2 expert sees (Q2, β, z):
        #   Gibbs-power on Q2 — ℓ(Q²)=ℓ₀·e^(αQ²) encodes perturbative transition
        #   Matérn-5/2 on (β, z)
        #
        # β expert sees (Q2, W2, z):
        #   Gibbs-sigmoid on Q2 — length scale transitions at perturbative threshold
        #   SM on W2 (dim 1) to capture Pomeron
        #   Actually: gibbs_power_x_matern52 for simplicity (Q2 is dim 0)
        #
        # z expert sees (Q2, W2, β):
        #   Full ARD Matérn-5/2 — z is smoothest axis (rapidity gap is regular)
        kernels={
            "q2": AxisKernelConfig(
                axis="q2",
                composite_kernel="sm_x_matern52",
                sm_num_mixtures=4,
                use_ard=True,
                use_scale_kernel=True,
            ),
            "w2": AxisKernelConfig(
                axis="w2",
                composite_kernel="gibbs_power_x_matern52",
                gibbs_variant="power_law",
                gibbs_alpha_init=-0.5,
                use_ard=True,
                use_scale_kernel=True,
            ),
            "beta": AxisKernelConfig(
                axis="beta",
                composite_kernel="gibbs_sigmoid_x_matern52",
                gibbs_variant="sigmoid",
                gibbs_slope_init=2.0,
                gibbs_x0_init=0.0,
                use_ard=True,
                use_scale_kernel=True,
            ),
            "z": AxisKernelConfig(
                axis="z",
                composite_kernel="matern52",
                use_ard=True,
                use_scale_kernel=True,
            ),
        },

        aggregation=AggregationConfig(
            strategy="rbcm",
            learn_weights=True,
            weight_lr=0.05,
            weight_epochs=200,
            weight_val_frac=0.20,
            axes_enabled=["q2", "w2", "beta", "z"],
        ),

        training=TrainingConfig(
            n_epochs=150,
            lr=0.1,
            device="auto",
            cg_tol=1e-3,
            max_cg_iter=1000,
            oom_fallback="ski",
            early_stop_patience=20,
        ),

        metrics=MetricConfig(
            track=["mae", "rmse", "nlpd", "crps", "coverage_90", "coverage_68",
                   "mean_uncertainty", "uncertainty_calibration_error"],
            primary_metric="mae",
        ),

        save_results=True,
        results_dir="./results",
    )


# Convenience singleton — import directly
MASTER_DEFAULT = make_master_default()


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built experiment variants (swap in for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def make_baseline_config(csv_path: str = "") -> ExperimentConfig:
    """All Matérn-5/2, rBCM aggregation — establishes the baseline."""
    cfg = make_master_default(csv_path)
    cfg.name = "baseline_matern52"
    cfg.description = "Baseline: all Matérn-5/2, rBCM aggregation"
    for axis in cfg.kernels:
        cfg.kernels[axis].composite_kernel = "matern52"
    return cfg


def make_all_sm_config(csv_path: str = "") -> ExperimentConfig:
    """All spectral mixture — tests if SM universally beats Matérn."""
    cfg = make_master_default(csv_path)
    cfg.name = "all_spectral_mixture"
    cfg.description = "All SM kernels, rBCM aggregation"
    for axis in cfg.kernels:
        cfg.kernels[axis].composite_kernel = "spectral_mixture"
        cfg.kernels[axis].sm_num_mixtures = 4
    return cfg


def make_aggregation_sweep_configs(csv_path: str = "") -> list:
    """One config per aggregation strategy — all use master-default kernels."""
    configs = []
    for strategy in ["hard", "poe", "gpoe", "rbcm", "precision", "learned"]:
        cfg = make_master_default(csv_path)
        cfg.name = f"agg_{strategy}"
        cfg.description = f"Master kernels + {strategy} aggregation"
        cfg.aggregation.strategy = strategy
        cfg.aggregation.learn_weights = (strategy == "learned")
        configs.append(cfg)
    return configs


def make_kernel_sweep_configs(csv_path: str = "") -> list:
    """Grid over per-axis kernel choices with fixed rBCM aggregation."""
    kernel_options = {
        "q2":   ["sm_x_matern52", "matern52", "gibbs_power_x_matern52", "spectral_mixture"],
        "w2":   ["gibbs_power_x_matern52", "matern52", "sm_x_matern52"],
        "beta": ["gibbs_sigmoid_x_matern52", "matern52", "gibbs_power_x_matern52"],
        "z":    ["matern52", "spectral_mixture"],
    }
    configs = []
    for axis, kernels in kernel_options.items():
        for k in kernels:
            cfg = make_master_default(csv_path)
            cfg.name = f"kernel_{axis}_{k}"
            cfg.description = f"Testing kernel={k} on {axis} axis"
            cfg.kernels[axis].composite_kernel = k
            configs.append(cfg)
    return configs
