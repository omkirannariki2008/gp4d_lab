"""
gp4d_lab
========
4-Axis Partitioned GP Ensemble for Inclusive Diffraction Cross-Section Prediction.

Quick start
-----------
    from gp4d_lab import run_experiment, ExperimentRunner
    from gp4d_lab.config import make_master_default, make_aggregation_sweep_configs

    # Single run
    result = run_experiment("data.csv")

    # Grid sweep
    runner = ExperimentRunner("data.csv")
    results = runner.sweep(make_aggregation_sweep_configs("data.csv"))
    runner.print_leaderboard(results)

Architecture
------------
  MultiAxisGPEnsemble trains 4 GP experts, one per axis (Q2, W2, beta, z).
  Each expert partitions data by its axis variable and fits one GP per bin.
  At inference, all experts predict (μ, σ), and an Aggregator combines
  them using the configured strategy (rBCM by default).
"""

import logging
from pathlib import Path

# Configure library-level logging with both console and persistent file handlers
log_dir = Path("logs")
try:
    log_dir.mkdir(exist_ok=True)
except Exception:
    pass

formatter = logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")

# Reset root logger to avoid duplicate handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

# 1. Console stream handler (INFO and above)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

# 2. File handler for all logs (INFO and above)
try:
    info_handler = logging.FileHandler(log_dir / "gp4d_lab.log", mode="a", encoding="utf-8")
    info_handler.setFormatter(formatter)
    info_handler.setLevel(logging.INFO)
    root_logger.addHandler(info_handler)
except Exception as e:
    root_logger.warning(f"Could not create info log file: {e}")

# 3. File handler for warnings and errors (WARNING and above)
try:
    error_handler = logging.FileHandler(log_dir / "gp4d_lab_errors.log", mode="a", encoding="utf-8")
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.WARNING)
    root_logger.addHandler(error_handler)
except Exception as e:
    root_logger.warning(f"Could not create error log file: {e}")


from .config import (
    ExperimentConfig,
    DataConfig,
    AxisKernelConfig,
    AggregationConfig,
    TrainingConfig,
    MetricConfig,
    GridSpec,
    MASTER_DEFAULT,
    make_master_default,
    make_baseline_config,
    make_all_sm_config,
    make_aggregation_sweep_configs,
    make_kernel_sweep_configs,
)

from .kernels import GibbsKernel1D, build_axis_kernel
from .scalers import get_scaler, PhysicsInputScaler4D
from .data import load_dataset, Dataset, AxisPartitioner
from .models import ExactGPModel, train_gp, predict_gp, fit_partition
from .aggregation import Aggregator, AggregationResult
from .ensemble import MultiAxisGPEnsemble
from .metrics import evaluate, format_metrics
from .experiment import ExperimentRunner, RunResult


def run_experiment(
    csv_path: str,
    cfg: ExperimentConfig | None = None,
    **overrides,
) -> "RunResult":
    """
    One-liner: load data, train ensemble, evaluate, return results.

    Parameters
    ----------
    csv_path  : path to the CSV file
    cfg       : ExperimentConfig; defaults to MASTER_DEFAULT
    **overrides : dotted-key overrides, e.g. aggregation_strategy="gpoe"
    """
    from .experiment import _set_dotted

    if cfg is None:
        cfg = make_master_default(csv_path)

    # Apply overrides like aggregation_strategy="gpoe"
    for k, v in overrides.items():
        dotted = k.replace("_", ".", 1)  # aggregation_strategy → aggregation.strategy
        try:
            _set_dotted(cfg, dotted, v)
        except (AttributeError, KeyError):
            setattr(cfg, k, v)

    runner = ExperimentRunner(csv_path, cfg)
    return runner.run(cfg)


__version__ = "0.1.0"
__all__ = [
    "ExperimentConfig", "DataConfig", "AxisKernelConfig",
    "AggregationConfig", "TrainingConfig", "MetricConfig", "GridSpec",
    "MASTER_DEFAULT", "make_master_default", "make_baseline_config",
    "make_all_sm_config", "make_aggregation_sweep_configs", "make_kernel_sweep_configs",
    "GibbsKernel1D", "build_axis_kernel",
    "get_scaler", "PhysicsInputScaler4D",
    "load_dataset", "Dataset", "AxisPartitioner",
    "MultiAxisGPEnsemble",
    "Aggregator", "AggregationResult",
    "evaluate", "format_metrics",
    "ExperimentRunner", "RunResult",
    "run_experiment",
]
