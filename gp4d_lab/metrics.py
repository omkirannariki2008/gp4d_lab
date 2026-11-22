"""
gp4d_lab.metrics
================
Evaluation metrics for probabilistic GP predictions.

All functions operate in ORIGINAL (unscaled) target space.
The caller is responsible for inverse-transforming predictions
before passing to these functions.

Metrics
-------
  mae, rmse, mape        — point-prediction accuracy
  nlpd                   — Negative Log Predictive Density
  crps                   — Continuous Ranked Probability Score
  coverage_k             — Empirical coverage at k% CI
  mean_uncertainty       — mean(σ)
  uncertainty_calibration_error  — mean |empirical_coverage - nominal|
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.stats import norm

log = logging.getLogger("gp4d_lab.metrics")

EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Individual metrics
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def nlpd(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Negative Log Predictive Density for Gaussian predictions.
    NLPD = 0.5·log(2πσ²) + (y−μ)²/(2σ²)  — lower is better.
    """
    sigma = np.maximum(sigma, EPS)
    return float(np.mean(
        0.5 * np.log(2 * np.pi * sigma ** 2)
        + (y_true - mu) ** 2 / (2 * sigma ** 2)
    ))


def crps(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score for Gaussian predictions.
    CRPS = σ·[z·(2Φ(z)−1) + 2φ(z) − 1/√π]  where z = (y−μ)/σ
    Lower is better.
    """
    sigma = np.maximum(sigma, EPS)
    z     = (y_true - mu) / sigma
    crps_val = sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_val))


def coverage(
    y_true: np.ndarray,
    mu:     np.ndarray,
    sigma:  np.ndarray,
    level:  float = 0.90,
) -> float:
    """
    Empirical coverage at a given confidence level.
    Returns the fraction of test points inside the ±z·σ interval.
    """
    z    = norm.ppf((1 + level) / 2)
    lo   = mu - z * sigma
    hi   = mu + z * sigma
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def uncertainty_calibration_error(
    y_true: np.ndarray,
    mu:     np.ndarray,
    sigma:  np.ndarray,
    levels: Sequence[float] = (0.50, 0.68, 0.90, 0.95, 0.99),
) -> float:
    """
    Mean absolute calibration error.
    For each nominal CI level, compute |empirical coverage − level|.
    Returns the mean across levels.
    """
    errors = []
    for lv in levels:
        emp = coverage(y_true, mu, sigma, level=lv)
        errors.append(abs(emp - lv))
    return float(np.mean(errors))


def mean_uncertainty(sigma: np.ndarray) -> float:
    return float(np.mean(sigma))


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    y_true:  np.ndarray,
    mu:      np.ndarray,
    sigma:   np.ndarray,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute all requested metrics in one call.

    Parameters
    ----------
    y_true, mu, sigma : arrays in ORIGINAL (unscaled) target space
    metrics           : list of metric names; None → all

    Returns
    -------
    dict of {metric_name: value}
    """
    all_metrics = {
        "mae":                         lambda: mae(y_true, mu),
        "rmse":                        lambda: rmse(y_true, mu),
        "mape":                        lambda: mape(y_true, mu),
        "nlpd":                        lambda: nlpd(y_true, mu, sigma),
        "crps":                        lambda: crps(y_true, mu, sigma),
        "coverage_50":                 lambda: coverage(y_true, mu, sigma, 0.50),
        "coverage_68":                 lambda: coverage(y_true, mu, sigma, 0.68),
        "coverage_90":                 lambda: coverage(y_true, mu, sigma, 0.90),
        "coverage_95":                 lambda: coverage(y_true, mu, sigma, 0.95),
        "mean_uncertainty":            lambda: mean_uncertainty(sigma),
        "uncertainty_calibration_error": lambda: uncertainty_calibration_error(y_true, mu, sigma),
    }

    if metrics is None:
        metrics = list(all_metrics.keys())

    results = {}
    for name in metrics:
        if name not in all_metrics:
            log.warning(f"Unknown metric '{name}' — skipping")
            continue
        try:
            results[name] = all_metrics[name]()
        except Exception as e:
            log.warning(f"Metric '{name}' failed: {e}")
            results[name] = float("nan")

    return results


def format_metrics(metrics_dict: dict[str, float]) -> str:
    """Pretty-print a metrics dict."""
    lines = []
    for k, v in metrics_dict.items():
        lines.append(f"  {k:40s}: {v:.6f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc uncertainty calibration
# ─────────────────────────────────────────────────────────────────────────────

def find_calibration_temperature(
    y_true: np.ndarray,
    mu:     np.ndarray,
    sigma:  np.ndarray,
    method: str = "nlpd",
) -> float:
    """
    Find T such that sigma_calibrated = sigma / T gives best calibration.

    Under-confident (coverage_90 > 0.90) → T > 1 → intervals narrow.
    Over-confident  (coverage_90 < 0.90) → T < 1 → intervals widen.

    method="nlpd"     — minimise negative log predictive density
    method="coverage" — match empirical to 90% nominal coverage
    """
    from scipy.optimize import minimize_scalar
    sigma = np.maximum(sigma, EPS)

    if method == "nlpd":
        def objective(log_T):
            s = sigma / np.exp(log_T)
            return float(np.mean(
                0.5 * np.log(2 * np.pi * s**2)
                + (y_true - mu)**2 / (2 * s**2)
            ))
        res = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
        return float(np.exp(res.x))

    elif method == "coverage":
        def cov_err(log_T):
            s  = sigma / np.exp(log_T)
            z  = norm.ppf(0.95)
            lo = mu - z * s
            hi = mu + z * s
            return float(np.mean((y_true >= lo) & (y_true <= hi))) - 0.90
        try:
            from scipy.optimize import brentq
            log_T = brentq(cov_err, -4, 4)
            return float(np.exp(log_T))
        except ValueError:
            return find_calibration_temperature(y_true, mu, sigma, "nlpd")

    raise ValueError(f"method must be 'nlpd' or 'coverage', got '{method}'")


def calibrated_metrics(
    y_true:  np.ndarray,
    mu:      np.ndarray,
    sigma:   np.ndarray,
    metrics: list | None = None,
) -> tuple:
    """
    Auto-calibrate sigma then evaluate metrics.

    Returns (metrics_dict, temperature_T, sigma_calibrated)

    Example
    -------
        m, T, sigma_cal = calibrated_metrics(y_true, y_pred, y_std)
        print(f"T={T:.3f}  coverage_90={m['coverage_90']:.4f}")
    """
    T         = find_calibration_temperature(y_true, mu, sigma, "nlpd")
    sigma_cal = sigma / T
    m         = evaluate(y_true, mu, sigma_cal, metrics)
    return m, T, sigma_cal


def uncertainty_decomposition(
    y_true: np.ndarray,
    mu:     np.ndarray,
    sigma:  np.ndarray,
) -> dict:
    """
    Decompose prediction error to diagnose whether the problem is
    in the mean predictions (bias/variance) or the uncertainty (σ).

    Key output:
      ratio_sigma_to_residual > 1  →  σ too wide (under-confident)
      ratio_sigma_to_residual < 1  →  σ too narrow (over-confident)
      calibration_T ≈ 1.0          →  well calibrated
    """
    residuals = mu - y_true
    T = find_calibration_temperature(y_true, mu, sigma, "nlpd")
    ratio = float(np.mean(sigma) / (float(np.std(residuals)) + EPS))
    return {
        "bias":                    float(np.mean(residuals)),
        "residual_std":            float(np.std(residuals)),
        "mean_sigma":              float(np.mean(sigma)),
        "ratio_sigma_to_residual": ratio,
        "calibration_T":           T,
        "interpretation": (
            "UNDER-CONFIDENT (σ too wide)"   if T > 1.05 else
            "OVER-CONFIDENT (σ too narrow)"  if T < 0.95 else
            "WELL CALIBRATED"
        ),
    }


def evaluate_stratified(
    y_true:          np.ndarray,
    mu:              np.ndarray,
    sigma:           np.ndarray,
    floor_threshold: float = -17.955,   # median - 3*IQR from your diagnostics
    metrics:         list  | None = None,
) -> dict:
    """
    Evaluate separately on physical-regime vs floor/zero-cross-section points.

    The floor points (logA ≈ -87.33, A ≈ 0) are a kinematic boundary — the GP
    cannot and should not be expected to predict them as -87.  The meaningful
    metric for physics quality is the error on the physical regime only.

    Returns
    -------
    dict with keys:
        "physical"   — metrics on points where y_true >= floor_threshold
        "floor"      — metrics on points where y_true < floor_threshold
        "all"        — metrics on all points (for comparison)
        "floor_frac" — fraction of test points that are floor points
        "floor_detected_frac" — fraction of floor points where model
                                  predicted below floor_threshold too
                                  (did model identify the zero-CS region?)
    """
    mask_physical = y_true > floor_threshold + 1e-3
    mask_floor    = ~mask_physical

    n_floor    = mask_floor.sum()
    n_physical = mask_physical.sum()
    n_total    = len(y_true)

    result = {
        "floor_frac":     float(n_floor / n_total),
        "n_floor":        int(n_floor),
        "n_physical":     int(n_physical),
    }

    # Physical regime metrics — the number that matters for physics quality
    if n_physical > 0:
        result["physical"] = evaluate(
            y_true[mask_physical], mu[mask_physical],
            sigma[mask_physical], metrics
        )
    else:
        result["physical"] = {}

    # Floor region metrics — model will fail here, but report honestly
    if n_floor > 0:
        result["floor"] = evaluate(
            y_true[mask_floor], mu[mask_floor],
            sigma[mask_floor], metrics
        )
        # Key question: does the model at least IDENTIFY these as low-CS?
        # A good model should predict mu < floor_threshold for floor points
        predicted_as_floor = (mu[mask_floor] <= floor_threshold + 1e-3).sum()
        result["floor_detected_frac"] = float(predicted_as_floor / n_floor)
    else:
        result["floor"] = {}
        result["floor_detected_frac"] = float("nan")

    # All-points metrics (dominated by floor errors — for reference only)
    result["all"] = evaluate(y_true, mu, sigma, metrics)

    return result


def print_stratified_report(
    y_true:          np.ndarray,
    mu:              np.ndarray,
    sigma:           np.ndarray,
    floor_threshold: float = -17.955,
) -> None:
    """
    Print a clean stratified report separating physics quality from floor behaviour.
    """
    r = evaluate_stratified(
        y_true, mu, sigma, floor_threshold,
        metrics=["mae", "rmse", "nlpd", "coverage_90", "coverage_68",
                 "mean_uncertainty"]
    )

    print(f"\n{'='*60}")
    print(f"  STRATIFIED EVALUATION  (floor_threshold={floor_threshold})")
    print(f"{'='*60}")
    print(f"  Test set: {r['n_physical']:,} physical  +  {r['n_floor']:,} floor "
          f"({100*r['floor_frac']:.2f}%)")

    print(f"\n  PHYSICAL REGIME (logA >= {floor_threshold})  ← USE THESE FOR PHYSICS")
    print(f"  {'─'*56}")
    for k, v in r["physical"].items():
        print(f"    {k:<40s}: {v:.6f}")

    print(f"\n  FLOOR / ZERO CROSS-SECTION (logA < {floor_threshold})")
    print(f"  {'─'*56}")
    for k, v in r["floor"].items():
        print(f"    {k:<40s}: {v:.6f}")
    print(f"    {'floor_detected_frac':<40s}: "
          f"{r['floor_detected_frac']:.4f}  "
          f"(fraction GP correctly identifies as low-CS)")

    print(f"\n  ALL POINTS (dominated by floor errors — reference only)")
    print(f"  {'─'*56}")
    for k, v in r["all"].items():
        print(f"    {k:<40s}: {v:.6f}")
    print(f"{'='*60}\n")


def local_calibration_temperature(
    y_true: np.ndarray,
    mu:     np.ndarray,
    sigma:  np.ndarray,
    region_key: np.ndarray,   # integer array: 0=smooth, 1=near-dip, etc.
) -> dict:
    """Fit a separate T per region. Apply at test time by looking up region."""
    from scipy.optimize import minimize_scalar
    temperatures = {}
    for region in np.unique(region_key):
        mask = region_key == region
        if mask.sum() < 10:
            continue
        def obj(log_T, mask=mask):
            s = sigma[mask] / np.exp(log_T)
            return float(np.mean(
                0.5*np.log(2*np.pi*s**2) + (y_true[mask]-mu[mask])**2/(2*s**2)
            ))
        res = minimize_scalar(obj, bounds=(-4, 4), method="bounded")
        temperatures[int(region)] = float(np.exp(res.x))
    return temperatures
