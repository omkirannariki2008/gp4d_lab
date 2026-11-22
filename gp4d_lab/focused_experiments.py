"""
gp4d_lab.focused_experiments
=============================
Physics-motivated experiment grid for inclusive diffraction (H1/ZEUS).

Instead of testing all combinations (~96 runs), this module defines a
3-phase coordinate-descent strategy that requires only 16 targeted runs.
Every choice is justified by published HERA physics results.

Physics Justification
---------------------

log(Q²):  transition variable — soft Pomeron at low Q², DGLAP scaling
          violations at high Q². Pomeron intercept rises with Q².
          → NON-STATIONARY kernel required when Q² appears as an INPUT
            dimension (W2-expert, β-expert): Gibbs(power_law) or Gibbs(sigmoid).

log(W²):  σ ∝ W^(4ε) → in log W² space this is smooth & long-range
          (single Pomeron pole). No regime transition expected within a W²-bin.
          → RBF (∞-differentiable) preferred over Matérn when W² is INPUT dim
            (Q2-expert). SpectralMixture as backup for soft+hard Pomeron coexistence.

β:        Two simultaneous scales documented in H1/ZEUS data:
            β→0: perturbative rise, gluon-dominated, long-range
            β→1: elastic/quasi-elastic peak, short-range & sharp
          → SpectralMixture (2+ components) preferred when β is INPUT dim.
            GibbsSigmoid as backup (variable length-scale).

z:        Rapidity gap variable with exponential suppression e^{-0.01/(1-z)}
          in H1 DPDF parametrisation. Smooth, no known transitions.
          → RBF preferred (exponential → smooth). Matérn-5/2 as backup.

Aggregation:
  Skip hard (inconsistent) and poe (overconfident with overlapping data).
  Must test: rbcm, gpoe, precision, learned.

Usage
-----
    from gp4d_lab.focused_experiments import (
        phase1_aggregation_sweep,
        phase2_kernel_sweep,
        phase3_final,
        run_all_phases,
        EXPERIMENT_RATIONALE,
    )
    from gp4d_lab.experiment import ExperimentRunner

    runner = ExperimentRunner("your_data.csv")
    results = run_all_phases(runner, "your_data.csv")
    runner.print_leaderboard(results, primary_metric="mae")
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Optional

from .config import (
    ExperimentConfig, AxisKernelConfig, AggregationConfig,
    make_master_default,
)

log = logging.getLogger("gp4d_lab.focused_experiments")


# ─────────────────────────────────────────────────────────────────────────────
# Physics rationale registry — why each kernel is tested
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_RATIONALE: dict[str, str] = {
    # Q2-expert kernels (Q2 is partitioned out; expert sees W2, β, z)
    "q2:sm_x_matern52": (
        "SM on W2-dim captures both soft (long-range, low-W²) and hard Pomeron "
        "(shorter-range, high-W²) components simultaneously. "
        "Matérn-5/2 on (β,z) is robust for smooth variables."
    ),
    "q2:rbf_x_matern52": (
        "RBF on W2-dim: σ∝W^{4ε} is infinitely smooth in log W². "
        "Within a fixed Q²-bin the Pomeron exponent ε is near-constant, "
        "so a single long-range smooth kernel is optimal."
    ),

    # W2-expert kernels (W2 is partitioned out; expert sees Q2, β, z)
    "w2:gibbs_power_x_matern52": (
        "GibbsPower on Q2-dim: length-scale shrinks as Q² increases, "
        "directly encoding the soft→hard transition (Pomeron intercept rise). "
        "Backed by H1/ZEUS observation that intercept grows with Q²."
    ),
    "w2:gibbs_sigmoid_x_matern52": (
        "GibbsSigmoid on Q2-dim: explicit transition at a learnable Q₀². "
        "Preferred if the non-perturbative→perturbative transition is abrupt "
        "rather than gradual."
    ),

    # β-expert kernels (β is partitioned out; expert sees Q2, W2, z)
    "beta:sm_x_matern52": (
        "SM on Q2-dim within β-slices: captures both soft and hard Pomeron "
        "energy modes visible in each β-bin. Q² variation inside a β-bin "
        "spans the full DGLAP range."
    ),
    "beta:gibbs_sigmoid_x_matern52": (
        "GibbsSigmoid on Q2-dim: length-scale transitions at the perturbative "
        "threshold. Natural when Q² is the primary non-stationarity driver "
        "within each β-slice."
    ),

    # z-expert kernels (z is partitioned out; expert sees Q2, W2, β)
    "z:rbf": (
        "Full ARD RBF: z-partitioned data shows smooth variation in all three "
        "remaining variables. The exponential e^{-0.01/(1-z)} suppression is "
        "smooth → ∞-differentiable kernel appropriate."
    ),
    "z:matern52": (
        "ARD Matérn-5/2: robust baseline, less strong smoothness assumption "
        "than RBF. Preferred if the cross-section has 'kinks' at kinematic "
        "boundaries within z-bins."
    ),

    # Aggregation strategies
    "agg:rbcm": (
        "Robust BCM: differential entropy weights naturally down-weight experts "
        "whose posterior ≈ prior (uninformative). Best-documented for "
        "overlapping, non-disjoint data partitions."
    ),
    "agg:gpoe": (
        "Generalised PoE: β=1/M uniform weights. Conservative — tends to "
        "overestimate uncertainty, which improves calibration for physics data "
        "where tails matter."
    ),
    "agg:precision": (
        "Precision weighting: β∝1/σ²ᵢ normalised. Intuitive baseline — "
        "simply gives more weight to whichever expert is most confident. "
        "Fast and interpretable."
    ),
    "agg:learned": (
        "Softmax(w) weights trained on validation NLL. Discovers which axes "
        "contribute most signal for THIS dataset. If Q² expert dominates, "
        "data confirms the known Q²-driven structure of σ_diff."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base(csv_path: str, n_epochs: int = 150, axes=None) -> ExperimentConfig:
    """Master default with configurable epochs and axes."""
    cfg = make_master_default(csv_path)
    cfg.training.n_epochs            = n_epochs
    cfg.training.early_stop_patience = max(20, n_epochs // 8)
    cfg.training.verbose             = False
    cfg.aggregation.learn_weights    = False
    cfg.aggregation.axes_enabled     = axes or ["q2", "w2", "beta", "z"]
    cfg.save_results                 = True
    return cfg


def _patch_axis_kernel(
    cfg: ExperimentConfig, axis: str, composite: str
) -> ExperimentConfig:
    c = copy.deepcopy(cfg)
    c.kernels[axis].composite_kernel = composite
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Aggregation sweep (4 runs, master-default kernels)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_aggregation_sweep(
    csv_path: str,
    n_epochs: int = 150,
) -> list[ExperimentConfig]:
    """
    Fix master-default kernels. Compare all 4 aggregation strategies.
    Identifies the best aggregator before touching kernels.

    Runs: 4
    """
    strategies = [
        ("rbcm",      False),
        ("gpoe",      False),
        ("precision", False),
        ("learned",   True),
    ]
    configs = []
    for strategy, do_learn in strategies:
        cfg = _base(csv_path, n_epochs)
        cfg.name                         = f"p1_agg_{strategy}"
        cfg.description                  = EXPERIMENT_RATIONALE[f"agg:{strategy}"]
        cfg.aggregation.strategy         = strategy
        cfg.aggregation.learn_weights    = do_learn
        cfg.aggregation.weight_epochs    = 200 if do_learn else 0
        configs.append(cfg)

    log.info(f"Phase 1: {len(configs)} aggregation configs")
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Per-axis kernel sweep (8 runs, best aggregation fixed)
# ─────────────────────────────────────────────────────────────────────────────

# Physics-motivated kernel pairs per axis
# Only 2 options per axis — no combinatorial explosion
AXIS_KERNEL_OPTIONS: dict[str, list[tuple[str, str]]] = {
    # (composite_kernel_name, short_tag)
    "q2": [
        ("sm_x_matern52",  "SM·M52"),    # SM on W2 captures multi-scale Pomeron
        ("rbf_x_matern52", "RBF·M52"),   # RBF on W2 for pure power-law W2
    ],
    "w2": [
        ("gibbs_power_x_matern52",   "GP·M52"),   # Non-stationary Q2 (MUST TEST)
        ("gibbs_sigmoid_x_matern52", "GS·M52"),   # Alternative sigmoid transition
    ],
    "beta": [
        ("sm_x_matern52",            "SM·M52"),   # Multi-scale β (two length scales)
        ("gibbs_sigmoid_x_matern52", "GS·M52"),   # Variable ℓ(Q2) within β-slices
    ],
    "z": [
        ("rbf",     "RBF"),    # ∞-differentiable for smooth exponential z
        ("matern52", "M52"),   # Robust baseline, less assumption
    ],
}


def phase2_kernel_sweep(
    csv_path: str,
    best_aggregation: str = "rbcm",
    learn_weights: bool   = False,
    n_epochs: int         = 150,
) -> list[ExperimentConfig]:
    """
    Fix the best aggregation from Phase 1. For each axis, test 2 kernel options
    while keeping all other axes at master-default. Coordinate descent approach.

    Runs: 8  (2 options × 4 axes)
    """
    configs = []
    for axis, options in AXIS_KERNEL_OPTIONS.items():
        for composite, tag in options:
            cfg = _base(csv_path, n_epochs)
            cfg = _patch_axis_kernel(cfg, axis, composite)
            cfg.name                      = f"p2_{axis}_{tag}"
            cfg.description               = EXPERIMENT_RATIONALE.get(
                f"{axis}:{composite}",
                f"Testing {composite} on {axis}-expert"
            )
            cfg.aggregation.strategy      = best_aggregation
            cfg.aggregation.learn_weights = learn_weights
            configs.append(cfg)

    log.info(f"Phase 2: {len(configs)} per-axis kernel configs")
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Final validation (4 runs: best kernels × top-2 aggregations)
# ─────────────────────────────────────────────────────────────────────────────

def phase3_final(
    csv_path:            str,
    best_kernels:        dict[str, str],          # {axis: composite_kernel}
    top_aggregations:    list[str] = ("rbcm", "learned"),
    n_epochs:            int       = 150,
) -> list[ExperimentConfig]:
    """
    Assemble the best per-axis kernel set found in Phase 2.
    Re-test with top-2 aggregations + learned weights as a final cross-check.

    Runs: 2 (or len(top_aggregations))
    """
    configs = []
    for strategy in top_aggregations:
        cfg = _base(csv_path, n_epochs)
        for axis, composite in best_kernels.items():
            cfg.kernels[axis].composite_kernel = composite
        do_learn = (strategy == "learned")
        cfg.name                      = f"p3_best_{strategy}"
        cfg.description               = f"Best kernels + {strategy} aggregation"
        cfg.aggregation.strategy      = strategy
        cfg.aggregation.learn_weights = do_learn
        cfg.aggregation.weight_epochs = 200 if do_learn else 0
        configs.append(cfg)

    log.info(f"Phase 3: {len(configs)} final configs")
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all 3 phases end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def run_all_phases(
    runner,                          # ExperimentRunner instance
    csv_path:      str,
    primary_metric: str = "mae",
    n_epochs:      int  = 150,
    verbose:       bool = True,
) -> list:
    """
    Execute all three phases in sequence, automatically propagating
    the best result from each phase to the next.

    Returns
    -------
    All RunResult objects from all phases (for full leaderboard).
    """
    all_results = []

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("  PHASE 1: Aggregation sweep (4 runs)")
    log.info("="*60)
    p1_cfgs = phase1_aggregation_sweep(csv_path, n_epochs)
    p1_results = runner.sweep(p1_cfgs)
    all_results.extend(p1_results)

    if not p1_results:
        log.error("Phase 1 produced no results — aborting")
        return all_results

    best_agg = min(p1_results, key=lambda r: r.metrics.get(primary_metric, float("inf")))
    best_agg_strategy = best_agg.config.aggregation.strategy
    log.info(f"\n  → Best aggregation: {best_agg_strategy} "
             f"({primary_metric}={best_agg.metrics.get(primary_metric):.5f})")

    if verbose:
        runner.print_leaderboard(p1_results, primary_metric=primary_metric)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info(f"  PHASE 2: Per-axis kernel sweep (8 runs, agg={best_agg_strategy})")
    log.info("="*60)
    p2_cfgs = phase2_kernel_sweep(csv_path, best_aggregation=best_agg_strategy,
                                  n_epochs=n_epochs)
    p2_results = runner.sweep(p2_cfgs)
    all_results.extend(p2_results)

    # Pick best kernel per axis
    best_kernels: dict[str, str] = {}
    for axis in ["q2", "w2", "beta", "z"]:
        axis_results = [r for r in p2_results if r.experiment_name.startswith(f"p2_{axis}_")]
        if axis_results:
            best_for_axis = min(axis_results,
                                key=lambda r: r.metrics.get(primary_metric, float("inf")))
            best_kernels[axis] = best_for_axis.config.kernels[axis].composite_kernel
            log.info(f"  → Best {axis} kernel: {best_kernels[axis]} "
                     f"({primary_metric}={best_for_axis.metrics.get(primary_metric):.5f})")
        else:
            # Fallback to master default
            best_kernels[axis] = make_master_default().kernels[axis].composite_kernel
            log.warning(f"  → No Phase 2 results for {axis}, using master default")

    if verbose:
        runner.print_leaderboard(p2_results, primary_metric=primary_metric)

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    # Second-best aggregation from Phase 1
    sorted_p1 = sorted(p1_results,
                       key=lambda r: r.metrics.get(primary_metric, float("inf")))
    top2_aggs = [r.config.aggregation.strategy for r in sorted_p1[:2]]
    if "learned" not in top2_aggs:
        top2_aggs.append("learned")   # always include learned as final check

    log.info("\n" + "="*60)
    log.info(f"  PHASE 3: Final validation ({len(top2_aggs)} runs)")
    log.info(f"  Best kernels: {best_kernels}")
    log.info(f"  Top aggregations: {top2_aggs}")
    log.info("="*60)
    p3_cfgs = phase3_final(csv_path, best_kernels, top2_aggs, n_epochs)
    p3_results = runner.sweep(p3_cfgs)
    all_results.extend(p3_results)

    if verbose:
        runner.print_leaderboard(all_results, primary_metric=primary_metric)

    # ── Final summary ─────────────────────────────────────────────────────────
    best_overall = min(all_results,
                       key=lambda r: r.metrics.get(primary_metric, float("inf")))
    log.info("\n" + "="*60)
    log.info("  FINAL BEST CONFIGURATION")
    log.info("="*60)
    log.info(f"  Run:        {best_overall.experiment_name}")
    log.info(f"  Aggregation: {best_overall.config.aggregation.strategy}")
    log.info(f"  Kernels:")
    for axis in ["q2", "w2", "beta", "z"]:
        log.info(f"    {axis}: {best_overall.config.kernels[axis].composite_kernel}")
    log.info(f"  {primary_metric}: {best_overall.metrics.get(primary_metric):.5f}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Print experiment plan (without running)
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_plan() -> None:
    """
    Print the full 3-phase experiment plan with physics justifications.
    Does not require a CSV or runner.
    """
    total = 0
    print("\n" + "="*68)
    print("  GP4D-LAB FOCUSED EXPERIMENT PLAN (physics-motivated)")
    print("="*68)

    print("\n  PHASE 1 — Aggregation sweep (fix master kernels, 4 runs)")
    print("  " + "-"*64)
    for strategy, _ in [("rbcm",""), ("gpoe",""), ("precision",""), ("learned","")]:
        print(f"  p1_agg_{strategy:<12}  {EXPERIMENT_RATIONALE[f'agg:{strategy}'][:55]}…")
    total += 4

    print("\n  PHASE 2 — Per-axis kernel sweep (fix best agg, 8 runs)")
    print("  " + "-"*64)
    for axis, options in AXIS_KERNEL_OPTIONS.items():
        for composite, tag in options:
            rationale = EXPERIMENT_RATIONALE.get(f"{axis}:{composite}", "—")
            print(f"  p2_{axis}_{tag:<10}  {composite:<28}  {rationale[:36]}…")
    total += 8

    print("\n  PHASE 3 — Final validation (best kernels + top-2 aggs, ~3 runs)")
    print("  " + "-"*64)
    print("  p3_best_<best_agg>         assembled best-per-axis kernels, best aggregation")
    print("  p3_best_<2nd_agg>          same kernels, 2nd-best aggregation for robustness")
    print("  p3_best_learned            always: learned weights on best kernel combo")
    total += 3

    print()
    print(f"  Total runs: ~{total}  vs {6 * 2**4} full grid (87% reduction)")
    print()
    print("  Skipped strategies (with reason):")
    print("    hard   — winner-takes-all is inconsistent with overlapping data")
    print("    poe    — overconfident (each expert sees ALL data → correlated)")
    print()
    print("  Key physics constraints:")
    print("    Q²   : non-stationary (DGLAP) → Gibbs kernels for W2/β experts")
    print("    W²   : power-law smooth → RBF preferred for Q2 expert")
    print("    β    : two-scale (elastic peak + perturbative rise) → SM")
    print("    z    : exponential suppression, smooth → RBF")
    print("="*68 + "\n")
