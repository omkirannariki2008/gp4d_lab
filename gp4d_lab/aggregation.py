"""
gp4d_lab.aggregation
====================
Combines predictions from the 4 axis-expert GPs into a single estimate.

Strategies
----------
  hard       — argmin σ² per test point (winner-takes-all)
  poe        — Product-of-Experts  (β_i = 1)
  gpoe       — Generalised PoE     (β_i = 1/M)
  rbcm       — Robust BCM          (β_i = differential entropy weight) [RECOMMENDED]
  precision  — Normalised precision weighting (no prior correction)
  learned    — Softmax weights trained on validation NLL

Math — common precision aggregation skeleton
--------------------------------------------
  σ_agg⁻²  = Σᵢ βᵢ · σᵢ⁻²   +   correction_term
  μ_agg     = σ_agg² · Σᵢ βᵢ · σᵢ⁻² · μᵢ

For rBCM (Deisenroth & Ng, 2015):
  βᵢ            = 0.5 · (log σ*² − log σᵢ²)   (differential entropy)
  correction    = (1 − Σ βᵢ) · σ*⁻²
  σ*²           = prior variance at test point (model.covar_module evaluated at x*)

References
----------
  Deisenroth & Ng 2015 — "Distributed Gaussian Processes" (rBCM)
  Cao & Fleet 2014     — gPoE
  MoGU 2025            — uncertainty-based gating
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .config import AggregationConfig

log = logging.getLogger("gp4d_lab.aggregation")

EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AggregationResult:
    mu:          np.ndarray    # combined predictive mean
    sigma:       np.ndarray    # combined predictive std
    weights:     np.ndarray    # (M, N) weight per expert per test point
    expert_means:  np.ndarray  # (M, N)
    expert_sigmas: np.ndarray  # (M, N)
    strategy:    str


# ─────────────────────────────────────────────────────────────────────────────
# Core aggregation functions
# ─────────────────────────────────────────────────────────────────────────────

def _precision_aggregate(
    means:   np.ndarray,   # (M, N)
    sigmas:  np.ndarray,   # (M, N)
    betas:   np.ndarray,   # (M, N) or (M,) broadcast
    prior_var: np.ndarray | None = None,  # (N,) for rBCM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Core precision-weighted aggregation.
    Returns (combined_mean, combined_sigma).
    """
    precisions = 1.0 / (sigmas ** 2 + EPS)             # (M, N)

    if betas.ndim == 1:
        betas = betas[:, None]                           # broadcast over N

    weighted_prec = betas * precisions                  # (M, N)
    sum_w_prec    = weighted_prec.sum(axis=0)            # (N,)

    # rBCM correction: prior precision term
    if prior_var is not None:
        prior_prec    = 1.0 / (prior_var + EPS)
        sum_beta      = betas.sum(axis=0)
        agg_prec      = sum_w_prec + (1.0 - sum_beta) * prior_prec
    else:
        agg_prec = sum_w_prec

    agg_var  = 1.0 / (agg_prec + EPS)
    agg_mean = agg_var * (weighted_prec * means).sum(axis=0)
    agg_sigma = np.sqrt(np.clip(agg_var, EPS, None))

    return agg_mean, agg_sigma


# ─────────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ─────────────────────────────────────────────────────────────────────────────

def agg_hard(
    means: np.ndarray, sigmas: np.ndarray, **_
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Winner-takes-all: pick the expert with lowest σ at each point."""
    best_idx = np.argmin(sigmas, axis=0)                   # (N,)
    N = means.shape[1]
    M = means.shape[0]
    agg_mu    = means[best_idx, np.arange(N)]
    agg_sigma = sigmas[best_idx, np.arange(N)]

    weights = np.zeros_like(sigmas)
    weights[best_idx, np.arange(N)] = 1.0
    return agg_mu, agg_sigma, weights


def agg_poe(
    means: np.ndarray, sigmas: np.ndarray, **_
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Product-of-Experts: β_i = 1 for all experts."""
    M, N = means.shape
    betas = np.ones(M)
    mu, sigma = _precision_aggregate(means, sigmas, betas, prior_var=None)
    weights = (1.0 / (sigmas ** 2 + EPS)) / ((1.0 / (sigmas ** 2 + EPS)).sum(axis=0))
    return mu, sigma, weights


def agg_gpoe(
    means: np.ndarray, sigmas: np.ndarray, **_
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generalised PoE: β_i = 1/M (uniform fractional weight)."""
    M, N = means.shape
    betas = np.full(M, 1.0 / M)
    mu, sigma = _precision_aggregate(means, sigmas, betas, prior_var=None)
    weights = (1.0 / (sigmas ** 2 + EPS)) / ((1.0 / (sigmas ** 2 + EPS)).sum(axis=0))
    return mu, sigma, weights


def agg_precision(
    means: np.ndarray, sigmas: np.ndarray, **_
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision-only: β_i proportional to 1/σ_i², normalised to sum=1."""
    prec    = 1.0 / (sigmas ** 2 + EPS)                 # (M, N)
    betas_n = prec / (prec.sum(axis=0) + EPS)            # (M, N) — normalised per point
    agg_prec = (betas_n * prec).sum(axis=0)
    agg_var  = 1.0 / (agg_prec + EPS)
    agg_mean = agg_var * (betas_n * prec * means).sum(axis=0)
    agg_sig  = np.sqrt(np.clip(agg_var, EPS, None))
    return agg_mean, agg_sig, betas_n


def agg_rbcm(
    means:     np.ndarray,
    sigmas:    np.ndarray,
    prior_var: np.ndarray,
    beta_floor: float = 1e-4,
    **_,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust BCM (Deisenroth & Ng 2015).

    β_i = 0.5 · (log σ*² − log σᵢ²)
    Weights are clipped at beta_floor to avoid zero/negative contribution.
    """
    log_prior = np.log(prior_var + EPS)           # (N,)
    log_sigmas = np.log(sigmas ** 2 + EPS)        # (M, N)

    betas = 0.5 * (log_prior[None, :] - log_sigmas)  # (M, N)
    betas = np.maximum(betas, beta_floor)

    mu, sigma = _precision_aggregate(means, sigmas, betas, prior_var=prior_var)

    # Normalise betas for reporting
    weights = betas / (betas.sum(axis=0) + EPS)
    return mu, sigma, weights


# ─────────────────────────────────────────────────────────────────────────────
# Learned-weight aggregator
# ─────────────────────────────────────────────────────────────────────────────

class LearnedWeightAggregator:
    """
    Learns a set of M global weights (one per expert) by minimising NLL
    on a held-out validation set.

    The combination is precision-weighted with softmax weights:
      β_i = softmax(w)[i]   ∀ i ∈ {0, …, M-1}

    The NLL for a Gaussian with precision-weighted combination:
      σ_agg⁻² = Σ_i β_i · σ_i⁻²
      μ_agg   = σ_agg² · Σ_i β_i · σ_i⁻² · μ_i
      NLL     = 0.5·log(2π/σ_agg²) + 0.5·σ_agg²·(y−μ_agg)²
    """

    def __init__(self, n_experts: int, cfg: AggregationConfig):
        self.n_experts  = n_experts
        self.cfg        = cfg
        self._raw_w:    torch.nn.Parameter | None = None
        self.trained_weights: np.ndarray | None = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        val_means:  np.ndarray,   # (M, N_val)
        val_sigmas: np.ndarray,   # (M, N_val)
        val_y:      np.ndarray,   # (N_val,)  — scaled
    ) -> "LearnedWeightAggregator":
        """Train raw_w on validation set via Adam."""
        M, N = val_means.shape
        device = "cpu"

        mu    = torch.tensor(val_means,  dtype=torch.float64)
        sig   = torch.tensor(val_sigmas, dtype=torch.float64)
        y     = torch.tensor(val_y,      dtype=torch.float64)

        EPS_t = torch.tensor(EPS)

        init = torch.zeros(M, dtype=torch.float64)
        if self.cfg.weight_init == "precision_based":
            avg_prec = (1.0 / (val_sigmas ** 2 + EPS)).mean(axis=1)
            init = torch.tensor(np.log(avg_prec + EPS), dtype=torch.float64)

        raw_w = torch.nn.Parameter(init)
        optimizer = torch.optim.Adam([raw_w], lr=self.cfg.weight_lr)

        log.info(f"[LearnedWeights] Training {M} expert weights, {self.cfg.weight_epochs} epochs")

        best_loss, best_w = float("inf"), raw_w.data.clone()

        for epoch in range(self.cfg.weight_epochs):
            optimizer.zero_grad()
            beta = torch.softmax(raw_w, dim=0)              # (M,)

            prec       = 1.0 / (sig ** 2 + EPS_t)           # (M, N)
            w_prec     = beta[:, None] * prec                # (M, N)
            agg_prec   = w_prec.sum(dim=0)                   # (N,)
            agg_var    = 1.0 / (agg_prec + EPS_t)
            agg_mu     = agg_var * (w_prec * mu).sum(dim=0)

            nll  = 0.5 * (torch.log(2 * torch.pi * agg_var + EPS_t)
                          + (y - agg_mu) ** 2 / (agg_var + EPS_t))
            loss = nll.mean() + self.cfg.weight_reg * (raw_w ** 2).mean()

            loss.backward()
            optimizer.step()

            val_loss = loss.item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_w    = raw_w.data.clone()

            if epoch % max(1, self.cfg.weight_epochs // 5) == 0:
                log.info(
                    f"[LearnedWeights] epoch {epoch:4d}  nll={nll.mean().item():.5f}"
                    f"  weights={torch.softmax(raw_w, 0).detach().numpy().round(3)}"
                )

        self._raw_w = best_w
        self.trained_weights = torch.softmax(best_w, dim=0).numpy()
        log.info(f"[LearnedWeights] Final weights: {self.trained_weights.round(4)}")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def combine(
        self,
        means:  np.ndarray,   # (M, N)
        sigmas: np.ndarray,   # (M, N)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply learned weights to (means, sigmas) → (agg_mean, agg_sigma, weights)."""
        if self.trained_weights is None:
            raise RuntimeError("Call fit() before combine()")
        betas = self.trained_weights                         # (M,)
        mu, sigma = _precision_aggregate(means, sigmas, betas, prior_var=None)
        weights = np.tile(betas[:, None], (1, means.shape[1]))
        return mu, sigma, weights


# ─────────────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class Aggregator:
    """
    Stateful aggregator — wraps all strategies under one interface.

    Usage
    -----
        agg = Aggregator(cfg)

        # During training — fit learned weights (optional)
        agg.fit(val_means, val_sigmas, val_y_scaled)

        # At inference
        result = agg.combine(expert_preds)   # list of (mu, sigma) per expert
    """

    def __init__(self, cfg: AggregationConfig):
        self.cfg = cfg
        self._learned: LearnedWeightAggregator | None = None
        self._n_experts = len(cfg.axes_enabled)

    # ── Fit learned weights ───────────────────────────────────────────────────

    def fit(
        self,
        val_means:  np.ndarray,   # (M, N_val)
        val_sigmas: np.ndarray,   # (M, N_val)
        val_y:      np.ndarray,   # (N_val,)
    ) -> "Aggregator":
        """Fit learned weights on validation set (no-op for non-learned strategies)."""
        if self.cfg.learn_weights or self.cfg.strategy == "learned":
            self._learned = LearnedWeightAggregator(
                n_experts=val_means.shape[0],
                cfg=self.cfg,
            ).fit(val_means, val_sigmas, val_y)
        return self

    # ── Combine ───────────────────────────────────────────────────────────────

    def combine(
        self,
        expert_predictions: list[tuple[np.ndarray, np.ndarray]],
        prior_var: np.ndarray | None = None,
    ) -> AggregationResult:
        """
        Parameters
        ----------
        expert_predictions : list of (mean_array, sigma_array) — length M
        prior_var          : (N,) prior variance for rBCM; required if strategy=="rbcm"

        Returns
        -------
        AggregationResult
        """
        means  = np.stack([p[0] for p in expert_predictions], axis=0)  # (M, N)
        sigmas = np.stack([p[1] for p in expert_predictions], axis=0)  # (M, N)

        strategy = self.cfg.strategy

        # Override to learned if weights were trained
        if self._learned is not None:
            mu, sigma, weights = self._learned.combine(means, sigmas)
            strategy = "learned"

        elif strategy == "hard":
            mu, sigma, weights = agg_hard(means, sigmas)

        elif strategy == "poe":
            mu, sigma, weights = agg_poe(means, sigmas)

        elif strategy == "gpoe":
            mu, sigma, weights = agg_gpoe(means, sigmas)

        elif strategy == "precision":
            mu, sigma, weights = agg_precision(means, sigmas)

        elif strategy == "rbcm":
            if prior_var is None:
                log.warning("rBCM: prior_var not provided, falling back to gpoe")
                mu, sigma, weights = agg_gpoe(means, sigmas)
            else:
                mu, sigma, weights = agg_rbcm(
                    means, sigmas, prior_var,
                    beta_floor=self.cfg.rbcm_beta_floor,
                )

        else:
            raise ValueError(f"Unknown aggregation strategy '{strategy}'")

        return AggregationResult(
            mu=mu,
            sigma=sigma,
            weights=weights,
            expert_means=means,
            expert_sigmas=sigmas,
            strategy=strategy,
        )

    @property
    def learned_weights(self) -> np.ndarray | None:
        if self._learned is not None:
            return self._learned.trained_weights
        return None
