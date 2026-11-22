"""
gp4d_lab.ensemble
=================
MultiAxisGPEnsemble — the main model class.

Architecture
------------
  For each enabled axis (q2, w2, beta, z):
    1. Partition training data by that axis into bins
    2. Train one ExactGP per bin (kernel from AxisKernelConfig)
    3. At inference: each axis-expert produces (μ, σ) for every test point
       using nearest-bin lookup for unseen axis values

  Aggregation:
    All M axis-experts produce (μᵢ, σᵢ). The Aggregator combines them
    using the configured strategy (rBCM by default).

Usage
-----
    from gp4d_lab.config import make_master_default
    from gp4d_lab.data import load_dataset
    from gp4d_lab.ensemble import MultiAxisGPEnsemble

    cfg     = make_master_default("data.csv")
    dataset = load_dataset(cfg.data)
    model   = MultiAxisGPEnsemble(cfg)
    model.fit(dataset)
    result  = model.predict(dataset.X_test)
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, AggregationConfig
from .data import Dataset, AxisPartitioner, AXIS_NAMES, AXIS_COL_IDX
from .models import fit_partition, predict_gp
from .aggregation import Aggregator, AggregationResult

log = logging.getLogger("gp4d_lab.ensemble")


# ─────────────────────────────────────────────────────────────────────────────
# Fitted expert container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AxisExpert:
    """One trained axis-expert: a partitioned GP over the remaining 3 dims."""

    axis:          str
    partitioner:   AxisPartitioner
    bin_models:    dict   # {bin_center: (model, likelihood, loss_history)}
    kernel_name:   str
    n_bins:        int
    train_time_s:  float
    bin_scalers:   dict = None   # {bin_center: local_scaler}


# ─────────────────────────────────────────────────────────────────────────────
# Main ensemble model
# ─────────────────────────────────────────────────────────────────────────────

class MultiAxisGPEnsemble:
    """
    4-axis partitioned GP ensemble with configurable kernels and aggregation.

    Each axis is treated as a "view" of the data. The ensemble combines
    all views at inference time using the configured aggregation strategy.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg        = cfg
        self._experts:  dict[str, AxisExpert] = {}
        self._aggregator: Aggregator | None = None
        self._device    = self._resolve_device(cfg.training.device)
        self._fitted    = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, dataset: Dataset) -> "MultiAxisGPEnsemble":
        """
        Train all axis-experts (or load them from checkpoints), then fit the aggregator's learned weights.
        """
        checkpoint_dir = Path(self.cfg.results_dir) / self.cfg.name / "models"
        
        # Try loading models first to support resumption
        if self.load_models(checkpoint_dir, dataset):
            log.info(f"[Ensemble] Models loaded successfully from checkpoint. Skipping training.")
            return self

        # Save target scaler reference
        self._scaler_y_fitted = dataset.scaler_y

        # ── Fit hurdle classifier ─────────────────────────────────────────────
        if self.cfg.data.use_hurdle_model:
            raw_y_train = dataset.scaler_y.inverse_transform(dataset.y_train).ravel()
            is_physical = raw_y_train > (self.cfg.data.hurdle_floor_val + 1e-4)
            
            clf_type = getattr(self.cfg.data, "hurdle_classifier_type", "hist_gb")
            if clf_type == "xgboost":
                try:
                    import xgboost as xgb
                    self._hurdle_classifier = xgb.XGBClassifier(
                        random_state=self.cfg.data.random_seed,
                        n_estimators=100,
                        max_depth=5,
                        eval_metric="logloss"
                    )
                except ImportError:
                    log.warning("xgboost not installed. Falling back to HistGradientBoostingClassifier.")
                    clf_type = "hist_gb"
            
            if clf_type == "hist_gb":
                try:
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    self._hurdle_classifier = HistGradientBoostingClassifier(random_state=self.cfg.data.random_seed)
                except ImportError:
                    log.warning("HistGradientBoostingClassifier not available. Falling back to RandomForestClassifier.")
                    clf_type = "rf"
                    
            if clf_type == "rf":
                from sklearn.ensemble import RandomForestClassifier
                self._hurdle_classifier = RandomForestClassifier(
                    random_state=self.cfg.data.random_seed,
                    n_estimators=100,
                    max_depth=8,
                    n_jobs=-1
                )
                
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight(class_weight="balanced", y=is_physical.astype(int))
            self._hurdle_classifier.fit(dataset.X_train, is_physical.astype(int), sample_weight=sample_weight)
            
            # Filter the training dataset for all downstream GP expert training and aggregation
            # using the bulk threshold (-17.955) to keep scaling parameters stable and clean.
            is_physical_gp = raw_y_train > (self.cfg.data.hurdle_floor_val + 1e-4)
            from .data import Dataset
            dataset = Dataset(
                X_train=dataset.X_train[is_physical_gp],
                y_train=dataset.y_train[is_physical_gp],
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                feature_cols=dataset.feature_cols,
                target_col=dataset.target_col,
                q2_bins_train=dataset.q2_bins_train[is_physical_gp],
                q2_bins_test=dataset.q2_bins_test,
                scaler_X=dataset.scaler_X,
                scaler_y=dataset.scaler_y,
                raw_X_train=dataset.raw_X_train[is_physical_gp],
                raw_X_test=dataset.raw_X_test,
            )

        # ── Handle validation split to avoid data leakage in learned weights ──
        if self.cfg.aggregation.learn_weights or self.cfg.aggregation.strategy == "learned":
            val_frac = self.cfg.aggregation.weight_val_frac
            N = len(dataset.X_train)
            n_val = min(N, max(2, int(N * val_frac)))
            
            rng = np.random.default_rng(self.cfg.data.random_seed + 999)
            all_idx = np.arange(N)
            rng.shuffle(all_idx)
            
            val_idx = all_idx[:n_val]
            train_idx = all_idx[n_val:]
            
            gp_train_dataset = Dataset(
                X_train=dataset.X_train[train_idx],
                y_train=dataset.y_train[train_idx],
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                feature_cols=dataset.feature_cols,
                target_col=dataset.target_col,
                q2_bins_train=dataset.q2_bins_train[train_idx],
                q2_bins_test=dataset.q2_bins_test,
                scaler_X=dataset.scaler_X,
                scaler_y=dataset.scaler_y,
                raw_X_train=dataset.raw_X_train[train_idx],
                raw_X_test=dataset.raw_X_test,
            )
            
            self._X_val = dataset.X_train[val_idx]
            self._y_val = dataset.y_train[val_idx]
            train_dataset = gp_train_dataset
        else:
            train_dataset = dataset
            self._X_val = None
            self._y_val = None

        axes = self.cfg.aggregation.axes_enabled

        for axis in axes:
            if axis not in self.cfg.kernels:
                log.warning(f"No kernel config for axis '{axis}' — skipping")
                continue
            log.info(f"\n{'='*60}")
            log.info(f"  Training axis-expert: {axis.upper()}")
            log.info(f"{'='*60}")
            expert = self._fit_axis_expert(axis, train_dataset)
            self._experts[axis] = expert

        # ── Aggregator setup ─────────────────────────────────────────────────
        self._aggregator = Aggregator(self.cfg.aggregation)

        if self.cfg.aggregation.learn_weights or self.cfg.aggregation.strategy == "learned":
            self._fit_learned_weights()

        self._fitted = True
        log.info("\n[Ensemble] Training complete.")
        
        # Save checkpoints immediately after training
        self.save_models(checkpoint_dir)
        
        # Re-load checkpoints immediately to re-anchor GP models on the full physical dataset for inference
        log.info("[Ensemble] Re-loading models from checkpoints to swap to physical inference anchors...")
        self.load_models(checkpoint_dir, dataset)
        
        return self

    def save_models(self, path: Path | str) -> None:
        """
        Save state dicts of all trained models and likelihoods to disk.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        log.info(f"[Ensemble] Saving model checkpoints to {path}...")
        
        # Save hurdle classifier if enabled
        if self.cfg.data.use_hurdle_model and hasattr(self, "_hurdle_classifier"):
            import pickle
            classifier_path = path / "hurdle_classifier.pkl"
            with open(classifier_path, "wb") as f:
                pickle.dump(self._hurdle_classifier, f)
            log.info(f"[Ensemble] Saved hurdle classifier to {classifier_path}")

        
        for axis, expert in self._experts.items():
            for bin_center, (model, likelihood, losses) in expert.bin_models.items():
                from .models import SKIGPModel
                local_scaler_y = None
                if expert.bin_scalers is not None:
                    local_scaler_y = expert.bin_scalers.get(bin_center)
                
                checkpoint = {
                    "model_state": model.state_dict(),
                    "likelihood_state": likelihood.state_dict(),
                    "is_ski": isinstance(model, SKIGPModel),
                    "losses": losses,
                    "local_scaler_median": getattr(local_scaler_y, "median", self._scaler_y_fitted.median),
                    "local_scaler_iqr": getattr(local_scaler_y, "iqr", self._scaler_y_fitted.iqr),
                }
                # Format to a fixed precision string to keep file names clean and consistent
                filename = f"{axis}_bin_{bin_center:.8f}.pt"
                torch.save(checkpoint, path / filename)
                
        # Save aggregator if weights are learned
        if self._aggregator and self._aggregator._learned is not None:
            agg_checkpoint = {
                "raw_w": self._aggregator._learned._raw_w,
                "trained_weights": self._aggregator._learned.trained_weights
            }
            torch.save(agg_checkpoint, path / "aggregator.pt")
            
        log.info("[Ensemble] All model checkpoints saved successfully.")

    def load_models(self, path: Path | str, dataset: Dataset) -> bool:
        """
        Attempt to load models from saved checkpoints.
        Returns True if successful, False otherwise.
        """
        path = Path(path)
        if not path.exists():
            log.info(f"[Ensemble] Checkpoint path {path} does not exist. Cannot load models.")
            return False
            
        self._scaler_y_fitted = dataset.scaler_y

        if self.cfg.data.use_hurdle_model:
            classifier_path = path / "hurdle_classifier.pkl"
            if not classifier_path.exists():
                log.info(f"[Ensemble] Hurdle classifier checkpoint {classifier_path} not found. Cannot load.")
                return False
            import pickle
            try:
                with open(classifier_path, "rb") as f:
                    self._hurdle_classifier = pickle.load(f)
                log.info("[Ensemble] Hurdle classifier loaded successfully.")
            except Exception as e:
                log.warning(f"[Ensemble] Failed to load hurdle classifier: {e}")
                return False
                
            raw_y_train = dataset.scaler_y.inverse_transform(dataset.y_train).ravel()
            is_physical = raw_y_train > (self.cfg.data.hurdle_floor_val + 1e-4)
            from .data import Dataset
            dataset = Dataset(
                X_train=dataset.X_train[is_physical],
                y_train=dataset.y_train[is_physical],
                X_test=dataset.X_test,
                y_test=dataset.y_test,
                feature_cols=dataset.feature_cols,
                target_col=dataset.target_col,
                q2_bins_train=dataset.q2_bins_train[is_physical],
                q2_bins_test=dataset.q2_bins_test,
                scaler_X=dataset.scaler_X,
                scaler_y=dataset.scaler_y,
                raw_X_train=dataset.raw_X_train[is_physical],
                raw_X_test=dataset.raw_X_test,
            )

        axes = self.cfg.aggregation.axes_enabled
        training = self.cfg.training
        
        loaded_experts = {}
        
        try:
            for axis in axes:
                if axis not in self.cfg.kernels:
                    continue
                    
                kernel_cfg = self.cfg.kernels[axis].resolve()
                
                partitioner = AxisPartitioner(axis=axis, n_bins=None)
                partitioner.fit(dataset.X_train)
                splits = list(partitioner.split(dataset.X_train, dataset.y_train))
                
                bin_models = {}
                bin_scalers = {}
                for bin_center, X_sub, y_sub in splits:
                    filename = f"{axis}_bin_{bin_center:.8f}.pt"
                    filepath = path / filename
                    if not filepath.exists():
                         log.info(f"[Ensemble] Checkpoint file {filepath} not found. Aborting load.")
                         return False
                         
                    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
                    
                    # Reconstruct local scaler from checkpoint parameters, falling back to global scaler
                    from .scalers import TargetScalerConditionalRobust
                    local_scaler = TargetScalerConditionalRobust(threshold=self.cfg.data.hurdle_floor_val)
                    local_scaler.median = checkpoint.get("local_scaler_median", dataset.scaler_y.median)
                    local_scaler.iqr = checkpoint.get("local_scaler_iqr", dataset.scaler_y.iqr)
                    bin_scalers[float(bin_center)] = local_scaler

                    # Map globally-scaled y_sub targets to local scaled space matching train bounds
                    y_sub_raw = dataset.scaler_y.inverse_transform(y_sub.reshape(-1, 1)).ravel()
                    y_sub_local = local_scaler.transform(y_sub_raw).ravel()

                    dtype = torch.float64 if getattr(training, 'use_double', False) else torch.float32
                    tx = torch.tensor(X_sub, dtype=dtype)
                    ty = torch.tensor(y_sub_local, dtype=dtype)
                    
                    # Reconstruct the model structure
                    if checkpoint.get("is_ski", False):
                        from .models import SKIGPModel
                        from gpytorch.likelihoods import GaussianLikelihood
                        lh = GaussianLikelihood()
                        model = SKIGPModel(tx, ty, lh, grid_size=training.ski_grid_size)
                    else:
                        from .models import build_gp
                        model, lh = build_gp(
                            tx, ty, kernel_cfg, training, init_from_data=False,
                            bin_center_scaled=bin_center, scaler_x=dataset.scaler_X,
                            scaler_y=local_scaler
                        )
                        
                    model.load_state_dict(checkpoint["model_state"])
                    lh.load_state_dict(checkpoint["likelihood_state"])
                    
                    # Fix GPyTorch loading bug where constraint gets converted to Interval with upper_bound inf,
                    # which combined with raw_noise = -inf produces NaN noise on CPU.
                    import gpytorch
                    fixed_noise = getattr(training, "fixed_noise", None)
                    if fixed_noise is not None:
                        lh.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(fixed_noise))
                        lh.noise_covar.raw_noise.requires_grad = False
                    
                    # Keep on CPU
                    model.cpu()
                    lh.cpu()
                    
                    bin_models[float(bin_center)] = (model, lh, checkpoint.get("losses", []))
                    
                loaded_experts[axis] = AxisExpert(
                    axis=axis,
                    partitioner=partitioner,
                    bin_models=bin_models,
                    kernel_name=kernel_cfg.composite_kernel or kernel_cfg.primary_kernel,
                    n_bins=partitioner.n_effective_bins,
                    train_time_s=0.0,
                    bin_scalers=bin_scalers,
                )
                
            self._experts = loaded_experts
            
            # ── Aggregator setup ─────────────────────────────────────────────────
            self._aggregator = Aggregator(self.cfg.aggregation)
            if self.cfg.aggregation.learn_weights or self.cfg.aggregation.strategy == "learned":
                agg_path = path / "aggregator.pt"
                if agg_path.exists():
                    agg_checkpoint = torch.load(agg_path, map_location="cpu", weights_only=False)
                    from .aggregation import LearnedWeightAggregator
                    self._aggregator._learned = LearnedWeightAggregator(
                        n_experts=len(axes),
                        cfg=self.cfg.aggregation
                    )
                    self._aggregator._learned._raw_w = agg_checkpoint["raw_w"]
                    self._aggregator._learned.trained_weights = agg_checkpoint["trained_weights"]
                else:
                    log.info("[Ensemble] Aggregator checkpoint not found, re-fitting learned weights...")
                    self._fit_learned_weights(dataset)
                    
            self._fitted = True
            log.info("[Ensemble] Successfully loaded all models from checkpoints.")
            return True
            
        except Exception as e:
            log.warning(f"[Ensemble] Failed to load models from checkpoints: {e}", exc_info=True)
            return False

    # ── Per-axis expert training ──────────────────────────────────────────────

    def _fit_axis_expert(self, axis: str, dataset: Dataset) -> AxisExpert:
        kernel_cfg = self.cfg.kernels[axis].resolve()
        training   = self.cfg.training
        col_idx    = AXIS_COL_IDX[axis]

        # Partition by unique grid values for all axes to ensure each partition
        # only sees the data related to its specific grid coordinate (max ~3600 points).
        # This keeps memory usage low and ensures Exact GP training succeeds without OOM.
        n_bins = None

        partitioner = AxisPartitioner(axis=axis, n_bins=n_bins)
        partitioner.fit(dataset.X_train)

        splits = partitioner.split(dataset.X_train, dataset.y_train)
        log.info(
            f"  [{axis}] {partitioner.n_effective_bins} bins  |  "
            f"kernel={kernel_cfg.composite_kernel or kernel_cfg.primary_kernel}"
        )

        bin_models: dict = {}
        bin_scalers: dict = {}
        t0 = time.time()

        for bin_center, X_sub, y_sub in splits:
            n_sub  = len(X_sub)
            prefix = f"    [{axis}|bin={bin_center:.3g}|n={n_sub}] "
            log.info(f"{prefix}")

            # Unscale y_sub using global scaler to get original raw target values
            y_sub_raw = dataset.scaler_y.inverse_transform(y_sub.reshape(-1, 1)).ravel()
            
            # Fit local target conditional robust scaler with threshold = -87.4
            # (so deep tail physical points are included)
            from .scalers import TargetScalerConditionalRobust
            local_scaler_y = TargetScalerConditionalRobust(threshold=-87.4)
            
            # Fallback: if bin has too few physical points (< 10), use global scaler parameters
            mask_phys_sub = y_sub_raw > (self.cfg.data.hurdle_floor_val + 1e-3)
            if np.sum(mask_phys_sub) < 10:
                local_scaler_y.median = dataset.scaler_y.median
                local_scaler_y.iqr = dataset.scaler_y.iqr
            else:
                local_scaler_y.fit(y_sub_raw)
            
            y_sub_scaled = local_scaler_y.transform(y_sub_raw).ravel()
            bin_scalers[float(bin_center)] = local_scaler_y

            model, lh, losses = fit_partition(
                X_sub=X_sub,
                y_sub=y_sub_scaled,
                kernel_cfg=kernel_cfg,
                training_cfg=training,
                device=self._device,
                prefix=prefix,
                bin_center_scaled=bin_center,
                scaler_x=dataset.scaler_X,
                scaler_y=local_scaler_y,
            )

            # Keep on CPU
            model.cpu(); lh.cpu()
            bin_models[float(bin_center)] = (model, lh, losses)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        log.info(f"  [{axis}] Done in {elapsed:.1f}s")

        return AxisExpert(
            axis=axis,
            partitioner=partitioner,
            bin_models=bin_models,
            kernel_name=kernel_cfg.composite_kernel or kernel_cfg.primary_kernel,
            n_bins=partitioner.n_effective_bins,
            train_time_s=elapsed,
            bin_scalers=bin_scalers,
        )

    # ── Learned weight fitting ────────────────────────────────────────────────

    def _fit_learned_weights(self) -> None:
        """
        Train aggregation weights using the held-out validation split.
        """
        if not hasattr(self, "_X_val") or self._X_val is None:
            raise RuntimeError("Validation split for learned weights not found. Call fit() first.")
            
        X_val = self._X_val
        y_val = self._y_val
        n_val = len(X_val)

        log.info(f"[LearnedWeights] Collecting expert predictions on {n_val} held-out validation points")
        expert_preds = self._get_expert_predictions(X_val)

        val_means  = np.stack([p[0] for p in expert_preds], axis=0)   # (M, N_val)
        val_sigmas = np.stack([p[1] for p in expert_preds], axis=0)

        self._aggregator.fit(val_means, val_sigmas, y_val)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        return_experts: bool = False,
    ) -> AggregationResult:
        """
        Predict on scaled test data X  →  AggregationResult.

        Parameters
        ----------
        X              : (N, 4) scaled feature array
        return_experts : if True, expert-level means/sigmas are included in result
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        N = len(X)
        log.info(f"[Ensemble] Predicting on {N} points...")
        
        if self.cfg.data.use_hurdle_model:
            # 1. Run hurdle classifier to get probabilities
            probs = self._hurdle_classifier.predict_proba(X)
            p_phys = probs[:, 1]
            
            # Gating routing threshold: evaluate GP on any point with p_phys > 0.05
            is_physical_pred = p_phys > 0.05
            n_phys = np.sum(is_physical_pred)
            n_floor = N - n_phys
            log.info(f"[HurdleModel] Evaluating GP on {n_phys} points (p_phys > 0.05); {n_floor} points are pure floor.")

            # 2. Allocate output arrays (all in global scaled space)
            floor_val_scaled = self._scaler_y_fitted.transform(np.array([[self.cfg.data.hurdle_floor_val]]))[0, 0]
            
            # Global scale factor for standard deviation mapping
            if hasattr(self._scaler_y_fitted, "iqr"):
                global_scale = self._scaler_y_fitted.iqr
            elif hasattr(self._scaler_y_fitted, "std_"):
                global_scale = self._scaler_y_fitted.std_
            else:
                global_scale = 1.0
                
            var_floor_scaled = (0.1 / global_scale) ** 2
            
            agg_mu = np.zeros(N)
            agg_sigma = np.ones(N)
            
            M = len(self.cfg.aggregation.axes_enabled)
            agg_weights = np.zeros((M, N))
            expert_means = np.zeros((M, N))
            expert_sigmas = np.ones((M, N))

            # Set defaults for pure floor points
            agg_mu[:] = floor_val_scaled
            agg_sigma[:] = 0.1 / global_scale
            for j in range(M):
                expert_means[j, :] = floor_val_scaled
                expert_sigmas[j, :] = 0.1 / global_scale
                agg_weights[j, :] = 1.0 / M

            # 4. Predict on physical points and combine using mixture formulas
            if n_phys > 0:
                X_phys = X[is_physical_pred]
                p = p_phys[is_physical_pred]
                
                expert_preds_phys = self._get_expert_predictions(X_phys)
                prior_var_phys = self._estimate_prior_var(X_phys)

                res_phys = self._aggregator.combine(expert_preds_phys, prior_var=prior_var_phys)

                # Banded Hurdle GP prediction gate:
                # physical if p > 0.5 OR (p > 0.05 and GP prediction >= -30.0)
                res_phys_mu_unscaled = self._scaler_y_fitted.inverse_transform(res_phys.mu.reshape(-1, 1)).ravel()
                is_phys_gate = (p > 0.5) | ((p > 0.05) & (res_phys_mu_unscaled >= -30.0))
                
                # Mixture calculations using is_phys_gate
                mu_comb = np.zeros_like(res_phys.mu)
                sigma_comb = np.ones_like(res_phys.sigma) * (0.1 / global_scale)
                
                phys_active = is_phys_gate
                floor_active = ~is_phys_gate
                
                # Active physical points
                if np.any(phys_active):
                    mu_comb[phys_active] = (1.0 - p[phys_active]) * floor_val_scaled + p[phys_active] * res_phys.mu[phys_active]
                    var_gp = res_phys.sigma[phys_active] ** 2
                    var_comb = (1.0 - p[phys_active]) * (var_floor_scaled + floor_val_scaled**2) + p[phys_active] * (var_gp + res_phys.mu[phys_active]**2) - mu_comb[phys_active]**2
                    var_comb = np.maximum(var_comb, 1e-8)
                    sigma_comb[phys_active] = np.sqrt(var_comb)
                
                # Floor-gate points
                if np.any(floor_active):
                    mu_comb[floor_active] = floor_val_scaled
                    sigma_comb[floor_active] = 0.1 / global_scale
                
                agg_mu[is_physical_pred] = mu_comb
                agg_sigma[is_physical_pred] = sigma_comb
                agg_weights[:, is_physical_pred] = res_phys.weights
                
                # Apply mixture calculation to individual experts as well
                for j in range(M):
                    exp_mu = expert_preds_phys[j][0]
                    exp_var = expert_preds_phys[j][1] ** 2
                    
                    exp_mu_comb = np.zeros_like(exp_mu)
                    exp_sigma_comb = np.ones_like(exp_mu) * (0.1 / global_scale)
                    
                    if np.any(phys_active):
                        exp_mu_comb[phys_active] = (1.0 - p[phys_active]) * floor_val_scaled + p[phys_active] * exp_mu[phys_active]
                        exp_var_comb = (1.0 - p[phys_active]) * (var_floor_scaled + floor_val_scaled**2) + p[phys_active] * (exp_var[phys_active] + exp_mu[phys_active]**2) - exp_mu_comb[phys_active]**2
                        exp_var_comb = np.maximum(exp_var_comb, 1e-8)
                        exp_sigma_comb[phys_active] = np.sqrt(exp_var_comb)
                        
                    if np.any(floor_active):
                        exp_mu_comb[floor_active] = floor_val_scaled
                        exp_sigma_comb[floor_active] = 0.1 / global_scale
                    
                    expert_means[j, is_physical_pred] = exp_mu_comb
                    expert_sigmas[j, is_physical_pred] = exp_sigma_comb

            # Clear caches of all models
            for expert in self._experts.values():
                for model, likelihood, _ in expert.bin_models.values():
                    model._clear_cache()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return AggregationResult(
                mu=agg_mu,
                sigma=agg_sigma,
                weights=agg_weights,
                expert_means=expert_means,
                expert_sigmas=expert_sigmas,
                strategy=self.cfg.aggregation.strategy,
            )

        # Predict on full test set by expert (handles chunking internally per bin)
        expert_preds = self._get_expert_predictions(X)
        prior_var = self._estimate_prior_var(X)
        
        # Clear caches of all models at the end of predict()
        for expert in self._experts.values():
            for model, likelihood, _ in expert.bin_models.values():
                model._clear_cache()
                
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        res = self._aggregator.combine(expert_preds, prior_var=prior_var)
        return res

    # ── Per-expert prediction ─────────────────────────────────────────────────

    def _get_expert_predictions(
        self, X: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        For each axis-expert, predict on X.
        Returns list of (mean, sigma) per expert.
        """
        predictions = []
        for axis, expert in self._experts.items():
            mu, sigma = self._predict_one_expert(expert, X)
            predictions.append((mu, sigma))
        return predictions

    def _predict_one_expert(
        self, expert: AxisExpert, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict using a single axis-expert.
        Maps all test points to their nearest bins, and queries each bin's GP in chunks.
        Reuses the Cholesky factor cache of the active model on GPU across chunks.
        """
        N = len(X)
        mu_out    = np.zeros(N)
        sigma_out = np.ones(N)

        bin_ids, X_reduced = expert.partitioner.reduce(X)
        bin_centers = list(expert.bin_models.keys())
        unique_bins = np.unique(bin_ids)
        n_bins_to_predict = len(unique_bins)

        log.info(f"  [{expert.axis}] Predicting on {N} points across {n_bins_to_predict} bins...")
        
        # For progress/ETA logging
        start_time = time.time()
        last_log_time = start_time
        
        for idx, bin_id in enumerate(unique_bins):
            # Look up nearest bin center
            center = list(expert.bin_models.keys())[
                min(bin_id, len(bin_centers) - 1)
            ]
            if center not in expert.bin_models:
                # Nearest-neighbor fallback
                keys = np.array(bin_centers)
                axis_val = X[bin_ids == bin_id, AXIS_COL_IDX[expert.axis]]
                if len(axis_val) > 0:
                    nearest = keys[np.argmin(np.abs(keys - axis_val.mean()))]
                    center = float(nearest)
                else:
                    center = bin_centers[0]

            model, lh, _ = expert.bin_models[float(center)]
            mask = bin_ids == bin_id
            n_sub = np.sum(mask)
            if n_sub == 0:
                continue

            # Move model and likelihood to device once for this bin
            model = model.to(self._device)
            lh = lh.to(self._device)

            X_sub = X_reduced[mask]
            sub_mu = np.zeros(n_sub)
            sub_std = np.ones(n_sub)

            # Chunk size to prevent GPU memory spikes (5,000 is a good size)
            chunk_size = getattr(self.cfg.training, "predict_chunk_size", 5000)
            
            # Predict in chunks, keeping the model on device to reuse cached Cholesky
            for chunk_start in range(0, n_sub, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_sub)
                tx = torch.tensor(X_sub[chunk_start:chunk_end], dtype=torch.float32)
                # Call predict_gp with keep_on_device=True to reuse caches!
                m, s = predict_gp(model, lh, tx, self._device, keep_on_device=True)
                sub_mu[chunk_start:chunk_end] = m
                sub_std[chunk_start:chunk_end] = s

            # Retrieve local scaler
            local_scaler_y = None
            if expert.bin_scalers is not None:
                local_scaler_y = expert.bin_scalers.get(float(center))
            if local_scaler_y is None:
                local_scaler_y = self._scaler_y_fitted

            sub_mu_unscaled = local_scaler_y.inverse_transform(sub_mu.reshape(-1, 1)).ravel()
            sub_std_unscaled = local_scaler_y.std_to_original(sub_std)

            # Map to global scaled space
            sub_mu_global = self._scaler_y_fitted.transform(sub_mu_unscaled.reshape(-1, 1)).ravel()
            
            if hasattr(self._scaler_y_fitted, "iqr"):
                global_scale = self._scaler_y_fitted.iqr
            elif hasattr(self._scaler_y_fitted, "std_"):
                global_scale = self._scaler_y_fitted.std_
            else:
                global_scale = 1.0
                
            sub_std_global = sub_std_unscaled / global_scale

            mu_out[mask] = sub_mu_global
            sigma_out[mask] = sub_std_global

            # Move model/likelihood back to CPU and clear GPU cache
            model.cpu()
            lh.cpu()
            model._clear_cache()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Progress logging and ETA calculation
            current_time = time.time()
            elapsed = current_time - start_time
            avg_time = elapsed / (idx + 1)
            eta = avg_time * (n_bins_to_predict - (idx + 1))
            
            # Log every 10% of bins or every 5 seconds to avoid spamming the log
            # while keeping the user informed of progress.
            if (idx + 1) % max(1, n_bins_to_predict // 10) == 0 or (current_time - last_log_time) > 5:
                log.info(
                    f"    [{expert.axis}] Progress: {idx+1}/{n_bins_to_predict} bins "
                    f"({(idx+1)/n_bins_to_predict*100:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                )
                last_log_time = current_time

        total_elapsed = time.time() - start_time
        log.info(f"  [{expert.axis}] Prediction complete in {total_elapsed:.1f}s")
        return mu_out, sigma_out

    def _estimate_prior_var(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate prior variance σ*²(x*) = k(x*, x*) for rBCM.

        The rBCM beta formula is β_k = 0.5 * (log σ*² − log σ_k²).
        σ*²(x*) is the GP prior predictive variance at x* — i.e. the kernel
        diagonal k(x*, x*).  For a ScaleKernel with any stationary base kernel,
        k(x*,x*) = outputscale * 1.0 = outputscale.

        Old implementation: took the outputscale from ONLY the first bin of the
        first expert and broadcast it as a constant — causing β to be off by
        up to 311% when bins differ in amplitude.

        New implementation: for each test point, look up the bin it falls into
        per expert, read that bin's outputscale, then average across experts.
        This gives a per-point, per-bin-adjusted prior variance.
        """
        N = len(X)
        # Accumulate per-point outputscale across experts; shape (M, N)
        all_pv = []

        for axis, expert in self._experts.items():
            if not expert.bin_models:
                continue

            # Get the bin assignment for this expert's partitioning axis
            bin_ids, _ = expert.partitioner.reduce(X)
            bin_centers = list(expert.bin_models.keys())
            pv_per_point = np.ones(N)  # default

            unique_bins = np.unique(bin_ids)
            for bin_id in unique_bins:
                # Resolve bin_id → center key (same logic as _predict_one_expert)
                center = bin_centers[min(bin_id, len(bin_centers) - 1)]
                if center not in expert.bin_models:
                    keys = np.array(bin_centers)
                    axis_vals = X[bin_ids == bin_id, AXIS_COL_IDX[axis]]
                    if len(axis_vals) > 0:
                        center = float(keys[np.argmin(np.abs(keys - axis_vals.mean()))])
                    else:
                        center = bin_centers[0]

                model, _, _ = expert.bin_models[float(center)]
                try:
                    with torch.no_grad():
                        covar_mod = model.covar_module
                        pv = float(covar_mod.outputscale.item()) \
                            if hasattr(covar_mod, "outputscale") else 1.0
                except Exception:
                    pv = 1.0

                # Retrieve local scaler and scale prior variance to global scaled scale
                local_scaler_y = None
                if expert.bin_scalers is not None:
                    local_scaler_y = expert.bin_scalers.get(float(center))
                if local_scaler_y is None:
                    local_scaler_y = self._scaler_y_fitted

                if hasattr(local_scaler_y, "iqr"):
                    local_scale = local_scaler_y.iqr
                elif hasattr(local_scaler_y, "std_"):
                    local_scale = local_scaler_y.std_
                else:
                    local_scale = 1.0

                if hasattr(self._scaler_y_fitted, "iqr"):
                    global_scale = self._scaler_y_fitted.iqr
                elif hasattr(self._scaler_y_fitted, "std_"):
                    global_scale = self._scaler_y_fitted.std_
                else:
                    global_scale = 1.0

                pv_global = pv * (local_scale ** 2) / (global_scale ** 2)
                mask = bin_ids == bin_id
                pv_per_point[mask] = pv_global

            all_pv.append(pv_per_point)

        if not all_pv:
            return np.ones(N)

        # Average outputscale over all experts at each test point
        return np.stack(all_pv, axis=0).mean(axis=0)   # (N,)


    def diagnose_experts(
        self,
        X: np.ndarray,
        y_true_scaled: np.ndarray,
        scaler_y,
    ) -> dict:
        """
        Run each expert independently and report per-expert metrics.
        Call with a validation sample (e.g. 10k points) to see which
        experts contribute signal and which add noise.

        Returns dict of {axis: {mae, rmse, mean_sigma_scaled, learned_weight}}
        """
        from .metrics import evaluate
        results = {}
        for axis, expert in self._experts.items():
            mu, sigma = self._predict_one_expert(expert, X)
            y_pred = scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
            y_std  = scaler_y.std_to_original(sigma)
            y_true = scaler_y.inverse_transform(
                y_true_scaled.reshape(-1, 1)
            ).ravel()
            m = evaluate(y_true, y_pred, y_std,
                         ["mae", "rmse", "mean_uncertainty"])
            results[axis] = {
                "mae":               m["mae"],
                "rmse":              m["rmse"],
                "mean_sigma":        m["mean_uncertainty"],
                "mean_sigma_scaled": float(sigma.mean()),
            }

        if self._aggregator and self._aggregator.learned_weights is not None:
            for i, axis in enumerate(self._experts.keys()):
                results[axis]["learned_weight"] = float(
                    self._aggregator.learned_weights[i]
                )
        return results

    def print_expert_diagnostics(
        self,
        X: np.ndarray,
        y_true_scaled: np.ndarray,
        scaler_y,
    ) -> None:
        diag = self.diagnose_experts(X, y_true_scaled, scaler_y)
        print(f"\n{'Expert':8s}  {'MAE':>10s}  {'RMSE':>10s}"
              f"  {'mean_σ_scaled':>14s}  {'learned_w':>10s}")
        print("-" * 62)
        for axis, m in diag.items():
            wt = f"{m.get('learned_weight', float('nan')):.4f}"
            print(
                f"  {axis:6s}  {m['mae']:10.5f}  {m['rmse']:10.5f}"
                f"  {m['mean_sigma_scaled']:14.5f}  {wt:>10s}"
            )
        print("-" * 62)

    def regional_competence_map(
        self,
        X: np.ndarray,            # scaled test/val features  (N, 4)
        y_true: np.ndarray,       # ORIGINAL-scale targets    (N,)
        scaler_y,
        n_bins: int = 5,
    ) -> dict:
        """
        For each axis expert, compute MAE per bin of each input variable.
        Returns a nested dict showing WHERE each expert is best/worst.

        Structure
        ---------
        result[axis][col_name][bin_range_str] = mae_float

        Usage
        -----
        After calling this, use print_competence_table() to display a
        human-readable summary, or inspect the dict directly.
        """
        from .metrics import mae as _mae
        from .data import AXIS_NAMES

        result = {}
        for axis, expert in self._experts.items():
            mu_scaled, _ = self._predict_one_expert(expert, X)
            y_pred = scaler_y.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()

            result[axis] = {}
            for col_idx, col_name in enumerate(AXIS_NAMES):
                col_vals = X[:, col_idx]
                edges = np.percentile(col_vals, np.linspace(0, 100, n_bins + 1))
                edges[0] -= 1e-9
                edges[-1] += 1e-9

                bin_maes = {}
                for b in range(n_bins):
                    mask = (col_vals >= edges[b]) & (col_vals < edges[b + 1])
                    if mask.sum() < 5:
                        continue
                    bin_label = f"{edges[b]:.3f}–{edges[b+1]:.3f}"
                    bin_maes[bin_label] = float(_mae(y_true[mask], y_pred[mask]))
                result[axis][col_name] = bin_maes

        return result

    def print_competence_table(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        scaler_y,
        n_bins: int = 5,
    ) -> None:
        """
        Print a human-readable lookup table of expert competence across input regions.

        For each input axis (q2, w2, beta, z) and each bin of that axis,
        the expert with the LOWEST MAE is highlighted with *.

        This shows at a glance:
          - Which expert dominates in high-Q2 vs low-Q2 regions
          - Which expert is most accurate near beta=1 vs beta=0
          - etc.
        """
        competence = self.regional_competence_map(X, y_true, scaler_y, n_bins)
        from .data import AXIS_NAMES
        expert_names = list(competence.keys())

        for col_name in AXIS_NAMES:
            print(f"\n{'='*70}")
            print(f"  EXPERT COMPETENCE BY {col_name.upper()} BINS")
            print(f"{'='*70}")
            # Collect all bins for this column
            all_bins: set = set()
            for ax in expert_names:
                all_bins.update(competence[ax].get(col_name, {}).keys())
            all_bins_sorted = sorted(all_bins)

            # Header
            header = f"  {'Bin Range':22s}"
            for ax in expert_names:
                header += f"  {ax.upper():>10s}"
            header += "  BEST"
            print(header)
            print("  " + "-" * 65)

            for bin_label in all_bins_sorted:
                row = f"  {bin_label:22s}"
                maes = {}
                for ax in expert_names:
                    v = competence[ax].get(col_name, {}).get(bin_label, float("nan"))
                    maes[ax] = v
                best_ax = min(maes, key=lambda a: maes[a] if not np.isnan(maes[a]) else 1e9)
                for ax in expert_names:
                    v = maes[ax]
                    cell = f"{v:10.4f}" if not np.isnan(v) else f"{'---':>10s}"
                    row += f"  {cell}"
                row += f"  {best_ax.upper()}*"
                print(row)
        print()

    def nn_routing_predict(
        self,
        X_val: np.ndarray,
        y_val_true: np.ndarray,
        X_test: np.ndarray,
        scaler_y,
        k: int = 5,
        cached_test_expert_preds: "tuple[np.ndarray, np.ndarray] | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Option B: Nearest-Neighbour expert routing.

        Parameters
        ----------
        X_val       : scaled validation features (N_val, 4)
        y_val_true  : original-scale validation targets (N_val,)
        X_test      : scaled test features (N_test, 4)
        scaler_y    : target scaler with inverse_transform()
        k           : KNN neighbours
        cached_test_expert_preds : optional (expert_means, expert_sigmas) both
            shape (M, N_test) in SCALED space from a prior AggregationResult
            (pass result_a.expert_means, result_a.expert_sigmas).  When given,
            the full test-set re-prediction is skipped entirely — saves 4×
            expert inference passes on 972K points.

        Returns
        -------
        (mu_routed, sigma_routed) in SCALED space.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from .metrics import mae as _mae

        expert_names = list(self._experts.keys())
        n_val = len(X_val)

        # --- Step 1: Get each expert's predictions on the validation set ---
        val_mus    = []
        val_sigmas = []
        for axis in expert_names:
            expert = self._experts[axis]
            mu_s, sig_s = self._predict_one_expert(expert, X_val)
            val_mus.append(mu_s)
            val_sigmas.append(sig_s)

        # --- Step 2: Find best expert per validation point (vectorised) ---
        # val_mus shape: (M, N_val) — inverse-transform each expert's predictions
        val_mus_arr = np.stack(val_mus, axis=0)          # (M, N_val)
        # Inverse transform: apply scaler_y to each row
        val_orig = np.array([
            scaler_y.inverse_transform(val_mus_arr[j].reshape(-1, 1)).ravel()
            for j in range(len(expert_names))
        ])                                               # (M, N_val)
        # Absolute error vs y_val_true (broadcast)
        errors = np.abs(val_orig - y_val_true[None, :])  # (M, N_val)
        best_expert_labels = np.argmin(errors, axis=0).astype(int)  # (N_val,)

        # --- Step 3: Train KNN classifier ---
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_val, best_expert_labels)

        # --- Step 4: Route test points to best expert ---
        test_labels = knn.predict(X_test)

        # --- Step 5: Build routed predictions ---
        n_test = len(X_test)
        mu_routed    = np.zeros(n_test)
        sigma_routed = np.ones(n_test)

        if cached_test_expert_preds is not None:
            # Reuse expert predictions from a prior predict() call — no re-inference
            cached_means, cached_sigmas = cached_test_expert_preds  # (M, N_test)
            log.info("[NN Routing] Using cached expert predictions (no re-inference).")
            for label_idx in range(len(expert_names)):
                mask = test_labels == label_idx
                mu_routed[mask]    = cached_means[label_idx][mask]
                sigma_routed[mask] = cached_sigmas[label_idx][mask]
        else:
            # Lazy per-expert prediction on test set (only for assigned experts)
            test_mus    = {}
            test_sigmas = {}
            for label in np.unique(test_labels):
                axis = expert_names[label]
                if axis not in test_mus:
                    expert = self._experts[axis]
                    mu_t, sig_t = self._predict_one_expert(expert, X_test)
                    test_mus[axis]    = mu_t
                    test_sigmas[axis] = sig_t
            for i in range(n_test):
                axis = expert_names[test_labels[i]]
                mu_routed[i]    = test_mus[axis][i]
                sigma_routed[i] = test_sigmas[axis][i]

        log.info(
            f"[NN Routing] Expert assignment on test set:\n"
            + "\n".join(
                f"    {expert_names[lbl]:8s}: {int((test_labels==lbl).sum()):,} points"
                for lbl in range(len(expert_names))
            )
        )
        return mu_routed, sigma_routed

    # ── Utilities ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["MultiAxisGPEnsemble Summary", "=" * 50]
        for axis, expert in self._experts.items():
            lines.append(
                f"  {axis.upper():5s} | "
                f"kernel={expert.kernel_name:30s} | "
                f"bins={expert.n_bins:3d} | "
                f"time={expert.train_time_s:.1f}s"
            )
        if self._aggregator and self._aggregator.learned_weights is not None:
            axes  = list(self._experts.keys())
            wts   = self._aggregator.learned_weights
            lines.append("\n  Learned weights:")
            for a, w in zip(axes, wts):
                lines.append(f"    {a}: {w:.4f}")
        lines.append(f"\n  Aggregation: {self.cfg.aggregation.strategy}")
        return "\n".join(lines)
