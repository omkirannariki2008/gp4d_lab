"""
gp4d_lab.models
===============
GPyTorch ExactGP wrapper used by each axis-partitioned expert.

Classes
-------
  ExactGPModel       — standard exact GP (Cholesky / CG solver)
  SKIGPModel         — SKI-GP fallback for OOM bins
  build_gp           — factory: picks model class and builds kernel
  train_gp           — training loop with early stopping
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP

from .config import AxisKernelConfig, TrainingConfig
from .kernels import build_axis_kernel, initialize_kernel_from_data, describe_kernel

log = logging.getLogger("gp4d_lab.models")


# ─────────────────────────────────────────────────────────────────────────────
# Model classes
# ─────────────────────────────────────────────────────────────────────────────

class ExactGPModel(ExactGP):
    """Standard exact GP with configurable covariance kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        covar_module: gpytorch.kernels.Kernel,
        mean_module: Optional[gpytorch.means.Mean] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = mean_module or gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class PhysicsMeanFunction(gpytorch.means.Mean):
    """
    Parametric physics-informed mean function mapping the Regge-like trajectories:
        mean(x) = c0 + c1 * log(Q2) + c2 * log(W2) + c3 * log(1 - beta)
    Unscales input features using the mean and scale of the fitted input scaler.
    """
    def __init__(self, axis: str, bin_center_scaled: float, scaler_x, scaler_y=None):
        super().__init__()
        self.axis = axis
        self.scaler_y = scaler_y
        
        # Extract means and scales from scaler_x if it's standard scaler wrapper
        if hasattr(scaler_x, "_sk") and hasattr(scaler_x._sk, "mean_"):
            sk = scaler_x._sk
            self.means = torch.nn.Parameter(torch.tensor(sk.mean_, dtype=torch.float32), requires_grad=False)
            self.scales = torch.nn.Parameter(torch.tensor(sk.scale_, dtype=torch.float32), requires_grad=False)
        else:
            self.means = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)
            self.scales = torch.nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=False)
            
        axis_col_idx = {"q2": 0, "w2": 1, "beta": 2, "z": 3}
        axis_idx = axis_col_idx.get(axis, 0)
        
        if hasattr(scaler_x, "_sk") and hasattr(scaler_x._sk, "mean_"):
            sk = scaler_x._sk
            self.partition_val = float(bin_center_scaled * sk.scale_[axis_idx] + sk.mean_[axis_idx])
        else:
            self.partition_val = float(bin_center_scaled)
            
        # Learnable parameters
        self.register_parameter("c0", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("c1", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("c2", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("c3", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = self.means.to(dtype=x.dtype, device=x.device)
        scales = self.scales.to(dtype=x.dtype, device=x.device)
        
        # Reconstruct 4D unscaled features
        if self.axis == "q2":
            logQ2 = torch.full(x.shape[:-1], self.partition_val, dtype=x.dtype, device=x.device)
            logW2 = x[..., 0] * scales[1] + means[1]
            beta  = x[..., 1] * scales[2] + means[2]
        elif self.axis == "w2":
            logQ2 = x[..., 0] * scales[0] + means[0]
            logW2 = torch.full(x.shape[:-1], self.partition_val, dtype=x.dtype, device=x.device)
            beta  = x[..., 1] * scales[2] + means[2]
        elif self.axis == "beta":
            logQ2 = x[..., 0] * scales[0] + means[0]
            logW2 = x[..., 1] * scales[1] + means[1]
            beta  = torch.full(x.shape[:-1], self.partition_val, dtype=x.dtype, device=x.device)
        else: # z
            logQ2 = x[..., 0] * scales[0] + means[0]
            logW2 = x[..., 1] * scales[1] + means[1]
            beta  = x[..., 2] * scales[2] + means[2]
            
        log_one_minus_beta = torch.log(torch.clamp(1.0 - beta, min=1e-5))
        physics_mean_unscaled = self.c0 + self.c1 * logQ2 + self.c2 * logW2 + self.c3 * log_one_minus_beta
        
        if self.scaler_y is not None:
            if hasattr(self.scaler_y, "median"):
                median = torch.tensor(self.scaler_y.median, dtype=x.dtype, device=x.device)
                iqr = torch.tensor(self.scaler_y.iqr, dtype=x.dtype, device=x.device)
                return (physics_mean_unscaled - median) / iqr
            elif hasattr(self.scaler_y, "mean_"):
                mean = torch.tensor(self.scaler_y.mean_, dtype=x.dtype, device=x.device)
                std = torch.tensor(self.scaler_y.std_, dtype=x.dtype, device=x.device)
                return (physics_mean_unscaled - mean) / std
                
        return physics_mean_unscaled



class SKIGPModel(ExactGP):
    """SKI / KISS-GP fallback — uses grid-interpolation for OOM bins."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        grid_size: int = 30,
        base_kernel: Optional[gpytorch.kernels.Kernel] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if base_kernel is None:
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)

        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            ScaleKernel(base_kernel),
            grid_size=grid_size,
            num_dims=train_x.shape[-1],
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def _apply_lengthscale_constraints(
    kernel,
    lo: float,
    hi: float,
) -> None:
    """
    Recursively apply Interval(lo, hi) to ALL lengthscale parameters
    in the kernel tree. Prevents collapse below grid spacing.
    """
    constraint = gpytorch.constraints.Interval(lo, hi)

    if hasattr(kernel, "lengthscale_constraint"):
        kernel.lengthscale_constraint = constraint
        with torch.no_grad():
            mid = (lo + hi) / 2.0
            try:
                ls = kernel.lengthscale
                kernel.lengthscale = torch.full_like(ls, mid)
            except Exception:
                pass   # Gibbs kernels manage their own ℓ; skip

    # Recurse into sub-kernels
    if hasattr(kernel, "kernels"):
        for sub in kernel.kernels:
            _apply_lengthscale_constraints(sub, lo, hi)
    if hasattr(kernel, "base_kernel"):
        _apply_lengthscale_constraints(kernel.base_kernel, lo, hi)
    if hasattr(kernel, "k1"):
        _apply_lengthscale_constraints(kernel.k1, lo, hi)
    if hasattr(kernel, "k2"):
        _apply_lengthscale_constraints(kernel.k2, lo, hi)


def build_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel_cfg: AxisKernelConfig,
    training_cfg: TrainingConfig,
    init_from_data: bool = True,
    bin_center_scaled: Optional[float] = None,
    scaler_x: Optional[object] = None,
    scaler_y: Optional[object] = None,
) -> tuple[ExactGP, GaussianLikelihood]:
    """
    Build (model, likelihood) from config.
    Initializes SM kernel parameters from data if applicable.
    """
    input_dim    = train_x.shape[-1]
    covar_module = build_axis_kernel(kernel_cfg, remaining_input_dim=input_dim)

    # Likelihood construction (fixed noise floor vs. optimized)
    fixed_noise = getattr(training_cfg, "fixed_noise", None)
    if fixed_noise is not None:
        likelihood = GaussianLikelihood()
        likelihood.noise = fixed_noise
        likelihood.noise_covar.raw_noise.requires_grad = False
    else:
        min_noise = getattr(training_cfg, "min_noise", 0.02)
        max_noise = getattr(training_cfg, "max_noise", None)

        if max_noise is not None:
            noise_constraint = gpytorch.constraints.Interval(min_noise, max_noise)
        else:
            noise_constraint = gpytorch.constraints.GreaterThan(min_noise)

        likelihood = GaussianLikelihood(noise_constraint=noise_constraint)

    # Mean module construction
    mean_module = None
    if getattr(training_cfg, "use_physics_mean", False):
        if bin_center_scaled is not None and scaler_x is not None:
            mean_module = PhysicsMeanFunction(
                axis=kernel_cfg.axis,
                bin_center_scaled=bin_center_scaled,
                scaler_x=scaler_x,
                scaler_y=scaler_y
            )
        else:
            log.warning("use_physics_mean is True but bin_center_scaled or scaler_x is None. Falling back to ConstantMean.")

    model = ExactGPModel(train_x, train_y, likelihood, covar_module, mean_module=mean_module)


    # Apply lengthscale constraints to prevent collapse
    if getattr(training_cfg, "constrain_lengthscale", True):
        lo = getattr(training_cfg, "min_lengthscale", 0.10)
        hi = getattr(training_cfg, "max_lengthscale", 10.0)
        _apply_lengthscale_constraints(covar_module, lo, hi)

    # SM-specific init — important for convergence
    if init_from_data:
        initialize_kernel_from_data(covar_module, train_x, train_y)

    log.debug(f"Built kernel:\n{describe_kernel(covar_module)}")
    return model, likelihood


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_gp(
    model:      ExactGP,
    likelihood: GaussianLikelihood,
    train_x:    torch.Tensor,
    train_y:    torch.Tensor,
    cfg:        TrainingConfig,
    device:     str,
    prefix:     str = "",
    jitter:     Optional[float] = None,
) -> tuple[ExactGP, GaussianLikelihood, list[float]]:
    """
    Adam + MLL training with early stopping.

    Returns
    -------
    model, likelihood, loss_history
    """
    model      = model.to(device)
    likelihood = likelihood.to(device)
    tx         = train_x.to(device)
    ty         = train_y.to(device)

    model.train()
    likelihood.train()

    # Deduplicate: ExactGP registers likelihood as submodule, so
    # model.parameters() already includes likelihood.parameters().
    # Passing both creates duplicate-param warning (pytorch#40967).
    nn_params = []
    gp_params = []
    seen: set = set()
    
    # Identify and separate parameters
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if "feature_extractor" in name or "net" in name:
            nn_params.append(param)
        else:
            gp_params.append(param)
            
    for param in likelihood.parameters():
        if id(param) not in seen:
            seen.add(id(param))
            gp_params.append(param)
            
    param_groups = []
    if gp_params:
        param_groups.append({"params": gp_params, "lr": cfg.lr})
    if nn_params:
        # Neural networks are highly sensitive; use 0.01 learning rate and 1e-4 weight decay
        param_groups.append({"params": nn_params, "lr": 0.01, "weight_decay": 1e-4})
        
    optimizer = torch.optim.Adam(param_groups)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss   = float("inf")
    patience    = 0
    loss_history: list[float] = []

    # Use Cholesky for bins small enough — far more stable on grid/duplicate data.
    # CG struggles when the kernel matrix is near-singular (duplicate points).
    # cholesky_jitter adds a small diagonal perturbation for numerical safety.
    n_train          = train_x.shape[0]
    chol_threshold   = getattr(cfg, "max_cholesky_size", 2000)
    chol_jitter      = getattr(cfg, "cholesky_jitter",   1e-4)

    with gpytorch.settings.cg_tolerance(cfg.cg_tol), \
         gpytorch.settings.max_cg_iterations(cfg.max_cg_iter), \
         gpytorch.settings.max_cholesky_size(max(chol_threshold, n_train + 1)), \
         gpytorch.settings.cholesky_jitter(chol_jitter, 1e-6):

        for epoch in range(cfg.n_epochs):
            optimizer.zero_grad()
            output = model(tx)
            loss   = -mll(output, ty)
            loss.backward()
            optimizer.step()

            val = loss.item()
            loss_history.append(val)

            if cfg.verbose and (epoch % max(1, cfg.n_epochs // 10) == 0):
                log.info(f"{prefix}epoch {epoch:4d}/{cfg.n_epochs}  loss={val:.5f}")

            # Early stopping
            if val < best_loss - cfg.early_stop_delta:
                best_loss = val
                patience  = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    log.info(f"{prefix}Early stop at epoch {epoch} (best loss={best_loss:.5f})")
                    break

    return model, likelihood, loss_history


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────────────────────────────────────

def predict_gp(
    model:      ExactGP,
    likelihood: GaussianLikelihood,
    test_x:     torch.Tensor,
    device:     str,
    keep_on_device: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mean, std) as numpy arrays. Model stays on cpu; moves temporarily.
    """
    model      = model.to(device)
    likelihood = likelihood.to(device)
    tx         = test_x.to(device)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), \
         gpytorch.settings.max_cholesky_size(5000), \
         gpytorch.settings.cholesky_jitter(1e-3):
        # max_cholesky_size(5000): forces exact Cholesky (not CG) for all our
        # per-bin training sets (≤3600 points).
        # cholesky_jitter(1e-3): Matérn-1/2 at n=3600 is barely PSD — the
        # default jitter sequence 1e-6→1e-5→1e-4 fails with NotPSDError.
        # Starting at 1e-3 gives the sequence 1e-3→1e-2→1e-1, which succeeds.
        pred = likelihood(model(tx))
        mu   = pred.mean.cpu().numpy()
        std  = pred.stddev.cpu().numpy().clip(1e-9)

    if not keep_on_device:
        model.cpu()
        likelihood.cpu()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return mu, std


# ─────────────────────────────────────────────────────────────────────────────
# Fit one partition with OOM fallback
# ─────────────────────────────────────────────────────────────────────────────

def fit_partition(
    X_sub:       np.ndarray,
    y_sub:       np.ndarray,
    kernel_cfg:  AxisKernelConfig,
    training_cfg: TrainingConfig,
    device:      str,
    prefix:      str = "",
    bin_center_scaled: Optional[float] = None,
    scaler_x: Optional[object] = None,
    scaler_y: Optional[object] = None,
) -> tuple[ExactGP, GaussianLikelihood, list[float]]:
    """
    Fit a single GP on a partition, with SKI-GP fallback on OOM,
    and automatic retries for PSD/numerical stability errors.
    """
    import copy
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Define retry strategies
    # use_double: use float64 (more numerically stable)
    # max_cholesky_size_override: if set, limits when we use Cholesky vs CG
    # jitter_override: diagonal jitter value to add
    strategies = [
        # Strategy 0: Standard configuration (usually float32)
        {"use_double": getattr(training_cfg, 'use_double', False), "max_cholesky_size_override": None, "jitter_override": getattr(training_cfg, 'cholesky_jitter', 1e-4)},
        # Strategy 1: Force CG solver for large matrix solves (set max_cholesky_size to 2000) with higher jitter
        {"use_double": getattr(training_cfg, 'use_double', False), "max_cholesky_size_override": 2000, "jitter_override": 1e-3},
        # Strategy 2: Switch to float64 (double precision)
        {"use_double": True, "max_cholesky_size_override": None, "jitter_override": getattr(training_cfg, 'cholesky_jitter', 1e-4)},
        # Strategy 3: Switch to float64 + CG solver with higher jitter
        {"use_double": True, "max_cholesky_size_override": 2000, "jitter_override": 1e-3},
    ]

    last_err = None
    for attempt, strategy in enumerate(strategies):
        # Skip Strategy 1 if the dataset is already small (e.g. n_train <= 2000), as it behaves identically to Strategy 0
        n_train = X_sub.shape[0]
        if attempt == 1 and n_train <= 2000:
            continue
        # Skip Strategy 2/3 if use_double is already True by default, as it would be redundant
        if attempt >= 2 and getattr(training_cfg, 'use_double', False):
            continue

        try:
            dtype = torch.float64 if strategy["use_double"] else torch.float32
            tx = torch.tensor(X_sub, dtype=dtype)
            ty = torch.tensor(y_sub, dtype=dtype)

            # Create a modified config copy
            cfg_copy = copy.deepcopy(training_cfg)
            cfg_copy.use_double = strategy["use_double"]
            if strategy["max_cholesky_size_override"] is not None:
                cfg_copy.max_cholesky_size = strategy["max_cholesky_size_override"]
            
            model, lh = build_gp(
                tx, ty, kernel_cfg, cfg_copy,
                bin_center_scaled=bin_center_scaled,
                scaler_x=scaler_x,
                scaler_y=scaler_y
            )
            model, lh, losses = train_gp(
                model, lh, tx, ty, cfg_copy, device, prefix,
                jitter=strategy["jitter_override"]
            )
            return model, lh, losses

        except Exception as e:
            err_msg = str(e).lower()
            # If it's a memory/OOM error, don't retry PSD strategies, fall back to SKI-GP/skip immediately
            if "out of memory" in err_msg or isinstance(e, MemoryError):
                log.warning(f"{prefix}OOM encountered during training attempt {attempt}. Proceeding to SKI-GP fallback.")
                break
            
            log.warning(
                f"{prefix}Attempt {attempt} failed with numerical error: {e}. "
                f"Retrying with strategy {attempt + 1}..."
            )
            last_err = e
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.warning(f"{prefix}All PSD recovery strategies failed. Falling back to SKI-GP.")
    
    # Fallback to SKI-GP on CPU/GPU
    try:
        dtype = torch.float64 if getattr(training_cfg, 'use_double', False) else torch.float32
        tx = torch.tensor(X_sub, dtype=dtype)
        ty = torch.tensor(y_sub, dtype=dtype)
        
        lh = GaussianLikelihood()
        model = SKIGPModel(tx, ty, lh, grid_size=training_cfg.ski_grid_size)
        model, lh, losses = train_gp(model, lh, tx, ty, training_cfg, device, prefix + "[SKI-FALLBACK]")
        return model, lh, losses
    except Exception as e:
        if training_cfg.oom_fallback == "skip":
            log.error(f"{prefix}SKI fallback failed: {e}. Skipping partition and returning dummy model.")
            lh = GaussianLikelihood()
            model = SKIGPModel(tx[:2], ty[:2], lh)
            return model, lh, []
        raise e
