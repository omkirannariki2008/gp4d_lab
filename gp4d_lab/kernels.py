"""
gp4d_lab.kernels
================
Kernel zoo for the 4-axis partitioned GP ensemble.

Includes
--------
  GibbsKernel1D      — Non-stationary variable-length-scale kernel (3 variants)
  build_axis_kernel  — Factory: builds the right product kernel given an AxisKernelConfig
  KERNEL_SIGNATURES  — Registry of all supported kernel names

Gibbs Kernel Math
-----------------
  k_Gibbs(x, x') = sqrt(2·ℓ(x)·ℓ(x') / (ℓ(x)² + ℓ(x')²))
                   · exp(-(x - x')² / (ℓ(x)² + ℓ(x')²))

  The length-scale function ℓ(x) can be:
    power_law : ℓ(x) = softplus(ℓ₀) · exp(α·x)          — shrinks/grows with x
    sigmoid   : ℓ(x) = ℓ_min + Δ · σ(−slope·(x − x₀))  — transitions at x₀
    linear    : ℓ(x) = softplus(a + b·x)                 — affine + softplus
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel, SpectralMixtureKernel

from .config import AxisKernelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Gibbs Kernel — 1D, applied to a single active dimension
# ─────────────────────────────────────────────────────────────────────────────

class PowerLawLengthscale(torch.nn.Module):
    def __init__(self, alpha_init: float = -0.5):
        super().__init__()
        self.register_parameter(
            "raw_ell0",
            torch.nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            "alpha",
            torch.nn.Parameter(torch.tensor([alpha_init]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        EPS = 1e-5
        ell0 = F.softplus(self.raw_ell0) + EPS
        return ell0 * torch.exp(self.alpha * x)


class SigmoidLengthscale(torch.nn.Module):
    def __init__(self, slope_init: float = 2.0, x0_init: float = 0.0):
        super().__init__()
        self.register_parameter(
            "raw_ell_min",
            torch.nn.Parameter(torch.tensor([-1.0]))
        )
        self.register_parameter(
            "raw_ell_range",
            torch.nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            "raw_slope",
            torch.nn.Parameter(torch.tensor([math.log(math.exp(slope_init) - 1)]))
        )
        self.register_parameter(
            "x0",
            torch.nn.Parameter(torch.tensor([x0_init]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        EPS = 1e-5
        ell_min = F.softplus(self.raw_ell_min) + EPS
        ell_range = F.softplus(self.raw_ell_range) + EPS
        slope = F.softplus(self.raw_slope) + EPS
        return ell_min + ell_range * torch.sigmoid(-slope * (x - self.x0))


class LinearLengthscale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "raw_a",
            torch.nn.Parameter(torch.zeros(1))
        )
        self.register_parameter(
            "raw_b",
            torch.nn.Parameter(torch.zeros(1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        EPS = 1e-5
        return F.softplus(self.raw_a + self.raw_b * x) + EPS


class MLPLengthscale(torch.nn.Module):
    """
    Non-stationary lengthscale function represented by a Multi-Layer Perceptron (MLP).
    Outputs a lengthscale constrained to [min_val, max_val] via sigmoid scaling.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, init_val: float = 1.0, min_val: float = 0.1, max_val: float = 10.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )
        # Initialize bias of the last linear layer to yield init_val after sigmoid
        with torch.no_grad():
            p = (init_val - min_val) / (max_val - min_val)
            p = max(1e-5, min(1.0 - 1e-5, p))
            raw_init = math.log(p / (1.0 - p))
            self.net[-1].bias.fill_(raw_init)
            self.net[-1].weight.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(self.net(x))



class GibbsKernel1D(gpytorch.kernels.GibbsKernel):
    """
    Non-stationary Gibbs kernel (Gibbs 1997, Paciorek & Schervish 2004)
    wrapping GPyTorch's native GibbsKernel.
    Operates on exactly ONE input dimension (use active_dims in product kernels).

    Parameters
    ----------
    variant : str
        "power_law" | "sigmoid" | "linear"
    active_dims : tuple/list, optional
        Which input column this kernel operates on.
    """
    def __init__(
        self,
        variant: str = "power_law",
        alpha_init: float = -0.5,
        slope_init: float = 2.0,
        x0_init: float = 0.0,
        mlp_hidden_dim: int = 16,
        **kwargs,
    ):
        if variant == "power_law":
            lengthscale_fn = PowerLawLengthscale(alpha_init)
        elif variant == "sigmoid":
            lengthscale_fn = SigmoidLengthscale(slope_init, x0_init)
        elif variant == "linear":
            lengthscale_fn = LinearLengthscale()
        elif variant == "mlp":
            lengthscale_fn = MLPLengthscale(input_dim=1, hidden_dim=mlp_hidden_dim, init_val=1.0)
        else:
            raise ValueError(
                f"Unknown GibbsKernel1D variant '{variant}'. "
                "Choose 'power_law', 'sigmoid', 'linear', or 'mlp'."
            )
        super().__init__(lengthscale_fn=lengthscale_fn, **kwargs)
        self.variant = variant


    def extra_repr(self) -> str:
        return f"variant={self.variant}"


class GibbsBellKernel1D(gpytorch.kernels.Kernel):
    """
    Non-stationary Gibbs kernel with Gaussian BELL-shaped length-scale.

    ℓ(x) = ℓ_min + Δ · exp(−γ · (x − x_peak)²)

    Designed for functions with a physical PEAK in the interior and
    floor/zero regions at BOTH extremes (e.g. logA vs Q² in diffraction):
      - Short ℓ at low Q² (photoproduction floor)
      - Long ℓ at mid Q² (physical cross-section peak)
      - Short ℓ at high Q² (kinematic saturation floor)

    All four parameters (ℓ_min, Δ, γ, x_peak) are learned by MLL.
    """

    has_lengthscale = False

    def __init__(self, peak_init: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter("raw_ell_min",    torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("raw_delta",       torch.nn.Parameter(torch.ones(1)))
        self.register_parameter("raw_log_gamma",   torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("peak",            torch.nn.Parameter(torch.tensor([peak_init])))

    def _ell(self, x: torch.Tensor) -> torch.Tensor:
        ell_min = torch.nn.functional.softplus(self.raw_ell_min) + 1e-5
        delta   = torch.nn.functional.softplus(self.raw_delta)   + 1e-5
        gamma   = torch.exp(self.raw_log_gamma)  # always positive
        return ell_min + delta * torch.exp(-gamma * (x - self.peak) ** 2)

    def forward(self, x1, x2, **params):
        x1_v = x1.squeeze(-1)
        x2_v = x2.squeeze(-1)
        ell1 = self._ell(x1_v).unsqueeze(-1)
        ell2 = self._ell(x2_v).unsqueeze(-2)
        ell1_sq = ell1 ** 2
        ell2_sq = ell2 ** 2
        sum_sq  = ell1_sq + ell2_sq + 1e-10
        prefactor = torch.sqrt(2.0 * ell1 * ell2 / sum_sq)
        diff_sq   = (x1_v.unsqueeze(-1) - x2_v.unsqueeze(-2)) ** 2
        return prefactor * torch.exp(-diff_sq / sum_sq)

    def extra_repr(self):
        return "variant=bell"


class MultiDimGibbsKernel(gpytorch.kernels.Kernel):
    """
    Fully multi-dimensional non-stationary Gibbs covariance kernel.
    The lengthscale vector l(x) = (l_1(x), ..., l_D(x)) depends on the entire
    multi-dimensional input state x via a PyTorch Multi-Layer Perceptron (MLP).
    """
    has_lengthscale = False

    def __init__(self, input_dim: int, hidden_dim: int = 16, init_val: float = 1.0, min_val: float = 0.1, max_val: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.min_val = min_val
        self.max_val = max_val
        
        # MLP outputs D lengthscales (one for each input dimension)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize bias of the last linear layer to yield init_val after sigmoid
        with torch.no_grad():
            p = (init_val - min_val) / (max_val - min_val)
            p = max(1e-5, min(1.0 - 1e-5, p))
            raw_init = math.log(p / (1.0 - p))
            self.net[-1].bias.fill_(raw_init)
            self.net[-1].weight.fill_(0.0)

    def _lengthscales(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(self.net(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        EPS = 1e-10
        L1 = self._lengthscales(x1)  # (..., N, D)
        
        if diag:
            L2 = self._lengthscales(x2)  # (..., N, D)
            sum_sq = L1**2 + L2**2 + EPS
            prefactor = torch.prod(torch.sqrt(2.0 * L1 * L2 / sum_sq), dim=-1)
            diff_sq = (x1 - x2)**2
            exponent = torch.sum(-diff_sq / sum_sq, dim=-1)
            return prefactor * torch.exp(exponent)
        
        # Pairwise covariance
        L2 = self._lengthscales(x2)  # (..., M, D)
        L1_exp = L1.unsqueeze(-2)    # (..., N, 1, D)
        L2_exp = L2.unsqueeze(-3)    # (..., 1, M, D)
        
        sum_sq = L1_exp**2 + L2_exp**2 + EPS
        prefactor = torch.prod(torch.sqrt(2.0 * L1_exp * L2_exp / sum_sq), dim=-1) # (..., N, M)
        
        diff_sq = (x1.unsqueeze(-2) - x2.unsqueeze(-3))**2
        exponent = torch.sum(-diff_sq / sum_sq, dim=-1) # (..., N, M)
        
        return prefactor * torch.exp(exponent)



# ─────────────────────────────────────────────────────────────────────────────
# ChangePoint Kernel — 1D, applied to a single active dimension
# ─────────────────────────────────────────────────────────────────────────────

class ChangePointKernel1D(gpytorch.kernels.Kernel):
    """
    ChangePoint kernel that switches between two Matérn-1/2 kernels
    at a transition location along a single dimension.
    """
    def __init__(self, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.k1 = MaternKernel(nu=0.5, active_dims=active_dims)
        self.k2 = MaternKernel(nu=0.5, active_dims=active_dims)
        self.register_parameter("raw_location", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("raw_steepness", torch.nn.Parameter(torch.tensor([2.0])))

    @property
    def location(self):
        return self.raw_location

    @location.setter
    def location(self, val):
        self.raw_location.data.copy_(val)

    @property
    def steepness(self):
        return F.softplus(self.raw_steepness)

    @steepness.setter
    def steepness(self, val):
        self.raw_steepness.data.copy_(torch.log(torch.exp(val) - 1.0))

    def forward(self, x1, x2, diag=False, **params):
        dim = self.active_dims[0] if self.active_dims is not None else 0
        v1 = x1[..., [dim]]
        v2 = x2[..., [dim]]

        # Compute sigmoid weights
        steepness = F.softplus(self.raw_steepness) + 1e-5
        sig1 = torch.sigmoid(steepness * (v1 - self.raw_location))
        sig2 = torch.sigmoid(steepness * (v2 - self.raw_location))

        covar1 = self.k1(x1, x2, diag=diag, **params)
        covar2 = self.k2(x1, x2, diag=diag, **params)

        if diag:
            w1 = sig1.squeeze(-1)
            w2 = sig2.squeeze(-1)
            return w1 * w2 * covar1 + (1.0 - w1) * (1.0 - w2) * covar2
        else:
            w1 = sig1
            w2 = sig2.transpose(-1, -2)
            return w1 * w2 * covar1 + (1.0 - w1) * (1.0 - w2) * covar2


# ─────────────────────────────────────────────────────────────────────────────
# Kernel factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_gibbs(variant: str, cfg: AxisKernelConfig, active_dims: list) -> GibbsKernel1D:
    return GibbsKernel1D(
        variant=variant,
        alpha_init=cfg.gibbs_alpha_init,
        slope_init=cfg.gibbs_slope_init,
        x0_init=cfg.gibbs_x0_init,
        mlp_hidden_dim=getattr(cfg, "gibbs_mlp_hidden_dim", 16),
        active_dims=active_dims,
    )



def _make_matern(nu: float, input_dim: int, use_ard: bool, active_dims: Optional[list]) -> MaternKernel:
    kw: dict = dict(nu=nu)
    if use_ard and input_dim > 0:
        kw["ard_num_dims"] = input_dim
    if active_dims is not None:
        kw["active_dims"] = active_dims
    return MaternKernel(**kw)


def _make_sm(num_mixtures: int, ard_num_dims: int, active_dims: list) -> SpectralMixtureKernel:
    return SpectralMixtureKernel(
        num_mixtures=num_mixtures,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
    )

class DKLKernel(gpytorch.kernels.Kernel):
    """
    Deep Kernel Learning (DKL) wrapper with a residual skip-connection.
    Passes inputs through a neural network feature extractor added to a linear projection,
    ensuring that at initialization the mapping is close to an identity/projection.
    """
    def __init__(self, base_kernel: gpytorch.kernels.Kernel, input_dim: int, latent_dim: int = 2, hidden_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        
        # Skip connection projection if dimensions don't match
        if input_dim != latent_dim:
            self.skip_proj = torch.nn.Linear(input_dim, latent_dim, bias=False)
            # Initialize skip projection to be standard orthographic projection/identity-like
            with torch.no_grad():
                torch.nn.init.eye_(self.skip_proj.weight[:min(input_dim, latent_dim), :min(input_dim, latent_dim)])
        else:
            self.skip_proj = None
            
        # Feature extractor neural network (residual correction)
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        
        # Initialize MLP weights to be very small, so residual correction starts at ~0
        with torch.no_grad():
            for m in self.feature_extractor.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params) -> torch.Tensor:
        proj1 = self.skip_proj(x1) if self.skip_proj is not None else x1
        proj2 = self.skip_proj(x2) if self.skip_proj is not None else x2
        
        z1 = proj1 + self.feature_extractor(x1)
        z2 = proj2 + self.feature_extractor(x2)
        return self.base_kernel.forward(z1, z2, diag=diag, **params)
def build_axis_kernel(cfg: AxisKernelConfig, remaining_input_dim: int = 3) -> Kernel:
    """
    Build the product kernel for one axis-partitioned GP expert.

    Parameters
    ----------
    cfg : AxisKernelConfig
        Resolved (composite_kernel expanded) kernel spec.
    remaining_input_dim : int
        Number of input dimensions the expert sees (D-1 after partitioning).
        Default 3 (full 4D problem, one axis removed).

    Returns
    -------
    gpytorch.kernels.Kernel — optionally wrapped in ScaleKernel.
    """

    # Expand composite shorthand
    cfg = cfg.resolve()
    D = remaining_input_dim

    # ── Deep Kernel Learning (DKL) early exits ───────────────────────────────
    if cfg.primary_kernel == "dkl_matern52":
        latent_dim = getattr(cfg, "dkl_latent_dim", 2)
        hidden_dim = getattr(cfg, "dkl_hidden_dim", 16)
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=latent_dim)
        k = DKLKernel(
            base_kernel=base_kernel,
            input_dim=D,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        return ScaleKernel(k) if cfg.use_scale_kernel else k

    if cfg.primary_kernel == "dkl_gibbs":
        latent_dim = getattr(cfg, "dkl_latent_dim", 2)
        hidden_dim = getattr(cfg, "dkl_hidden_dim", 16)
        base_kernel = MultiDimGibbsKernel(
            input_dim=latent_dim,
            hidden_dim=getattr(cfg, "gibbs_mlp_hidden_dim", 16),
            init_val=1.0
        )
        k = DKLKernel(
            base_kernel=base_kernel,
            input_dim=D,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        return ScaleKernel(k) if cfg.use_scale_kernel else k

    # ── Multi-dimensional Gibbs kernel early exit ─────────────────────────────
    if cfg.primary_kernel == "gibbs_multidim":
        k = MultiDimGibbsKernel(
            input_dim=D,
            hidden_dim=getattr(cfg, "gibbs_mlp_hidden_dim", 16),
            init_val=1.0
        )
        return ScaleKernel(k) if cfg.use_scale_kernel else k


    # ── Special 3-component product for W2-expert ─────────────────────────
    if cfg.composite_kernel == "gibbs_bell_x_gibbs_sigmoid_x_matern12":
        k_bell = GibbsBellKernel1D(
            peak_init=cfg.gibbs_bell_peak_init, active_dims=[0]
        )
        k_sig  = GibbsKernel1D(
            variant="sigmoid",
            x0_init=cfg.gibbs_x0_beta_init,
            slope_init=cfg.gibbs_slope_init,
            active_dims=[1]
        )
        k_mat  = MaternKernel(nu=0.5, active_dims=[2])
        k = k_bell * k_sig * k_mat
        return ScaleKernel(k) if cfg.use_scale_kernel else k

    pk  = cfg.primary_kernel
    sk  = cfg.secondary_kernel
    dim = cfg.primary_dim         # which dim gets the primary kernel

    # Dims for secondary kernel
    secondary_dims = [i for i in range(D) if i != dim]

    # ── Build primary kernel ──────────────────────────────────────────────────

    def _primary(k_name: str) -> Kernel:
        if k_name == "matern52":
            return _make_matern(2.5, 1, False, [dim])
        elif k_name == "matern32":
            return _make_matern(1.5, 1, False, [dim])
        elif k_name == "matern12":
            return _make_matern(0.5, 1, False, [dim])
        elif k_name == "changepoint":
            return ChangePointKernel1D(active_dims=[dim])
        elif k_name == "rbf":
            return RBFKernel(active_dims=[dim])
        elif k_name == "gibbs_power":
            return _make_gibbs("power_law", cfg, [dim])
        elif k_name == "gibbs_sigmoid":
            return _make_gibbs("sigmoid", cfg, [dim])
        elif k_name == "gibbs_bell":
            return GibbsBellKernel1D(
                peak_init=cfg.gibbs_bell_peak_init,
                active_dims=[dim]
            )
        elif k_name == "gibbs_linear":
            return _make_gibbs("linear", cfg, [dim])
        elif k_name == "gibbs_mlp":
            return _make_gibbs("mlp", cfg, [dim])
        elif k_name == "spectral_mixture":
            return _make_sm(cfg.sm_num_mixtures, 1, [dim])

        else:
            raise ValueError(f"Unknown primary kernel '{k_name}'")

    # ── Build secondary kernel ────────────────────────────────────────────────

    def _secondary(k_name: str) -> Optional[Kernel]:
        if not secondary_dims:
            return None
        n_sec = len(secondary_dims)
        if k_name == "matern52":
            return _make_matern(2.5, n_sec if cfg.use_ard else 1, cfg.use_ard, secondary_dims)
        elif k_name == "matern32":
            return _make_matern(1.5, n_sec if cfg.use_ard else 1, cfg.use_ard, secondary_dims)
        elif k_name == "matern12":
            return _make_matern(0.5, n_sec if cfg.use_ard else 1, cfg.use_ard, secondary_dims)
        elif k_name == "rbf":
            return RBFKernel(active_dims=secondary_dims)
        elif k_name == "spectral_mixture":
            return _make_sm(cfg.sm_num_mixtures, n_sec, secondary_dims)
        elif k_name in ("gibbs_power", "gibbs_sigmoid", "gibbs_linear"):
            # Gibbs as secondary — only makes sense for 1-dim secondary
            variant = k_name.replace("gibbs_", "")
            if n_sec == 1:
                return _make_gibbs(variant, cfg, secondary_dims)
            # fall back to matern for multi-dim secondary
            return _make_matern(2.5, n_sec, cfg.use_ard, secondary_dims)
        else:
            raise ValueError(f"Unknown secondary kernel '{k_name}'")

    # ── If both primary==secondary and no split needed, build full-D kernel ─

    if pk == sk and pk in ("matern52", "matern32", "matern12", "rbf", "spectral_mixture"):
        # No product — use single kernel on all dims
        k: Kernel
        if pk == "matern52":
            k = MaternKernel(nu=2.5, ard_num_dims=D if cfg.use_ard else 1)
        elif pk == "matern32":
            k = MaternKernel(nu=1.5, ard_num_dims=D if cfg.use_ard else 1)
        elif pk == "matern12":
            k = MaternKernel(nu=0.5, ard_num_dims=D if cfg.use_ard else 1)
        elif pk == "rbf":
            k = RBFKernel()
        elif pk == "spectral_mixture":
            k = SpectralMixtureKernel(num_mixtures=cfg.sm_num_mixtures, ard_num_dims=D)
        else:
            k = MaternKernel(nu=2.5, ard_num_dims=D)
    else:
        k1   = _primary(pk)
        k2   = _secondary(sk)
        k    = k1 * k2 if k2 is not None else k1

    # ── Wrap in ScaleKernel ───────────────────────────────────────────────────

    if cfg.use_scale_kernel:
        k = ScaleKernel(k)

    return k


def initialize_kernel_from_data(
    kernel: Kernel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
) -> None:
    """
    Initialize SM kernel parameters from data using the built-in method.
    Silently skips non-SM kernels (no-op for Gibbs/Matérn).
    If initialization results in inf or nan values, restores original parameters.
    """

    def _init_sm(k: Kernel) -> None:
        if isinstance(k, SpectralMixtureKernel):
            orig_params = {name: p.clone() for name, p in k.named_parameters()}
            try:
                active_x = train_x
                if k.active_dims is not None:
                    active_dims = list(k.active_dims)
                    active_x = train_x[:, active_dims]

                # Check if data is constant or too small to initialize
                is_constant = False
                if active_x.shape[0] <= 1:
                    is_constant = True
                else:
                    for col in range(active_x.shape[-1]):
                        if (active_x[:, col].max() - active_x[:, col].min()).item() < 1e-9:
                            is_constant = True
                            break
                
                if is_constant:
                    return

                k.initialize_from_data(active_x, train_y)
                
                # Check for inf or NaN
                has_invalid = False
                for p in k.parameters():
                    if torch.isinf(p).any() or torch.isnan(p).any():
                        has_invalid = True
                        break
                if has_invalid:
                    log = logging.getLogger("gp4d_lab.kernels")
                    log.warning("SpectralMixtureKernel initialization produced inf/nan values. Reverting to default initialization.")
                    for name, p in k.named_parameters():
                        p.data.copy_(orig_params[name].data)
            except Exception as e:
                log = logging.getLogger("gp4d_lab.kernels")
                log.warning(f"Error in SpectralMixtureKernel initialization: {e}. Reverting to default.")
                for name, p in k.named_parameters():
                    p.data.copy_(orig_params[name].data)

        if hasattr(k, "kernels"):      # ProductKernel / AdditiveKernel
            for sub in k.kernels:
                _init_sm(sub)
        if hasattr(k, "base_kernel"):  # ScaleKernel
            _init_sm(k.base_kernel)

    import logging
    _init_sm(kernel)


def describe_kernel(kernel: Kernel, indent: int = 0) -> str:
    """Return a human-readable summary of a kernel tree."""
    pad = "  " * indent
    name = type(kernel).__name__
    extra = kernel.extra_repr() if hasattr(kernel, "extra_repr") else ""
    lines = [f"{pad}{name}({extra})"]
    if hasattr(kernel, "kernels"):
        for sub in kernel.kernels:
            lines.append(describe_kernel(sub, indent + 1))
    if hasattr(kernel, "base_kernel"):
        lines.append(describe_kernel(kernel.base_kernel, indent + 1))
    return "\n".join(lines)
