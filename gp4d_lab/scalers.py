"""
gp4d_lab.scalers
================
Scaler registry — physics-informed + standard sklearn wrappers.

Available input scalers:
  "physics"  — PhysicsInputScaler4D  (log Q2/W2 already done; log+std beta/z)
  "standard" — per-column (x - mean) / std
  "robust"   — per-column (x - median) / IQR
  "minmax"   — per-column [0, 1]
  "power"    — Yeo-Johnson then standardize
  "quantile" — normal quantile transform
  "none"     — identity

Available target scalers:
  "robust"   — (y - median) / IQR          [default]
  "standard" — (y - mean) / std
  "minmax"   — [0, 1]
  "log"      — log(y) then standardize
  "none"     — identity
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler as _RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class ScalerRegistry:
    INPUT_SCALERS  = {"physics", "standard", "robust", "minmax", "power", "quantile", "none"}
    TARGET_SCALERS = {"robust", "standard", "minmax", "log", "conditional_robust", "yeo_johnson", "none"}


    @staticmethod
    def get_input(name: str, physics_cfg: dict | None = None):
        match name:
            case "physics":
                return PhysicsInputScaler4D(**(physics_cfg or {}))
            case "standard":
                return _SklearnInputWrapper(StandardScaler())
            case "robust":
                return _SklearnInputWrapper(_RobustScaler())
            case "minmax":
                return _SklearnInputWrapper(MinMaxScaler())
            case "power":
                return _SklearnInputWrapper(PowerTransformer(method="yeo-johnson"))
            case "quantile":
                return _SklearnInputWrapper(
                    QuantileTransformer(output_distribution="normal", n_quantiles=1000)
                )
            case "none":
                return _IdentityScaler()
            case _:
                raise ValueError(f"Unknown input scaler '{name}'")

    @staticmethod
    def get_target(name: str):
        match name:
            case "robust":   return TargetScalerRobust()
            case "standard": return TargetScalerStandard()
            case "minmax":   return TargetScalerMinMax()
            case "log":      return TargetScalerLog()
            case "conditional_robust": return TargetScalerConditionalRobust()
            case "yeo_johnson": return TargetScalerYeoJohnson()
            case "none":     return _IdentityScaler()
            case _:
                raise ValueError(f"Unknown target scaler '{name}'")


def get_scaler(kind: str, name: str, physics_cfg: dict | None = None):
    if kind == "input":
        return ScalerRegistry.get_input(name, physics_cfg)
    elif kind == "target":
        return ScalerRegistry.get_target(name)
    raise ValueError("kind must be 'input' or 'target'")


# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

class _IdentityScaler:
    def fit(self, x): return self
    def transform(self, x): return x
    def fit_transform(self, x): return x
    def inverse_transform(self, x): return x
    def std_to_original(self, s): return s


# ─────────────────────────────────────────────────────────────────────────────
# sklearn wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _SklearnInputWrapper:
    def __init__(self, sk_scaler):
        self._sk = sk_scaler

    def fit(self, X):
        self._sk.fit(X); return self

    def transform(self, X):
        return self._sk.transform(X)

    def fit_transform(self, X):
        return self._sk.fit_transform(X)

    def std_to_original(self, s):
        if hasattr(self._sk, "scale_"):
            return s * np.mean(self._sk.scale_)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Physics-informed 4D input scaler
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsInputScaler4D:
    """
    Col 0  logQ2  → already log-scaled in CSV → standardize
    Col 1  logW2  → already log-scaled in CSV → standardize
    Col 2  beta   → log(|beta| + eps) → standardize
    Col 3  z      → log(|z|   + eps) → standardize
    """

    def __init__(self, beta_eps_factor: float = 1e-3, z_eps_factor: float = 1e-3):
        self.beta_eps_factor = beta_eps_factor
        self.z_eps_factor    = z_eps_factor
        self._fitted         = False

    def fit(self, X: np.ndarray):
        logQ2, logW2, beta, z = X[:,0], X[:,1], X[:,2], X[:,3]
        self._fit_col(logQ2, "logQ2", log_it=False)
        self._fit_col(logW2, "logW2", log_it=False)

        beta_abs = np.abs(beta)
        self.beta_eps = max(self.beta_eps_factor * float(np.median(beta_abs)), 1e-6)
        log_beta = np.log(beta_abs + self.beta_eps)
        self.mean_log_beta = float(np.mean(log_beta))
        self.std_log_beta  = max(float(np.std(log_beta)), 1e-8)

        z_abs = np.abs(z)
        self.z_eps = max(self.z_eps_factor * float(np.median(z_abs)), 1e-6)
        log_z = np.log(z_abs + self.z_eps)
        self.mean_log_z = float(np.mean(log_z))
        self.std_log_z  = max(float(np.std(log_z)), 1e-8)

        self._fitted = True
        return self

    def _fit_col(self, col, name, log_it=False):
        if log_it:
            col = np.log(np.maximum(col, 1e-12))
        setattr(self, f"mean_{name}", float(np.mean(col)))
        setattr(self, f"std_{name}",  max(float(np.std(col)), 1e-8))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        Xs = np.empty_like(X, dtype=np.float64)
        Xs[:,0] = (X[:,0] - self.mean_logQ2) / self.std_logQ2
        Xs[:,1] = (X[:,1] - self.mean_logW2) / self.std_logW2
        Xs[:,2] = (np.log(np.abs(X[:,2]) + self.beta_eps) - self.mean_log_beta) / self.std_log_beta
        Xs[:,3] = (np.log(np.abs(X[:,3]) + self.z_eps)    - self.mean_log_z)    / self.std_log_z
        return Xs

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform()")
        X = np.empty_like(Xs, dtype=np.float64)
        X[:,0] = Xs[:,0] * self.std_logQ2 + self.mean_logQ2
        X[:,1] = Xs[:,1] * self.std_logW2 + self.mean_logW2
        X[:,2] = np.exp(Xs[:,2] * self.std_log_beta + self.mean_log_beta) - self.beta_eps
        X[:,3] = np.exp(Xs[:,3] * self.std_log_z + self.mean_log_z) - self.z_eps
        return X

    def std_to_original(self, s: float) -> float:
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Target scalers
# ─────────────────────────────────────────────────────────────────────────────

class TargetScalerRobust:
    def fit(self, y: np.ndarray):
        y = y.ravel()
        self.median = float(np.median(y))
        q75, q25 = np.percentile(y, [75, 25])
        self.iqr = float(q75 - q25) or 1.0
        return self

    def transform(self, y):
        return ((y.ravel() - self.median) / self.iqr).reshape(-1, 1)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return (y.ravel() * self.iqr + self.median).reshape(-1, 1)

    def std_to_original(self, s):
        return s * self.iqr


class TargetScalerStandard:
    def fit(self, y: np.ndarray):
        y = y.ravel()
        self.mean_ = float(np.mean(y))
        self.std_  = float(np.std(y)) or 1.0
        return self

    def transform(self, y):
        return ((y.ravel() - self.mean_) / self.std_).reshape(-1, 1)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return (y.ravel() * self.std_ + self.mean_).reshape(-1, 1)

    def std_to_original(self, s):
        return s * self.std_


class TargetScalerMinMax:
    def fit(self, y: np.ndarray):
        y = y.ravel()
        self.min_   = float(y.min())
        self.range_ = float(y.max() - y.min()) or 1.0
        return self

    def transform(self, y):
        return ((y.ravel() - self.min_) / self.range_).reshape(-1, 1)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return (y.ravel() * self.range_ + self.min_).reshape(-1, 1)

    def std_to_original(self, s):
        return s * self.range_


class TargetScalerLog:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, y: np.ndarray):
        y = y.ravel()
        self.shift_ = float(y.min()) - self.eps
        log_y = np.log(y - self.shift_)
        self.mean_ = float(np.mean(log_y))
        self.std_  = float(np.std(log_y)) or 1.0
        return self

    def transform(self, y):
        log_y = np.log(y.ravel() - self.shift_)
        return ((log_y - self.mean_) / self.std_).reshape(-1, 1)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        log_y = y.ravel() * self.std_ + self.mean_
        return (np.exp(log_y) + self.shift_).reshape(-1, 1)

    def std_to_original(self, s):
        return s * self.std_
        

class TargetScalerConditionalRobust:
    """
    Robust scaling using median and IQR computed ONLY on values above threshold.
    Avoids skewing the scaler parameters when a large fraction of points are
    clamped to a floor value (e.g. -17.955 or -87).
    """
    def __init__(self, threshold: float = -15.0):
        self.threshold = threshold
        self.median = 0.0
        self.iqr = 1.0

    def fit(self, y: np.ndarray):
        y = y.ravel()
        # Compute stats only on non-floor points
        y_phys = y[y > self.threshold]
        if len(y_phys) == 0:
            y_phys = y  # fallback if all points are below threshold
        self.median = float(np.median(y_phys))
        q75, q25 = np.percentile(y_phys, [75, 25])
        self.iqr = float(q75 - q25) or 1.0
        return self

    def transform(self, y):
        return ((y.ravel() - self.median) / self.iqr).reshape(-1, 1)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return (y.ravel() * self.iqr + self.median).reshape(-1, 1)

    def std_to_original(self, s):
        return s * self.iqr


class TargetScalerYeoJohnson:
    """
    Yeo-Johnson power transform for target scaling.
    """
    def __init__(self):
        self._pt = PowerTransformer(method="yeo-johnson", standardize=True)

    def fit(self, y: np.ndarray):
        self._pt.fit(y.reshape(-1, 1))
        return self

    def transform(self, y):
        return self._pt.transform(y.reshape(-1, 1))

    def fit_transform(self, y):
        return self._pt.fit_transform(y.reshape(-1, 1))

    def inverse_transform(self, y):
        return self._pt.inverse_transform(y.reshape(-1, 1))

    def std_to_original(self, s):
        # Linear approximation using the mean scaling factor if available,
        # otherwise returns standard deviation as-is in transformed space.
        # Note: asymmetric errors are exact by transforming the bounds manually.
        if hasattr(self._pt, "scale_"):
            return s * np.mean(self._pt.scale_)
        return s

