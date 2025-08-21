
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from scipy.interpolate import PchipInterpolator

@dataclass
class AgeModelResult:
    depth: np.ndarray
    age_kyr: np.ndarray
    model_type: str
    meta: Dict

def _ensure_sorted_by_depth(depth, age):
    idx = np.argsort(depth)
    return np.asarray(depth)[idx], np.asarray(age)[idx]

def build_age_model(anchors_depth, anchors_age_kyr, model: str = "pchip") -> Callable[[np.ndarray], np.ndarray]:
    """Build a monotonic age-depth model ('pchip' or 'linear'). Returns f(depth)->age_kyr."""
    d, a = _ensure_sorted_by_depth(anchors_depth, anchors_age_kyr)
    if model == "pchip":
        f = PchipInterpolator(d, a, extrapolate=True)
    elif model == "linear":
        def f(x):
            return np.interp(x, d, a, left=np.nan, right=np.nan)
    else:
        raise ValueError("model must be 'pchip' or 'linear'")
    return f

def apply_age_model(depth_series: np.ndarray, f_age) -> np.ndarray:
    return np.asarray(f_age(depth_series))

def mc_age_models(anchors_depth: np.ndarray, anchors_age_kyr: np.ndarray,
                  anchors_age_sigma_kyr: Optional[np.ndarray],
                  n: int = 200, model: str = "pchip", random_state: int = 42):
    """Monte Carlo ensemble of age models by perturbing anchor ages with Gaussian noise."""
    rng = np.random.default_rng(random_state)
    if anchors_age_sigma_kyr is None:
        anchors_age_sigma_kyr = np.zeros_like(anchors_age_kyr)
    for _ in range(n):
        a_mc = anchors_age_kyr + rng.normal(0, anchors_age_sigma_kyr)
        yield build_age_model(anchors_depth, a_mc, model=model)
