
import numpy as np
from typing import Dict
from .age_depth import build_age_model, apply_age_model, mc_age_models
from .spectrum import lomb_scargle_uneven, ar1_significance_levels

def run_pipeline(depth, value,
                 anchors_depth=None, anchors_age_kyr=None, anchors_age_sigma_kyr=None,
                 model="pchip", mc=0,
                 freq_min=0.002, freq_max=0.1, nfreq=1000) -> Dict:
    """Minimal pipeline: age-depth -> LS spectrum (+ AR1 threshold)."""
    depth = np.asarray(depth, float)
    value = np.asarray(value, float)
    if anchors_depth is None or anchors_age_kyr is None:
        dmin, dmax = np.nanmin(depth), np.nanmax(depth)
        t0, t1 = 0.0, (dmax - dmin) * 2.0
        f_age = build_age_model([dmin, dmax], [t0, t1], model="linear")
        anchors_meta = {"note":"No anchors provided; using arbitrary linear age scale (demo only)."}
    else:
        f_age = build_age_model(anchors_depth, anchors_age_kyr, model=model)
        anchors_meta = {"n_anchors": len(anchors_depth)}

    t_kyr = apply_age_model(depth, f_age)
    idx = np.argsort(t_kyr)
    t_kyr = t_kyr[idx]
    x = value[idx]

    freq_cpk, power = lomb_scargle_uneven(t_kyr, x, freq_min, freq_max, nfreq=nfreq)
    thr95, ar1meta = ar1_significance_levels(x, freq_cpk, alpha=0.95)

    out = {
        "t_kyr": t_kyr, "x": x,
        "freq_cpk": freq_cpk, "power": power,
        "ar1_95": thr95,
        "age_model_meta": anchors_meta | {"model": model},
        "ar1_meta": ar1meta
    }
    if anchors_depth is not None and anchors_age_kyr is not None and mc and mc > 0:
        peaks_stack = []
        for f_age_i in mc_age_models(anchors_depth, anchors_age_kyr, anchors_age_sigma_kyr, n=mc):
            t_i = apply_age_model(depth, f_age_i)
            idxi = np.argsort(t_i)
            t_i = t_i[idxi]; x_i = value[idxi]
            _, p_i = lomb_scargle_uneven(t_i, x_i, freq_min, freq_max, nfreq=nfreq)
            peaks_stack.append(p_i)
        out["power_mc_mean"] = np.mean(peaks_stack, axis=0)
        out["power_mc_std"] = np.std(peaks_stack, axis=0)
        out["mc"] = mc
    return out
