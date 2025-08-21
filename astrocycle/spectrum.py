
import numpy as np
from typing import Tuple, Dict
from scipy.signal import lombscargle, butter, filtfilt, hilbert

def lomb_scargle_uneven(t_kyr: np.ndarray, x: np.ndarray,
                        fmin_cpk: float, fmax_cpk: float, nfreq: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Lomb–Scargle for uneven sampling. Returns (freq_cpk, power)."""
    t = np.asarray(t_kyr, float)
    y = np.asarray(x, float) - np.nanmean(x)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    freq_cpk = np.linspace(fmin_cpk, fmax_cpk, nfreq)
    ang = 2*np.pi*freq_cpk  # rad/kyr
    p = lombscargle(t, y, ang, precenter=True, normalize=True)
    return freq_cpk, p

def ar1_significance_levels(y: np.ndarray, freq_cpk: np.ndarray,
                            alpha: float = 0.95) -> Tuple[np.ndarray, Dict[str, float]]:
    """Approximate AR(1) red-noise significance curve across frequencies."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    y = y - y.mean()
    if len(y) < 5:
        thr = np.full_like(freq_cpk, np.nan, dtype=float)
        return thr, {"rho": np.nan, "sigma2": np.nan}
    y0 = y[:-1]; y1 = y[1:]
    rho = np.corrcoef(y0, y1)[0,1]
    rho = np.clip(rho, -0.99, 0.99)
    sigma2 = np.var(y, ddof=1)
    dt_eff = 1.0
    cos_term = np.cos(2*np.pi*freq_cpk*dt_eff)
    red_bg = sigma2*(1-rho**2) / (1 + rho**2 - 2*rho*cos_term)
    q = {0.90: 2.3, 0.95: 3.0, 0.99: 4.6}.get(alpha, 3.0)
    thr = red_bg * q
    return thr, {"rho": float(rho), "sigma2": float(sigma2)}

def bandpass_hilbert(t_kyr: np.ndarray, x: np.ndarray,
                     f_lo_cpk: float, f_hi_cpk: float,
                     order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-phase Butterworth bandpass + Hilbert envelope. Returns (x_filt, envelope)."""
    t = np.asarray(t_kyr, float)
    y = np.asarray(x, float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    dt = np.median(np.diff(np.sort(t)))
    t_uni = np.arange(t.min(), t.max()+dt/2, dt)
    y_uni = np.interp(t_uni, t, y)
    fs = 1.0/dt
    nyq = 0.5*fs
    low = max(f_lo_cpk/nyq, 1e-6)
    high = min(f_hi_cpk/nyq, 0.999)
    if low >= high:
        raise ValueError("Invalid band edges after normalization")
    b, a = butter(order, [low, high], btype="band")
    y_f = filtfilt(b, a, y_uni)
    env = np.abs(hilbert(y_f))
    y_f_back = np.interp(t, t_uni, y_f)
    env_back = np.interp(t, t_uni, env)
    return y_f_back, env_back
