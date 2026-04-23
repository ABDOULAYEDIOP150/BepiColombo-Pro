"""
Inter-instrument comparison: correlation, RMSE, MAE, find_peaks, interpolation.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

def compare_instruments(ds1, ds2):
    """
    Compare two instrument datasets.
    Aligns time via linear interpolation, computes metrics.
    """
    # Extract 1D signals (spatial average)
    sig1 = ds1["temperature"].mean(dim=["latitude", "longitude"]).values
    sig2 = ds2["temperature"].mean(dim=["latitude", "longitude"]).values
    
    # Get time coordinates safely
    if "time" in ds1.coords:
        t1 = ds1["time"].values
    else:
        t1 = np.arange(len(sig1))
    
    if "time" in ds2.coords:
        t2 = ds2["time"].values
    else:
        t2 = np.arange(len(sig2))
    
    # Flatten if needed
    if hasattr(t1, 'ndim') and t1.ndim > 1:
        t1 = t1.flatten()
    if hasattr(t2, 'ndim') and t2.ndim > 1:
        t2 = t2.flatten()
    
    sig1 = sig1.flatten() if hasattr(sig1, 'flatten') else sig1
    sig2 = sig2.flatten() if hasattr(sig2, 'flatten') else sig2
    
    # Find common time range
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    
    if t_start >= t_end:
        # Fallback: use indices
        min_len = min(len(sig1), len(sig2))
        sig1 = sig1[:min_len]
        sig2 = sig2[:min_len]
        return {
            "MAE": float(np.mean(np.abs(sig1 - sig2))),
            "RMSE": float(np.sqrt(np.mean((sig1 - sig2)**2))),
            "correlation": float(pearsonr(sig1, sig2)[0]) if len(sig1) > 1 else 0.0,
            "peaks_instrument1": len(find_peaks(sig1, distance=10)[0]),
            "peaks_instrument2": len(find_peaks(sig2, distance=10)[0])
        }
    
    # Create common time grid
    common_t = np.linspace(t_start, t_end, 500)
    
    # Interpolate both signals
    interp1 = interp1d(t1, sig1, kind="linear", fill_value="extrapolate")
    interp2 = interp1d(t2, sig2, kind="linear", fill_value="extrapolate")
    
    sig1_interp = interp1(common_t)
    sig2_interp = interp2(common_t)
    
    # Find peaks
    peaks1, _ = find_peaks(sig1_interp, distance=10)
    peaks2, _ = find_peaks(sig2_interp, distance=10)
    
    # Correlation
    corr, _ = pearsonr(sig1_interp, sig2_interp)
    
    return {
        "MAE": float(np.mean(np.abs(sig1_interp - sig2_interp))),
        "RMSE": float(np.sqrt(np.mean((sig1_interp - sig2_interp)**2))),
        "correlation": float(corr),
        "peaks_instrument1": len(peaks1),
        "peaks_instrument2": len(peaks2)
    }