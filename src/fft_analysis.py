import numpy as np
from scipy.signal import detrend, find_peaks


def _interpolate_nans_1d(signal):
    signal = np.asarray(signal, dtype=float)
    mask_nan = np.isnan(signal)

    if np.all(mask_nan):
        return None

    if np.any(mask_nan):
        x = np.arange(len(signal))
        signal = np.interp(x, x[~mask_nan], signal[~mask_nan])

    return signal


def compute_fft(signal, fs=1.0, detrend_signal=True):
    """
    FFT 1D robuste avec interpolation des NaN.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("compute_fft attend un signal 1D.")

    clean_signal = _interpolate_nans_1d(signal)
    if clean_signal is None:
        return np.array([]), np.array([])

    if detrend_signal:
        clean_signal = detrend(clean_signal)

    n = len(clean_signal)
    fft_vals = np.fft.rfft(clean_signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(fft_vals) / n

    return freqs, magnitude


def get_dominant_frequencies(freqs, magnitude, n_peaks=10, min_relative_height=0.1):
    """
    Retourne les fréquences dominantes.
    """
    if len(freqs) == 0 or len(magnitude) == 0:
        return []

    if len(magnitude) < 3:
        return []

    peaks, properties = find_peaks(
        magnitude,
        height=np.max(magnitude) * min_relative_height
    )

    if len(peaks) == 0:
        peak_idx = np.argsort(magnitude[1:])[-n_peaks:] + 1 if len(magnitude) > 1 else np.array([0])
    else:
        peak_idx = peaks[np.argsort(magnitude[peaks])[::-1]]

    peak_idx = peak_idx[:n_peaks]

    results = []
    for idx in peak_idx:
        freq = freqs[idx]
        mag = magnitude[idx]
        period = np.inf if freq == 0 else 1.0 / freq

        results.append({
            "frequency_hz": float(freq),
            "magnitude": float(mag),
            "period_s": float(period)
        })

    return results


def count_frequencies_below_cutoff(freqs, cutoff):
    """
    Nombre de composantes fréquentielles sous la coupure.
    """
    return int(np.sum(freqs <= cutoff))