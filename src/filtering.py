"""
Filtering utilities for BepiColombo DBSC signals.

Contenu
-------
- interpolation des NaN
- filtre Butterworth passe-bas
- filtre Savitzky-Golay
- filtre combo : Butterworth + Savitzky-Golay
- métriques de filtrage
- validation croisée des paramètres
- comparaison des modèles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter


def interpolate_nans(signal):
    signal = np.asarray(signal, dtype=float)
    mask_nan = np.isnan(signal)

    if np.all(mask_nan):
        return signal.copy(), mask_nan

    if np.any(mask_nan):
        x = np.arange(len(signal))
        signal_interp = np.interp(x, x[~mask_nan], signal[~mask_nan])
    else:
        signal_interp = signal.copy()

    return signal_interp, mask_nan


def butterworth_filter(signal, cutoff=0.0015, order=4, fs=1.0):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("butterworth_filter attend un signal 1D.")

    nyquist = fs / 2.0
    if cutoff <= 0 or cutoff >= nyquist:
        raise ValueError(f"La coupure doit être dans ]0, {nyquist}[ Hz. Reçu: {cutoff}")

    signal_interp, mask_nan = interpolate_nans(signal)

    if np.all(mask_nan):
        return signal.copy()

    b, a = butter(order, cutoff / nyquist, btype="low", analog=False)
    filtered = filtfilt(b, a, signal_interp)
    filtered[mask_nan] = np.nan
    return filtered


def savgol_filter_signal(signal, window_length=21, polyorder=3):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("savgol_filter_signal attend un signal 1D.")

    signal_interp, mask_nan = interpolate_nans(signal)

    if np.all(mask_nan):
        return signal.copy()

    if window_length % 2 == 0:
        window_length += 1

    if window_length <= polyorder:
        raise ValueError("window_length doit être > polyorder.")

    if window_length >= len(signal_interp):
        window_length = len(signal_interp) - 1 if len(signal_interp) % 2 == 0 else len(signal_interp)
        if window_length <= polyorder:
            raise ValueError("Signal trop court pour Savitzky-Golay.")

    filtered = savgol_filter(signal_interp, window_length=window_length, polyorder=polyorder, mode="interp")
    filtered[mask_nan] = np.nan
    return filtered


def combo_filter(signal, fs, cutoff, order, window_length, polyorder):
    buttered = butterworth_filter(signal, cutoff=cutoff, order=order, fs=fs)
    combo = savgol_filter_signal(buttered, window_length=window_length, polyorder=polyorder)
    return combo


def compare_to_true_signal(estimated_signal, true_signal):
    estimated_signal = np.asarray(estimated_signal, dtype=float)
    true_signal = np.asarray(true_signal, dtype=float)

    mask = ~(np.isnan(estimated_signal) | np.isnan(true_signal))

    if np.sum(mask) < 3:
        return {"MAE": np.nan, "RMSE": np.nan, "correlation": np.nan}

    err = estimated_signal[mask] - true_signal[mask]
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    corr = np.corrcoef(estimated_signal[mask], true_signal[mask])[0, 1]

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "correlation": float(corr)
    }


def analyze_filter_effectiveness(raw_signal, filtered_signal):
    raw_std = np.nanstd(raw_signal)
    filtered_std = np.nanstd(filtered_signal)

    if filtered_std == 0:
        reduction_db = np.inf
    else:
        reduction_db = 20 * np.log10(raw_std / filtered_std)

    return {
        "raw_std": float(raw_std),
        "filtered_std": float(filtered_std),
        "noise_reduction_db": float(reduction_db)
    }


def evaluate_filter_candidate(raw_signal, true_signal, filtered_signal):
    raw_vs_true = compare_to_true_signal(raw_signal, true_signal)
    filt_vs_true = compare_to_true_signal(filtered_signal, true_signal)
    filt_effect = analyze_filter_effectiveness(raw_signal, filtered_signal)

    gain_mae = raw_vs_true["MAE"] - filt_vs_true["MAE"]
    gain_rmse = raw_vs_true["RMSE"] - filt_vs_true["RMSE"]
    gain_corr = filt_vs_true["correlation"] - raw_vs_true["correlation"]

    score = (
        3.0 * gain_mae +
        3.0 * gain_rmse +
        2.0 * gain_corr +
        0.2 * filt_effect["noise_reduction_db"]
    )

    return {
        "raw_vs_true": raw_vs_true,
        "filtered_vs_true": filt_vs_true,
        "filter_effectiveness": filt_effect,
        "gain_mae": float(gain_mae),
        "gain_rmse": float(gain_rmse),
        "gain_corr": float(gain_corr),
        "score": float(score)
    }


def grid_search_butterworth_parameters(
    raw_signal,
    true_signal,
    fs,
    max_dominant_freq,
    orders=(3, 4, 5, 6),
    n_cutoffs=8,
    cutoff_multiplier_min=1.0,
    cutoff_multiplier_max=1.5,
):
    raw_signal = np.asarray(raw_signal, dtype=float)
    true_signal = np.asarray(true_signal, dtype=float)

    nyquist = fs / 2.0
    cutoff_min = max(max_dominant_freq * cutoff_multiplier_min, 1e-8)
    cutoff_max = min(max_dominant_freq * cutoff_multiplier_max, nyquist * 0.99)

    if cutoff_min >= cutoff_max:
        cutoff_max = min(cutoff_min * 1.2, nyquist * 0.99)

    cutoffs = np.linspace(cutoff_min, cutoff_max, n_cutoffs)
    all_results = []

    for order in orders:
        for cutoff in cutoffs:
            try:
                filtered_signal = butterworth_filter(raw_signal, cutoff=cutoff, order=order, fs=fs)
                metrics = evaluate_filter_candidate(raw_signal, true_signal, filtered_signal)

                all_results.append({
                    "model": "Butterworth",
                    "order": int(order),
                    "cutoff": float(cutoff),
                    "filtered_signal": filtered_signal,
                    **metrics
                })
            except Exception:
                continue

    if len(all_results) == 0:
        raise RuntimeError("Aucun couple Butterworth valide trouvé.")

    best_result = max(all_results, key=lambda x: x["score"])
    return best_result, all_results


def grid_search_savgol_parameters(
    raw_signal,
    true_signal,
    window_lengths=(11, 21, 31, 41),
    polyorders=(2, 3, 4),
):
    raw_signal = np.asarray(raw_signal, dtype=float)
    true_signal = np.asarray(true_signal, dtype=float)

    all_results = []

    for window_length in window_lengths:
        for polyorder in polyorders:
            if window_length <= polyorder:
                continue
            try:
                filtered_signal = savgol_filter_signal(
                    raw_signal,
                    window_length=window_length,
                    polyorder=polyorder
                )
                metrics = evaluate_filter_candidate(raw_signal, true_signal, filtered_signal)

                all_results.append({
                    "model": "Savitzky-Golay",
                    "window_length": int(window_length),
                    "polyorder": int(polyorder),
                    "filtered_signal": filtered_signal,
                    **metrics
                })
            except Exception:
                continue

    if len(all_results) == 0:
        raise RuntimeError("Aucun couple Savitzky-Golay valide trouvé.")

    best_result = max(all_results, key=lambda x: x["score"])
    return best_result, all_results


def grid_search_combo_parameters(
    raw_signal,
    true_signal,
    fs,
    max_dominant_freq,
    butter_orders=(3, 4, 5, 6),
    n_cutoffs=6,
    cutoff_multiplier_min=1.0,
    cutoff_multiplier_max=1.5,
    window_lengths=(11, 21, 31),
    polyorders=(2, 3),
):
    raw_signal = np.asarray(raw_signal, dtype=float)
    true_signal = np.asarray(true_signal, dtype=float)

    nyquist = fs / 2.0
    cutoff_min = max(max_dominant_freq * cutoff_multiplier_min, 1e-8)
    cutoff_max = min(max_dominant_freq * cutoff_multiplier_max, nyquist * 0.99)

    if cutoff_min >= cutoff_max:
        cutoff_max = min(cutoff_min * 1.2, nyquist * 0.99)

    cutoffs = np.linspace(cutoff_min, cutoff_max, n_cutoffs)
    all_results = []

    for order in butter_orders:
        for cutoff in cutoffs:
            for window_length in window_lengths:
                for polyorder in polyorders:
                    if window_length <= polyorder:
                        continue
                    try:
                        filtered_signal = combo_filter(
                            raw_signal,
                            fs=fs,
                            cutoff=cutoff,
                            order=order,
                            window_length=window_length,
                            polyorder=polyorder
                        )
                        metrics = evaluate_filter_candidate(raw_signal, true_signal, filtered_signal)

                        all_results.append({
                            "model": "Combo",
                            "order": int(order),
                            "cutoff": float(cutoff),
                            "window_length": int(window_length),
                            "polyorder": int(polyorder),
                            "filtered_signal": filtered_signal,
                            **metrics
                        })
                    except Exception:
                        continue

    if len(all_results) == 0:
        raise RuntimeError("Aucun couple Combo valide trouvé.")

    best_result = max(all_results, key=lambda x: x["score"])
    return best_result, all_results


def summarize_grid_search_results(all_results):
    rows = []
    for r in all_results:
        row = {
            "model": r.get("model", ""),
            "score": r["score"],
            "gain_mae": r["gain_mae"],
            "gain_rmse": r["gain_rmse"],
            "gain_corr": r["gain_corr"],
            "noise_reduction_db": r["filter_effectiveness"]["noise_reduction_db"],
            "filtered_mae": r["filtered_vs_true"]["MAE"],
            "filtered_rmse": r["filtered_vs_true"]["RMSE"],
            "filtered_corr": r["filtered_vs_true"]["correlation"],
        }
        if "order" in r:
            row["order"] = r["order"]
        if "cutoff" in r:
            row["cutoff"] = r["cutoff"]
        if "window_length" in r:
            row["window_length"] = r["window_length"]
        if "polyorder" in r:
            row["polyorder"] = r["polyorder"]
        rows.append(row)
    return rows


def plot_filter_and_fft(raw_signal, filtered_signal, freqs_raw, mag_raw, freqs_filt, mag_filt, fs, cutoff, x_max, title_prefix=""):
    time_h = np.arange(len(raw_signal)) / fs / 3600.0
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(time_h, raw_signal, alpha=0.5, label="Brut")
    axes[0, 0].plot(time_h, filtered_signal, linewidth=1.5, label="Filtré")
    axes[0, 0].set_title(f"{title_prefix}Signal temporel brut vs filtré")
    axes[0, 0].set_xlabel("Temps [heures]")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    zoom_start = int(0.2 * len(time_h))
    zoom_end = min(zoom_start + 300, len(time_h))
    axes[0, 1].plot(time_h[zoom_start:zoom_end], raw_signal[zoom_start:zoom_end], alpha=0.5, label="Brut")
    axes[0, 1].plot(time_h[zoom_start:zoom_end], filtered_signal[zoom_start:zoom_end], linewidth=1.5, label="Filtré")
    axes[0, 1].set_title(f"{title_prefix}Zoom temporel")
    axes[0, 1].set_xlabel("Temps [heures]")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].semilogy(freqs_raw, mag_raw, alpha=0.7, label="FFT brut")
    if cutoff is not None:
        axes[1, 0].axvline(cutoff, color="black", linestyle="--", label=f"Coupure = {cutoff:.6f} Hz")
    axes[1, 0].set_xlim(0, x_max)
    axes[1, 0].set_title(f"{title_prefix}FFT avant filtrage")
    axes[1, 0].set_xlabel("Fréquence [Hz]")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(freqs_filt, mag_filt, color="red", label="FFT filtré")
    if cutoff is not None:
        axes[1, 1].axvline(cutoff, color="black", linestyle="--", label=f"Coupure = {cutoff:.6f} Hz")
    axes[1, 1].set_xlim(0, x_max)
    axes[1, 1].set_title(f"{title_prefix}FFT après filtrage")
    axes[1, 1].set_xlabel("Fréquence [Hz]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_filter_vs_true(raw_signal, filtered_signal, true_signal, fs, model_name="Filtre"):
    time_h = np.arange(len(raw_signal)) / fs / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(time_h, true_signal, label="Signal réel", linewidth=1.8)
    axes[0].plot(time_h, raw_signal, label="Signal mesuré", alpha=0.45)
    axes[0].plot(time_h, filtered_signal, label=f"Signal filtré {model_name}", linewidth=1.6)
    axes[0].set_title(f"Comparaison au signal réel après filtrage {model_name}")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_h, raw_signal - true_signal, label="Écart brut - réel", linewidth=1.1)
    axes[1].plot(time_h, filtered_signal - true_signal, label=f"Écart {model_name} - réel", linewidth=1.1)
    axes[1].set_title("Écarts au signal réel")
    axes[1].set_xlabel("Temps [heures]")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def plot_filter_benchmark(raw_signal, true_signal, filtered_signals, fs):
    time_h = np.arange(len(raw_signal)) / fs / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(time_h, true_signal, linewidth=2.0, label="Réel")
    axes[0].plot(time_h, raw_signal, alpha=0.35, label="Brut")
    for name, sig in filtered_signals.items():
        axes[0].plot(time_h, sig, linewidth=1.5, label=name)
    axes[0].set_title("Comparaison des modèles de filtrage")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_h, raw_signal - true_signal, alpha=0.6, label="Brut - réel")
    for name, sig in filtered_signals.items():
        axes[1].plot(time_h, sig - true_signal, linewidth=1.3, label=f"{name} - réel")
    axes[1].set_title("Écarts au signal réel")
    axes[1].set_xlabel("Temps [heures]")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    return fig