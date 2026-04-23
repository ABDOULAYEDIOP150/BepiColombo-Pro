"""
Instrument calibration for filtered 1D signals.

Objectif
--------
- calibrer plusieurs signaux filtrés
- supprimer une dérive lente (polynomiale)
- appliquer un gain / offset par instrument
- comparer les modèles calibrés
- choisir le meilleur modèle final
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


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


def compare_signals(signal_a, signal_b):
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)

    mask = ~(np.isnan(signal_a) | np.isnan(signal_b))
    if np.sum(mask) < 3:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "correlation": np.nan,
            "bias": np.nan,
            "std_error": np.nan
        }

    err = signal_a[mask] - signal_b[mask]
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    corr = np.corrcoef(signal_a[mask], signal_b[mask])[0, 1]
    bias = np.mean(err)
    std_error = np.std(err)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "correlation": float(corr),
        "bias": float(bias),
        "std_error": float(std_error)
    }


def estimate_drift(time_1d, signal_1d, degree=2):
    time_1d = np.asarray(time_1d, dtype=float)
    signal_1d = np.asarray(signal_1d, dtype=float)

    valid = ~(np.isnan(time_1d) | np.isnan(signal_1d))
    if np.sum(valid) < degree + 2:
        return np.full_like(signal_1d, np.nan), np.array([])

    coeffs = np.polyfit(time_1d[valid], signal_1d[valid], deg=degree)
    drift_curve = np.polyval(coeffs, time_1d)
    return drift_curve, coeffs


def get_instrument_gain_offset(instrument="DBSC"):
    if instrument == "DBSC":
        gain = 1.02
        offset = -0.5
    elif instrument == "MAG":
        gain = 0.98
        offset = 0.3
    else:
        gain = 1.0
        offset = 0.0
    return float(gain), float(offset)


def calibrate_filtered_signal(signal_1d, time_1d, instrument="DBSC", drift_degree=2, preserve_mean=True):
    signal_1d = np.asarray(signal_1d, dtype=float)
    time_1d = np.asarray(time_1d, dtype=float)

    if signal_1d.ndim != 1:
        raise ValueError("signal_1d doit être 1D.")
    if time_1d.ndim != 1:
        raise ValueError("time_1d doit être 1D.")
    if len(signal_1d) != len(time_1d):
        raise ValueError("signal_1d et time_1d doivent avoir la même longueur.")

    signal_interp, mask_nan = interpolate_nans(signal_1d)

    if np.all(mask_nan):
        return {
            "calibrated_signal": signal_1d.copy(),
            "drift_curve": np.full_like(signal_1d, np.nan),
            "drift_coeffs": np.array([]),
            "gain": np.nan,
            "offset": np.nan,
            "drift_degree": drift_degree
        }

    drift_curve, coeffs = estimate_drift(time_1d, signal_interp, degree=drift_degree)

    if np.all(np.isnan(drift_curve)):
        drift_corrected = signal_interp.copy()
    else:
        if preserve_mean:
            drift_corrected = signal_interp - drift_curve + np.nanmean(signal_interp)
        else:
            drift_corrected = signal_interp - drift_curve

    gain, offset = get_instrument_gain_offset(instrument)
    calibrated_signal = drift_corrected * gain + offset
    calibrated_signal[mask_nan] = np.nan

    return {
        "calibrated_signal": calibrated_signal,
        "drift_curve": drift_curve,
        "drift_coeffs": coeffs,
        "gain": float(gain),
        "offset": float(offset),
        "drift_degree": int(drift_degree)
    }


def inject_calibrated_signal_in_dataset(dataset, calibrated_signal_1d):
    if not isinstance(dataset, xr.Dataset):
        raise ValueError("dataset doit être un xarray.Dataset.")

    calibrated_signal_1d = np.asarray(calibrated_signal_1d, dtype=float)

    if "temperature" not in dataset:
        raise ValueError("Le dataset doit contenir la variable 'temperature'.")

    ds_cal = dataset.copy()
    target_shape = ds_cal["temperature"].shape

    calibrated_3d = np.broadcast_to(calibrated_signal_1d[:, None, None], target_shape).copy()
    ds_cal["temperature"] = (["time", "latitude", "longitude"], calibrated_3d)
    ds_cal.attrs["calibrated"] = True
    ds_cal.attrs["calibration_source"] = "filtered_signal"
    return ds_cal


def compute_drift_metric(signal_1d):
    signal_1d = np.asarray(signal_1d, dtype=float)
    valid = ~np.isnan(signal_1d)

    if np.sum(valid) < 10:
        return np.nan

    x = signal_1d[valid]
    n = len(x)

    if n >= 200:
        start_mean = np.mean(x[:100])
        end_mean = np.mean(x[-100:])
    else:
        start_mean = np.mean(x[:max(1, n // 10)])
        end_mean = np.mean(x[-max(1, n // 10):])

    return float(end_mean - start_mean)


def compute_calibration_metrics(raw_signal, filtered_signal, calibrated_signal, true_signal):
    raw_vs_true = compare_signals(raw_signal, true_signal)
    filtered_vs_true = compare_signals(filtered_signal, true_signal)
    calibrated_vs_true = compare_signals(calibrated_signal, true_signal)
    calibrated_vs_raw = compare_signals(calibrated_signal, raw_signal)
    calibrated_vs_filtered = compare_signals(calibrated_signal, filtered_signal)

    drift_filtered = compute_drift_metric(filtered_signal)
    drift_calibrated = compute_drift_metric(calibrated_signal)

    return {
        "raw_vs_true": raw_vs_true,
        "filtered_vs_true": filtered_vs_true,
        "calibrated_vs_true": calibrated_vs_true,
        "calibrated_vs_raw": calibrated_vs_raw,
        "calibrated_vs_filtered": calibrated_vs_filtered,
        "gain_mae_vs_filtered": float(filtered_vs_true["MAE"] - calibrated_vs_true["MAE"]),
        "gain_rmse_vs_filtered": float(filtered_vs_true["RMSE"] - calibrated_vs_true["RMSE"]),
        "gain_corr_vs_filtered": float(calibrated_vs_true["correlation"] - filtered_vs_true["correlation"]),
        "drift_filtered": float(drift_filtered) if np.isfinite(drift_filtered) else np.nan,
        "drift_calibrated": float(drift_calibrated) if np.isfinite(drift_calibrated) else np.nan,
        "drift_removed": float(drift_filtered - drift_calibrated)
        if np.isfinite(drift_filtered) and np.isfinite(drift_calibrated) else np.nan,
    }


def benchmark_calibration_models(filtered_models, raw_signal, true_signal, time_1d, instrument="DBSC", drift_degree=2, preserve_mean=True):
    results = {}

    for model_name, filtered_signal in filtered_models.items():
        cal = calibrate_filtered_signal(
            signal_1d=filtered_signal,
            time_1d=time_1d,
            instrument=instrument,
            drift_degree=drift_degree,
            preserve_mean=preserve_mean
        )
        metrics = compute_calibration_metrics(
            raw_signal=raw_signal,
            filtered_signal=filtered_signal,
            calibrated_signal=cal["calibrated_signal"],
            true_signal=true_signal
        )

        score = (
            3.0 * metrics["gain_mae_vs_filtered"] +
            3.0 * metrics["gain_rmse_vs_filtered"] +
            2.0 * metrics["gain_corr_vs_filtered"] +
            0.2 * metrics["drift_removed"]
        )

        results[model_name] = {
            "filtered_signal": filtered_signal,
            "calibrated_signal": cal["calibrated_signal"],
            "gain": cal["gain"],
            "offset": cal["offset"],
            "drift_degree": cal["drift_degree"],
            "drift_curve": cal["drift_curve"],
            "metrics": metrics,
            "score": float(score)
        }

    best_model_name = max(results.keys(), key=lambda k: results[k]["score"])
    return best_model_name, results


def summarize_calibration_benchmark(results):
    rows = []
    for model_name, result in results.items():
        m = result["metrics"]
        rows.append({
            "model": model_name,
            "score": result["score"],
            "gain": result["gain"],
            "offset": result["offset"],
            "drift_degree": result["drift_degree"],
            "calibrated_mae": m["calibrated_vs_true"]["MAE"],
            "calibrated_rmse": m["calibrated_vs_true"]["RMSE"],
            "calibrated_corr": m["calibrated_vs_true"]["correlation"],
            "gain_mae_vs_filtered": m["gain_mae_vs_filtered"],
            "gain_rmse_vs_filtered": m["gain_rmse_vs_filtered"],
            "gain_corr_vs_filtered": m["gain_corr_vs_filtered"],
            "drift_removed": m["drift_removed"],
        })
    return rows


def prove_calibration_quality(raw_signal, true_signal, calibrated_models, time_1d):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    axes[0, 0].plot(time_1d, raw_signal, alpha=0.30, label="Brut")
    axes[0, 0].plot(time_1d, true_signal, linewidth=2.0, label="Réel")
    for model_name, sig in calibrated_models.items():
        axes[0, 0].plot(time_1d, sig, linewidth=1.5, label=model_name)
    axes[0, 0].set_title("Comparaison globale des modèles calibrés")
    axes[0, 0].set_xlabel("Temps [heures]")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(time_1d, raw_signal - true_signal, alpha=0.6, label="Brut - réel")
    for model_name, sig in calibrated_models.items():
        axes[0, 1].plot(time_1d, sig - true_signal, linewidth=1.3, label=f"{model_name} - réel")
    axes[0, 1].set_title("Écarts au signal réel")
    axes[0, 1].set_xlabel("Temps [heures]")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    for model_name, sig in calibrated_models.items():
        valid = sig[~np.isnan(sig)]
        if len(valid) > 0:
            axes[1, 0].hist(valid, bins=40, alpha=0.4, density=True, label=model_name)
    axes[1, 0].set_title("Distribution des signaux calibrés")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    first_key = list(calibrated_models.keys())[0]
    ref_sig = calibrated_models[first_key]
    for model_name, sig in list(calibrated_models.items())[1:]:
        mask = ~(np.isnan(ref_sig) | np.isnan(sig))
        if np.sum(mask) >= 3:
            axes[1, 1].scatter(ref_sig[mask], sig[mask], s=6, alpha=0.4, label=f"{model_name} vs {first_key}")
    axes[1, 1].set_title("Relation entre modèles calibrés")
    axes[1, 1].set_xlabel(first_key)
    axes[1, 1].set_ylabel("Autres modèles")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig