"""
Complete processing pipeline for BepiColombo DBSC and companion instruments.

Objectif
--------
Pipeline complet avec :
1. génération réaliste des données
2. analyse exploratoire des données brutes
3. FFT d'analyse fréquentielle
4. benchmark de filtrage :
   - Butterworth
   - Savitzky-Golay
   - Combo Butterworth + Savitzky-Golay
5. calibration des signaux filtrés
6. sélection du meilleur modèle
7. validation physique
8. comparaison inter-instruments
"""

import numpy as np
import pandas as pd

from src.simulation import generate_realistic_instrument_data
from src.filtering import (
    grid_search_butterworth_parameters,
    grid_search_savgol_parameters,
    grid_search_combo_parameters,
    summarize_grid_search_results,
    compare_to_true_signal,
)
from src.calibration import (
    calibrate_filtered_signal,
    benchmark_calibration_models,
    summarize_calibration_benchmark,
    inject_calibrated_signal_in_dataset,
)
from src.validation import validate
from src.logger import setup_logger


# =============================================================================
# UTILITAIRES LOCAUX
# =============================================================================
def get_spatial_mean(ds):
    return ds["temperature"].mean(dim=["latitude", "longitude"]).values


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


def compute_fft(signal, fs=1.0, detrend_signal=True):
    signal = np.asarray(signal, dtype=float)
    signal_interp, _ = interpolate_nans(signal)

    if np.all(np.isnan(signal_interp)):
        return np.array([]), np.array([])

    if detrend_signal:
        signal_interp = signal_interp - np.mean(signal_interp)

    n = len(signal_interp)
    fft_vals = np.fft.rfft(signal_interp)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(fft_vals) / n

    return freqs, magnitude


def get_dominant_frequencies(freqs, magnitude, n_peaks=5, min_relative_height=0.1):
    if len(freqs) == 0 or len(magnitude) == 0:
        return []

    if len(magnitude) < 3:
        return []

    from scipy.signal import find_peaks

    peaks, _ = find_peaks(
        magnitude,
        height=np.max(magnitude) * min_relative_height
    )

    if len(peaks) == 0:
        if len(magnitude) > 1:
            idx_sorted = np.argsort(magnitude[1:])[::-1] + 1
        else:
            idx_sorted = np.array([0])
    else:
        idx_sorted = peaks[np.argsort(magnitude[peaks])[::-1]]

    idx_sorted = idx_sorted[:n_peaks]

    results = []
    for idx in idx_sorted:
        f = freqs[idx]
        mag = magnitude[idx]
        results.append({
            "frequency_hz": float(f),
            "magnitude": float(mag)
        })

    return results


def compare_instruments(ds1, ds2):
    sig1 = ds1["temperature"].mean(dim=["latitude", "longitude"]).values
    sig2 = ds2["temperature"].mean(dim=["latitude", "longitude"]).values

    mask = ~(np.isnan(sig1) | np.isnan(sig2))

    if np.sum(mask) < 3:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "correlation": np.nan
        }

    err = sig1[mask] - sig2[mask]
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    corr = np.corrcoef(sig1[mask], sig2[mask])[0, 1]

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "correlation": float(corr)
    }


def analyze_raw_data(ds, instrument_name="DBSC"):
    raw_signal = get_spatial_mean(ds)
    true_signal = ds["physical_signal"].values

    raw_vs_true = compare_to_true_signal(raw_signal, true_signal)

    quality_flags = ds.attrs.get("quality_flags", {})

    return {
        "instrument": instrument_name,
        "n_samples": int(len(raw_signal)),
        "raw_vs_true": raw_vs_true,
        "missing_data_percent": float(quality_flags.get("missing_data_percent", np.nan)),
        "glitches_count": int(quality_flags.get("glitches_count", 0)),
        "mean": float(np.nanmean(raw_signal)),
        "std": float(np.nanstd(raw_signal)),
        "min": float(np.nanmin(raw_signal)),
        "max": float(np.nanmax(raw_signal)),
    }


def summarize_final_results(filter_models, calibration_results):
    rows = []

    for model_name in filter_models.keys():
        filt = filter_models[model_name]
        cal = calibration_results[model_name]

        rows.append({
            "model": model_name,
            "filter_score": filt["score"],
            "filtered_mae": filt["filtered_vs_true"]["MAE"],
            "filtered_rmse": filt["filtered_vs_true"]["RMSE"],
            "filtered_corr": filt["filtered_vs_true"]["correlation"],
            "calibration_score": cal["score"],
            "calibrated_mae": cal["metrics"]["calibrated_vs_true"]["MAE"],
            "calibrated_rmse": cal["metrics"]["calibrated_vs_true"]["RMSE"],
            "calibrated_corr": cal["metrics"]["calibrated_vs_true"]["correlation"],
            "drift_removed": cal["metrics"]["drift_removed"],
        })

    return pd.DataFrame(rows).sort_values("calibration_score", ascending=False)


def compute_reliability_score(
    best_filter_result,
    best_calibration_result,
    dbsc_validation,
    mag_validation,
    comparison_metrics
):
    score = 0
    max_score = 5
    details = []

    # 1. Filtrage améliore les métriques
    if (
        best_filter_result["gain_mae"] > 0 and
        best_filter_result["gain_rmse"] > 0
    ):
        score += 1
        details.append("✅ Filtrage: amélioration de MAE et RMSE")
    else:
        details.append("⚠️ Filtrage: amélioration insuffisante")

    # 2. Calibration améliore encore
    cal_metrics = best_calibration_result["metrics"]
    if (
        cal_metrics["gain_mae_vs_filtered"] > 0 and
        cal_metrics["gain_rmse_vs_filtered"] > 0
    ):
        score += 1
        details.append("✅ Calibration: amélioration vs signal filtré")
    else:
        details.append("⚠️ Calibration: amélioration insuffisante")

    # 3. Validation physique DBSC et MAG
    if dbsc_validation["valid_range"] and mag_validation["valid_range"]:
        score += 1
        details.append("✅ Validation: DBSC et MAG dans la plage physique")
    else:
        details.append("⚠️ Validation: violation de plage physique")

    # 4. Corrélation calibrée élevée
    if cal_metrics["calibrated_vs_true"]["correlation"] > 0.95:
        score += 1
        details.append("✅ Corrélation calibrée > 0.95")
    else:
        details.append("⚠️ Corrélation calibrée insuffisante")

    # 5. Cohérence inter-instruments
    if comparison_metrics["correlation"] > 0.7:
        score += 1
        details.append("✅ Cohérence inter-instruments acceptable")
    else:
        details.append("⚠️ Cohérence inter-instruments faible")

    return score, max_score, details


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================
def run_pipeline(
    duration_hours=48,
    sampling_rate_hz=1.0,
    include_gaps=True,
    include_glitches=True,
    drift_degree=2
):
    logger = setup_logger()

    logger.info("=" * 80)
    logger.info("BEPICOLOMBO DBSC PIPELINE - FULL ENGINEERING ANALYSIS")
    logger.info("=" * 80)

    # =========================================================================
    # STEP 1 - GENERATION
    # =========================================================================
    logger.info("[STEP 1] Generating realistic instrument data...")

    dbsc_raw = generate_realistic_instrument_data(
        duration_hours=duration_hours,
        sampling_rate_hz=sampling_rate_hz,
        instrument_name="DBSC",
        seed=42,
        include_gaps=include_gaps,
        include_glitches=include_glitches,
        include_saturation=False
    )

    mag_raw = generate_realistic_instrument_data(
        duration_hours=duration_hours,
        sampling_rate_hz=sampling_rate_hz,
        instrument_name="MAG",
        seed=123,
        include_gaps=include_gaps,
        include_glitches=include_glitches,
        include_saturation=False
    )

    # =========================================================================
    # STEP 2 - EXPLORATORY ANALYSIS
    # =========================================================================
    logger.info("[STEP 2] Exploratory analysis on raw DBSC and MAG data...")

    raw_stats_dbsc = analyze_raw_data(dbsc_raw, "DBSC")
    raw_stats_mag = analyze_raw_data(mag_raw, "MAG")

    logger.info(f"DBSC raw correlation to true signal: {raw_stats_dbsc['raw_vs_true']['correlation']:.4f}")
    logger.info(f"MAG raw correlation to true signal:  {raw_stats_mag['raw_vs_true']['correlation']:.4f}")

    # =========================================================================
    # STEP 3 - FFT ANALYSIS
    # =========================================================================
    logger.info("[STEP 3] FFT analysis on raw DBSC signal...")

    raw_signal_dbsc = get_spatial_mean(dbsc_raw)
    true_signal_dbsc = dbsc_raw["physical_signal"].values
    fs = dbsc_raw.attrs["sampling_rate_hz"]

    freqs, mag = compute_fft(raw_signal_dbsc, fs=fs, detrend_signal=True)
    dominant = get_dominant_frequencies(freqs, mag, n_peaks=5, min_relative_height=0.1)

    if len(dominant) > 0:
        max_dominant_freq = max(item["frequency_hz"] for item in dominant)
    else:
        max_dominant_freq = 0.001

    logger.info(f"Max dominant frequency from FFT: {max_dominant_freq:.6f} Hz")

    # =========================================================================
    # STEP 4 - FILTERING BENCHMARK
    # =========================================================================
    logger.info("[STEP 4] Filtering benchmark: Butterworth / Savitzky-Golay / Combo...")

    best_butter, all_butter = grid_search_butterworth_parameters(
        raw_signal=raw_signal_dbsc,
        true_signal=true_signal_dbsc,
        fs=fs,
        max_dominant_freq=max_dominant_freq,
        orders=(3, 4, 5, 6),
        n_cutoffs=8,
        cutoff_multiplier_min=1.0,
        cutoff_multiplier_max=1.5,
    )

    best_sg, all_sg = grid_search_savgol_parameters(
        raw_signal=raw_signal_dbsc,
        true_signal=true_signal_dbsc,
        window_lengths=(11, 21, 31, 41),
        polyorders=(2, 3, 4),
    )

    best_combo, all_combo = grid_search_combo_parameters(
        raw_signal=raw_signal_dbsc,
        true_signal=true_signal_dbsc,
        fs=fs,
        max_dominant_freq=max_dominant_freq,
        butter_orders=(3, 4, 5, 6),
        n_cutoffs=6,
        cutoff_multiplier_min=1.0,
        cutoff_multiplier_max=1.5,
        window_lengths=(11, 21, 31),
        polyorders=(2, 3),
    )

    filter_models = {
        "Butterworth": best_butter,
        "Savitzky-Golay": best_sg,
        "Combo": best_combo,
    }

    best_filter_model_name = max(filter_models.keys(), key=lambda k: filter_models[k]["score"])
    best_filter_result = filter_models[best_filter_model_name]

    logger.info(f"Best filtering model: {best_filter_model_name}")
    logger.info(f"Best filtering score: {best_filter_result['score']:.4f}")

    # =========================================================================
    # STEP 5 - CALIBRATION BENCHMARK
    # =========================================================================
    logger.info("[STEP 5] Calibration benchmark on filtered models...")

    filtered_models = {
        model_name: result["filtered_signal"]
        for model_name, result in filter_models.items()
    }

    best_calibrated_model_name, calibration_results = benchmark_calibration_models(
        filtered_models=filtered_models,
        raw_signal=raw_signal_dbsc,
        true_signal=true_signal_dbsc,
        time_1d=dbsc_raw["time_hours"].values,
        instrument=dbsc_raw.attrs.get("instrument", "DBSC"),
        drift_degree=drift_degree,
        preserve_mean=True
    )

    best_calibration_result = calibration_results[best_calibrated_model_name]
    best_calibrated_signal = best_calibration_result["calibrated_signal"]

    logger.info(f"Best calibrated model: {best_calibrated_model_name}")
    logger.info(f"Best calibrated score: {best_calibration_result['score']:.4f}")

    dbsc_calibrated = inject_calibrated_signal_in_dataset(dbsc_raw, best_calibrated_signal)

    # MAG : calibration simple
    mag_signal = get_spatial_mean(mag_raw)
    mag_cal = calibrate_filtered_signal(
        signal_1d=mag_signal,
        time_1d=mag_raw["time_hours"].values,
        instrument=mag_raw.attrs.get("instrument", "MAG"),
        drift_degree=drift_degree,
        preserve_mean=True
    )
    mag_calibrated = inject_calibrated_signal_in_dataset(mag_raw, mag_cal["calibrated_signal"])

    # =========================================================================
    # STEP 6 - VALIDATION
    # =========================================================================
    logger.info("[STEP 6] Validation against physical standards...")

    dbsc_valid = validate(dbsc_calibrated)
    mag_valid = validate(mag_calibrated)

    logger.info(f"DBSC validation status: {dbsc_valid['validation_status']}")
    logger.info(f"MAG validation status:  {mag_valid['validation_status']}")

    # =========================================================================
    # STEP 7 - INTER-INSTRUMENT COMPARISON
    # =========================================================================
    logger.info("[STEP 7] Inter-instrument comparison...")

    comparison_metrics = compare_instruments(dbsc_calibrated, mag_calibrated)

    logger.info(f"Inter-instrument correlation: {comparison_metrics['correlation']:.4f}")

    # =========================================================================
    # STEP 8 - FINAL ASSESSMENT
    # =========================================================================
    logger.info("[STEP 8] Final assessment...")

    reliability_score, max_score, reliability_details = compute_reliability_score(
        best_filter_result=best_filter_result,
        best_calibration_result=best_calibration_result,
        dbsc_validation=dbsc_valid,
        mag_validation=mag_valid,
        comparison_metrics=comparison_metrics
    )

    final_table = summarize_final_results(filter_models, calibration_results)

    logger.info("=" * 80)
    logger.info("FINAL ASSESSMENT - DATA RELIABILITY")
    logger.info("=" * 80)

    for line in reliability_details:
        logger.info(line)

    logger.info(f"Reliability score: {reliability_score}/{max_score}")

    if reliability_score >= 4:
        conclusion = "RELIABLE for scientific use"
    elif reliability_score >= 2:
        conclusion = "USABLE but requires additional validation"
    else:
        conclusion = "NOT reliable - review processing parameters"

    logger.info(f"Conclusion: {conclusion}")

    return {
        "dbsc_raw": dbsc_raw,
        "mag_raw": mag_raw,

        "raw_statistics_dbsc": raw_stats_dbsc,
        "raw_statistics_mag": raw_stats_mag,

        "fft_dominant_frequencies": dominant,
        "max_dominant_frequency": max_dominant_freq,

        "filter_models": filter_models,
        "best_filter_model_name": best_filter_model_name,
        "best_filter_result": best_filter_result,
        "all_butter_results": all_butter,
        "all_savgol_results": all_sg,
        "all_combo_results": all_combo,
        "filter_summary_table": pd.DataFrame(summarize_grid_search_results(all_butter + all_sg + all_combo)),

        "calibration_results": calibration_results,
        "best_calibrated_model_name": best_calibrated_model_name,
        "best_calibration_result": best_calibration_result,
        "final_model_table": final_table,

        "dbsc_calibrated": dbsc_calibrated,
        "mag_calibrated": mag_calibrated,

        "dbsc_validation": dbsc_valid,
        "mag_validation": mag_valid,
        "comparison_metrics": comparison_metrics,

        "reliability_score": reliability_score,
        "reliability_max_score": max_score,
        "reliability_details": reliability_details,
        "final_conclusion": conclusion,
    }


if __name__ == "__main__":
    results = run_pipeline()

    print("\n" + "=" * 80)
    print("FINAL MODEL TABLE")
    print("=" * 80)
    print(results["final_model_table"].to_string(index=False))

    print("\n" + "=" * 80)
    print("FINAL CONCLUSION")
    print("=" * 80)
    print(results["final_conclusion"])