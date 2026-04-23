"""
Validation utilities for BepiColombo DBSC project.

Objectif
--------
Valider un dataset calibré ou non en calculant :
- statistiques globales
- qualité des données
- conformité à une plage physique
- détection d'anomalies (z-score + IQR)
- score synthétique de validation
"""

import numpy as np
import xarray as xr


def _extract_temperature_array(dataset):
    """
    Extrait la variable temperature depuis un xarray.Dataset,
    un xarray.DataArray, ou accepte directement un ndarray.
    """
    if isinstance(dataset, xr.Dataset):
        if "temperature" not in dataset:
            raise ValueError("Le dataset doit contenir la variable 'temperature'.")
        temp = dataset["temperature"].values
    elif isinstance(dataset, xr.DataArray):
        temp = dataset.values
    else:
        temp = np.asarray(dataset, dtype=float)

    return np.asarray(temp, dtype=float)


def _safe_stats(x):
    """
    Calcule des statistiques robustes sur un vecteur 1D sans NaN.
    """
    if len(x) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "variance": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "iqr": np.nan,
        }

    q01 = np.percentile(x, 1)
    q05 = np.percentile(x, 5)
    q25 = np.percentile(x, 25)
    q75 = np.percentile(x, 75)
    q95 = np.percentile(x, 95)
    q99 = np.percentile(x, 99)

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "variance": float(np.var(x)),
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "iqr": float(q75 - q25),
    }


def _detect_anomalies_zscore(values, sigma=3.0):
    """
    Détection d'anomalies par z-score.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.array([], dtype=bool)

    std_val = np.std(values)
    mean_val = np.mean(values)

    if std_val == 0 or np.isnan(std_val):
        return np.zeros(len(values), dtype=bool)

    z_scores = np.abs((values - mean_val) / std_val)
    return z_scores > sigma


def _detect_anomalies_iqr(values, iqr_factor=1.5):
    """
    Détection d'anomalies par méthode IQR.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.array([], dtype=bool)

    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25

    if iqr == 0 or np.isnan(iqr):
        return np.zeros(len(values), dtype=bool)

    lower_bound = q25 - iqr_factor * iqr
    upper_bound = q75 + iqr_factor * iqr

    return (values < lower_bound) | (values > upper_bound)


def _compute_validation_score(
    nan_percent,
    within_range_percent,
    anomaly_percent_zscore,
    anomaly_percent_iqr
):
    """
    Score synthétique simple sur 100.
    """
    score = 100.0

    score -= 2.0 * nan_percent
    score -= 1.0 * max(0.0, 100.0 - within_range_percent)
    score -= 0.5 * anomaly_percent_zscore
    score -= 0.5 * anomaly_percent_iqr

    return float(max(0.0, min(100.0, score)))


def validate(
    dataset,
    min_valid=-50.0,
    max_valid=100.0,
    anomaly_sigma=3.0,
    iqr_factor=1.5
):
    """
    Validate dataset against a physical admissible range.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray or ndarray
        Dataset contenant 'temperature' ou tableau numérique.
    min_valid : float
        Valeur minimale admissible.
    max_valid : float
        Valeur maximale admissible.
    anomaly_sigma : float
        Seuil de détection d'anomalies par z-score.
    iqr_factor : float
        Facteur de détection d'anomalies par IQR.

    Returns
    -------
    dict
        Résumé complet de validation.
    """
    temp = _extract_temperature_array(dataset)

    total_points = int(temp.size)
    temp_flat = temp.flatten()

    nan_mask = np.isnan(temp_flat)
    nan_count = int(np.sum(nan_mask))
    valid_values = temp_flat[~nan_mask]

    if len(valid_values) == 0:
        return {
            "total_points": total_points,
            "valid_points": 0,
            "nan_count": nan_count,
            "nan_percent": 100.0 if total_points > 0 else np.nan,

            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "variance": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "iqr": np.nan,

            "valid_range": False,
            "within_range_percent": 0.0,
            "out_of_range_count": 0,
            "out_of_range_percent": 0.0,
            "below_min_count": 0,
            "above_max_count": 0,

            "anomaly_count_zscore": 0,
            "anomaly_percent_zscore": 0.0,
            "anomaly_count_iqr": 0,
            "anomaly_percent_iqr": 0.0,

            "min_valid_threshold": float(min_valid),
            "max_valid_threshold": float(max_valid),
            "anomaly_sigma_threshold": float(anomaly_sigma),
            "iqr_factor_threshold": float(iqr_factor),

            "validation_score": 0.0,
            "validation_status": "invalid_no_data",
        }

    stats = _safe_stats(valid_values)

    in_range_mask = (valid_values >= min_valid) & (valid_values <= max_valid)
    out_of_range_mask = ~in_range_mask

    valid_points = int(len(valid_values))
    out_of_range_count = int(np.sum(out_of_range_mask))
    below_min_count = int(np.sum(valid_values < min_valid))
    above_max_count = int(np.sum(valid_values > max_valid))

    within_range_percent = 100.0 * np.sum(in_range_mask) / valid_points
    out_of_range_percent = 100.0 * out_of_range_count / valid_points
    nan_percent = 100.0 * nan_count / total_points if total_points > 0 else np.nan

    anomaly_mask_zscore = _detect_anomalies_zscore(valid_values, sigma=anomaly_sigma)
    anomaly_count_zscore = int(np.sum(anomaly_mask_zscore))
    anomaly_percent_zscore = 100.0 * anomaly_count_zscore / valid_points

    anomaly_mask_iqr = _detect_anomalies_iqr(valid_values, iqr_factor=iqr_factor)
    anomaly_count_iqr = int(np.sum(anomaly_mask_iqr))
    anomaly_percent_iqr = 100.0 * anomaly_count_iqr / valid_points

    valid_range = bool(out_of_range_count == 0)

    validation_score = _compute_validation_score(
        nan_percent=nan_percent,
        within_range_percent=within_range_percent,
        anomaly_percent_zscore=anomaly_percent_zscore,
        anomaly_percent_iqr=anomaly_percent_iqr
    )

    if valid_range and nan_percent == 0 and anomaly_percent_zscore < 1.0 and anomaly_percent_iqr < 1.0:
        validation_status = "excellent"
    elif valid_range and nan_percent < 2.0 and anomaly_percent_zscore < 3.0:
        validation_status = "good"
    elif valid_range and nan_percent < 5.0:
        validation_status = "acceptable"
    elif within_range_percent >= 95.0:
        validation_status = "warning"
    else:
        validation_status = "invalid"

    return {
        "total_points": total_points,
        "valid_points": valid_points,
        "nan_count": nan_count,
        "nan_percent": float(nan_percent),

        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "median": stats["median"],
        "variance": stats["variance"],
        "q01": stats["q01"],
        "q05": stats["q05"],
        "q25": stats["q25"],
        "q75": stats["q75"],
        "q95": stats["q95"],
        "q99": stats["q99"],
        "iqr": stats["iqr"],

        "valid_range": valid_range,
        "within_range_percent": float(within_range_percent),
        "out_of_range_count": out_of_range_count,
        "out_of_range_percent": float(out_of_range_percent),
        "below_min_count": below_min_count,
        "above_max_count": above_max_count,

        "anomaly_count_zscore": anomaly_count_zscore,
        "anomaly_percent_zscore": float(anomaly_percent_zscore),
        "anomaly_count_iqr": anomaly_count_iqr,
        "anomaly_percent_iqr": float(anomaly_percent_iqr),

        "min_valid_threshold": float(min_valid),
        "max_valid_threshold": float(max_valid),
        "anomaly_sigma_threshold": float(anomaly_sigma),
        "iqr_factor_threshold": float(iqr_factor),

        "validation_score": float(validation_score),
        "validation_status": validation_status,
    }