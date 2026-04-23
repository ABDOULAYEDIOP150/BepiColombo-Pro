import numpy as np
import pytest

from src.simulation import generate_realistic_instrument_data
from src.filtering import (
    butterworth_filter,
    savgol_filter_signal,
    combo_filter,
    analyze_filter_effectiveness,
    compare_to_true_signal,
    grid_search_butterworth_parameters,
    grid_search_savgol_parameters,
    grid_search_combo_parameters,
)


def _get_spatial_mean(ds):
    return ds["temperature"].mean(dim=["latitude", "longitude"]).values


def test_butterworth_filter_returns_same_length():
    x = np.sin(np.linspace(0, 10, 500)) + 0.2 * np.random.randn(500)
    y = butterworth_filter(x, cutoff=0.1, order=4, fs=1.0)

    assert len(y) == len(x)


def test_butterworth_filter_reduces_noise_std_on_noisy_signal():
    np.random.seed(0)
    t = np.arange(1000)
    clean = np.sin(2 * np.pi * 0.01 * t)
    noisy = clean + 0.5 * np.random.randn(len(t))

    filtered = butterworth_filter(noisy, cutoff=0.05, order=4, fs=1.0)
    metrics = analyze_filter_effectiveness(noisy, filtered)

    assert metrics["filtered_std"] < metrics["raw_std"]
    assert np.isfinite(metrics["noise_reduction_db"])


def test_butterworth_filter_raises_on_invalid_cutoff():
    x = np.random.randn(100)

    with pytest.raises(ValueError):
        butterworth_filter(x, cutoff=0.0, order=4, fs=1.0)

    with pytest.raises(ValueError):
        butterworth_filter(x, cutoff=1.0, order=4, fs=1.0)  # fs=1 => nyquist=0.5


def test_savgol_filter_returns_same_length():
    x = np.sin(np.linspace(0, 10, 501)) + 0.1 * np.random.randn(501)
    y = savgol_filter_signal(x, window_length=21, polyorder=3)

    assert len(y) == len(x)


def test_combo_filter_returns_same_length():
    x = np.sin(np.linspace(0, 10, 501)) + 0.2 * np.random.randn(501)
    y = combo_filter(
        x,
        fs=1.0,
        cutoff=0.05,
        order=4,
        window_length=21,
        polyorder=3
    )

    assert len(y) == len(x)


def test_compare_to_true_signal_returns_valid_metrics():
    true_signal = np.array([1.0, 2.0, 3.0, 4.0])
    estimated_signal = np.array([1.1, 1.9, 3.2, 3.8])

    metrics = compare_to_true_signal(estimated_signal, true_signal)

    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert -1.0 <= metrics["correlation"] <= 1.0


def test_grid_search_butterworth_returns_best_result():
    ds = generate_realistic_instrument_data(
        duration_hours=12,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=10,
        include_gaps=True,
        include_glitches=True,
        include_saturation=False
    )

    raw_signal = _get_spatial_mean(ds)
    true_signal = ds["physical_signal"].values

    best_result, all_results = grid_search_butterworth_parameters(
        raw_signal=raw_signal,
        true_signal=true_signal,
        fs=1.0,
        max_dominant_freq=0.001,
        orders=(3, 4),
        n_cutoffs=4,
        cutoff_multiplier_min=1.0,
        cutoff_multiplier_max=1.2,
    )

    assert isinstance(all_results, list)
    assert len(all_results) > 0
    assert "filtered_signal" in best_result
    assert "score" in best_result
    assert len(best_result["filtered_signal"]) == len(raw_signal)


def test_grid_search_savgol_returns_best_result():
    ds = generate_realistic_instrument_data(
        duration_hours=12,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=11,
        include_gaps=True,
        include_glitches=True,
        include_saturation=False
    )

    raw_signal = _get_spatial_mean(ds)
    true_signal = ds["physical_signal"].values

    best_result, all_results = grid_search_savgol_parameters(
        raw_signal=raw_signal,
        true_signal=true_signal,
        window_lengths=(11, 21),
        polyorders=(2, 3),
    )

    assert isinstance(all_results, list)
    assert len(all_results) > 0
    assert "filtered_signal" in best_result
    assert "score" in best_result


def test_grid_search_combo_returns_best_result():
    ds = generate_realistic_instrument_data(
        duration_hours=12,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=12,
        include_gaps=True,
        include_glitches=True,
        include_saturation=False
    )

    raw_signal = _get_spatial_mean(ds)
    true_signal = ds["physical_signal"].values

    best_result, all_results = grid_search_combo_parameters(
        raw_signal=raw_signal,
        true_signal=true_signal,
        fs=1.0,
        max_dominant_freq=0.001,
        butter_orders=(3, 4),
        n_cutoffs=4,
        cutoff_multiplier_min=1.0,
        cutoff_multiplier_max=1.2,
        window_lengths=(11, 21),
        polyorders=(2, 3),
    )

    assert isinstance(all_results, list)
    assert len(all_results) > 0
    assert "filtered_signal" in best_result
    assert "score" in best_result


def test_filtering_handles_nans():
    x = np.sin(np.linspace(0, 10, 500))
    x[100:110] = np.nan

    y = butterworth_filter(x, cutoff=0.05, order=4, fs=1.0)

    assert len(y) == len(x)
    assert np.all(np.isnan(y[100:110]))


def test_analyze_filter_effectiveness_returns_expected_keys():
    x = np.sin(np.linspace(0, 10, 500)) + 0.2 * np.random.randn(500)
    y = butterworth_filter(x, cutoff=0.05, order=4, fs=1.0)

    metrics = analyze_filter_effectiveness(x, y)

    expected_keys = {"raw_std", "filtered_std", "noise_reduction_db"}
    assert expected_keys.issubset(metrics.keys())