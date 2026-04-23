import numpy as np
import pytest

from src.simulation import generate_realistic_instrument_data
from src.calibration import (
    calibrate_filtered_signal,
    inject_calibrated_signal_in_dataset,
    compute_calibration_metrics,
    compare_signals,
)


def _get_spatial_mean(ds):
    return ds["temperature"].mean(dim=["latitude", "longitude"]).values


def test_calibrate_filtered_signal_returns_expected_keys():
    ds = generate_realistic_instrument_data(
        duration_hours=12,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=42,
        include_gaps=True,
        include_glitches=True,
        include_saturation=False
    )

    signal = _get_spatial_mean(ds)
    time_h = ds["time_hours"].values

    result = calibrate_filtered_signal(
        signal_1d=signal,
        time_1d=time_h,
        instrument="DBSC",
        drift_degree=2,
        preserve_mean=True
    )

    expected_keys = {
        "calibrated_signal",
        "drift_curve",
        "drift_coeffs",
        "gain",
        "offset",
        "drift_degree",
    }

    assert set(result.keys()) == expected_keys


def test_calibrated_signal_has_same_length_as_input():
    ds = generate_realistic_instrument_data(
        duration_hours=8,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=1,
        include_gaps=True,
        include_glitches=False,
        include_saturation=False
    )

    signal = _get_spatial_mean(ds)
    time_h = ds["time_hours"].values

    result = calibrate_filtered_signal(signal, time_h, instrument="DBSC")
    calibrated = result["calibrated_signal"]

    assert len(calibrated) == len(signal)
    assert len(result["drift_curve"]) == len(signal)


def test_inject_calibrated_signal_in_dataset_preserves_shape():
    ds = generate_realistic_instrument_data(
        duration_hours=6,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=2,
        include_gaps=False,
        include_glitches=False,
        include_saturation=False
    )

    signal = _get_spatial_mean(ds)
    time_h = ds["time_hours"].values

    result = calibrate_filtered_signal(signal, time_h, instrument="DBSC")
    ds_cal = inject_calibrated_signal_in_dataset(ds, result["calibrated_signal"])

    assert ds_cal["temperature"].shape == ds["temperature"].shape
    assert ds_cal.attrs["calibrated"] is True
    assert ds_cal.attrs["calibration_source"] == "filtered_signal"


def test_compare_signals_returns_expected_metrics():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.1, 2.1, 2.9, 3.8])

    metrics = compare_signals(x, y)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "correlation" in metrics
    assert "bias" in metrics
    assert "std_error" in metrics

    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert -1.0 <= metrics["correlation"] <= 1.0


def test_compute_calibration_metrics_returns_consistent_structure():
    ds = generate_realistic_instrument_data(
        duration_hours=10,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=3,
        include_gaps=True,
        include_glitches=True,
        include_saturation=False
    )

    raw_signal = _get_spatial_mean(ds)
    true_signal = ds["physical_signal"].values
    time_h = ds["time_hours"].values

    cal = calibrate_filtered_signal(
        signal_1d=raw_signal,
        time_1d=time_h,
        instrument="DBSC",
        drift_degree=2,
        preserve_mean=True
    )

    metrics = compute_calibration_metrics(
        raw_signal=raw_signal,
        filtered_signal=raw_signal,
        calibrated_signal=cal["calibrated_signal"],
        true_signal=true_signal
    )

    expected_top_keys = {
        "raw_vs_true",
        "filtered_vs_true",
        "calibrated_vs_true",
        "calibrated_vs_raw",
        "calibrated_vs_filtered",
        "gain_mae_vs_filtered",
        "gain_rmse_vs_filtered",
        "gain_corr_vs_filtered",
        "drift_filtered",
        "drift_calibrated",
        "drift_removed",
    }

    assert expected_top_keys.issubset(metrics.keys())


def test_calibration_handles_nan_only_signal():
    signal = np.array([np.nan, np.nan, np.nan], dtype=float)
    time_h = np.array([0.0, 1.0, 2.0], dtype=float)

    result = calibrate_filtered_signal(signal, time_h, instrument="DBSC")

    assert np.all(np.isnan(result["calibrated_signal"]))
    assert np.all(np.isnan(result["drift_curve"]))
    assert np.isnan(result["gain"])
    assert np.isnan(result["offset"])


def test_calibration_raises_on_shape_mismatch():
    signal = np.array([1.0, 2.0, 3.0])
    time_h = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        calibrate_filtered_signal(signal, time_h, instrument="DBSC")


def test_calibration_output_is_not_constant_on_realistic_data():
    ds = generate_realistic_instrument_data(
        duration_hours=12,
        sampling_rate_hz=1.0,
        instrument_name="DBSC",
        seed=123,
        include_gaps=False,
        include_glitches=False,
        include_saturation=False
    )

    signal = _get_spatial_mean(ds)
    time_h = ds["time_hours"].values

    result = calibrate_filtered_signal(signal, time_h, instrument="DBSC")
    calibrated = result["calibrated_signal"]

    assert np.nanstd(calibrated) > 0