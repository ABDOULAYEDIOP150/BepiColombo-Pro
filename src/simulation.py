"""
Simulation réaliste des données instrumentales pour BepiColombo DBSC
1. Bruit thermique réaliste (dépendant de la température)
2. Dérive non-linéaire
3. Artefacts instrumentaux (glitches, gaps, saturation)
4. Effets d'orbite réels
5. Bruit en 1/f (bruit rose)
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta

def generate_realistic_instrument_data(
    duration_hours=24,
    sampling_rate_hz=1.0,
    instrument_name="DBSC",
    seed=42,
    include_gaps=True,
    include_glitches=True,
    include_saturation=False
):
    """
    Génère des données instrumentales réalistes pour BepiColombo DBSC.

    Returns:
        xarray.Dataset avec dimensions (time, latitude, longitude)
    """
    np.random.seed(seed)

    # Paramètres temporels
    n_samples = int(duration_hours * 3600 * sampling_rate_hz)
    time = np.arange(n_samples)
    time_hours = time / (3600 * sampling_rate_hz)

    # Paramètres spatiaux
    lat, lon = 2, 3
    latitude = np.linspace(-90, 90, lat)
    longitude = np.linspace(-180, 180, lon)

    # ---- 1. Signal physique réel ----
    orbital_period_hours = 2.3
    orbital_freq = 1.0 / (orbital_period_hours * 3600) * (1 / sampling_rate_hz)
    orbital_signal = 5.0 * np.sin(2 * np.pi * orbital_freq * time)

    mercury_rotation_period_hours = 59 * 24
    rotation_freq = 1.0 / (mercury_rotation_period_hours * 3600) * (1 / sampling_rate_hz)
    rotation_signal = 2.0 * np.sin(2 * np.pi * rotation_freq * time)

    local_variations = 1.5 * np.sin(2 * np.pi * 0.0005 * time) + \
                       0.8 * np.sin(2 * np.pi * 0.001 * time + 0.5)

    physical_signal = orbital_signal + rotation_signal + local_variations

    # Variation spatiale (latitude/longitude)
    lat_factor = 10.0 * np.cos(np.radians(latitude))[:, None]
    lon_factor = 5.0 * np.sin(np.radians(longitude))[None, :]
    spatial_pattern = lat_factor + lon_factor

    # ---- 2. Dérive instrumentale réaliste ----
    if instrument_name == "DBSC":
        tau_drift = 6 * 3600 * sampling_rate_hz
        warmup_drift = 8.0 * (1 - np.exp(-time / tau_drift))
        linear_drift = 0.008 * time_hours
        instrument_drift = warmup_drift + linear_drift
        base_noise_std = 0.5
    elif instrument_name == "MAG":
        tau_drift = 8 * 3600 * sampling_rate_hz
        warmup_drift = 6.0 * (1 - np.exp(-time / tau_drift))
        linear_drift = 0.006 * time_hours
        instrument_drift = warmup_drift + linear_drift
        base_noise_std = 0.4
    else:  # SWA
        tau_drift = 4 * 3600 * sampling_rate_hz
        warmup_drift = 10.0 * (1 - np.exp(-time / tau_drift))
        linear_drift = 0.010 * time_hours
        instrument_drift = warmup_drift + linear_drift
        base_noise_std = 0.7

    # ---- 3. Bruit thermique dépendant de la température ----
    instrument_temp = 20.0 + instrument_drift
    thermal_noise_std = base_noise_std * (1 + 0.02 * (instrument_temp - 20))
    thermal_noise = np.random.normal(0, thermal_noise_std[:, None, None],
                                     (n_samples, lat, lon))

    # ---- 4. Bruit en 1/f (bruit rose) ----
    def pink_noise(n_samples, alpha=1.0):
        freqs = np.fft.rfftfreq(n_samples)
        spectrum = freqs.copy()
        spectrum[0] = 1.0
        spectrum[1:] = 1.0 / (spectrum[1:] ** (alpha / 2))
        phase = np.random.rand(len(spectrum)) * 2 * np.pi
        fft_signal = spectrum * (np.cos(phase) + 1j * np.sin(phase))
        pink = np.fft.irfft(fft_signal, n=n_samples)
        return pink / np.std(pink) * 0.3

    pink_noise_1d = pink_noise(n_samples, alpha=0.8)
    pink_noise_3d = pink_noise_1d[:, None, None] * np.random.randn(lat, lon) * 0.2

    # ---- 5. Assemblage du signal de base ----
    data = np.zeros((n_samples, lat, lon))
    for i in range(n_samples):
        data[i] = physical_signal[i] + spatial_pattern + instrument_drift[i]

    data = data + thermal_noise + pink_noise_3d

    # ---- 6. Glitches (artefacts) ----
    if include_glitches:
        n_glitches = np.random.randint(5, 15)
        for _ in range(n_glitches):
            glitch_time = np.random.randint(100, n_samples - 100)
            glitch_amplitude = np.random.uniform(-3.0, 3.0)
            glitch_duration = np.random.randint(1, 5)
            glitch_spread = np.random.choice([1, 2, 3])
            if glitch_spread == 1:
                lat_idx = np.random.randint(0, lat)
                lon_idx = np.random.randint(0, lon)
                data[glitch_time:glitch_time+glitch_duration, lat_idx, lon_idx] += glitch_amplitude
            elif glitch_spread == 2:
                lat_idx = np.random.randint(0, lat)
                data[glitch_time:glitch_time+glitch_duration, lat_idx, :] += glitch_amplitude
            else:
                data[glitch_time:glitch_time+glitch_duration, :, :] += glitch_amplitude

    # ---- 7. Saturation optionnelle ----
    if include_saturation:
        data = np.clip(data, -15.0, 35.0)

    # ---- 8. Gaps (données manquantes) ----
    if include_gaps:
        n_gaps = np.random.randint(2, 6)
        gap_mask = np.ones(n_samples, dtype=bool)
        for _ in range(n_gaps):
            gap_start = np.random.randint(500, n_samples - 500)
            gap_duration = np.random.randint(50, 200)
            gap_mask[gap_start:gap_start+gap_duration] = False
        data[~gap_mask] = np.nan

    # ---- 9. Signal de calibration périodique ----
    cal_signal = 2.0 * np.sin(2 * np.pi * 0.01 * time / (3600 * sampling_rate_hz))
    cal_mask = (np.abs(physical_signal) < 0.5) & (time_hours % 6 < 0.1)
    data[cal_mask] += cal_signal[cal_mask, None, None] * 0.5

    # ---- 10. Création du Dataset xarray ----
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_coords = [start_time + timedelta(seconds=int(t / sampling_rate_hz)) for t in range(n_samples)]

    ds = xr.Dataset(
        {
            "temperature": (["time", "latitude", "longitude"], data),
            "instrument_temperature": (["time"], instrument_temp),
            "physical_signal": (["time"], physical_signal),
            "calibration_flag": (["time"], cal_mask)
        },
        coords={
            "time": time_coords,
            "latitude": latitude,
            "longitude": longitude,
            "time_hours": ("time", time_hours)
        },
        attrs={
            "instrument": instrument_name,
            "mission": "BepiColombo",
            "description": f"Simulation réaliste {instrument_name}",
            "sampling_rate_hz": sampling_rate_hz,
            "duration_hours": duration_hours,
            "has_gaps": include_gaps,
            "has_glitches": include_glitches,
            "generation_seed": seed
        }
    )

    # Métadonnées qualité
    ds.attrs["quality_flags"] = {
        "missing_data_percent": float(100 * np.isnan(data).sum() / data.size),
        "glitches_count": n_glitches if include_glitches else 0,
        "saturation_applied": include_saturation
    }

    return ds


def add_temperature_dependency(dataset):
    """
    Ajoute une dépendance réaliste à la température pour l'instrument.
    """
    temp = dataset.instrument_temperature.values
    temp_offset = (temp - 20) * 0.05
    dataset.temperature.values += temp_offset[:, None, None]
    dataset.attrs["temperature_correction_applied"] = False
    dataset.attrs["temperature_sensitivity"] = "0.05°C/°C"
    return dataset