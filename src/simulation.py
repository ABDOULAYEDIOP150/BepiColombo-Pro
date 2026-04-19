import numpy as np
import xarray as xr

def generate_spatial_temperature_data(t=100, lat=30, lon=30, seed=42):
    np.random.seed(seed)

    time = np.arange(t)
    latitude = np.linspace(-90, 90, lat)
    longitude = np.linspace(-180, 180, lon)

    temp = np.zeros((t, lat, lon))

    for i in range(t):
        for j in range(lat):
            for k in range(lon):

                # 🌍 physique réelle (gradient solaire)
                base = 20 - 0.04 * abs(latitude[j])

                # 🌊 onde spatiale (rotation + dynamique planète)
                wave = 5 * np.sin(0.1 * i + longitude[k] * 0.05)

                # 📡 bruit instrument (capteur réel)
                noise = np.random.normal(0, 0.8)

                # 📉 dérive instrumentale (vieillissement capteur)
                drift = 0.015 * i

                temp[i, j, k] = base + wave + noise + drift

    ds = xr.Dataset(
        {
            "temperature": (["time", "latitude", "longitude"], temp)
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude
        }
    )

    return ds