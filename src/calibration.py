import numpy as np

def calibrate(dataset):
    corrected = dataset.copy()

    drift = 0.015 * corrected["time"]

    corrected["temperature"] = corrected["temperature"] - drift

    return corrected