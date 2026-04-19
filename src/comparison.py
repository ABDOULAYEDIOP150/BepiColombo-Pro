import numpy as np

def compare(raw, processed):
    return {
        "MAE": float(np.mean(np.abs(raw["temperature"] - processed["temperature"])))
    }