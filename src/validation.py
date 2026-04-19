import numpy as np

def validate(dataset):
    temp = dataset["temperature"]

    return {
        "mean": float(temp.mean()),
        "std": float(temp.std()),
        "min": float(temp.min()),
        "max": float(temp.max()),
        "valid": bool(temp.std() > 0)
    }