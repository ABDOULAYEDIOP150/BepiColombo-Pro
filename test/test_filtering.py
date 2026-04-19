from src.filtering import butterworth_filter
import numpy as np

def test_filtering():
    data = np.sin(np.linspace(0, 10, 100))
    out = butterworth_filter(data)

    assert len(out) == len(data)