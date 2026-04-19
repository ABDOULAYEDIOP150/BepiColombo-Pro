import numpy as np

def compute_fft(signal):
    freq = np.fft.fftfreq(len(signal))
    spectrum = np.abs(np.fft.fft(signal))
    return freq, spectrum