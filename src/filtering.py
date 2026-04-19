from scipy.signal import butter, filtfilt

def butterworth_filter(data, cutoff=0.05, order=4):
    b, a = butter(order, cutoff, btype="low")
    return filtfilt(b, a, data)