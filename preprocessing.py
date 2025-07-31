import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=500, order=3):
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq
    from scipy.signal import butter, filtfilt
    b,a=butter(order,[low,high],btype='band')
    return filtfilt(b,a,signal)

def normalize(signal):
    return (signal - np.mean(signal))/(np.std(signal)+1e-8)
