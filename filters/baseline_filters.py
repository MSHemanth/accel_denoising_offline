# filters/baseline_filters.py

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt

def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def savgol_denoise(signal, window_length=11, polyorder=2):
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

def butter_lowpass_denoise(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)
