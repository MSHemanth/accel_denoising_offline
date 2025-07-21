# ml/baseline_filters.py

import numpy as np
from scipy.signal import savgol_filter
from pykalman import KalmanFilter

def moving_average_filter(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def savgol_filter_denoise(signal, window_length=11, polyorder=2):
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

def kalman_filter_denoise(signal):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(signal, n_iter=5)
    state_means, _ = kf.filter(signal)
    return state_means.flatten()
