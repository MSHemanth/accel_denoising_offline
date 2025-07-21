# helpers.py

import numpy as np

def create_sliding_window_dataset(noisy_signal, clean_signal, window_size):
    half_w = window_size // 2
    X, y = [], []

    for i in range(half_w, len(noisy_signal) - half_w):
        window = noisy_signal[i - half_w:i + half_w + 1]
        X.append(window)
        y.append(clean_signal[i])

    return np.array(X), np.array(y)
