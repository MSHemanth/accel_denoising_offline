# utils/simulate_data.py

import numpy as np
import os

def generate_accel_data(freq=1.0, dt=0.01, N=1000, noise_std=0.2, seed=42):
    """
    Generate clean and noisy synthetic accelerometer signal.
    
    Args:
        freq (float): Frequency of the synthetic acceleration signal.
        dt (float): Time step between samples.
        N (int): Number of time steps.
        noise_std (float): Standard deviation of Gaussian noise.
        seed (int): Random seed for reproducibility.

    Returns:
        t (np.ndarray): Time vector.
        clean (np.ndarray): Clean signal.
        noisy (np.ndarray): Noisy signal.
    """
    np.random.seed(seed)
    t = np.linspace(0, dt*N, N)
    clean = np.sin(2 * np.pi * freq * t)  # You can replace this with a more complex signal later
    noise = np.random.normal(0, noise_std, size=N)
    noisy = clean + noise
    return t, clean, noisy

def save_data(t, clean, noisy, out_dir="data/synthetic"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "time.npy"), t)
    np.save(os.path.join(out_dir, "clean.npy"), clean)
    np.save(os.path.join(out_dir, "noisy.npy"), noisy)
    print(f"Saved time, clean, and noisy data to {out_dir}/")

if __name__ == "__main__":
    t, clean, noisy = generate_accel_data(freq=2.0, dt=0.01, N=1000, noise_std=0.3)
    save_data(t, clean, noisy)
