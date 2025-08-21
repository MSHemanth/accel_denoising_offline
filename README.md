# Accelerometer Signal Denoising (Offline)

This repository demonstrates how to denoise synthetic accelerometer signals using both traditional signal processing methods and machine learning models. The workflow is fully offline, assuming the entire time series is available for processing.

---

## 📁 Directory Structure

```
accel_denoising_offline/
├── data/                # Synthetic data (clean, noisy, train sets)
├── filters/             # Traditional filter implementations and helpers
├── ml/                  # ML model training and baseline ML filters
├── models/              # Saved ML models (LinearRegression, MLP, RandomForest)
├── noteooks/            # Jupyter notebooks for experimentation
├── utils/               # Data simulation utilities
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🧑‍💻 Step-by-Step Workflow

### 1. **Synthetic Data Generation**

- **File:** [`utils/simulate_data.py`](utils/simulate_data.py)
- **Function:** [`generate_accel_data`](utils/simulate_data.py)
- Generates a clean sinusoidal signal and adds Gaussian noise.
- Saves time, clean, and noisy signals as `.npy` files in `data/synthetic/`.

**Usage:**
```python
# utils/simulate_data.py
if __name__ == "__main__":
    t, clean, noisy = generate_accel_data(freq=2.0, dt=0.01, N=1000, noise_std=0.3)
    save_data(t, clean, noisy)
```

---

### 2. **Traditional Baseline Filters**

- **File:** [`filters/baseline_filters.py`](filters/baseline_filters.py)
- Implements:
  - [`moving_average`](filters/baseline_filters.py): Simple windowed average.
  - [`savgol_denoise`](filters/baseline_filters.py): Savitzky–Golay polynomial filter.
  - [`butter_lowpass_denoise`](filters/baseline_filters.py): Butterworth low-pass filter.

**Example usage in notebook:**
```python
from filters.baseline_filters import moving_average, savgol_denoise, butter_lowpass_denoise

ma_signal = moving_average(noisy, window_size=15)
sg_signal = savgol_denoise(noisy, window_length=31, polyorder=3)
bw_signal = butter_lowpass_denoise(noisy, cutoff=3.0, fs=100.0)
```

---

### 3. **Sliding Window Dataset Preparation**

- **File:** [`filters/helpers.py`](filters/helpers.py)
- **Function:** [`create_sliding_window_dataset`](filters/helpers.py)
- Converts the noisy signal into overlapping windows (features) and aligns the clean signal as targets for supervised ML training.

**Example usage:**
```python
from filters.helpers import create_sliding_window_dataset

X, y = create_sliding_window_dataset(noisy, clean, window_size=21)
np.save("../data/synthetic/X_train.npy", X)
np.save("../data/synthetic/y_train.npy", y)
```

---

### 4. **Machine Learning Model Training**

- **File:** [`ml/train_models.py`](ml/train_models.py)
- **Function:** [`train_and_evaluate_models`](ml/train_models.py)
- Trains three models:
  - Linear Regression
  - Random Forest Regressor
  - MLP Regressor (Neural Network)
- Splits data into train/test, evaluates MSE and R², and saves models to `models/`.

**Example usage:**
```python
from ml.train_models import train_and_evaluate_models

results = train_and_evaluate_models(X, y)
```

---

### 5. **ML Baseline Filters**

- **File:** [`ml/baseline_filters.py`](ml/baseline_filters.py)
- Implements:
  - [`moving_average_filter`](ml/baseline_filters.py): Same as traditional.
  - [`savgol_filter_denoise`](ml/baseline_filters.py): Same as traditional.
  - [`kalman_filter_denoise`](ml/baseline_filters.py): Kalman filter using `pykalman`.

**Example usage:**
```python
from ml.baseline_filters import moving_average_filter, savgol_filter_denoise, kalman_filter_denoise

ma = moving_average_filter(y_noisy)
sg = savgol_filter_denoise(y_noisy)
kalman = kalman_filter_denoise(y_noisy)
```

---

### 6. **Evaluation and Visualization**

- **Notebook:** [`noteooks/01_baseline_denoising.ipynb`](noteooks/01_baseline_denoising.ipynb)
- Loads all signals and predictions, plots them for visual comparison.
- Computes metrics (MSE, R²) for each method.

**Example plotting:**
```python
plt.plot(y, label="Clean Signal")
plt.plot(y_noisy, label="Noisy", alpha=0.4)
plt.plot(y_pred_mlp, label="MLP", linestyle='--')
plt.plot(ma, label="Moving Avg")
plt.plot(sg, label="Savitzky-Golay")
plt.plot(kalman, label="Kalman")
plt.legend()
plt.title("Denoising Comparison")
plt.show()
```

**Example metrics:**
```python
from sklearn.metrics import mean_squared_error

def print_metrics(y_true, preds, labels):
    for p, label in zip(preds, labels):
        mse = mean_squared_error(y_true[-len(p):], p)
        print(f"{label} → MSE: {mse:.6f}")
```

---

## 🧠 Methods Implemented

- **Moving Average:** Smooths signal using a sliding window.
- **Savitzky–Golay Filter:** Fits polynomials in a moving window.
- **Butterworth Filter:** Low-pass filtering for noise reduction.
- **Kalman Filter:** State estimation for time-series denoising.
- **MLP Regressor:** Neural network trained to map noisy windows to clean center values.

---

## 📊 Evaluation Metrics

- **Mean Squared Error (MSE)**
- **R² Score**
- **Visual comparison plots**

---

## 🚀 Getting Started

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Generate synthetic data:
    ```sh
    python utils/simulate_data.py
    ```
3. Run notebooks for filtering, dataset creation, ML training, and evaluation.

---

## ✅ TODO

- Add real-world data support
- Implement additional ML models (e.g., CNN, LSTM)
- Add SNR metric
- Improve Kalman filter tuning
