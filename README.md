# Accelerometer Signal Denoising (Offline)

This repository implements and compares various signal denoising methods (traditional and machine learning-based) for accelerometer time-series data in an **offline** setting. It provides a simple but extensible framework for testing different denoising techniques using synthetic or real-world data.

---

## 📌 Problem Statement

Accelerometer signals are frequently corrupted by noise due to environmental disturbances, sensor inaccuracies, and electronic interference. This repository demonstrates how to clean such signals using:

- Traditional signal processing methods: Moving Average, Savitzky–Golay filter, Kalman Filter  
- Machine Learning methods: Multi-Layer Perceptron (MLP) Regression

All processing is done **offline**, assuming the entire time series is available beforehand.

---

## 📁 Directory Structure

```
accel_denoising_offline/
├── data/          # Data folder (raw and processed signals)
├── models/        # Trained ML models (MLP, etc.)
├── notebooks/     # Jupyter notebooks for experimentation
├── results/       # Visualizations and output signals
├── src/           # Core source code
│   ├── preprocess.py   # Signal generation and preprocessing
│   ├── filters.py      # MA, SG, Kalman filters
│   ├── ml_model.py     # MLP training and prediction
│   └── utils.py        # Metric calculation and plotting
├── main.py        # Driver script for full pipeline
└── README.md      # This file
```

---

## 🧠 Methods Implemented

### 1. **Moving Average (MA)**
- Smoothens the signal using a sliding average window  
- Hyperparameter: `window_size`

### 2. **Savitzky–Golay Filter**
- Polynomial fitting within a moving window  
- Hyperparameters: `window_size`, `polyorder`

### 3. **Kalman Filter**
- Dynamic state estimation assuming a linear system with Gaussian noise  
- Assumes constant acceleration model

### 4. **MLP Regressor (Offline)**
- Fully connected neural network trained to learn a denoising mapping  
- Input: Noisy signal segments  
- Output: Estimated clean value at center of window  
- Hyperparameters: `window_size`, learning rate, hidden layers

---

## 📊 Evaluation Metrics

- **Mean Squared Error (MSE)**  
- **Signal-to-Noise Ratio (SNR)** (optional)  
- Visual comparison plots: Clean vs. Noisy vs. Denoised

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MSHemanth/accel_denoising_offline.git
cd accel_denoising_offline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`

### 3. Run the Main Pipeline

```bash
python main.py
```

This will:

- Generate synthetic signal (clean + noisy)  
- Apply each filter (MA, SG, Kalman)  
- Train and apply MLP  
- Compare outputs and print metrics

---

## 🧪 Example Output

| Method           | MSE     |
|------------------|---------|
| Noisy Signal     | 0.0274  |
| Moving Average   | 0.0142  |
| Savitzky-Golay   | 0.0119  |
| Kalman Filter    | 0.0093  |
| MLP Regressor    | 0.0064  |

---

## 📝 Notes

- All filtering is **offline**, assuming access to both future and past data.
- For real-time (online) filtering, a different implementation would be needed.
- You can modify the synthetic signal generation to simulate various motion profiles or load real accelerometer logs.

---

## ✅ TODO

- [ ] Add GRU/LSTM models for temporal learning  
- [ ] Add real accelerometer dataset loader  
- [ ] Compare runtime performance of methods  
- [ ] Integrate windowed denoising as a streaming wrapper

---

## 👨‍💻 Author

**Hemanth Madduri**  
MTech, Aerospace Engineering  
IIT Madras  
🌐 [hemanth.madduri.xyz](https://hemanth.madduri.xyz)
