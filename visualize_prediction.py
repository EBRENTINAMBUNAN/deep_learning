import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load model
MODEL_PATH = "models/pm25_model.h5"
if not os.path.exists(MODEL_PATH):
    print("Model tidak ditemukan. Jalankan main.py terlebih dahulu.")
    exit()

model = load_model(MODEL_PATH)
print("[INFO] Model berhasil dimuat.")

# 2. Load dataset
DATA_PATH = "data/PRSA_data_2010.1.1-2014.12.31.csv"
if not os.path.exists(DATA_PATH):
    print("Dataset tidak ditemukan di folder 'data/'.")
    exit()

df = pd.read_csv(DATA_PATH)
df = df.dropna()
df.columns = df.columns.str.strip().str.upper()

# 3. Ambil fitur dan target
features = ['DEWP', 'TEMP', 'PRES', 'IWS']
X = df[features]
y = df['PM2.5']

# 4. Normalisasi fitur (ulang)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Prediksi
print("[INFO] Melakukan prediksi data...")
y_pred = model.predict(X_scaled)

# 6. Visualisasi 100 data pertama
plt.figure(figsize=(12, 5))
plt.plot(y.values[:100], label='Asli', marker='o', linestyle='-', alpha=0.6)
plt.plot(y_pred[:100], label='Prediksi', marker='x', linestyle='--', alpha=0.7)
plt.title('Prediksi vs Nilai Asli PM2.5 (100 data pertama)')
plt.xlabel('Index')
plt.ylabel('PM2.5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()