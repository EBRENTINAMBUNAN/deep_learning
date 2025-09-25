import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import save_model

# 1. Load dataset
DATA_PATH = "data/PRSA_data_2010.1.1-2014.12.31.csv"
if not os.path.exists(DATA_PATH):
    print("Dataset tidak ditemukan di folder 'data/'. Silakan unduh dan simpan sebagai 'PRSA_data_2010.1.1-2014.12.31.csv'")
    exit()

print("\n[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# 2. Bersihkan data (hapus nilai NaN)
print("[INFO] Membersihkan data...")
df = df.dropna()

# 3. Normalisasi nama kolom agar seragam
print("[INFO] Standarisasi nama kolom...")
df.columns = df.columns.str.strip().str.upper()

# 4. Pilih fitur yang digunakan
features = ['DEWP', 'TEMP', 'PRES', 'IWS']  # embun, suhu, tekanan, kecepatan angin
X = df[features]
y = df['PM2.5']

# 5. Normalisasi data
print("[INFO] Normalisasi fitur...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split data latih dan uji
print("[INFO] Membagi data latih dan uji...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Bangun model
print("[INFO] Membangun model deep learning...")
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # output numerik
])

# 8. Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 9. Latih model
print("[INFO] Melatih model...")
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=1)

# 10. Evaluasi
print("[INFO] Evaluasi model...")
loss, mae = model.evaluate(X_test, y_test)
print(f"\n✅ Mean Absolute Error (MAE): {mae:.2f}")

# 11. Simpan model & scaler
os.makedirs("models", exist_ok=True)
save_model(model, "models/pm25_model.h5")
print("\n💾 Model disimpan di 'models/pm25_model.h5'")