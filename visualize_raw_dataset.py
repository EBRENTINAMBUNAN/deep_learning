import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
DATA_PATH = "data/PRSA_data_2010.1.1-2014.12.31.csv"
if not os.path.exists(DATA_PATH):
    print("Dataset tidak ditemukan di folder 'data/'.")
    exit()

print("[INFO] Membaca dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna()
df.columns = df.columns.str.strip().str.upper()

print("[INFO] Menampilkan 5 baris pertama:")
print(df.head())

# 2. Plot distribusi nilai PM2.5
plt.figure(figsize=(10, 5))
sns.histplot(df['PM2.5'], bins=50, kde=True)
plt.title("Distribusi Nilai PM2.5")
plt.xlabel("PM2.5")
plt.ylabel("Frekuensi")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Korelasi antar fitur
plt.figure(figsize=(10, 8))
corr_matrix = df[['PM2.5', 'DEWP', 'TEMP', 'PRES', 'IWS']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap Korelasi Fitur")
plt.tight_layout()
plt.show()
