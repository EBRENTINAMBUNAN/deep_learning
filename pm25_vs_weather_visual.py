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

# 2. Visualisasi PM2.5 vs DEWP, TEMP, PRES, IWS dalam satu figure
features = ['DEWP', 'TEMP', 'PRES', 'IWS']
plt.figure(figsize=(14, 10))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=feature, y='PM2.5', data=df, alpha=0.3, edgecolor=None)
    plt.title(f'PM2.5 vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('PM2.5')
    plt.grid(True)

plt.suptitle("Korelasi Visual: PM2.5 terhadap Fitur Cuaca", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 3. Korelasi numerik
print("\n[INFO] Korelasi Numerik:")
print(df[['PM2.5', 'DEWP', 'TEMP', 'PRES', 'IWS']].corr())
