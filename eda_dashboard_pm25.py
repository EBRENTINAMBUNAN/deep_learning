import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Data PM2.5", layout="wide")
st.title("📊 Dashboard Analisis Data Kualitas Udara - PM2.5")

# Load dataset
DATA_PATH = "data/PRSA_data_2010.1.1-2014.12.31.csv"
if not os.path.exists(DATA_PATH):
    st.error("Dataset tidak ditemukan di folder 'data/'. Harap simpan file .csv di sana.")
    st.stop()

# Baca dan bersihkan data
df = pd.read_csv(DATA_PATH)
df = df.dropna()
df.columns = df.columns.str.strip().str.upper()

# Split data (sekali di awal agar tetap konsisten)
fitur = ['DEWP', 'TEMP', 'PRES', 'IWS']
target = 'PM2.5'
X = df[fitur]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gabungkan kembali untuk tampilan
train_data = X_train.copy()
train_data[target] = y_train

test_data = X_test.copy()
test_data[target] = y_test

# Sidebar
with st.sidebar:
    st.header("🧭 Navigasi")
    options = st.radio("Pilih visualisasi:", [
        "Preview Data Mentah",
        "Data Training",
        "Data Testing",
        "Distribusi Fitur",
        "Boxplot Fitur",
        "Heatmap Korelasi",
        "Trend PM2.5 dari Waktu"
    ])

# Preview
if options == "Preview Data Mentah":
    st.subheader("📄 Seluruh Data Mentah")
    st.dataframe(df)
    st.write("Jumlah Baris:", df.shape[0], "| Jumlah Kolom:", df.shape[1])

elif options == "Data Training":
    st.subheader("📘 Data Training (80%)")
    st.dataframe(train_data)
    st.write("Jumlah Baris:", train_data.shape[0])

elif options == "Data Testing":
    st.subheader("📙 Data Testing (20%)")
    st.dataframe(test_data)
    st.write("Jumlah Baris:", test_data.shape[0])

# Distribusi
elif options == "Distribusi Fitur":
    st.subheader("📊 Distribusi Setiap Fitur")
    fitur_numerik = ['PM2.5', 'DEWP', 'TEMP', 'PRES', 'IWS']
    for col in fitur_numerik:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col], bins=50, kde=True, ax=ax)
        ax.set_title(f"Distribusi: {col}")
        st.pyplot(fig)

# Boxplot
elif options == "Boxplot Fitur":
    st.subheader("📦 Deteksi Outlier dengan Boxplot")
    fitur_numerik = ['PM2.5', 'DEWP', 'TEMP', 'PRES', 'IWS']
    for col in fitur_numerik:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot: {col}")
        st.pyplot(fig)

# Korelasi
elif options == "Heatmap Korelasi":
    st.subheader("🔥 Korelasi Antar Fitur")
    fitur_corr = ['PM2.5', 'DEWP', 'TEMP', 'PRES', 'IWS']
    corr = df[fitur_corr].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi Fitur")
    st.pyplot(fig)

# Trend waktu
elif options == "Trend PM2.5 dari Waktu":
    st.subheader("📈 Trend PM2.5 dari Waktu")
    if all(col in df.columns for col in ['YEAR', 'MONTH', 'DAY', 'HOUR']):
        df['DATETIME'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR']])
        df_sorted = df.sort_values('DATETIME')
        df_daily = df_sorted.groupby('DATETIME')['PM2.5'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df_daily['DATETIME'], df_daily['PM2.5'], color='steelblue')
        ax.set_title("Rata-rata Harian PM2.5 dari Waktu")
        ax.set_ylabel("PM2.5")
        ax.set_xlabel("Tanggal")
        st.pyplot(fig)
    else:
        st.warning("Kolom waktu (YEAR, MONTH, DAY, HOUR) tidak tersedia di dataset.")