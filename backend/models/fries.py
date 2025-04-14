import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

print(tf.__version__) 

# Load data
file_path = "Data Warung Fotkop.xlsx"
df = pd.read_excel(file_path)

# Filter data khusus Garlic Fries
data_garlic_fries = df[df['Item Name'] == 'Garlic Fries'].copy()

# Konversi Date ke datetime
data_garlic_fries['Date'] = pd.to_datetime(data_garlic_fries['Date'])

# Tambahkan kolom Month dalam format string
data_garlic_fries['Month'] = data_garlic_fries['Date'].dt.to_period('M').astype(str)

# Daftar kolom numerik (hanya yang boleh di-sum)
kolom_numerik = [col for col in data_garlic_fries.columns if col not in ['Date', 'Month', 'Item Name', 'Category Name']]

# Agregasi ke level bulanan, hanya menjumlahkan kolom numerik
data_bulanan = data_garlic_fries.groupby('Month')[kolom_numerik].sum().reset_index()

# Fungsi menghapus outliers (Interquartile Range - IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Hapus outliers pada kolom Item Sold dan semua bahan baku
data_bulanan = remove_outliers(data_bulanan, 'Item Sold')
bahan_baku = [col for col in data_bulanan.columns if col not in ['Month', 'Item Sold']]
for bahan in bahan_baku:
    data_bulanan = remove_outliers(data_bulanan, bahan)

# Hitung rasio bahan baku per porsi
for bahan in bahan_baku:
    data_bulanan[f'Per Porsi {bahan}'] = data_bulanan[bahan] / data_bulanan['Item Sold']

# Normalisasi dengan RobustScaler
scaler_features = RobustScaler()
scaler_target = RobustScaler()
features = ['Item Sold'] + [f'Per Porsi {bahan}' for bahan in bahan_baku]
X_scaled = scaler_features.fit_transform(data_bulanan[features])
y_scaled = scaler_target.fit_transform(data_bulanan[['Item Sold']])

# Simpan scaler agar bisa dipakai nanti saat prediksi
joblib.dump(scaler_features, "scaler_features_garlic_fries.pkl")
joblib.dump(scaler_target, "scaler_target_garlic_fries.pkl")
print("\nScaler telah disimpan sebagai 'scaler_features_garlic_fries.pkl' dan 'scaler_target_garlic_fries.pkl'")

# Split data train-test (80% train, 20% test)
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# Bentuk ulang untuk input LSTM (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Bangun model LSTM
model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='huber')

# Latih model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_split=0.2)

# Simpan model
model.save("lstm_garlic_fries.h5")
print("\nModel telah disimpan sebagai 'lstm_garlic_fries.h5'")

# Prediksi pada data uji
y_pred_scaled = model.predict(X_test)
y_pred = scaler_target.inverse_transform(y_pred_scaled).flatten()
y_test_original = scaler_target.inverse_transform(y_test).flatten()

# Evaluasi performa model
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred)

# print("\n Evaluasi Model Garlic Fries")
# print(f"MSE: {mse:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")

# Tampilkan grafik training loss dan validation loss jika model baru dilatih
# if 'history' in locals():
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['loss'], label='Training Loss', color='blue')
#     plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training vs Validation Loss - Garlic Fries")
#     plt.legend()
#     plt.grid()
#     plt.show()

# Grafik Actual vs Predicted
# plt.figure(figsize=(10, 5))
# plt.plot(y_test_original, label="Actual", marker='o', linestyle='dashed', color='blue')
# plt.plot(y_pred, label="Predicted", marker='s', linestyle='dashed', color='red')
# plt.xlabel("Data Points")
# plt.ylabel("Item Sold")
# plt.title("Actual vs Predicted - Garlic Fries")
# plt.legend()
# plt.grid()
# plt.show()

# ==========================================================
# Prediksi Item Sold dan Bahan Baku untuk 1 Bulan ke Depan
# ==========================================================
last_input = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])
next_pred_scaled = model.predict(last_input)
next_pred = int(round(scaler_target.inverse_transform(next_pred_scaled)[0][0]))

# Hitung total kebutuhan bahan baku untuk 1 bulan
print(f"\n Prediksi Item Sold Garlic Fries Bulan Depan: {next_pred}")

bahan_baku_total = {}
for bahan in bahan_baku:
    mean_ratio = data_bulanan[f'Per Porsi {bahan}'].median()
    total_bahan = int(round(next_pred * mean_ratio))
    bahan_baku_total[bahan] = max(total_bahan, 0)

print("\n Prediksi Kebutuhan Bahan Baku Bulan Depan:")
for bahan, jumlah in bahan_baku_total.items():
    print(f"{bahan}: {jumlah} gram/ml")

# Simpan hasil prediksi ke file Excel
# output_data = {'Item Sold': [next_pred]}
# for bahan, jumlah in bahan_baku_total.items():
#     output_data[bahan] = [jumlah]

# output_df = pd.DataFrame(output_data)
# output_file = "Prediksi_GarlicFries_Bulanan.xlsx"
# output_df.to_excel(output_file, index=False)

# print(f"\n Hasil prediksi telah disimpan dalam file: {output_file}")