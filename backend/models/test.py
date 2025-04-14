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

# Filter data khusus Americano
data_americano = df[df['Item Name'] == 'Americano'].copy()

# Konversi Date ke datetime
data_americano['Date'] = pd.to_datetime(data_americano['Date'])

# Tambahkan kolom Month dalam format string
data_americano['Month'] = data_americano['Date'].dt.to_period('M').astype(str)

# Daftar kolom numerik (hanya yang boleh di-sum)
kolom_numerik = [col for col in data_americano.columns if col not in ['Date', 'Month', 'Item Name', 'Category Name']]

# Agregasi ke level bulanan, hanya menjumlahkan kolom numerik
data_bulanan = data_americano.groupby('Month')[kolom_numerik].sum().reset_index()

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
joblib.dump(scaler_features, "scaler_features_americano.pkl")
joblib.dump(scaler_target, "scaler_target_americano.pkl")
print("\nScaler telah disimpan sebagai 'scaler_features_americano.pkl' dan 'scaler_target_americano.pkl'")

scaler_features = joblib.load("scaler_features_americano.pkl")
scaler_target = joblib.load("scaler_target_americano.pkl")

print("Feature Scaler Mean:", scaler_features.center_)
print("Feature Scaler Scale:", scaler_features.scale_)
print("Target Scaler Mean:", scaler_target.center_)
print("Target Scaler Scale:", scaler_target.scale_)


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
model.save("lstm_model_americano.h5")
print("\nModel telah disimpan sebagai 'lstm_model_americano.h5'")

# Prediksi pada data uji
y_pred_scaled = model.predict(X_test)
y_pred = scaler_target.inverse_transform(y_pred_scaled).flatten()
y_test_original = scaler_target.inverse_transform(y_test).flatten()

# Evaluasi performa model
mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred)

# print("\nEvaluasi untuk Menu: Americano")
# print(f"MSE: {mse:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")

# ==========================================================
# Prediksi Item Sold untuk 1 Bulan ke Depan
# ==========================================================
# Prediksi item sold
last_input = X_scaled[-1].reshape(1, 1, X_scaled.shape[1])
next_pred_scaled = model.predict(last_input)
predicted_sold = int(round(scaler_target.inverse_transform(next_pred_scaled)[0][0]))

# Hitung kebutuhan bahan baku berdasarkan rasio per porsi
bahan_baku_total = {}
for bahan in bahan_baku:
    # Ambil rata-rata rasio bahan per porsi
    mean_ratio = data_bulanan[f'Per Porsi {bahan}'].mean()
    total_bahan = max(int(round(predicted_sold * mean_ratio)), 0)
    bahan_baku_total[bahan] = f"{total_bahan} gram/ml"

print(f"Predicted Sold: {predicted_sold}")
print("Bahan Baku Bulanan:", bahan_baku_total)


# Simpan hasil prediksi ke Excel
# output_data = {'Item Sold': [next_pred]}
# for bahan, jumlah in bahan_baku_total.items():
#     output_data[bahan] = [jumlah]

# output_df = pd.DataFrame(output_data)
# output_file = "Prediksi_Americano_Bulanan.xlsx"
# output_df.to_excel(output_file, index=False)

# print(f"\nHasil prediksi telah disimpan dalam file: {output_file}")

# ==========================================================
# Grafik Training Loss vs Validation Loss
# ==========================================================
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'], label='Training Loss', color='blue')
# plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training vs Validation Loss - Americano")
# plt.legend()
# plt.grid()
# plt.show()

# ==========================================================
# Grafik Actual vs Predicted
# ==========================================================
# plt.figure(figsize=(10, 5))
# plt.plot(y_test_original, label="Actual", marker='o', linestyle='dashed', color='blue')
# plt.plot(y_pred, label="Predicted", marker='s', linestyle='dashed', color='red')
# plt.xlabel("Months")
# plt.ylabel("Item Sold")
# plt.title("Actual vs Predicted Item Sold (Monthly) - Americano")
# plt.legend()
# plt.grid()
# plt.show()
