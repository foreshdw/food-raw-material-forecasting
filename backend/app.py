from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib
from flask_cors import CORS
import calendar
from openpyxl import load_workbook
from openpyxl.styles import NamedStyle, Alignment
from werkzeug.utils import secure_filename
from flask import send_file
import os

app = Flask(__name__)
CORS(app)  # Mengizinkan akses CORS dari React

latest_uploaded_filename = None # Variabel global file data terbaru

# Konfigurasi folder upload dan fungsi helper
UPLOAD_FOLDER = "../backend"
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model dan scaler
model_americano = tf.keras.models.load_model("./models/lstm_model_americano.h5")
model_garlic = tf.keras.models.load_model("./models/lstm_garlic_fries.h5")
scaler_features_americano = joblib.load("./models/scaler_features_americano.pkl")
scaler_target_americano = joblib.load("./models/scaler_target_americano.pkl")
scaler_features_garlic = joblib.load("./models/scaler_features_garlic_fries.pkl")
scaler_target_garlic = joblib.load("./models/scaler_target_garlic_fries.pkl")

# Load default dataset saat server start
DEFAULT_FILENAME = "Data Warung Fotkop.xlsx"
df = pd.read_excel(DEFAULT_FILENAME)
latest_uploaded_filename = DEFAULT_FILENAME  # Default saat belum ada upload

# Fungsi untuk mendapatkan tanggal terakhir dari bulan
def get_last_day_of_month(year, month):
    last_day = calendar.monthrange(year, month)[1]
    return pd.to_datetime(f"{year}-{month:02d}-{last_day}")

# Route prediksi
@app.route('/predict', methods=['POST'])
def predict():
    global df
    global latest_uploaded_filename

    # Gunakan nama file terbaru (upload) atau default
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_uploaded_filename)
    df = pd.read_excel(file_path)

    data = request.get_json()
    menu = data.get('menu')

    if menu not in ['Americano', 'Garlic Fries']:
        return jsonify({"error": "Menu tidak ditemukan"}), 400

    # Filter data berdasarkan menu yang dipilih
    data_menu = df[df['Item Name'] == menu].copy()
    data_menu['Date'] = pd.to_datetime(data_menu['Date'])
    data_menu['Month'] = data_menu['Date'].dt.to_period('M').astype(str)

    # Agregasi bulanan
    kolom_numerik = [col for col in data_menu.columns if col not in ['Date', 'Month', 'Item Name', 'Category Name']]
    data_bulanan = data_menu.groupby('Month')[kolom_numerik].sum().reset_index()

    # Rasio bahan baku per porsi
    bahan_baku = [col for col in data_bulanan.columns if col not in ['Month', 'Item Sold']]
    for bahan in bahan_baku:
        data_bulanan[f'Per Porsi {bahan}'] = data_bulanan[bahan] / data_bulanan['Item Sold']

    # Ambil data terakhir untuk prediksi
    last_data = data_bulanan.iloc[-1][['Item Sold']].values.reshape(1, -1)

    # Prediksi penjualan berdasarkan menu
    if menu == 'Americano':
        last_data_scaled = scaler_features_americano.transform(last_data).reshape(1, 1, -1)
        pred_scaled = model_americano.predict(last_data_scaled)
        predicted_sold = int(round(scaler_target_americano.inverse_transform(pred_scaled)[0][0]))
    else:  # Garlic Fries
        last_data_scaled = scaler_features_garlic.transform(last_data).reshape(1, 1, -1)
        pred_scaled = model_garlic.predict(last_data_scaled)
        predicted_sold = int(round(scaler_target_garlic.inverse_transform(pred_scaled)[0][0]))

    # Hitung kebutuhan bahan baku berdasarkan hasil prediksi
    bahan_baku_total = {}
    for bahan in bahan_baku:
        mean_ratio = data_bulanan[f'Per Porsi {bahan}'].mean() # Mean digunakan untuk menghindari perbedaan rasio bahan baku
        total_bahan = max(int(round(predicted_sold * mean_ratio)), 0)
        bahan_baku_total[bahan] = total_bahan

    # Tentukan bulan prediksi
    last_month = pd.to_datetime(data_bulanan['Month'].iloc[-1]).to_period('M')
    next_month = last_month + 1
    next_year = next_month.year
    next_month_num = next_month.month

    # Tentukan tanggal terakhir bulan tersebut
    predicted_date = get_last_day_of_month(next_year, next_month_num)

    # Simpan hasil prediksi ke dalam dataset baru
    new_entry = {
        'Date': predicted_date.strftime('%Y-%m-%d'),  # Memastikan format tetap YYYY-MM-DD
        'Item Name': menu,
        'Category Name': 'Coffee' if menu == 'Americano' else 'Food',
        'Item Sold': predicted_sold
    }

    # Tambahkan bahan baku yang digunakan ke entry baru
    for bahan in bahan_baku:
        new_entry[bahan] = bahan_baku_total[bahan]

    # Konversi ke DataFrame dan tambahkan ke dataset
    df_new_entry = pd.DataFrame([new_entry])
    df = pd.concat([df, df_new_entry], ignore_index=True)

    # Hanya simpan kolom yang diperlukan ke dalam dataframe baru
    required_columns = ['Date', 'Item Name', 'Category Name', 'Item Sold'] + bahan_baku
    df = df[required_columns]

    # Simpan ke file Excel dengan format tanggal yang benar
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

        # Format tanggal di Excel agar tetap dalam bentuk YYYY-MM-DD
        workbook = writer.book
        sheet = workbook.active
        date_column = sheet['A']  # Mengakses Kolom 'Date'

        # Terapkan format tanggal ke seluruh kolom 'Date'
        date_style = NamedStyle(name="date_style", number_format="YYYY-MM-DD")
        for cell in date_column[1:]:  # Lewati header
            cell.style = date_style

        # Atur lebar kolom sebanyak 15 karakter agar tidak terlalu sempit
        sheet.column_dimensions['A'].width = 15
        sheet.column_dimensions['B'].width = 15

        # Menengahkan seluruh isi tabel
        center_alignment = Alignment(horizontal='center')

        # Iterasi ke semua sel untuk menengahkan teks
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = center_alignment

        # Simpan perubahan ke file Excel
        workbook.save(file_path)

    import locale

    # Setel lokal ke bahasa Indonesia
    try:
        locale.setlocale(locale.LC_TIME, 'id_ID.utf8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_TIME, 'indonesian')  # Untuk Windows
        except locale.Error:
            pass  # Gunakan default jika locale tidak tersedia

    # Tentukan bulan prediksi
    last_month = pd.to_datetime(data_bulanan['Month'].iloc[-1]).to_period('M')
    next_month = last_month + 1
    next_year = next_month.year
    next_month_num = next_month.month

    # Tentukan tanggal terakhir bulan tersebut
    predicted_date = get_last_day_of_month(next_year, next_month_num)

    # Ambil nama bulan dalam bahasa Indonesia
    bulan_indonesia = predicted_date.strftime('%B')  # Contoh output: 'Oktober'
    tahun_prediksi = predicted_date.strftime('%Y')   # Contoh output: '2024'

    # Kirimkan bulan dan tahun ke frontend
    return jsonify({
        "menu": menu,
        "jumlah_terjual": predicted_sold,
        "bahan_baku": bahan_baku_total,
        "bulan_prediksi": bulan_indonesia,
        "tahun_prediksi": tahun_prediksi,
        "message": "Dataset berhasil diperbarui"
    })

# Route upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    global latest_uploaded_filename

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dalam request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        latest_uploaded_filename = filename  # Simpan nama file
        df = pd.read_excel(path)
        
        return jsonify({'message': 'File berhasil diunggah dan data diperbarui'}), 200
    else:
        return jsonify({'error': 'Format file tidak diizinkan (hanya .xlsx)'}), 400

# Route download file
@app.route('/download', methods=['GET'])
def download_file():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], latest_uploaded_filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File tidak ditemukan.'}), 404
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
