from flask import Flask, request, jsonify
# from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
# CORS(app)

# Load model dan scaler
model = joblib.load('xgb_model_real.pkl')
scaler = joblib.load('scaler_real.pkl')

# Simpan data dari web berdasarkan nama
user_inputs = {}

@app.route('/')
def home():
    return "🧪 API Prediksi Gula Darah Aktif"

@app.route('/input_web', methods=['POST'])
def input_web():
    data = request.json
    nama = data.get('nama')
    if not nama:
        return jsonify({'error': 'Nama tidak boleh kosong'})

    user_inputs[nama] = {
        'jenis_kulit': data.get('jenis_kulit'),
        'suhu': data.get('suhu')
    }
    return jsonify({'message': f'Data dari web untuk {nama} disimpan. Silakan lanjutkan dengan pengukuran dari alat.'})

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    pd1 = data.get('pd1')

    if not user_inputs:
        return jsonify({'error': 'Belum ada data dari web yang dikirim.'})

    # Ambil nama terakhir yang mengisi form
    nama_terakhir = list(user_inputs.keys())[-1]
    input_web = user_inputs[nama_terakhir]

    try:
        # Simpan nilai pd1 agar bisa diambil nanti endpoint /hasil_terakhir
        user_inputs[nama_terakhir]['pd1'] = pd1

        df = pd.DataFrame([[input_web['jenis_kulit'], pd1, input_web['suhu']]],
                          columns=['Jenis Kulit', 'PD1', 'Suhu Ruangan'])
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)

        return jsonify({
            'nama': nama_terakhir,
            'gula_darah': float(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/hasil_terakhir', methods=['GET'])
def hasil_terakhir():
    if not user_inputs:
        return jsonify({'error': 'Belum ada data dari user.'})
    # Ambil nama terakhir (asumsi ini adalah data yang relevan)
    nama_terakhir = list(user_inputs.keys())[-1]
    input_web = user_inputs[nama_terakhir]
    # Pastikan 'pd1' sudah tersimpan
    if 'pd1' not in input_web:
        return jsonify({'error': 'Nilai PD1 belum diupdate dari alat.'})
        
    try:
        df = pd.DataFrame([[input_web['jenis_kulit'], input_web['pd1'], input_web['suhu']]],
                          columns=['Jenis Kulit', 'PD1', 'Suhu Ruangan'])
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)

        return jsonify({
            'nama': nama_terakhir,
            'gula_darah': float(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
