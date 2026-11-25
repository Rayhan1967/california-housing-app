# California Housing Price Prediction App 🏠

Aplikasi web interaktif berbasis Machine Learning untuk memprediksi harga rumah di California menggunakan Streamlit dan Scikit-Learn.

## Fitur
- **Eksplorasi Data**: Analisis deskriptif, visualisasi korelasi, dan peta persebaran geografis.
- **Modeling & Evaluasi**: Komparasi model regresi (Random Forest, HistGradientBoosting, Linear Regression, Ridge).
- **Hyperparameter Tuning**: Pencarian parameter terbaik dengan RandomizedSearchCV.
- **Prediksi Interaktif**: Input parameter rumah secara manual atau melalui batch CSV.

## Cara Menjalankan Secara Lokal
```bash
# Buat dan aktifkan virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run aplikasi.py
```
