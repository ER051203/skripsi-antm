# config.py
"""
Konfigurasi untuk aplikasi prediksi saham
"""

# Konfigurasi data saham
TICKER = "ANTM.JK"
START_DATE = "2015-01-01"
END_DATE = None  # None berarti sampai hari ini

# Konfigurasi model
WINDOW_SIZE = 60
TEST_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Konfigurasi LSTM
LSTM_UNITS_1 = 50
LSTM_UNITS_2 = 25
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

# Konfigurasi Flask
SECRET_KEY = "skripsi-saham-antm-2024"
DEBUG = True
PORT = 5000