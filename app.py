# app.py
import os
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import plotly
import plotly.graph_objs as go
from flask import Flask, render_template, request

# Import modul buatan sendiri
from utils.preprocessing import prepare_data, calculate_metrics
from models.linear_regression_model import LinearRegressionModel
from models.lstm_model import LSTMModel

# ============================================
# KONFIGURASI
# ============================================
TICKER = "ANTM.JK"
START_DATE = "2015-01-01"
WINDOW_SIZE = 60
TEST_SIZE = 0.2
EPOCHS = 50 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PORT = 5000
DEBUG = True
SECRET_KEY = "skripsi-saham-antm-2024"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

os.makedirs('templates', exist_ok=True)
os.makedirs('runs', exist_ok=True)

def download_stock_data(ticker, start_date, end_date=None):
    logger.info(f"Mengambil data {ticker} dari Yahoo Finance...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df.reset_index(inplace=True)
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        data = pd.DataFrame({
            'Date': df['Date'],
            'Close': df['Close'].values.flatten()
        })
        return data
    except Exception as e:
        logger.error(f"Gagal mengambil data: {e}")
        raise e

@app.route('/')
def index():
    stock_info = {'name': 'PT Aneka Tambang Tbk (ANTM)', 'ticker': TICKER}
    return render_template('index.html', stock_info=stock_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form.get('ticker', TICKER)
        prediction_date = request.form.get('prediction_date') or datetime.now().strftime('%Y-%m-%d')
        
        # 1. Ambil Data
        data = download_stock_data(ticker, START_DATE, datetime.now().strftime('%Y-%m-%d'))
        last_data_date = data['Date'].iloc[-1]
        
        # Hitung berapa hari ke depan untuk diprediksi
        target_date = datetime.strptime(prediction_date, '%Y-%m-%d')
        days_to_predict = (target_date - last_data_date).days
        if days_to_predict < 0:
            days_to_predict = 0

        # 2. Preprocessing
        processed = prepare_data(data, window_size=WINDOW_SIZE, test_size=TEST_SIZE)
        scaler = processed['scaler']
        y_test_original = processed['y_test_original']
        
        # Perbaiki alignment tanggal untuk plotting
        aligned_test_dates = processed['test_dates'][WINDOW_SIZE:]
        
        # 3. Model Linear Regression
        lr_model = LinearRegressionModel()
        lr_model.train(processed['X_train'], processed['y_train'])
        lr_pred_scaled = lr_model.predict(processed['X_test'])
        lr_pred_asli = scaler.inverse_transform(lr_pred_scaled.reshape(-1, 1)).flatten()
        lr_metrics = calculate_metrics(y_test_original, lr_pred_asli)
        
        # 4. Model LSTM
        lstm_model = LSTMModel(window_size=WINDOW_SIZE, learning_rate=LEARNING_RATE)
        model_path = 'runs/lstm_model.keras'
        if os.path.exists(model_path):
            lstm_model.load_model(model_path)
        else:
            lstm_model.train(processed['X_train_lstm'], processed['y_train'], epochs=EPOCHS, batch_size=BATCH_SIZE)
            lstm_model.save_model(model_path)
            
        lstm_pred_scaled = lstm_model.predict(processed['X_test_lstm'])
        lstm_pred_asli = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
        lstm_metrics = calculate_metrics(y_test_original, lstm_pred_asli)

        # 5. PREDIKSI MASA DEPAN (FORECASTING)
        future_predictions = None
        if days_to_predict > 0:
            last_seq_lr = processed['X_test'][-1]
            last_seq_lstm = processed['X_test_lstm'][-1]
            
            future_lr_scaled = lr_model.predict_future(last_seq_lr, days=days_to_predict)
            future_lstm_scaled = lstm_model.predict_future(last_seq_lstm, days=days_to_predict)
            
            future_lr = scaler.inverse_transform(future_lr_scaled.reshape(-1, 1)).flatten().tolist()
            future_lstm = scaler.inverse_transform(future_lstm_scaled.reshape(-1, 1)).flatten().tolist()
            future_dates = [(last_data_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_to_predict + 1)]
            
            future_predictions = {
                'dates': future_dates,
                'lr': future_lr,
                'lstm': future_lstm
            }

        # 6. Buat Grafik Plotly
        best_model = "LSTM" if lstm_metrics['rmse'] < lr_metrics['rmse'] else "Linear Regression"
        best_rmse = min(lstm_metrics['rmse'], lr_metrics['rmse'])

        plot_range = -50 
        
        # WAJIB: Konversi ke .tolist() agar skala grafik tidak menjadi 0-50
        x_plot = pd.to_datetime(aligned_test_dates[plot_range:]).strftime('%Y-%m-%d').tolist()
        y_aktual_plot = y_test_original[plot_range:].tolist()
        y_lr_plot = lr_pred_asli[plot_range:].tolist()
        y_lstm_plot = lstm_pred_asli[plot_range:].tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_plot, y=y_aktual_plot, mode='lines', name='Harga Aktual', line=dict(color='white', width=3)))
        fig.add_trace(go.Scatter(x=x_plot, y=y_lr_plot, mode='lines', name='Linear Regression', line=dict(color='#00FF00', width=2)))
        fig.add_trace(go.Scatter(x=x_plot, y=y_lstm_plot, mode='lines', name='LSTM', line=dict(color='#FFD700', width=2, dash='dash')))
        
        # Tambahkan garis titik-titik untuk masa depan di grafik jika ada prediksi
        if future_predictions:
            fig.add_trace(go.Scatter(
                x=future_predictions['dates'], y=future_predictions['lr'], 
                mode='lines', name='Forecast LR', line=dict(color='#00FF00', width=2, dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=future_predictions['dates'], y=future_predictions['lstm'], 
                mode='lines', name='Forecast LSTM', line=dict(color='#FFD700', width=2, dash='dot')
            ))

        fig.update_layout(
            title=f'Perbandingan Prediksi Saham {ticker}',
            xaxis_title='Tanggal', yaxis_title='Harga (Rp)',
            template='plotly_dark', plot_bgcolor='#800000', paper_bgcolor='#800000', font=dict(color='white')
        )
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        result_data = {
            'ticker': ticker, 'start_date': START_DATE, 'end_date': prediction_date,
            'lr_metrics': lr_metrics, 'lstm_metrics': lstm_metrics,
            'best_model': best_model, 'best_rmse': best_rmse, 'graphJSON': graphJSON,
            'future_predictions': future_predictions
        }
        return render_template('result.html', **result_data)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=DEBUG, port=PORT)