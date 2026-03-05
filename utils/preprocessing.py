# utils/preprocessing.py
"""
Fungsi untuk preprocessing data dan evaluasi model
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

def prepare_data(data, window_size=60, test_size=0.2):
    """
    Mempersiapkan data untuk training dan testing
    
    Parameters:
    -----------
    data : DataFrame
        Data saham dengan kolom Date dan Close
    window_size : int
        Ukuran window untuk sliding window
    test_size : float
        Proporsi data testing (0-1)
    
    Returns:
    --------
    Dictionary berisi data yang sudah diproses
    """
    try:
        # Ambil harga penutupan
        prices = data['Close'].values.reshape(-1, 1)
        dates = data['Date'].values
        
        # Split data secara kronologis
        split_idx = int(len(prices) * (1 - test_size))
        
        train_data = prices[:split_idx]
        test_data = prices[split_idx - window_size:]  # Ambil extra untuk window
        
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        # Scaling dengan MinMaxScaler (fit hanya pada training)
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # Fungsi untuk membuat dataset dengan sliding window
        def create_sequences(data, window_size):
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        
        # Buat sequences untuk training dan testing
        X_train, y_train = create_sequences(train_scaled, window_size)
        X_test, y_test = create_sequences(test_scaled, window_size)
        
        # Reshape untuk LSTM: (samples, timesteps, features)
        X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Harga asli untuk testing (tanpa window)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        logger.info(f"Data shape - X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_train_lstm': X_train_lstm,
            'y_train': y_train,
            'X_test': X_test,
            'X_test_lstm': X_test_lstm,
            'y_test': y_test,
            'y_test_original': y_test_original,
            'train_dates': train_dates[window_size:],
            'test_dates': test_dates,
            'scaler': scaler,
            'window_size': window_size,
            'split_idx': split_idx
        }
    
    except Exception as e:
        logger.error(f"Error dalam preprocessing: {str(e)}")
        raise

def calculate_metrics(y_true, y_pred):
    """
    Menghitung metrik evaluasi MAE, RMSE, R²
    
    Parameters:
    -----------
    y_true : array
        Nilai aktual
    y_pred : array
        Nilai prediksi
    
    Returns:
    --------
    Dictionary berisi MAE, RMSE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 4)
    }