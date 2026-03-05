# models/lstm_model.py
"""
Model LSTM untuk prediksi saham
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, window_size=60, lstm_units_1=50, lstm_units_2=25, 
                 dense_units=16, dropout_rate=0.2, learning_rate=0.001):
        
        self.window_size = window_size
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.is_trained = False
    
    def build_model(self):
        """Membangun arsitektur LSTM"""
        model = Sequential([
            # LSTM Layer 1 dengan return sequences
            LSTM(self.lstm_units_1, return_sequences=True, 
                 input_shape=(self.window_size, 1)),
            Dropout(self.dropout_rate),
            
            # LSTM Layer 2
            LSTM(self.lstm_units_2, return_sequences=False),
            
            # Dense Layer
            Dense(self.dense_units, activation='relu'),
            
            # Output Layer
            Dense(1, activation='linear')
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info("Arsitektur LSTM berhasil dibangun")
        
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """
        Melatih model LSTM
        
        Parameters:
        -----------
        X_train : array
            Fitur training dengan shape (samples, window_size, 1)
        y_train : array
            Target training
        epochs : int
            Jumlah epoch
        batch_size : int
            Ukuran batch
        validation_split : float
            Proporsi data validasi
        """
        if self.model is None:
            self.build_model()
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained = True
        
        # Log hasil training
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        logger.info(f"Training selesai - Final loss: {final_loss:.6f}, Val loss: {final_val_loss:.6f}")
        
        return self.history
    
    def predict(self, X_test):
        """
        Melakukan prediksi dengan model yang sudah dilatih
        
        Parameters:
        -----------
        X_test : array
            Fitur testing dengan shape (samples, window_size, 1)
        
        Returns:
        --------
        array : Hasil prediksi dalam skala normalized
        """
        if not self.is_trained:
            raise ValueError("Model belum dilatih!")
        
        return self.model.predict(X_test, verbose=0).flatten()
    
    def predict_future(self, last_sequence, days=1):
        """
        Memprediksi harga untuk hari-hari berikutnya
        
        Parameters:
        -----------
        last_sequence : array
            Window terakhir dengan shape (window_size, 1)
        days : int
            Jumlah hari ke depan
        
        Returns:
        --------
        array : Prediksi untuk hari-hari berikutnya
        """
        if not self.is_trained:
            raise ValueError("Model belum dilatih!")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape untuk input LSTM: (1, window_size, 1)
            current_input = current_sequence.reshape(1, self.window_size, 1)
            
            # Prediksi hari berikutnya
            next_pred = self.model.predict(current_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence untuk prediksi selanjutnya
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        return np.array(predictions)
    
    def save_model(self, filepath='runs/lstm_model.keras'):
        """Menyimpan model ke file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model disimpan ke {filepath}")
    
    def load_model(self, filepath='runs/lstm_model.keras'):
        """Memuat model dari file"""
        self.model = load_model(filepath)
        self.is_trained = True
        logger.info(f"Model dimuat dari {filepath}")