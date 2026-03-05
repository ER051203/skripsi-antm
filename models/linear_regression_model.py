# models/linear_regression_model.py
"""
Model Linear Regression untuk prediksi saham
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import logging
 
logger = logging.getLogger(__name__)

class LinearRegressionModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def build_model(self):
        """Membangun model Linear Regression"""
        self.model = LinearRegression(fit_intercept=True)
        logger.info("Model Linear Regression siap")
        return self.model
    
    def train(self, X_train, y_train):
        """
        Melatih model Linear Regression
        
        Parameters:
        -----------
        X_train : array
            Fitur training (n_samples, n_features)
        y_train : array
            Target training (n_samples,)
        """
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("Training Linear Regression selesai")
        
        # Log koefisien (hanya beberapa)
        coef_mean = np.mean(self.model.coef_)
        logger.info(f"Rata-rata koefisien: {coef_mean:.6f}")
        logger.info(f"Intercept: {self.model.intercept_:.6f}")
        
        return self.model
    
    def predict(self, X_test):
        """
        Melakukan prediksi dengan model yang sudah dilatih
        
        Parameters:
        -----------
        X_test : array
            Fitur testing
        
        Returns:
        --------
        array : Hasil prediksi dalam skala normalized
        """
        if not self.is_trained:
            raise ValueError("Model belum dilatih!")
        
        return self.model.predict(X_test)
    
    def predict_future(self, last_sequence, days=1):
        """
        Memprediksi harga untuk hari berikutnya
        
        Parameters:
        -----------
        last_sequence : array
            Window terakhir (window_size,)
        days : int
            Jumlah hari ke depan (default: 1)
        
        Returns:
        --------
        array : Prediksi untuk hari berikutnya
        """
        if not self.is_trained:
            raise ValueError("Model belum dilatih!")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Prediksi hari berikutnya
            next_pred = self.model.predict(current_sequence.reshape(1, -1))[0]
            predictions.append(next_pred)
            
            # Update sequence untuk prediksi selanjutnya
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        return np.array(predictions)