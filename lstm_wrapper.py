import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
import joblib

class LSTMForecastWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_path = 'final_lstm_model.keras',lookback = 30, horizon = 7):
        self.model = tf.keras.models.load_model(model_path)
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = MinMaxScaler(feature_range=(0,1))
    
    def fit(self, X, y= None):
        return self
    
    def predict(self, X):
        scaled = self.scaler.fit_transform(np.array(X).reshape(-1,1))

        if len(scaled)<self.lookback:
            print(f"Need atleast {self.lookback} data points")

        seq = scaled[-self.lookback:,0]
        predictions = []

        current_seq = seq.copy()
        
        for _ in range(self.horizon):
            pred = self.model.predict(current_seq.reshape(1,self.lookback,1))
            predictions.append(pred[0][0])

        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        return predictions.flatten().tolist()
