from sklearn.pipeline import Pipeline
from lstm_wrapper import LSTMForecastWrapper
import joblib

pipe = Pipeline([
    ('lstm_forecast', LSTMForecastWrapper())
])

joblib.dump(pipe, "final_lstm_pipline.pkl")