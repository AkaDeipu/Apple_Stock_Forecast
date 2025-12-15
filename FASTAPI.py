# FASTAPI.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from lstm_wrapper import LSTMForecastWrapper

app = FastAPI()
pipeline = joblib.load("final_lstm_pipline.pkl")

class ForecastRequest(BaseModel):
    data: list
    horizon: int = 7

@app.post("/predict")
def predict(request: ForecastRequest):
    preds = pipeline.named_steps['lstm_forecast'].predict(
        request.data, horizon=request.horizon
    )
    return {"predictions": preds}

