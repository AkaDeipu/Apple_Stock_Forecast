# FASTAPI.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from lstm_wrapper import LSTMForecastWrapper

app = FastAPI()
templates = Jinja2Templates(directory=".")
pipeline = joblib.load("final_lstm_pipline.pkl")

class ForecastRequest(BaseModel):
    data: list
    horizon: int = 7

@app.get("/", response_class=HTMLResponse) 
def home(request: Request): 
    return templates.TemplateResponse("ui.html", {"request": request})

@app.post("/predict")
def predict(request: ForecastRequest):
    try:
        preds = pipeline.named_steps['lstm_forecast'].predict(
        request.data, horizon=request.horizon)
        return {"predictions": preds}
    except Exception as e:
        return {"error": str(e)}

