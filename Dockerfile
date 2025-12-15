#Start from Python base image
FROM python:3.10-slim

#Set working directory
WORKDIR /app

#Copy requirements first (better caching)
COPY requirements.txt /app/

##Copy the model and pipeline
COPY final_lstm_pipline.pkl /app/
COPY final_lstm_model.keras /app/

#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Copy the rest of the project
COPY . /app

#Expose FastAPI port
EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
