# 1. Start from Python base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first (better caching)
COPY requirements.txt /app/

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the project
COPY . /app

# 6. Expose FastAPI port
EXPOSE 8080

# 7. Run FastAPI
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
