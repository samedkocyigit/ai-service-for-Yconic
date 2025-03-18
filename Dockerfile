FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY clip_api.py .

EXPOSE 8000
CMD ["uvicorn", "clip_api:app", "--host", "0.0.0.0", "--port", "8000"]
