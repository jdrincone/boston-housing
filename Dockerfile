# Stage 1: Builder - Installs dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install uv

COPY requirements.txt .
RUN uv pip install --no-cache-dir --system -r requirements.txt

FROM python:3.11-slim

# --- INICIO: Instalar dependencias del sistema ---
# LightGBM (usado por FLAML) necesita la librería libgomp1 para funcionar.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application source code
COPY ./app /app/app
COPY ./src /app/src

COPY ./models /app/models
COPY ./params.yaml /app/params.yaml

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]