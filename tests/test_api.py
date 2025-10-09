# tests/test_api.py

from fastapi.testclient import TestClient
from app.main import app  # Importa tu app de FastAPI

# Creamos un cliente de prueba
client = TestClient(app)


def test_health_check():
    """Prueba que el endpoint de health check ('/') funcione."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running!"}


def test_prediction_success():
    """Prueba que el endpoint de predicción funcione con datos válidos."""
    payload = {
        "CRIM": 0.02731,
        "INDUS": 7.07,
        "NOX": 0.469,
        "RM": 6.421,
        "AGE": 78.9,
        "DIS": 4.9671,
        "TAX": 242,
        "PTRATIO": 17.8,
        "B": 396.9,
        "LSTAT": 9.14,
        # Dejamos ZN, CHAS, RAD como opcionales
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_prediction_invalid_data():
    """Prueba que la API devuelva un error 422 con datos incompletos (faltan obligatorios)."""
    # Enviamos un payload al que le falta 'LSTAT' (que es obligatorio)
    payload = {
        "CRIM": 0.02731,
        "INDUS": 7.07,
        "NOX": 0.469,
        "RM": 6.421,
        "AGE": 78.9,
        "DIS": 4.9671,
        "TAX": 242,
        "PTRATIO": 17.8,
        "B": 396.9,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # Unprocessable Entity
