# tests/test_training.py

import joblib
import pandas as pd
from src.config import MODEL_PATH


def test_pipeline_generates_valid_model():
    """Verifica que el pipeline se puede cargar y puede hacer una predicción."""
    assert MODEL_PATH.exists(), "El archivo del modelo no fue creado."

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        assert False, f"El modelo no se pudo cargar: {e}"

    # Crear un dato de prueba (debe tener todas las columnas esperadas)
    sample_data = {
        "CRIM": [0.027],
        "ZN": [0.0],
        "INDUS": [7.07],
        "CHAS": [0],
        "NOX": [0.469],
        "RM": [6.421],
        "AGE": [78.9],
        "DIS": [4.967],
        "RAD": [2],
        "TAX": [242],
        "PTRATIO": [17.8],
        "B": [396.9],
        "LSTAT": [9.14],
    }
    sample_df = pd.DataFrame(sample_data)

    try:
        prediction = pipeline.predict(sample_df)
        assert isinstance(prediction[0], float)
    except Exception as e:
        assert False, f"La predicción falló con un dato de muestra: {e}"
