import logging
import pandas as pd
from fastapi import FastAPI, HTTPException

# ¡Importamos HousingFeatures directamente!
from app.schemas import HousingFeatures
from src.data_manager import load_pipeline
from src.config import FEATURES

# (La configuración del logger no cambia)
LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("boston.api")

app = FastAPI(
    title="Boston Housing Price Prediction API — Juan David Rincón",
    version="1.0.0",
    description="Predicción de precios de vivienda con pipeline serializado (scikit-learn)"
)

logger.info("Starting up API… loading pipeline.")
pipeline = load_pipeline()


# (El resto de la configuración no cambia)

@app.post("/predict", tags=["Predictions"])
def predict(payload: HousingFeatures):  # <--- CAMBIO 1: El tipo de entrada es ahora HousingFeatures
    """
    Realiza una predicción sobre un único registro.
    - Espera un objeto JSON con las features.
    - Devuelve un único valor de predicción.
    """
    logger.info(f"Received prediction request: {payload.dict()}")
    input_df = pd.DataFrame([payload.dict()])
    input_df = input_df[FEATURES]


    try:
        # --- CAMBIO 3: Devolvemos una sola predicción, no una lista ---
        prediction = pipeline.predict(input_df)[0]  # Obtenemos el primer (y único) elemento
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")