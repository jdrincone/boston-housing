import logging
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HousingFeatures
from src.data_manager import load_pipeline
from src.config import FEATURES

LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("boston.api")

app = FastAPI(title="Boston Housing Price Prediction API")
pipeline = load_pipeline()


# --- RUTA AÑADIDA PARA ARREGLAR EL ERROR 404 ---
@app.get("/", tags=["Health Check"])
def health_check():
    """Verifica que la API esté funcionando."""
    return {"status": "ok", "message": "API is running!"}


@app.post("/predict", tags=["Predictions"])
def predict(payload: HousingFeatures):
    """Realiza una predicción sobre un único registro."""
    # --- SINTAXIS ACTUALIZADA A PYDANTIC V2 ---
    logger.info(f"Received prediction request: {payload.model_dump()}")
    input_df = pd.DataFrame([payload.model_dump()])

    input_df = input_df[FEATURES]

    try:
        prediction = pipeline.predict(input_df)[0]
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
