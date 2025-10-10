import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session

from . import database
from .schemas import HousingFeatures
from src.data_manager import load_pipeline
from src.config import FEATURES

database.init_db()

logger = logging.getLogger("boston.api")
app = FastAPI(title="Boston Housing Price Prediction API")
pipeline = load_pipeline()


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "API is running!"}


@app.post("/predict", tags=["Predictions"])
def predict(payload: HousingFeatures, db: Session = Depends(get_db)):
    """Realiza una predicción y la guarda en la base de datos."""
    logger.info(f"Received prediction request: {payload.model_dump()}")

    input_df = pd.DataFrame([payload.model_dump()])
    input_df = input_df[FEATURES]

    try:
        prediction_value = pipeline.predict(input_df)[0]
        prediction_inputs = {key.lower(): value for key, value in payload.model_dump().items()}

        db_prediction = database.Prediction(
            prediction_value=prediction_value,
            **prediction_inputs  # Usamos el nuevo diccionario
        )

        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        logger.info(f"Prediction result: {prediction_value}. Saved with id: {db_prediction.id}")
        return {"prediction": prediction_value}

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")