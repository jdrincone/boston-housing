import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.config import TRAIN_FILE, MODEL_PATH, MODEL_DIR, METRICS_PATH

logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Loads a CSV file from the data directory."""
    path = Path(TRAIN_FILE.parent, file_name)
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)  # Si falla, dejamos que la excepción se propague
    logger.info("Dataset loaded: shape=%s", df.shape)
    return df

def save_pipeline(*, pipeline_to_persist: object) -> None:
    """Saves the pipeline to the models directory."""
    logger.info("Saving pipeline to %s", MODEL_PATH)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_to_persist, MODEL_PATH)
    logger.info("Pipeline saved to: %s", MODEL_PATH)

def load_pipeline() -> object:
    """Loads the trained pipeline."""
    if not MODEL_PATH.exists():
        logger.error("Model not found at %s", MODEL_PATH)
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logger.info("Loading pipeline from %s", MODEL_PATH)
    pipeline = joblib.load(MODEL_PATH)
    logger.info("Pipeline loaded from: %s", MODEL_PATH)
    return pipeline

def save_metrics(*, metrics: Dict) -> None:
    """Saves model metrics to the main metrics.json file."""
    logger.info("Saving metrics to %s (keys=%s)", METRICS_PATH, list(metrics.keys()))
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved to: %s", METRICS_PATH)
