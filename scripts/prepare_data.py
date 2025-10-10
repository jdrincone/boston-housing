import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Rutas y Parámetros ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
RAW_DATA_FILE = PROJECT_ROOT / "data" / "HousingData.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train_data.csv"
BACKTEST_DATA_PATH = PROJECT_ROOT / "data" / "backtest_data.csv"


def prepare_data():
    """Divide los datos crudos en un set de entrenamiento y un set de backtesting."""
    logging.info("🚀 Iniciando la preparación de datos...")

    # Cargar parámetros
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)

    backtest_size = params["backtest"]["split_size"]
    random_state = params["train"]["random_state"]

    # Cargar datos crudos
    df = pd.read_csv(RAW_DATA_FILE)
    logging.info(f"Datos crudos cargados. Forma: {df.shape}")

    # --- CORRECCIÓN: Manejar valores nulos en la columna de estratificación ---
    # Rellenamos los NaN en la columna 'CHAS' con 0 (el valor más común) antes de usarla.
    df['CHAS'].fillna(0, inplace=True)

    # Dividir los datos
    train_df, backtest_df = train_test_split(
        df,
        test_size=backtest_size,
        random_state=random_state,
        stratify=df['CHAS']
    )

    logging.info(
        f"Datos divididos. Tamaño de entrenamiento: {train_df.shape[0]}, Tamaño de backtesting: {backtest_df.shape[0]}")

    # Guardar los nuevos archivos
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    backtest_df.to_csv(BACKTEST_DATA_PATH, index=False)

    logging.info(f"✅ Archivo de entrenamiento guardado en: {TRAIN_DATA_PATH}")
    logging.info(f"✅ Archivo de backtesting guardado en: {BACKTEST_DATA_PATH}")


if __name__ == "__main__":
    prepare_data()