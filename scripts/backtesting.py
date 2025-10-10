import pandas as pd
import requests
import os
import logging
from typing import Dict, Any

from src.config import BACKTEST_FILE, REPORTS_DIR

# Configuración
API_URL = "http://localhost:8000/predict"
OUTPUT_REPORT_PATH = REPORTS_DIR / "backtest_report.csv"
LOG_FILE_PATH = REPORTS_DIR / "backtest.log"

# Configurar logging
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'),
        logging.StreamHandler()
    ]
)


def run_backtest():
    """
    Carga datos de backtesting, envía cada fila a la API de predicción,
    y guarda los resultados en un informe.
    """
    try:
        logging.info("Iniciando el proceso de backtesting...")

        # 1. Cargar los datos de backtesting
        if not BACKTEST_FILE.exists():
            raise FileNotFoundError(f"Archivo de datos no encontrado: {BACKTEST_FILE}")

        data = pd.read_csv(BACKTEST_FILE)
        logging.info(f"Cargados {len(data)} registros para backtesting.")

        # Asegurarse de que el directorio de salida existe
        os.makedirs(OUTPUT_REPORT_PATH.parent, exist_ok=True)

        # 2. Inicializar la lista para guardar los resultados
        results = []

        # 3. Iterar sobre cada fila de datos, enviar a la API y obtener la predicción
        for index, row in data.iterrows():
            payload = row.drop('MEDV').to_dict()

            try:
                print(payload)
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()

                prediction = response.json().get("prediction")
                actual_value = row['MEDV']

                results.append({
                    "id": index,
                    "actual_value": actual_value,
                    "predicted_value": prediction,
                    "payload_sent": payload
                })

                logging.info(f"Registro {index}: Predicción recibida: {prediction}, Valor real: {actual_value}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error al conectar con la API para el registro {index}: {e}")
                results.append({
                    "id": index,
                    "actual_value": row['MEDV'],
                    "predicted_value": None,
                    "error": str(e)
                })
            except KeyError:
                logging.error(
                    f"El campo 'prediction' no se encontró en la respuesta de la API para el registro {index}.")
                results.append({
                    "id": index,
                    "actual_value": row['MEDV'],
                    "predicted_value": None,
                    "error": "Campo de predicción faltante en la respuesta"
                })

        # 4. Crear un DataFrame de los resultados y guardarlo en un CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_REPORT_PATH, index=False)

        logging.info(f"Backtesting completado. Informe guardado en: {OUTPUT_REPORT_PATH}")

    except Exception as e:
        logging.critical(f"Ha ocurrido un error inesperado durante el backtesting: {e}", exc_info=True)


if __name__ == "__main__":
    run_backtest()