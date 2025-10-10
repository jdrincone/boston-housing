import pandas as pd
import requests
import os
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import BACKTEST_FILE, REPORTS_DIR

API_URL = "http://localhost:8000/predict"
OUTPUT_REPORT_PATH = REPORTS_DIR / "backtest_report.csv"
METRICS_REPORT_PATH = REPORTS_DIR / "metrics_summary.csv"
LOG_FILE_PATH = REPORTS_DIR / "backtest.log"


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
        data = pd.read_csv(BACKTEST_FILE,)
        logging.info(f"Cargados {len(data)} registros para backtesting.")

        os.makedirs(OUTPUT_REPORT_PATH.parent, exist_ok=True)
        results = []
        for index, row in data.iterrows():
            payload = row.drop('MEDV').to_dict()

            try:
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

        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_REPORT_PATH, index=False)

        if not results_df.empty and 'predicted_value' in results_df and 'actual_value' in results_df:
            valid_results = results_df.dropna(subset=['predicted_value'])

            if not valid_results.empty:
                mae = mean_absolute_error(valid_results['actual_value'], valid_results['predicted_value'])
                mse = mean_squared_error(valid_results['actual_value'], valid_results['predicted_value'])

                # Crear un DataFrame para las métricas
                metrics_df = pd.DataFrame([{
                    "mae": mae,
                    "mse": mse,
                    "num_predictions": len(valid_results)
                }])

                if os.path.exists(METRICS_REPORT_PATH):
                    existing_metrics_df = pd.read_csv(METRICS_REPORT_PATH)
                    metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)

                metrics_df.to_csv(METRICS_REPORT_PATH, index=False)

                logging.info("--- Resumen de Métricas del Backtesting ---")
                logging.info(f"MAE (Mean Absolute Error): {mae:.2f}")
                logging.info(f"MSE (Mean Squared Error): {mse:.2f}")
                logging.info("----------------------------------------")
            else:
                logging.warning("No hay predicciones válidas para calcular métricas.")
        else:
            logging.warning("El DataFrame de resultados está vacío o faltan columnas.")

        logging.info(f"Backtesting completado. Informe de backtesting guardado en: {OUTPUT_REPORT_PATH}")

    except Exception as e:
        logging.critical(f"Ha ocurrido un error inesperado durante el backtesting: {e}", exc_info=True)


if __name__ == "__main__":
    run_backtest()