import kagglehub
import shutil
from pathlib import Path
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración ---
DATASET_SLUG = "altavish/boston-housing-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "boston_housing.csv"
KAGGLE_FILENAME = "BostonHousing.csv"  # Nombre del archivo dentro del dataset de Kaggle


def fetch():
    """Descarga el dataset desde Kaggle y lo copia al directorio de datos."""
    logging.info("🚀 Iniciando la descarga del dataset desde Kaggle...")

    # Asegurarse de que el directorio de datos exista
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Descarga el dataset a una carpeta temporal
        download_path_str = kagglehub.dataset_download(DATASET_SLUG)
        source_file = Path(download_path_str) / KAGGLE_FILENAME

        # Copia y renombra el archivo a nuestro directorio 'data'
        shutil.copy(source_file, RAW_DATA_FILE)

        logging.info(f"✅ Dataset descargado y guardado en: {RAW_DATA_FILE}")

    except Exception as e:
        logging.error(f"❌ Error durante la descarga: {e}")
        # Salir con un código de error para que el pipeline de DVC se detenga
        exit(1)


if __name__ == "__main__":
    fetch()