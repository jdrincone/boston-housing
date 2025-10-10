import pathlib
import yaml

# --- Directories ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# --- Files ---
TRAIN_FILE = DATA_DIR / "HousingData.csv"
MODEL_PATH = MODEL_DIR / "best_pipeline.pkl"
SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary.png"
METRICS_PATH = REPORTS_DIR / "metrics.json"
AUTOML_SUMMARY_REPORT_PATH = REPORTS_DIR / "automl_summary.txt"
MAIN_LOG_PATH = REPORTS_DIR / "main.log"
FEATURE_IMPORTANCE_PLOT_PATH = REPORTS_DIR / "feature_importance.png"


# --- Training Parameters ---
with open(BASE_DIR / "params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

TARGET = params["target"]
TEST_SIZE = params["test_size"]
RANDOM_STATE = params["random_state"]
AUTOML_TIME_BUDGET = params["automl_budget_secs"]
FEATURES = params["features"]
