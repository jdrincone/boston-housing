from flaml import AutoML
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, AUTOML_TIME_BUDGET
from src.processing import transformers as t


def create_pipeline() -> Pipeline:
    """Ensambla y devuelve el pipeline completo de Scikit-learn."""

    automl_settings = {
        "time_budget": AUTOML_TIME_BUDGET,
        "metric": 'r2',
        "task": 'regression',
        "log_file_name": 'automl_flaml.log',
        "seed": RANDOM_STATE,
        "n_splits": 5,
    }

    price_prediction_pipeline = Pipeline([
        ('fix_column_names', t.ColumnNameFixer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', AutoML(**automl_settings))
    ])

    return price_prediction_pipeline