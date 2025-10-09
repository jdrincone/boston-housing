import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    AUTOML_SUMMARY_REPORT_PATH,
    FEATURE_IMPORTANCE_PLOT_PATH,
    MAIN_LOG_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_SUMMARY_PATH,
    TARGET,
    TEST_SIZE,
    TRAIN_FILE,
)
from src.data_manager import load_dataset, save_metrics, save_pipeline
from src.pipeline import create_pipeline


REPORTS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(MAIN_LOG_PATH)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def run_training() -> None:
    """Orquesta el entrenamiento, evaluación y guardado de artefactos y métricas."""
    logger.info("Starting the training process...")

    logger.info("Loading and splitting data...")
    data = load_dataset(file_name=TRAIN_FILE.name)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    pipeline = create_pipeline()
    logger.info("Training the pipeline with AutoML (logs will be shown)...")
    pipeline.fit(X_train, y_train)
    logger.info("AutoML training complete.")

    logger.info("---Detailed Model Evaluation ---")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    metrics = {
        "train": {
            "r2_score": r2_score(y_train, y_pred_train),
            "mse": mean_squared_error(y_train, y_pred_train),
            "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "mae": mean_absolute_error(y_train, y_pred_train),
        },
        "test": {
            "r2_score": r2_score(y_test, y_pred_test),
            "mse": mean_squared_error(y_test, y_pred_test),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae": mean_absolute_error(y_test, y_pred_test),
        },
        "best_model_name": pipeline.named_steps[
            "regressor"
        ].model.estimator.__class__.__name__,
    }

    logger.info(f"  Best Model: {metrics['best_model_name']}")
    logger.info(f"  Train R^2 Score: {metrics['train']['r2_score']:.4f}")
    logger.info(f"  Test R^2 Score: {metrics['test']['r2_score']:.4f}")
    save_metrics(metrics=metrics)

    logger.info("Building and saving summary report...")
    automl = pipeline.named_steps["regressor"]
    preprocessor = Pipeline(pipeline.steps[:-1])
    X_train_processed = pd.DataFrame(
        preprocessor.transform(X_train), columns=X_train.columns
    )
    final_model = automl.model.estimator

    summary = [
        "=" * 50,
        "      AutoML Final Summary Report",
        "=" * 50,
        f"\nBest Model Found: {final_model.__class__.__name__}",
        f"Best R2 Score (during CV): {-automl.best_loss:.4f}",
        "\n--- Best Model Configuration ---",
        *[f"  - {key}: {value}" for key, value in automl.best_config.items()],
    ]

    importances = []
    if hasattr(final_model, "feature_importances_"):
        importances = sorted(
            zip(X_train.columns, final_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        summary.append("\n--- Feature Importances (from final model) ---")
        summary.extend(
            [f"  - {feature}: {importance:.4f}" for feature, importance in importances]
        )

    with open(AUTOML_SUMMARY_REPORT_PATH, "w") as f:
        f.write("\n".join(summary))
    logger.info(f"AutoML summary report saved to: {AUTOML_SUMMARY_REPORT_PATH}")

    if importances:
        logger.info("Generating Feature Importance plot (model-based)...")
        features = [item[0] for item in importances]
        values = [item[1] for item in importances]

        plt.figure(figsize=(10, 6))
        plt.barh(features, values)
        plt.xlabel("Feature Importance Value")
        plt.ylabel("Feature")
        plt.title("Model-based Feature Importance")
        plt.gca().invert_yaxis()  # Poner la más importante arriba
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH)
        plt.close()
        logger.info(f"Feature Importance plot saved to: {FEATURE_IMPORTANCE_PLOT_PATH}")
    else:
        logger.warning(
            "Final model does not have 'feature_importances_'. Skipping model-based feature importance plot."
        )

    logger.info("Generating SHAP feature importance plot...")
    if not isinstance(X_train_processed, pd.DataFrame):
        X_train_processed = pd.DataFrame(X_train_processed, columns=X_train.columns)

    explainer = shap.Explainer(final_model, X_train_processed)
    shap_values = explainer(X_train_processed)

    plt.figure()
    shap.summary_plot(shap_values, X_train_processed, show=False)
    plt.title("SHAP Feature Importance (Dot/Scatter)")
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH)
    plt.close()
    logger.info(f"SHAP plot saved to: {SHAP_SUMMARY_PATH}")

    save_pipeline(pipeline_to_persist=pipeline)
    logger.info("Training pipeline finished successfully!")


if __name__ == "__main__":
    run_training()
