# BA/src/models/train_model.py
import datetime
import logging
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Import config
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config

# Access config parameters safely with getattr, providing defaults
REPORTS_DIR = getattr(config, "REPORTS_DIR", "reports")
FIGURES_DIR = getattr(config, "FIGURES_DIR", "figures")
LOG_FILE = getattr(config, "LOG_FILE", "logs/ml_pipeline.log")
LOG_LEVEL = getattr(config, "LOG_LEVEL", logging.INFO)
MODELS_DIR = getattr(config, "MODELS_DIR", "models")
RANDOM_STATE = getattr(config, "RANDOM_STATE", 42)
TARGET_VARIABLE = getattr(config, "TARGET_VARIABLE", "comment_score")
TEST_SIZE = getattr(config, "TEST_SIZE", 0.2)

# Setup logging (must be done before importing model_utils if model_utils also sets up logging)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)  # Get a logger instance for this module


from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import custom modules
try:
    from BA.src.data.database_utils import load_data_from_sqlite
    from BA.src.models.model_utils import (
        interpret_model_shap,
        preprocess_features,
        save_artifacts,
        train_and_evaluate_linear_regression,
        train_and_tune_xgboost,
    )
except ImportError as e:
    logger.error(
        f"Failed to import custom modules. Please check your PYTHONPATH and module paths: {e}"
    )
    sys.exit(1)  # Exit if core modules cannot be imported


def save_ml_results_summary(
    r2_linear_test: float,
    r2_linear_cv: float,
    r2_xgboost_tuned_test: float,
    r2_xgboost_tuned_cv: float,
    best_xgboost_params: Dict[str, Any],
) -> None:
    """
    Saves a summary of the ML model results to a text file in the REPORTS_DIR.

    Args:
        r2_linear_test (float): R-squared value for Linear Regression on the test set.
        r2_linear_cv (float): Mean cross-validation R-squared for Linear Regression.
        r2_xgboost_tuned_test (float): R-squared value for Tuned XGBoost on the test set.
        r2_xgboost_tuned_cv (float): Mean cross-validation R-squared for Tuned XGBoost.
        best_xgboost_params (Dict[str, Any]): Dictionary of best hyperparameters for XGBoost.
    """
    summary_filename = (
        f"ml_results_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    summary_filepath = os.path.join(REPORTS_DIR, summary_filename)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    try:
        with open(summary_filepath, "w") as f:
            f.write("--- Machine Learning Model Results Summary ---\n\n")
            f.write(
                f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("1. Linear Regression Model:\n")
            f.write(f"   R² (Test Set): {r2_linear_test:.4f}\n")
            f.write(f"   Mean CV R²: {r2_linear_cv:.4f}\n\n")

            f.write("2. XGBoost Regressor Model (Tuned):\n")
            f.write(f"   R² (Test Set): {r2_xgboost_tuned_test:.4f}\n")
            f.write(f"   Mean CV R²: {r2_xgboost_tuned_cv:.4f}\n")
            f.write("   Best Hyperparameters:\n")
            if best_xgboost_params:
                for param, value in best_xgboost_params.items():
                    f.write(f"     {param}: {value}\n")
            else:
                f.write("     No best parameters found (tuning might have failed).\n")
            f.write("\n")

            f.write("--- End of Summary ---\n")
        logger.info(f"ML results summary saved to: {summary_filepath}")
    except Exception as e:
        logger.error(f"Error saving ML results summary to '{summary_filepath}': {e}")


def run_model_pipeline() -> None:
    """
    Führt den gesamten Modellierungs-Workflow aus:
    1. Lädt Daten aus der SQLite-Datenbank.
    2. Bereitet Features vor (erstellt Preprocessing-Pipeline).
    3. Führt Train-Test-Split durch.
    4. Trainiert und evaluiert Lineare Regression und XGBoost-Modelle.
    5. Interpretiert das XGBoost-Modell mit SHAP.
    6. Speichert trainierte Modelle und Preprocessing-Objekte.
    7. Speichert eine Zusammenfassung der Ergebnisse.
    """
    logger.info("--- Starting Model Pipeline ---")

    # 1. Daten laden
    logger.info("\n--- Loading Data ---")
    df_comments = load_data_from_sqlite()
    if df_comments.empty:
        logger.error(
            "No data loaded. Please ensure the database contains data. Exiting model pipeline."
        )
        return

    # Check if TARGET_VARIABLE exists in the DataFrame
    if TARGET_VARIABLE not in df_comments.columns:
        logger.error(
            f"Target variable '{TARGET_VARIABLE}' not found in the loaded DataFrame. Exiting model pipeline."
        )
        return

    # 2. Features vorbereiten (Preprocessing Pipeline erstellen)
    logger.info("\n--- Preparing Features ---")
    preprocessor: ColumnTransformer
    all_feature_names: list[str]
    try:
        # preprocess_features now returns preprocessor and all_feature_names
        preprocessor, all_feature_names = preprocess_features(df_comments.copy())
    except ValueError as e:
        logger.error(
            f"Error during feature preprocessing: {e}. Exiting model pipeline."
        )
        return
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during feature preprocessing: {e}. Exiting model pipeline."
        )
        return

    # 3. Train-Test Split (auf den Rohdaten, da die Pipeline die Transformation übernimmt)
    logger.info("\n--- Performing Train-Test Split ---")
    # X_raw sind die Features, die direkt aus df_comments kommen, bevor die Pipeline angewendet wird
    X_raw = df_comments.drop(columns=[TARGET_VARIABLE])
    Y = df_comments[TARGET_VARIABLE]

    if X_raw.empty or Y.empty:
        logger.error(
            "Features or target variable are empty after dropping target. Exiting model pipeline."
        )
        return

    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X_raw, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    logger.info(f"  X_train_raw shape: {X_train_raw.shape}")
    logger.info(f"  X_test_raw shape: {X_test_raw.shape}")
    logger.info(f"  Y_train shape: {Y_train.shape}")
    logger.info(f"  Y_test shape: {Y_test.shape}")

    # Initialize results with NaN in case a model fails
    r2_linear_test, r2_linear_cv = np.nan, np.nan
    r2_xgboost_tuned_test, r2_xgboost_tuned_cv = np.nan, np.nan
    best_xgboost_params: Dict[str, Any] = {}
    linear_model_pipeline: Pipeline = Pipeline(steps=[])
    xgboost_model_tuned_pipeline: Pipeline = Pipeline(steps=[])

    # 4. Modelle trainieren und evaluieren
    logger.info("\n--- Training and Evaluating Models ---")
    try:
        linear_model_pipeline, r2_linear_test, r2_linear_cv = (
            train_and_evaluate_linear_regression(
                preprocessor, X_train_raw, Y_train, X_test_raw, Y_test, X_raw, Y
            )
        )
    except Exception as e:
        logger.error(f"Error during Linear Regression training/evaluation: {e}")

    try:
        (
            xgboost_model_tuned_pipeline,
            r2_xgboost_tuned_test,
            r2_xgboost_tuned_cv,
            best_xgboost_params,
        ) = train_and_tune_xgboost(
            preprocessor, X_train_raw, Y_train, X_test_raw, Y_test, X_raw, Y
        )
    except Exception as e:
        logger.error(f"Error during XGBoost training/tuning/evaluation: {e}")

    # 5. Modellvergleich
    logger.info("\n--- Final Model Comparison ---")
    logger.info(
        f"Linear Regression R² (Test Set): {r2_linear_test:.2f}, Mean CV R²: {r2_linear_cv:.2f}"
    )
    logger.info(
        f"XGBoost Regressor R² (Test Set - Tuned): {r2_xgboost_tuned_test:.2f}, Mean CV R²: {r2_xgboost_tuned_cv:.2f}"
    )

    # 6. Modell interpretieren (SHAP)
    logger.info("\n--- Interpreting XGBoost Model with SHAP ---")
    if not xgboost_model_tuned_pipeline.named_steps.get("regressor"):
        logger.warning(
            "XGBoost model not successfully trained. Skipping SHAP interpretation."
        )
    else:
        try:
            interpret_model_shap(
                xgboost_model_tuned_pipeline, X_test_raw, all_feature_names
            )
        except Exception as e:
            logger.error(f"Error during SHAP interpretation: {e}")

    # 7. Artefakte speichern
    logger.info("\n--- Saving Model Artifacts ---")
    # save_artifacts now takes preprocessor directly, not tfidf_vectorizer separately
    try:
        save_artifacts(
            linear_model_pipeline,
            xgboost_model_tuned_pipeline,
            preprocessor,
        )
    except Exception as e:
        logger.error(f"Error saving model artifacts: {e}")

    # 8. Ergebnisse zusammenfassen und speichern
    logger.info("\n--- Saving ML Results Summary ---")
    save_ml_results_summary(
        r2_linear_test,
        r2_linear_cv,
        r2_xgboost_tuned_test,
        r2_xgboost_tuned_cv,
        best_xgboost_params,
    )

    logger.info("\n--- Model Pipeline Completed Successfully ---")


if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_model_pipeline()
