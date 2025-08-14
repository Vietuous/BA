import datetime
import logging
import os
import shutil  # Added for file copying
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
LOG_LEVEL = getattr(config, "LOG_LEVEL", logging.INFO)
MODELS_DIR = getattr(config, "MODELS_DIR", "models")
RANDOM_STATE = getattr(config, "RANDOM_STATE", 42)
TARGET_VARIABLE = getattr(config, "TARGET_VARIABLE", "comment_score")
TEST_SIZE = getattr(config, "TEST_SIZE", 0.2)

# Setup logging (must be done before importing model_utils if model_utils also sets up logging)
os.makedirs(config.LOGS_DIR, exist_ok=True)

# Get a logger instance for this module
logger = logging.getLogger("ml.pipeline")  # Get a logger instance for this module


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
    from BA.src.utils.logging_utils import setup_logger
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
    subfolder: str = "",
    filename_prefix: str = "",
) -> None:
    """
    Saves a summary of the ML model results to a text file in the REPORTS_DIR,
    within a specified subfolder and with an optional filename prefix.

    Args:
        r2_linear_test (float): R-squared value for Linear Regression on the test set.
        r2_linear_cv (float): Mean cross-validation R-squared for Linear Regression.
        r2_xgboost_tuned_test (float): R-squared value for Tuned XGBoost on the test set.
        r2_xgboost_tuned_cv (float): Mean cross-validation R-squared for Tuned XGBoost.
        best_xgboost_params (Dict[str, Any]): Dictionary of best hyperparameters for XGBoost.
        subfolder (str, optional): The subfolder within REPORTS_DIR to save the summary. Defaults to "".
        filename_prefix (str, optional): Prefix for the saved summary filename. Defaults to "".
    """
    # Create the full path for the subfolder
    target_dir = os.path.join(REPORTS_DIR, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    summary_filename = f"{filename_prefix}ml_results_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary_filepath = os.path.join(target_dir, summary_filename)

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

    # Define subfolders for ML plots and reports
    ML_PLOTS_FOLDER = "ml_plots"
    FINAL_PLOTS_FOLDER = "final_plots"  # Define final_plots folder here too

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
            # Generate SHAP plots in the ml_plots folder with RQ7_ prefix
            interpret_model_shap(
                xgboost_model_tuned_pipeline,
                X_test_raw,
                all_feature_names,
                output_subfolder=ML_PLOTS_FOLDER,
                output_filename_prefix="RQ7_",
            )

            # Copy specific SHAP plots to final_plots folder
            source_dir = os.path.join(FIGURES_DIR, ML_PLOTS_FOLDER)
            target_dir_final_plots = os.path.join(FIGURES_DIR, FINAL_PLOTS_FOLDER)
            os.makedirs(
                target_dir_final_plots, exist_ok=True
            )  # Ensure final_plots folder exists

            files_to_copy = [
                "RQ7_xgboost_feature_importance_full.png",
                "RQ7_shap_summary_plot_full.png",
            ]

            for filename in files_to_copy:
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir_final_plots, filename)
                if os.path.exists(source_path):
                    shutil.copy2(source_path, target_path)
                    logger.info(f"Copied {filename} to {target_dir_final_plots}")
                else:
                    logger.warning(f"Source file not found for copying: {source_path}")

        except Exception as e:
            logger.error(f"Error during SHAP interpretation or copying: {e}")

    # 7. Artefakte speichern
    logger.info("\n--- Saving Model Artifacts ---")
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
        subfolder=ML_PLOTS_FOLDER,
        filename_prefix="",
    )

    logger.info("\n--- Model Pipeline completed Successfully ---")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, "ml_plots"), exist_ok=True)
    os.makedirs(os.path.join(REPORTS_DIR, "ml_plots"), exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, "final_plots"), exist_ok=True)

    setup_logger("ml.pipeline", config.LOG_FILES["ml.pipeline"], config.LOG_LEVEL)

    logger.info("Running train_model.py...")
    try:
        run_model_pipeline()
        logger.info("Training pipeline completed.")
    except Exception:
        logging.exception("Error while running ML pipeline")
