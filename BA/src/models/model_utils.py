# BA/src/models/model_utils.py
import logging
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_style("whitegrid")

# Import config
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.sys.path.insert(0, project_root_for_import)
import config

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# Access config parameters safely with getattr, providing defaults
BOOLEAN_FEATURES = getattr(config, "BOOLEAN_FEATURES", [])
CATEGORICAL_FEATURES = getattr(config, "CATEGORICAL_FEATURES", [])
CV_FOLDS = getattr(config, "CV_FOLDS", 5)
FIGURES_DIR = getattr(config, "FIGURES_DIR", "figures")
MODELS_DIR = getattr(config, "MODELS_DIR", "models")
NUMERICAL_FEATURES = getattr(config, "NUMERICAL_FEATURES", [])
RANDOM_STATE = getattr(config, "RANDOM_STATE", 42)
SHAP_TOP_FEATURES_TO_PLOT = getattr(config, "SHAP_TOP_FEATURES_TO_PLOT", 10)
TARGET_VARIABLE = getattr(config, "TARGET_VARIABLE", "comment_score")
TEST_SIZE = getattr(config, "TEST_SIZE", 0.2)
TEXT_FEATURE = getattr(config, "TEXT_FEATURE", "processed_comment_body")
TFIDF_MAX_DF = getattr(config, "TFIDF_MAX_DF", 0.95)
TFIDF_MAX_FEATURES = getattr(config, "TFIDF_MAX_FEATURES", 1000)
TFIDF_MIN_DF = getattr(config, "TFIDF_MIN_DF", 5)
TFIDF_NGRAM_RANGE = getattr(config, "TFIDF_NGRAM_RANGE", (1, 1))
XGB_TUNING_CV_FOLDS = getattr(config, "XGB_TUNING_CV_FOLDS", 3)
XGB_TUNING_N_ITER = getattr(config, "XGB_TUNING_N_ITER", 50)


# BA/src/models/model_utils.py

# ... (bestehender Code) ...


def preprocess_features(
    df_comments: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Führt Textvektorisierung (TF-IDF), One-Hot-Encoding und Feature Scaling durch
    und erstellt eine Preprocessing-Pipeline (ColumnTransformer).

    Args:
        df_comments (pd.DataFrame): Der DataFrame mit den Kommentardaten.
                                    Erwartet Spalten für Text, Kategorien, Numerik und Booleans.

    Returns:
        tuple:
            - preprocessor (sklearn.compose.ColumnTransformer): Die trainierte Preprocessing-Pipeline.
            - all_feature_names (List[str]): Eine Liste aller Feature-Namen nach der Transformation.
    """
    logger.info("--- Starting Feature Preprocessing ---")

    if df_comments.empty:
        logger.error("Input DataFrame for preprocessing is empty. Cannot proceed.")
        raise ValueError("Input DataFrame is empty.")

    # Ensure TEXT_FEATURE exists and is string type, handle NaNs
    if TEXT_FEATURE not in df_comments.columns:
        logger.error(f"Text feature column '{TEXT_FEATURE}' not found in DataFrame.")
        raise ValueError(f"Missing required column: {TEXT_FEATURE}")
    df_comments[TEXT_FEATURE] = df_comments[TEXT_FEATURE].astype(str).fillna("")

    # Debugging: Print a sample of the text feature
    logger.info(
        f"  Sample of '{TEXT_FEATURE}' column (first 5 entries):\n{df_comments[TEXT_FEATURE].head().to_string()}"
    )

    # Check number of non-empty text documents
    non_empty_text_docs = (
        df_comments[TEXT_FEATURE]
        .loc[df_comments[TEXT_FEATURE].str.strip() != ""]
        .shape[0]
    )
    if non_empty_text_docs == 0:
        logger.error(
            "No non-empty text documents found after preprocessing. TF-IDF cannot be applied."
        )
        raise ValueError("No non-empty text documents for TF-IDF.")
    logger.info(
        f"  Number of non-empty text documents for TF-IDF: {non_empty_text_docs}"
    )

    # Check number of non-empty text documents
    non_empty_text_docs = (
        df_comments[TEXT_FEATURE]
        .loc[df_comments[TEXT_FEATURE].str.strip() != ""]
        .shape[0]
    )
    if non_empty_text_docs == 0:
        logger.error(
            "No non-empty text documents found after preprocessing. TF-IDF cannot be applied."
        )
        raise ValueError("No non-empty text documents for TF-IDF.")
    logger.info(
        f"  Number of non-empty text documents for TF-IDF: {non_empty_text_docs}"
    )

    # Define preprocessing steps for different types of features
    # 1. TF-IDF for text
    text_transformer = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    min_df=1,  # Temporarily set to 1 to be very permissive
                    max_df=1,  # Temporarily set to 1 to be very permissive
                    ngram_range=TFIDF_NGRAM_RANGE,
                    stop_words="english",  # Add English stop words
                ),
            )
        ]
    )

    # 2. One-Hot Encoding for categorical features
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]
    )

    # 3. Scaling for numerical features
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Create a ColumnTransformer to apply different transformations to different columns
    # Filter columns to ensure they exist in the DataFrame
    # BA/src/models/model_utils.py

    # ... (bestehender Code) ...

    # Filter columns to ensure they exist in the DataFrame
    # ÄNDERUNG HIER: cols_for_text wird ein String oder None, nicht eine Liste
    cols_for_text = TEXT_FEATURE if TEXT_FEATURE in df_comments.columns else None
    cols_for_cat = [col for col in CATEGORICAL_FEATURES if col in df_comments.columns]
    cols_for_num = [col for col in NUMERICAL_FEATURES if col in df_comments.columns]
    cols_for_bool = [col for col in BOOLEAN_FEATURES if col in df_comments.columns]

    if cols_for_text is None:  # Angepasste Warnmeldung
        logger.warning(
            f"Text feature '{TEXT_FEATURE}' not found. Text pipeline will be skipped."
        )
    if not cols_for_cat:
        logger.warning(
            "No categorical features found in DataFrame. Categorical pipeline will be skipped."
        )
    if not cols_for_num:
        logger.warning(
            "No numerical features found in DataFrame. Numerical pipeline will be skipped."
        )
    if not cols_for_bool:
        logger.warning(
            "No boolean features found in DataFrame. Boolean passthrough will be skipped."
        )

    transformers_list = []
    if cols_for_text:  # Hier wird der String TEXT_FEATURE übergeben
        transformers_list.append(("text_pipeline", text_transformer, cols_for_text))
    if cols_for_cat:
        transformers_list.append(
            ("cat_pipeline", categorical_transformer, cols_for_cat)
        )
    if cols_for_num:
        transformers_list.append(("num_pipeline", numerical_transformer, cols_for_num))
    if cols_for_bool:
        transformers_list.append(("bool_passthrough", "passthrough", cols_for_bool))

    if not transformers_list:
        logger.error(
            "No valid features found to preprocess. Returning empty preprocessor and feature names."
        )
        return ColumnTransformer(transformers=[]), []

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder="drop",  # Drop any columns not specified
        sparse_threshold=0.3,  # Output sparse matrix if sparsity is above this threshold
    )

    # Fit the preprocessor to the data to learn transformations
    logger.info("  Fitting preprocessor to data...")
    preprocessor.fit(df_comments)

    # Extract all feature names after transformation using get_feature_names_out()
    # This is the most robust way for ColumnTransformer (Scikit-learn 0.23+)
    try:
        all_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        logger.warning(
            "preprocessor.get_feature_names_out() not available. Attempting manual feature name extraction."
        )
        # Fallback for older scikit-learn versions or complex cases
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if transformer == "passthrough":
                feature_names.extend(cols)
            elif hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols))
            elif hasattr(transformer, "get_feature_names"):  # Older versions
                feature_names.extend(transformer.get_feature_names(cols))
            else:
                # For pipelines, access the last step's feature names
                if isinstance(transformer, Pipeline) and hasattr(
                    transformer.steps[-1][1], "get_feature_names_out"
                ):
                    feature_names.extend(
                        transformer.steps[-1][1].get_feature_names_out(cols)
                    )
                elif isinstance(transformer, Pipeline) and hasattr(
                    transformer.steps[-1][1], "get_feature_names"
                ):
                    feature_names.extend(
                        transformer.steps[-1][1].get_feature_names(cols)
                    )
                else:
                    logger.warning(
                        f"Could not extract feature names for transformer '{name}'. Skipping."
                    )
        all_feature_names = feature_names

    logger.info(f"  Total features after preprocessing: {len(all_feature_names)}")
    logger.info("--- Feature Preprocessing Complete ---")
    return preprocessor, all_feature_names


def train_and_evaluate_linear_regression(
    preprocessor: ColumnTransformer,
    X_train_raw: pd.DataFrame,
    Y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    Y_test: pd.Series,
    X_full_raw: pd.DataFrame,
    Y_full: pd.Series,
) -> Tuple[Pipeline, float, float]:
    """
    Trainiert und evaluiert ein Lineares Regressionsmodell.

    Args:
        preprocessor (ColumnTransformer): Die Scikit-learn Preprocessing Pipeline.
        X_train_raw (pd.DataFrame): Trainingsdaten (Rohdaten).
        Y_train (pd.Series): Trainings-Zielvariable.
        X_test_raw (pd.DataFrame): Testdaten (Rohdaten).
        Y_test (pd.Series): Test-Zielvariable.
        X_full_raw (pd.DataFrame): Gesamter Datensatz (Rohdaten) für Cross-Validation.
        Y_full (pd.Series): Gesamte Zielvariable für Cross-Validation.

    Returns:
        tuple:
            - linear_model_pipeline (sklearn.pipeline.Pipeline): Das trainierte lineare Regressionsmodell.
            - r2_linear_test (float): R-squared Wert auf dem Testset.
            - r2_linear_cv (float): Durchschnittlicher R-squared Wert aus der Kreuzvalidierung.
    """
    logger.info("\n--- Training Multiple Linear Regression Model ---")

    if X_train_raw.empty or Y_train.empty or X_test_raw.empty or Y_test.empty:
        logger.error(
            "One or more input datasets for Linear Regression are empty. Skipping training."
        )
        return Pipeline(steps=[]), np.nan, np.nan

    # Create a full pipeline for Linear Regression
    linear_model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
    )

    try:
        linear_model_pipeline.fit(X_train_raw, Y_train)
        Y_pred_linear = linear_model_pipeline.predict(X_test_raw)
        mse_linear = mean_squared_error(Y_test, Y_pred_linear)
        r2_linear = r2_score(Y_test, Y_pred_linear)
        logger.info(f"\nLinear Regression Model Performance (Test Set):")
        logger.info(f"  Mean Squared Error (MSE): {mse_linear:.2f}")
        logger.info(f"  R-squared (R²): {r2_linear:.2f}")

        logger.info("\n--- Performing Cross-Validation for Linear Regression ---")
        cv_results_linear = cross_val_score(
            linear_model_pipeline,
            X_full_raw,
            Y_full,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        )
        logger.info(f"  Cross-Validation R² scores: {cv_results_linear}")
        logger.info(
            f"  Mean CV R²: {np.mean(cv_results_linear):.2f} (+/- {np.std(cv_results_linear):.2f})"
        )
        return linear_model_pipeline, r2_linear, np.mean(cv_results_linear)
    except Exception as e:
        logger.error(f"Error during Linear Regression training or evaluation: {e}")
        return linear_model_pipeline, np.nan, np.nan


def train_and_tune_xgboost(
    preprocessor: ColumnTransformer,
    X_train_raw: pd.DataFrame,
    Y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    Y_test: pd.Series,
    X_full_raw: pd.DataFrame,
    Y_full: pd.Series,
) -> Tuple[Pipeline, float, float, Dict[str, Any]]:
    """
    Trainiert, tunet und evaluiert ein XGBoost Regressionsmodell.

    Args:
        preprocessor (ColumnTransformer): Die Scikit-learn Preprocessing Pipeline.
        X_train_raw (pd.DataFrame): Trainingsdaten (Rohdaten).
        Y_train (pd.Series): Trainings-Zielvariable.
        X_test_raw (pd.DataFrame): Testdaten (Rohdaten).
        Y_test (pd.Series): Test-Zielvariable.
        X_full_raw (pd.DataFrame): Gesamter Datensatz (Rohdaten) für Cross-Validation und Tuning.
        Y_full (pd.Series): Gesamte Zielvariable für Cross-Validation und Tuning.

    Returns:
        tuple:
            - xgboost_model_tuned_pipeline (sklearn.pipeline.Pipeline): Das getunte XGBoost Modell.
            - r2_xgboost_tuned_test (float): R-squared Wert auf dem Testset des getunten Modells.
            - r2_xgboost_tuned_cv (float): Durchschnittlicher R-squared Wert aus der Kreuzvalidierung des getunten Modells.
            - best_xgboost_params (Dict[str, Any]): Die besten gefundenen Hyperparameter.
    """
    logger.info("\n--- Training XGBoost Regressor Model (Initial) ---")

    if X_train_raw.empty or Y_train.empty or X_test_raw.empty or Y_test.empty:
        logger.error(
            "One or more input datasets for XGBoost are empty. Skipping training."
        )
        return Pipeline(steps=[]), np.nan, np.nan, {}

    # Initial XGBoost model within a pipeline
    xgboost_initial_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    try:
        xgboost_initial_pipeline.fit(X_train_raw, Y_train)
        Y_pred_xgboost_initial = xgboost_initial_pipeline.predict(X_test_raw)
        mse_xgboost_initial = mean_squared_error(Y_test, Y_pred_xgboost_initial)
        r2_xgboost_initial = r2_score(Y_test, Y_pred_xgboost_initial)
        logger.info(f"\nXGBoost Regressor Model Performance (Test Set - Initial):")
        logger.info(f"  Mean Squared Error (MSE): {mse_xgboost_initial:.2f}")
        logger.info(f"  R-squared (R²): {r2_xgboost_initial:.2f}")

        logger.info("\n--- Performing Cross-Validation for XGBoost (Initial) ---")
        cv_results_xgboost_initial = cross_val_score(
            xgboost_initial_pipeline,
            X_full_raw,
            Y_full,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        )
        logger.info(f"  Cross-Validation R² scores: {cv_results_xgboost_initial}")
        logger.info(
            f"  Mean CV R²: {np.mean(cv_results_xgboost_initial):.2f} (+/- {np.std(cv_results_xgboost_initial):.2f})"
        )

        logger.info(
            "\n--- Starting Hyperparameter Tuning for XGBoost (RandomizedSearchCV) ---"
        )
        # Define the parameter distribution for RandomizedSearchCV
        param_distributions = {
            "regressor__n_estimators": randint(100, 1000),
            "regressor__learning_rate": uniform(0.01, 0.2),
            "regressor__max_depth": randint(3, 10),
            "regressor__subsample": uniform(0.6, 0.4),
            "regressor__colsample_bytree": uniform(0.6, 0.4),
            "regressor__gamma": uniform(0, 0.5),
            "regressor__reg_lambda": uniform(1, 2),  # Use reg_lambda for L2
            "regressor__reg_alpha": uniform(0, 1),  # Use reg_alpha for L1
        }

        # Initialize RandomizedSearchCV with the pipeline
        random_search = RandomizedSearchCV(
            estimator=xgboost_initial_pipeline,  # Pass the pipeline as estimator
            param_distributions=param_distributions,
            n_iter=XGB_TUNING_N_ITER,
            cv=XGB_TUNING_CV_FOLDS,
            scoring="r2",
            verbose=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        random_search.fit(X_full_raw, Y_full)  # Tune on the full dataset

        logger.info("\n--- Hyperparameter Tuning Complete ---")
        logger.info(
            f"Best R² score found during tuning: {random_search.best_score_:.2f}"
        )
        logger.info(f"Best parameters found: {random_search.best_params_}")

        # The best estimator from random_search is already a fitted pipeline
        xgboost_model_tuned_pipeline = random_search.best_estimator_

        Y_pred_xgboost_tuned = xgboost_model_tuned_pipeline.predict(X_test_raw)
        mse_xgboost_tuned = mean_squared_error(Y_test, Y_pred_xgboost_tuned)
        r2_xgboost_tuned = r2_score(Y_test, Y_pred_xgboost_tuned)
        logger.info(f"\nFinal Tuned XGBoost Regressor Model Performance (Test Set):")
        logger.info(f"  Mean Squared Error (MSE): {mse_xgboost_tuned:.2f}")
        logger.info(f"  R-squared (R²): {r2_xgboost_tuned:.2f}")

        logger.info("\n--- Performing Cross-Validation for Tuned XGBoost ---")
        cv_results_xgboost_tuned = cross_val_score(
            xgboost_model_tuned_pipeline,
            X_full_raw,
            Y_full,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        )
        logger.info(f"  Cross-Validation R² scores: {cv_results_xgboost_tuned}")
        logger.info(
            f"  Mean CV R²: {np.mean(cv_results_xgboost_tuned):.2f} (+/- {np.std(cv_results_xgboost_tuned):.2f})"
        )

        return (
            xgboost_model_tuned_pipeline,
            r2_xgboost_tuned,
            np.mean(cv_results_xgboost_tuned),
            random_search.best_params_,  # Return best params
        )
    except Exception as e:
        logger.error(f"Error during XGBoost training, tuning or evaluation: {e}")
        return xgboost_initial_pipeline, np.nan, np.nan, {}


def interpret_model_shap(
    model_pipeline: Pipeline, X_test_raw: pd.DataFrame, all_feature_names: List[str]
) -> None:
    """
    Führt Modellinterpretierbarkeit mit SHAP durch und generiert Plots.

    Args:
        model_pipeline (sklearn.pipeline.Pipeline): Die trainierte Scikit-learn Pipeline (mit XGBoost Regressor).
        X_test_raw (pd.DataFrame): Der Testdatensatz (Rohdaten).
        all_feature_names (List[str]): Liste aller Feature-Namen nach Preprocessing.
    """
    logger.info("\n--- Starting Model Interpretability with SHAP ---")

    if X_test_raw.empty:
        logger.warning("X_test_raw is empty. Skipping SHAP interpretation.")
        return
    if not all_feature_names:
        logger.warning("Feature names list is empty. Skipping SHAP interpretation.")
        return
    if "regressor" not in model_pipeline.named_steps:
        logger.error(
            "Model pipeline does not contain a 'regressor' step. Cannot perform SHAP interpretation."
        )
        return

    # Extract the fitted XGBoost model from the pipeline
    xgboost_model = model_pipeline.named_steps["regressor"]

    # Transform X_test_raw using the preprocessor within the pipeline
    logger.info("  Transforming X_test_raw for SHAP analysis...")
    X_test_transformed = model_pipeline.named_steps["preprocessor"].transform(
        X_test_raw
    )

    # Create a SHAP Explainer for the tuned XGBoost model
    explainer = shap.TreeExplainer(xgboost_model)

    # Calculate SHAP values for the transformed test set
    # Convert to dense array if it's sparse, as SHAP sometimes prefers dense for plotting
    X_test_transformed_dense = (
        X_test_transformed.toarray()
        if hasattr(X_test_transformed, "toarray")
        else X_test_transformed
    )

    # Ensure the number of features matches
    if X_test_transformed_dense.shape[1] != len(all_feature_names):
        logger.error(
            f"Mismatch between transformed feature count ({X_test_transformed_dense.shape[1]}) and provided feature names count ({len(all_feature_names)}). SHAP plots might be incorrect. Skipping."
        )
        return

    try:
        shap_values = explainer.shap_values(X_test_transformed_dense)
    except Exception as e:
        logger.error(
            f"Error calculating SHAP values: {e}. This might be due to a mismatch in feature names or data format."
        )
        return

    # Create a DataFrame for the feature names for SHAP plots
    X_test_df = pd.DataFrame(X_test_transformed_dense, columns=all_feature_names)

    # SHAP Summary Plot (Global Interpretability)
    logger.info("  Generating SHAP Summary Plot...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type="dot", show=False)
        plt.title("SHAP Summary Plot: Feature Importance and Impact")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_plot.png"))
        plt.close()  # Close plot to free memory
        logger.info(
            f"  SHAP Summary Plot saved to: {os.path.join(FIGURES_DIR, 'shap_summary_plot.png')}"
        )
    except Exception as e:
        logger.error(f"Error generating SHAP Summary Plot: {e}")

    # SHAP Dependence Plots (Local Interpretability for specific features)
    logger.info("  Generating SHAP Dependence Plots for key features...")
    # Use the model's feature importances to get top features
    if hasattr(xgboost_model, "feature_importances_"):
        importances = xgboost_model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"Feature": all_feature_names, "Importance": importances}
        )
        feature_importance_df = feature_importance_df.sort_values(
            by="Importance", ascending=False
        )
        top_features = (
            feature_importance_df["Feature"].head(SHAP_TOP_FEATURES_TO_PLOT).tolist()
        )
    else:
        logger.warning(
            "XGBoost model does not have 'feature_importances_'. Cannot determine top features for dependence plots."
        )
        top_features = []  # No top features to plot

    # Function to sanitize filenames
    def sanitize_filename(filename: str) -> str:
        """Replaces invalid characters in a filename with an underscore."""
        return re.sub(r'[<>:"/\\|?*]', "_", filename)

    for feature in top_features:
        if feature in X_test_df.columns:
            try:
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(
                    feature, shap_values, X_test_df, interaction_index=None, show=False
                )
                plt.title(f"SHAP Dependence Plot for: {feature}")
                plt.tight_layout()
                # Sanitize the feature name before saving
                sanitized_feature_name = sanitize_filename(feature)
                plt.savefig(
                    os.path.join(
                        FIGURES_DIR,
                        f"shap_dependence_plot_{sanitized_feature_name}.png",
                    )
                )
                plt.close()  # Close plot to free memory
                logger.info(f"  SHAP Dependence Plot for {feature} saved.")
            except Exception as e:
                logger.error(
                    f"Error generating SHAP Dependence Plot for feature '{feature}': {e}"
                )
        else:
            logger.warning(
                f"  Warning: Feature '{feature}' not found in X_test_df for dependence plot. Skipping."
            )

    logger.info("\n--- Model Interpretability Complete ---")


def save_artifacts(
    linear_model_pipeline: Pipeline,
    xgboost_model_tuned_pipeline: Pipeline,
    preprocessor: ColumnTransformer,
) -> None:
    """
    Speichert die trainierten Modell-Pipelines und Preprocessing-Objekte.

    Args:
        linear_model_pipeline (sklearn.pipeline.Pipeline): Die trainierte Lineare Regressions Pipeline.
        xgboost_model_tuned_pipeline (sklearn.pipeline.Pipeline): Die getunte XGBoost Pipeline.
        preprocessor (sklearn.compose.ColumnTransformer): Die gesamte Preprocessing Pipeline (ColumnTransformer).
    """
    logger.info("\n--- Saving Trained Models and Preprocessing Objects ---")
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        joblib.dump(
            linear_model_pipeline,
            os.path.join(MODELS_DIR, "linear_regression_model_pipeline.joblib"),
        )
        logger.info(
            f"Linear Regression model saved to: {os.path.join(MODELS_DIR, 'linear_regression_model_pipeline.joblib')}"
        )
    except Exception as e:
        logger.error(f"Error saving Linear Regression model: {e}")

    try:
        joblib.dump(
            xgboost_model_tuned_pipeline,
            os.path.join(MODELS_DIR, "xgboost_model_tuned_pipeline.joblib"),
        )
        logger.info(
            f"XGBoost model saved to: {os.path.join(MODELS_DIR, 'xgboost_model_tuned_pipeline.joblib')}"
        )
    except Exception as e:
        logger.error(f"Error saving XGBoost model: {e}")

    try:
        joblib.dump(
            preprocessor, os.path.join(MODELS_DIR, "full_preprocessor_pipeline.joblib")
        )
        logger.info(
            f"Preprocessor pipeline saved to: {os.path.join(MODELS_DIR, 'full_preprocessor_pipeline.joblib')}"
        )
    except Exception as e:
        logger.error(f"Error saving preprocessor pipeline: {e}")

    logger.info(f"Models and preprocessing objects saved to: {MODELS_DIR}")
