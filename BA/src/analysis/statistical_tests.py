import logging
import os
import sys
from typing import Any, Dict, List, Union

import pandas as pd
from scipy import stats

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def perform_independent_t_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group1_name: Any,
    group2_name: Any,
) -> Dict[str, Any]:
    """
    Performs an independent samples t-test (Welch's t-test, assuming unequal variances)
    to compare the means of a numerical 'value_col' between two specified groups
    in a 'group_col'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): The name of the column containing the group labels (e.g., 'event_name').
        value_col (str): The name of the column containing the numerical values to compare (e.g., 'comment_score').
        group1_name (Any): The label of the first group in 'group_col'.
        group2_name (Any): The label of the second group in 'group_col'.

    Returns:
        Dict[str, Any]: A dictionary containing the test status, t-statistic, p-value,
                        and an interpretation. Returns 'skipped' or 'error' status
                        with a reason if the test cannot be performed.
    """
    logger.info(
        f"\n--- Performing Independent Samples t-test for '{value_col}' between '{group1_name}' and '{group2_name}' ---"
    )

    # Validate input columns
    if group_col not in df.columns:
        logger.error(
            f"Group column '{group_col}' not found in DataFrame. Skipping t-test."
        )
        return {"status": "error", "reason": f"Group column '{group_col}' not found."}
    if value_col not in df.columns:
        logger.error(
            f"Value column '{value_col}' not found in DataFrame. Skipping t-test."
        )
        return {"status": "error", "reason": f"Value column '{value_col}' not found."}
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        logger.error(f"Value column '{value_col}' is not numeric. Skipping t-test.")
        return {
            "status": "error",
            "reason": f"Value column '{value_col}' is not numeric.",
        }

    # Filter data for each group and drop NaNs in the value column
    group1_data = df[df[group_col] == group1_name][value_col].dropna()
    group2_data = df[df[group_col] == group2_name][value_col].dropna()

    # Check if groups have sufficient data
    if group1_data.empty or group2_data.empty:
        logger.warning(
            f"One or both groups ('{group1_name}', '{group2_name}') have no valid data for '{value_col}'. Skipping t-test."
        )
        return {
            "status": "skipped",
            "reason": "Insufficient data in one or both groups.",
        }
    if len(group1_data) < 2 or len(group2_data) < 2:
        logger.warning(
            f"One or both groups ('{group1_name}', '{group2_name}') have fewer than 2 observations for '{value_col}'. Skipping t-test as it requires at least 2 samples per group."
        )
        return {
            "status": "skipped",
            "reason": "Not enough observations in one or both groups.",
        }

    try:
        # Perform independent t-test (Welch's t-test, assuming unequal variances)
        t_statistic, p_value = stats.ttest_ind(
            group1_data, group2_data, equal_var=False
        )

        interpretation = ""
        if p_value < 0.05:
            interpretation = f"There is a statistically significant difference (p={p_value:.3f}) in '{value_col}' between '{group1_name}' (Mean: {group1_data.mean():.2f}) and '{group2_name}' (Mean: {group2_data.mean():.2f})."
        else:
            interpretation = f"There is no statistically significant difference (p={p_value:.3f}) in '{value_col}' between '{group1_name}' (Mean: {group1_data.mean():.2f}) and '{group2_name}' (Mean: {group2_data.mean():.2f})."

        logger.info(f"  t-statistic: {t_statistic:.3f}")
        logger.info(f"  p-value: {p_value:.3f}")
        logger.info(f"  Interpretation: {interpretation}")

        return {
            "status": "completed",
            "t_statistic": t_statistic,
            "p_value": p_value,
            "interpretation": interpretation,
            "group1_mean": group1_data.mean(),
            "group2_mean": group2_data.mean(),
        }
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during t-test for '{value_col}' between '{group1_name}' and '{group2_name}': {e}"
        )
        return {"status": "error", "reason": str(e)}


def perform_anova_test(
    df: pd.DataFrame, group_col: str, value_col: str
) -> Dict[str, Any]:
    """
    Performs a one-way ANOVA test to compare the means of a numerical 'value_col'
    across multiple groups in a 'group_col'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): The name of the column containing the group labels.
        value_col (str): The name of the column containing the numerical values to compare.

    Returns:
        Dict[str, Any]: A dictionary containing the test status, F-statistic, p-value,
                        and an interpretation. Returns 'skipped' or 'error' status
                        with a reason if the test cannot be performed.
    """
    logger.info(
        f"\n--- Performing One-Way ANOVA test for '{value_col}' across groups in '{group_col}' ---"
    )

    # Validate input columns
    if group_col not in df.columns:
        logger.error(
            f"Group column '{group_col}' not found in DataFrame. Skipping ANOVA."
        )
        return {"status": "error", "reason": f"Group column '{group_col}' not found."}
    if value_col not in df.columns:
        logger.error(
            f"Value column '{value_col}' not found in DataFrame. Skipping ANOVA."
        )
        return {"status": "error", "reason": f"Value column '{value_col}' not found."}
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        logger.error(f"Value column '{value_col}' is not numeric. Skipping ANOVA.")
        return {
            "status": "error",
            "reason": f"Value column '{value_col}' is not numeric.",
        }

    # Get unique groups and filter out empty data for each group
    unique_groups = df[group_col].dropna().unique()
    if len(unique_groups) == 0:
        logger.warning(f"No unique groups found in '{group_col}'. Skipping ANOVA.")
        return {"status": "skipped", "reason": "No unique groups found."}

    groups_data: List[pd.Series] = []
    for g in unique_groups:
        data = df[df[group_col] == g][value_col].dropna()
        if not data.empty:
            groups_data.append(data)
        else:
            logger.warning(
                f"Group '{g}' has no valid data for '{value_col}'. It will be excluded from ANOVA."
            )

    if len(groups_data) < 2:
        logger.warning(
            f"Less than two non-empty groups with valid data for '{value_col}' in '{group_col}'. Skipping ANOVA."
        )
        return {
            "status": "skipped",
            "reason": "Insufficient non-empty groups for ANOVA.",
        }

    try:
        f_statistic, p_value = stats.f_oneway(*groups_data)

        interpretation = ""
        if p_value < 0.05:
            interpretation = f"There is a statistically significant difference (p={p_value:.3f}) in '{value_col}' across at least two groups in '{group_col}'."
        else:
            interpretation = f"There is no statistically significant difference (p={p_value:.3f}) in '{value_col}' across groups in '{group_col}'."

        logger.info(f"  F-statistic: {f_statistic:.3f}")
        logger.info(f"  p-value: {p_value:.3f}")
        logger.info(f"  Interpretation: {interpretation}")

        return {
            "status": "completed",
            "f_statistic": f_statistic,
            "p_value": p_value,
            "interpretation": interpretation,
        }
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during ANOVA for '{value_col}' across '{group_col}': {e}"
        )
        return {"status": "error", "reason": str(e)}


def perform_chi_squared_test(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """
    Performs a Chi-squared test of independence between two categorical columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first categorical column.
        col2 (str): The name of the second categorical column.

    Returns:
        Dict[str, Any]: A dictionary containing the test status, chi-squared statistic,
                        p-value, degrees of freedom, expected frequencies, and interpretation.
                        Returns 'skipped' or 'error' status with a reason if the test
                        cannot be performed.
    """
    logger.info(
        f"\n--- Performing Chi-squared test of independence between '{col1}' and '{col2}' ---"
    )

    # Validate input columns
    if col1 not in df.columns:
        logger.error(
            f"Column '{col1}' not found in DataFrame. Skipping Chi-squared test."
        )
        return {"status": "error", "reason": f"Column '{col1}' not found."}
    if col2 not in df.columns:
        logger.error(
            f"Column '{col2}' not found in DataFrame. Skipping Chi-squared test."
        )
        return {"status": "error", "reason": f"Column '{col2}' not found."}

    # Create a contingency table, dropping NaNs from the relevant columns
    contingency_table = pd.crosstab(df[col1], df[col2], dropna=True)

    if contingency_table.empty:
        logger.warning(
            f"Contingency table is empty for '{col1}' and '{col2}' after dropping NaNs. Skipping Chi-squared test."
        )
        return {"status": "skipped", "reason": "Empty contingency table."}

    # Check if contingency table has at least 2x2 dimensions for meaningful test
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        logger.warning(
            f"Contingency table for '{col1}' and '{col2}' has dimensions {contingency_table.shape}. "
            "Chi-squared test requires at least 2 rows and 2 columns. Skipping."
        )
        return {
            "status": "skipped",
            "reason": "Contingency table too small (less than 2x2).",
        }

    try:
        chi2_statistic, p_value, dof, expected_freq = stats.chi2_contingency(
            contingency_table
        )

        interpretation = ""
        if p_value < 0.05:
            interpretation = f"There is a statistically significant association (p={p_value:.3f}) between '{col1}' and '{col2}'. This suggests that the two variables are not independent."
        else:
            interpretation = f"There is no statistically significant association (p={p_value:.3f}) between '{col1}' and '{col2}'. This suggests that the two variables are independent."

        logger.info(f"  Chi-squared statistic: {chi2_statistic:.3f}")
        logger.info(f"  p-value: {p_value:.3f}")
        logger.info(f"  Degrees of freedom: {dof}")
        # logger.info(f"  Expected Frequencies:\n{expected_freq}") # Can be very large, log only if needed
        logger.info(f"  Interpretation: {interpretation}")

        return {
            "status": "completed",
            "chi2_statistic": chi2_statistic,
            "p_value": p_value,
            "dof": dof,
            "expected_freq": expected_freq.tolist(),  # Convert numpy array to list for JSON compatibility if returned
            "interpretation": interpretation,
        }
    except ValueError as e:
        logger.error(
            f"ValueError during Chi-squared test for '{col1}' and '{col2}': {e}. "
            "This often means expected frequencies are too low. Consider combining categories or "
            "using Fisher's exact test for small samples (e.g., if any expected frequency is < 5)."
        )
        return {"status": "error", "reason": str(e)}
    except Exception as e:
        logger.error(f"An unexpected error occurred during Chi-squared test: {e}")
        return {"status": "error", "reason": str(e)}
