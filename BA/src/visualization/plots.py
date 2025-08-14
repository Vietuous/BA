import logging
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config

# Logger configuration
logger = logging.getLogger("eda.stats")
sns.set_style("whitegrid")

# Ensure output directories exist (base directories)
os.makedirs(config.FIGURES_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)

# Import custom modules from src/
from BA.src.utils.logging_utils import setup_logger


def save_plot(fig: plt.Figure, filename: str, subfolder: str = "") -> None:
    """
    Helper function to save plots with consistent settings to a specified subfolder.

    Args:
        fig (plt.Figure): The matplotlib figure object to save.
        filename (str): The name of the file to save the plot as.
        subfolder (str, optional): The subfolder within FIGURES_DIR to save the plot. Defaults to "".
    """
    target_dir = os.path.join(config.FIGURES_DIR, subfolder)
    os.makedirs(target_dir, exist_ok=True)  # Ensure subfolder exists
    filepath = os.path.join(target_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    logger.info(f"Plot saved to: {filepath}")


def _save_dataframe_as_text(
    df: pd.DataFrame, filename: str, title: str = "", subfolder: str = ""
) -> None:
    """
    Helper function to save a DataFrame to a text file in a specified subfolder.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the text file.
        title (str, optional): An optional title for the text file. Defaults to "".
        subfolder (str, optional): The subfolder within REPORTS_DIR to save the text file. Defaults to "".
    """
    target_dir = os.path.join(config.REPORTS_DIR, subfolder)
    os.makedirs(target_dir, exist_ok=True)  # Ensure subfolder exists
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        if title:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
        f.write(df.to_string(index=False))
    logger.info(f"Table data saved to: {filepath}")


# --- Core Plotting Functions (Generalised) ---


def plot_average_metric_by_time_and_event(
    df: pd.DataFrame,
    metric: str,
    filename_prefix: str,
    title: str,
    y_label: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Plots the average of a given metric by time period and event, and saves extended statistics.
    This function generalizes plot_comment_score_by_time_period and plot_sentiment_score_by_time_period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        metric (str): The name of the column containing the metric to plot (e.g., "comment_score", "compound_sentiment").
        filename_prefix (str): Prefix for the saved filename and text file (e.g., "avg_comment_score").
        title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [metric, "event_name", "time_period"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for '{metric}' plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data to plot average '{metric}' after filtering NaNs. Skipping plot."
        )
        return

    all_periods = ["Before Event", "During Event", "After Event", "Outside Window"]
    all_events = df_filtered["event_name"].unique()

    grouped = (
        df_filtered.groupby(["time_period", "event_name"])[metric]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    grouped["lower_bound"] = grouped["mean"] - grouped["std"]
    grouped["upper_bound"] = grouped["mean"] + grouped["std"]

    # Reindex to ensure all periods/events are present, filling missing with NaN
    idx = pd.MultiIndex.from_product(
        [all_periods, all_events], names=["time_period", "event_name"]
    )
    grouped = (
        grouped.set_index(["time_period", "event_name"]).reindex(idx).reset_index()
    )

    _save_dataframe_as_text(
        grouped,
        f"{output_filename_prefix}{filename_prefix}_by_time_period_and_event.txt",
        f"Average {metric.replace('_', ' ').title()} by Time Period and Event (incl. Range)",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=grouped,
        x="time_period",
        y="mean",
        hue="event_name",
        errorbar="sd",
        palette="viridis",
    )
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=15)
    plt.legend(title="Event")
    plt.tight_layout()

    # Annotate bars with mean values
    for container in ax.containers:
        for patch in container.patches:
            height = patch.get_height()
            if not pd.isna(height):
                ax.annotate(
                    f"{height:.2f}",
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    save_plot(
        fig,
        f"{output_filename_prefix}{filename_prefix}_by_time_period_and_event_extended.png",
        subfolder=output_subfolder,
    )


def plot_metric_by_event_and_type(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    selected_types: list = None,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Plots the average of a given metric by post type and event, and saves extended statistics.
    This function generalizes plot_word_count_by_post_type_and_event and plot_engagement_by_post_type.

    Args:
        df (pd.DataFrame): The input DataFrame.
        metric (str): The name of the column containing the metric to plot (e.g., "word_count", "comment_score").
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        selected_types (list, optional): A list of post types to include. If None, all types are included.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [metric, "event_name", "post_type"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for '{metric}' by post type plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if selected_types:
        df_filtered = df_filtered[df_filtered["post_type"].isin(selected_types)]

    if df_filtered.empty:
        logger.warning(
            f"No valid data to plot average '{metric}' by post type after filtering NaNs/types. Skipping plot."
        )
        return

    post_types = df_filtered["post_type"].unique()
    events = df_filtered["event_name"].unique()

    grouped = (
        df_filtered.groupby(["post_type", "event_name"])[metric]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    grouped["lower_bound"] = grouped["mean"] - grouped["std"]
    grouped["upper_bound"] = grouped["mean"] + grouped["std"]

    # Reindex to ensure all post types/events are present, filling missing with NaN
    idx = pd.MultiIndex.from_product(
        [post_types, events], names=["post_type", "event_name"]
    )
    grouped = grouped.set_index(["post_type", "event_name"]).reindex(idx).reset_index()

    _save_dataframe_as_text(
        grouped,
        f"{output_filename_prefix}{metric}_by_post_type_and_event.txt",
        f"{metric.replace('_', ' ').title()} by Post Type and Event",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=grouped,
        x="post_type",
        y="mean",
        hue="event_name",
        errorbar="sd",
        palette="viridis",
    )
    plt.title(title)
    plt.ylabel(f"Mean {metric.replace('_', ' ').title()}")
    plt.xticks(rotation=20)
    plt.legend(title="Event")
    plt.tight_layout()

    # Annotate bars with mean values
    for container in ax.containers:
        for patch in container.patches:
            height = patch.get_height()
            if not pd.isna(height):
                ax.annotate(
                    f"{height:.2f}",
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    title: str,
    x_label: str,
    bin_count: int = 30,
    filename_prefix: str = "dist_",
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a histogram with KDE for a given feature, and saves summary statistics and KDE data.
    Can also plot by 'event_name' if available.
    This function generalizes plot_comment_score_distribution, plot_sentiment_distribution, and plot_word_count_distribution.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The name of the feature column to plot.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        bin_count (int): The number of bins for the histogram.
        filename_prefix (str): Prefix for the saved filename.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    if feature not in df.columns:
        logger.error(
            f"Feature column '{feature}' not found in DataFrame. Skipping plot."
        )
        return

    values = df[feature].dropna()
    if values.empty:
        logger.warning(
            f"No valid data for feature '{feature}'. Skipping distribution plot."
        )
        return

    mean, std, median, count = (
        values.mean(),
        values.std(),
        values.median(),
        values.count(),
    )

    _save_dataframe_as_text(
        pd.DataFrame.from_dict(
            {
                "Mean": [mean],
                "Median": [median],
                "Std": [std],
                "Count": [count],
                "Lower Bound (Mean - Std)": [mean - std],
                "Upper Bound (Mean + Std)": [mean + std],
            }
        ),
        f"{output_filename_prefix}{filename_prefix}{feature}_summary.txt",
        f"{feature} Distribution Summary",
        subfolder=output_subfolder,
    )

    if len(values) > 1:  # gaussian_kde requires at least 2 data points
        kde = gaussian_kde(values)
        x_vals = np.linspace(values.min(), values.max(), 200)
        kde_vals = kde(x_vals)
        _save_dataframe_as_text(
            pd.DataFrame({"x": x_vals, "kde_density": kde_vals}),
            f"{output_filename_prefix}{filename_prefix}{feature}_kde_curve.txt",
            f"{feature} KDE Curve Data",
            subfolder=output_subfolder,
        )
    else:
        logger.warning(f"Not enough data points to generate KDE curve for '{feature}'.")

    fig = plt.figure(figsize=(10, 6))
    if "event_name" in df.columns and df["event_name"].nunique() > 1:
        df_filtered = df.dropna(subset=[feature, "event_name"]).copy()
        if not df_filtered.empty:
            sns.histplot(
                data=df_filtered,
                x=feature,
                hue="event_name",
                kde=True,
                bins=bin_count,
                palette="viridis",
            )
        else:
            logger.warning(
                f"No valid data for feature '{feature}' with 'event_name' after filtering NaNs. Plotting without hue."
            )
            sns.histplot(values, bins=bin_count, kde=True, color="skyblue")
    else:
        sns.histplot(values, bins=bin_count, kde=True, color="skyblue")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.tight_layout()
    save_plot(
        fig,
        f"{output_filename_prefix}{filename_prefix}{feature}_histogram_extended.png",
        subfolder=output_subfolder,
    )


def plot_distribution_comparison(
    df: pd.DataFrame,
    feature: str,
    title: str,
    y_label: str,
    filename_prefix: str = "dist_comp_",
    plot_type: str = "violin",
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a box plot or violin plot to compare the distribution of a feature
    across different events, and saves grouped statistics.
    This function generalizes plot_comment_score_violin_by_event, plot_sentiment_violin_by_event, and plot_word_count_violin_by_event.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The name of the feature column to plot.
        title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        filename_prefix (str): Prefix for the saved filename.
        plot_type (str): Type of plot to generate ('box' or 'violin').
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [feature, "event_name"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for distribution comparison plot. Skipping."
        )
        return

    data = df.dropna(subset=required_cols).copy()
    if data.empty:
        logger.warning(
            f"No valid data for feature '{feature}' after filtering NaNs. Skipping distribution comparison plot."
        )
        return

    grouped = (
        data.groupby("event_name")[feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    grouped["lower_bound"] = grouped["mean"] - grouped["std"]
    grouped["upper_bound"] = grouped["mean"] + grouped["std"]

    _save_dataframe_as_text(
        grouped,
        f"{output_filename_prefix}{filename_prefix}{feature}_{plot_type}_stats.txt",
        f"{feature} Distribution by Event ({plot_type.title()} Plot Basis)",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(10, 6))
    if plot_type == "violin":
        sns.violinplot(
            data=data,
            x="event_name",
            y=feature,
            inner="quartile",
            palette="viridis",
            hue="event_name",
            legend=False,
        )
    elif plot_type == "box":
        sns.boxplot(
            data=data,
            x="event_name",
            y=feature,
            palette="viridis",
            hue="event_name",
            legend=False,
        )
    else:
        logger.error(
            f"Invalid plot_type '{plot_type}'. Must be 'box' or 'violin'. Skipping plot."
        )
        plt.close(fig)
        return

    plt.title(title)
    plt.xlabel("Event Name")
    plt.ylabel(y_label)
    plt.tight_layout()
    save_plot(
        fig,
        f"{output_filename_prefix}{filename_prefix}{feature}_{plot_type}_extended.png",
        subfolder=output_subfolder,
    )


# --- Specific Plotting Functions (from original and new additions) ---


def plot_posting_behavior_hist(
    df: pd.DataFrame,
    x_label: str,
    hue_label: str,
    title: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a histogram for posting behavior (e.g., comments per hour or day of week).

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        x_label (str): The column name for the x-axis.
        hue_label (str): The column name to use for hue (grouping).
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    if df.empty:
        logger.warning(f"DataFrame is empty, skipping histogram plot for {title}.")
        return
    if x_label not in df.columns or hue_label not in df.columns:
        logger.error(
            f"Missing required columns '{x_label}' or '{hue_label}' for histogram plot. Skipping."
        )
        return

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=x_label,
        hue=hue_label,
        multiple="dodge",
        shrink=0.8,
        palette="viridis",
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_posting_behavior_heatmap(
    df: pd.DataFrame,
    title: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a heatmap for posting behavior.
    The input DataFrame 'df' should be pre-aggregated (e.g., a pivot table of counts).

    Args:
        df (pd.DataFrame): The input DataFrame, pre-aggregated for the heatmap.
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    if df.empty:
        logger.warning(f"DataFrame is empty, skipping heatmap plot for {title}.")
        return

    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_score_development(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    hue_label: str,
    title: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a line plot for the development of scores over time.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_label (str): The column name for the x-axis.
        y_label (str): The column name for the y-axis.
        hue_label (str): The column name to use for hue (grouping).
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    if df.empty:
        logger.warning(
            f"DataFrame is empty, skipping score development plot for {title}."
        )
        return
    if not all(col in df.columns for col in [x_label, y_label, hue_label]):
        logger.error(
            f"Missing one or more required columns ({x_label}, {y_label}, {hue_label}) for score development plot. Skipping."
        )
        return

    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, x=x_label, y=y_label, hue=hue_label, marker="o", palette="viridis"
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_pearson_correlation(
    df: pd.DataFrame,
    filename: str = "pearson_correlation_matrix.png",
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Calculates and visualizes the Pearson correlation matrix for numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    if df.empty:
        logger.warning("DataFrame is empty, skipping Pearson correlation analysis.")
        return

    # Attempt to get numerical columns from config, if available
    numerical_cols_from_config = getattr(config, "NUMERICAL_FEATURES", []) + getattr(
        config, "BOOLEAN_FEATURES", []
    )
    # Add common numerical columns that might not be in config
    additional_numerical_cols = [
        "comment_score",
        "post_score",
        "post_num_comments",
        "upvote_ratio",
        "compound_sentiment",
        "word_count",
    ]

    all_potential_numerical_cols = list(
        set(numerical_cols_from_config + additional_numerical_cols)
    )

    available_numerical_cols = [
        col
        for col in all_potential_numerical_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(available_numerical_cols) < 2:
        logger.warning(
            "Not enough numerical columns available for correlation analysis (need at least 2). Skipping plot."
        )
        return

    logger.info("\n--- Calculating Pearson Correlations ---")
    correlation_matrix = df[available_numerical_cols].corr(method="pearson")
    logger.info("Correlation Matrix:\n%s", correlation_matrix)

    # Save correlation matrix to text file
    _save_dataframe_as_text(
        correlation_matrix,
        f"{output_filename_prefix}pearson_correlation_matrix_data.txt",
        "Pearson Correlation Matrix Data",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Pearson Correlation Matrix of Numerical Features")
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_eventwise_heatmap(
    df: pd.DataFrame,
    index_col: str,
    value_col: str,
    title: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a heatmap showing the mean of a value column across events, indexed by another column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        index_col (str): The column to use as index for the heatmap.
        value_col (str): The column whose mean value will be displayed.
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [index_col, "event_name", value_col]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for event-wise heatmap. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data for event-wise heatmap after filtering NaNs. Skipping plot."
        )
        return

    pivot = df_filtered.pivot_table(
        index=index_col, columns="event_name", values=value_col, aggfunc="mean"
    )

    _save_dataframe_as_text(
        pivot,
        f"{output_filename_prefix}eventwise_heatmap_data.txt",
        f"Event-wise Heatmap Data ({value_col} by {index_col})",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_boxplot_with_stats(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a box plot with statistical grouping.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x (str): The column for the x-axis.
        y (str): The column for the y-axis.
        hue (str): The column for hue grouping.
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [x, y, hue]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for boxplot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data for boxplot after filtering NaNs. Skipping plot."
        )
        return

    # Save grouped statistics for boxplot
    grouped_stats = (
        df_filtered.groupby([x, hue])[y]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    _save_dataframe_as_text(
        grouped_stats,
        f"{output_filename_prefix}boxplot_stats_{y}_by_{x}_{hue}.txt",
        f"Boxplot Statistics: {y} by {x} and {hue}",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x=x, y=y, hue=hue, palette="Set2")
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_event_keyword_distribution(
    df: pd.DataFrame,
    keyword_col: str,
    title: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a heatmap showing the percentage distribution of keywords per event.

    Args:
        df (pd.DataFrame): The input DataFrame.
        keyword_col (str): The column containing keywords.
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = ["event_name", keyword_col]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for keyword distribution plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data for keyword distribution after filtering NaNs. Skipping plot."
        )
        return

    keyword_counts = (
        df_filtered.groupby(["event_name", keyword_col])
        .size()
        .reset_index(name="count")
    )
    pivot = keyword_counts.pivot(
        index=keyword_col, columns="event_name", values="count"
    ).fillna(0)
    pivot_percent = pivot.div(pivot.sum(axis=0), axis=1) * 100  # Calculate percentages

    _save_dataframe_as_text(
        pivot_percent,
        f"{output_filename_prefix}keyword_distribution_percentages.txt",
        "Keyword Distribution Percentages by Event",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_percent, annot=True, fmt=".1f", cmap="Purples")
    plt.title(title)
    plt.ylabel("Keyword")
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_dual_distribution(
    df: pd.DataFrame,
    feature1: str,
    feature2: str,
    title: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Plots the distribution of two features on the same histogram.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature1 (str): The name of the first feature column.
        feature2 (str): The name of the second feature column.
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [feature1, feature2]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for dual distribution plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data for dual distribution after filtering NaNs. Skipping plot."
        )
        return

    fig = plt.figure(figsize=(12, 6))
    sns.histplot(
        df_filtered[feature1], bins=30, kde=True, color="skyblue", label=feature1
    )
    sns.histplot(
        df_filtered[feature2], bins=30, kde=True, color="salmon", label=feature2
    )
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_keyword_score_heatmap(
    df: pd.DataFrame,
    keyword_col: str,
    score_col: str,
    filename: str,
    title: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Generates a heatmap showing the average score for keywords across events.

    Args:
        df (pd.DataFrame): The input DataFrame.
        keyword_col (str): The column containing keywords.
        score_col (str): The column containing the score.
        filename (str): The name of the file to save the plot.
        title (str): The title of the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [keyword_col, "event_name", score_col]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for keyword score heatmap. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data for keyword score heatmap after filtering NaNs. Skipping plot."
        )
        return

    pivot = df_filtered.pivot_table(
        index=keyword_col, columns="event_name", values=score_col, aggfunc="mean"
    ).fillna(0)

    _save_dataframe_as_text(
        pivot,
        f"{output_filename_prefix}keyword_score_heatmap_data.txt",
        f"Average {score_col.replace('_', ' ').title()} by Keyword and Event",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title(title)
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def plot_engagement_per_day(
    df: pd.DataFrame,
    time_col: str,
    engagement_col: str,
    filename: str,
    output_subfolder: str = "",
    output_filename_prefix: str = "",
) -> None:
    """
    Plots the average engagement metric by day of the week and event.

    Args:
        df (pd.DataFrame): The input DataFrame.
        time_col (str): The column containing datetime information.
        engagement_col (str): The column containing the engagement metric.
        filename (str): The name of the file to save the plot.
        output_subfolder (str, optional): Subfolder for saving plots/text files. Defaults to "".
        output_filename_prefix (str, optional): Prefix for the saved plot filename. Defaults to "".
    """
    required_cols = [time_col, engagement_col, "event_name"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for engagement per day plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data for engagement per day after filtering NaNs. Skipping plot."
        )
        return

    df_filtered[time_col] = pd.to_datetime(df_filtered[time_col])
    df_filtered["day_of_week"] = df_filtered[time_col].dt.day_name()

    grouped = (
        df_filtered.groupby(["day_of_week", "event_name"])[engagement_col]
        .mean()
        .reset_index()
    )
    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    _save_dataframe_as_text(
        grouped,
        f"{output_filename_prefix}engagement_per_day_data.txt",
        f"Average {engagement_col.replace('_', ' ').title()} by Day of Week and Event",
        subfolder=output_subfolder,
    )

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(
        data=grouped,
        x="day_of_week",
        y=engagement_col,
        hue="event_name",
        order=order,
        palette="viridis",
    )
    plt.title(
        f"Average {engagement_col.replace('_', ' ').title()} by Day of Week and Event"
    )
    plt.xticks(rotation=20)
    plt.tight_layout()
    save_plot(fig, f"{output_filename_prefix}{filename}", subfolder=output_subfolder)


def generate_all_eda_plots(df: pd.DataFrame) -> None:
    """
    Generates a comprehensive set of EDA plots for the given DataFrame,
    organizing them into specified subfolders based on their relevance to the BA.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data for EDA.
    """
    logger.info("\n--- Generating Exploratory Data Analysis Plots ---")

    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping all EDA plot generation.")
        return

    # Define subfolders
    FINAL_PLOTS_FOLDER = "final_plots"
    OTHER_EDA_PLOTS_FOLDER = "other_eda_plots"
    PEARSON_CORRELATION_FOLDER = "pearson_correlation"

    # --- Distribution of Key Features (Histograms with KDE and Stats) ---
    logger.info("\n--- Generating Feature Distribution Plots ---")
    # dist_comment_score_histogram_extended.png -> final_plots (RQ5)
    plot_feature_distribution(
        df,
        "comment_score",
        "Distribution of Comment Score",
        "Comment Score",
        bin_count=50,
        filename_prefix="dist_comment_score_",
        output_subfolder=FINAL_PLOTS_FOLDER,
        output_filename_prefix="RQ5_",
    )
    plot_feature_distribution(
        df,
        "compound_sentiment",
        "Distribution of Compound Sentiment Score",
        "Compound Sentiment Score (-1 to 1)",
        bin_count=30,
        filename_prefix="dist_sentiment_",
        output_subfolder=OTHER_EDA_PLOTS_FOLDER,
    )
    plot_feature_distribution(
        df,
        "word_count",
        "Distribution of Comment Word Count",
        "Word Count",
        bin_count=50,
        filename_prefix="dist_word_count_",
        output_subfolder=OTHER_EDA_PLOTS_FOLDER,
    )

    # --- Distribution Comparison (Violin/Box Plots) ---
    logger.info("\n--- Generating Distribution Comparison Plots (Violin/Box) ---")
    # dist_comp_word_count_word_count_violin_extended.png -> final_plots (RQ4)
    plot_distribution_comparison(
        df,
        "word_count",
        "Distribution of Comment Word Count Across Events",
        "Word Count",
        plot_type="violin",
        filename_prefix="dist_comp_word_count_",
        output_subfolder=FINAL_PLOTS_FOLDER,
        output_filename_prefix="RQ4_",
    )
    plot_distribution_comparison(
        df,
        "compound_sentiment",
        "Distribution of Compound Sentiment Score Across Events",
        "Compound Sentiment Score (-1 to 1)",
        plot_type="violin",
        filename_prefix="dist_comp_sentiment_",
        output_subfolder=OTHER_EDA_PLOTS_FOLDER,
    )
    plot_distribution_comparison(
        df,
        "comment_score",
        "Distribution of Comment Score Across Events",
        "Comment Score",
        plot_type="violin",
        filename_prefix="dist_comp_comment_score_",
        output_subfolder=OTHER_EDA_PLOTS_FOLDER,
    )

    # --- Comparative Analysis of Time Periods ---
    logger.info("\n--- Generating Time Period Analysis Plots ---")
    if "time_period" in df.columns and "event_name" in df.columns:
        # comment_distribution_time_periods.png -> final_plots (RQ3)
        df_filtered_time_period = df.dropna(subset=["time_period", "event_name"]).copy()
        if not df_filtered_time_period.empty:
            fig = plt.figure(figsize=(12, 7))
            ax = sns.countplot(
                data=df_filtered_time_period,
                x="time_period",
                hue="event_name",
                palette="viridis",
                order=["Before Event", "During Event", "After Event", "Outside Window"],
            )
            plt.title("Comment Distribution Across Time Periods by Event")
            plt.xlabel("Time Period Relative to Event")
            plt.ylabel("Number of Comments")
            plt.legend(title="Event")
            plt.tight_layout()

            for container in ax.containers:
                for patch in container.patches:
                    height = patch.get_height()
                    if height > 0:
                        ax.annotate(
                            f"{int(height)}",
                            xy=(patch.get_x() + patch.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            counts = (
                df_filtered_time_period.groupby(["event_name", "time_period"])
                .size()
                .unstack(fill_value=0)
            )
            percentages = counts.apply(lambda x: x / x.sum() * 100, axis=1)

            combined_table = pd.concat([counts, percentages.add_suffix(" (%)")], axis=1)
            sorted_cols = sorted(
                combined_table.columns, key=lambda x: (x.endswith(" (%)"), x)
            )
            combined_table = combined_table[sorted_cols]

            _save_dataframe_as_text(
                combined_table,
                "RQ3_comment_distribution_time_periods_table.txt",
                "Comment Distribution Across Time Periods by Event (Counts and Percentages)",
                subfolder=FINAL_PLOTS_FOLDER,  # Save text file to final_plots folder
            )
            save_plot(
                fig,
                "RQ3_comment_distribution_time_periods.png",
                subfolder=FINAL_PLOTS_FOLDER,
            )
        else:
            logger.warning(
                "No valid data for 'Comment Distribution Across Time Periods by Event' plot after filtering."
            )
    else:
        logger.warning(
            "Missing 'time_period' or 'event_name' column for 'Comment Distribution Across Time Periods by Event' plot. Skipping."
        )

    # avg_comment_score_by_time_period_and_event_extended.png -> final_plots (RQ3)
    plot_average_metric_by_time_and_event(
        df,
        "comment_score",
        "avg_comment_score",
        "Average Comment Score by Time Period and Event",
        "Average Comment Score",
        output_subfolder=FINAL_PLOTS_FOLDER,
        output_filename_prefix="RQ3_",
    )
    plot_average_metric_by_time_and_event(
        df,
        "compound_sentiment",
        "avg_sentiment",
        "Average Compound Sentiment by Time Period and Event",
        "Average Compound Sentiment Score",
        output_subfolder=OTHER_EDA_PLOTS_FOLDER,
    )

    # --- Engagement by Post Type ---
    logger.info("\n--- Generating Engagement by Post Type Plots ---")
    if "post_type" in df.columns and "event_name" in df.columns:
        # avg_comment_score_by_post_type.png -> final_plots (RQ1)
        plot_metric_by_event_and_type(
            df,
            "comment_score",
            "Average Comment Score by Post Type and Event",
            "avg_comment_score_by_post_type.png",
            output_subfolder=FINAL_PLOTS_FOLDER,
            output_filename_prefix="RQ1_",
        )
        # avg_sentiment_by_post_type.png -> final_plots (RQ1)
        plot_metric_by_event_and_type(
            df,
            "compound_sentiment",
            "Average Compound Sentiment by Post Type and Event",
            "avg_sentiment_by_post_type.png",
            output_subfolder=FINAL_PLOTS_FOLDER,
            output_filename_prefix="RQ1_",
        )
        plot_metric_by_event_and_type(
            df,
            "word_count",
            "Average Word Count by Post Type and Event",
            "avg_word_count_by_post_type.png",
            output_subfolder=OTHER_EDA_PLOTS_FOLDER,  # Save text file to other_eda_plots folder
        )
        # filtered_avg_word_count_by_post_type_and_event.png -> final_plots (RQ2)
        plot_metric_by_event_and_type(
            df,
            "word_count",
            "Average Word Count by Selected Post Types and Event",
            "filtered_avg_word_count_by_post_type_and_event.png",
            selected_types=[
                "Player Transfer",
                "Other",
                "Tournament Result",
                "Ranking Update",
            ],
            output_subfolder=FINAL_PLOTS_FOLDER,
            output_filename_prefix="RQ2_",
        )
    else:
        logger.warning(
            "Missing 'post_type' or 'event_name' column for 'Engagement by Post Type' plots. Skipping."
        )

    # --- Pearson Correlation Matrix ---
    logger.info("\n--- Generating Pearson Correlation Matrix ---")
    # pearson_correlation_matrix.png -> pearson_correlation (RQ5)
    plot_pearson_correlation(
        df,
        filename="pearson_correlation_matrix.png",
        output_subfolder=PEARSON_CORRELATION_FOLDER,  # Save text file to pearson_correlation folder
        output_filename_prefix="RQ5_",
    )

    # --- Additional Plots (if data available) ---
    logger.info("\n--- Generating Additional Specific Plots ---")
    # Check for 'created_utc' and 'event_name' for time-based plots
    if "created_utc" in df.columns and "event_name" in df.columns:
        plot_engagement_per_day(
            df,
            "created_utc",
            "comment_score",
            "avg_comment_score_by_day_of_week.png",
            output_subfolder=OTHER_EDA_PLOTS_FOLDER,
        )
        plot_engagement_per_day(
            df,
            "created_utc",
            "compound_sentiment",
            "avg_sentiment_by_day_of_week.png",
            output_subfolder=OTHER_EDA_PLOTS_FOLDER,
        )

    logger.info("\n--- All EDA Plots generated. ---")
