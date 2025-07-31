import logging
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")

# Ensure FIGURES_DIR and REPORTS_DIR exist
os.makedirs(config.FIGURES_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)  # Ensure REPORTS_DIR exists for tables


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Helper function to save plots with consistent settings."""
    filepath = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    logger.info(f"Plot saved to: {filepath}")


def _save_dataframe_as_text(
    df_to_save: pd.DataFrame, filename: str, title: str = ""
) -> None:
    """Helper function to save a DataFrame to a text file."""
    filepath = os.path.join(config.REPORTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        if title:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
        f.write(df_to_save.to_string())
    logger.info(f"Table data saved to: {filepath}")


def plot_posting_behavior_hist(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    hue_label: str,
    title: str,
    filename: str = "posting_behavior_hist.png",
) -> None:
    """
    Generates a histogram for posting behavior (e.g., comments per hour or day of week).

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        x_label (str): The column name for the x-axis.
        y_label (str): The label for the y-axis.
        hue_label (str): The column name to use for hue (grouping).
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
    """
    if df.empty:
        logger.warning(f"DataFrame is empty, skipping histogram plot for {title}.")
        return
    if x_label not in df.columns or hue_label not in df.columns:
        logger.error(
            f"Missing required columns '{x_label}' or '{hue_label}' for histogram plot. Skipping."
        )
        return

    fig = plt.figure(figsize=(12, 7))
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
    plt.ylabel(y_label)
    plt.tight_layout()
    save_plot(fig, filename)


def plot_posting_behavior_heatmap(
    df: pd.DataFrame,
    title: str,
    filename: str = "posting_behavior_heatmap.png",
) -> None:
    """
    Generates a heatmap for posting behavior.
    The input DataFrame 'df' should be pre-aggregated (e.g., a pivot table of counts).

    Args:
        df (pd.DataFrame): The input DataFrame, pre-aggregated for the heatmap.
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot.
    """
    if df.empty:
        logger.warning(f"DataFrame is empty, skipping heatmap plot for {title}.")
        return

    fig = plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis")
    plt.title(title)
    plt.xlabel(df.columns.name if df.columns.name else "X-axis")
    plt.ylabel(df.index.name if df.index.name else "Y-axis")
    plt.tight_layout()
    save_plot(fig, filename)


def plot_score_development(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    hue_label: str,
    filename: str = "score_development.png",
) -> None:
    """
    Generates a line plot for the development of scores over time.

    Args:
        df (pd.DataFrame): The input DataFrame.
        title (str): The title of the plot.
        x_label (str): The column name for the x-axis.
        y_label (str): The column name for the y-axis.
        hue_label (str): The column name to use for hue (grouping).
        filename (str): The name of the file to save the plot.
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

    fig = plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df, x=x_label, y=y_label, hue=hue_label, marker="o", palette="viridis"
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    save_plot(fig, filename)


def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    title: str,
    x_label: str,
    bin_count: int = 30,
    filename_prefix: str = "feature_distribution_",
) -> None:
    """
    Generates a histogram for the distribution of a given feature, separated by event name.
    KDE is set to True for a smoother representation of the distribution shape.
    Filters out event_name groups that have no valid data for the feature.
    Displays quantitative metrics.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The name of the feature column to plot.
        title (str): The title of the plot.
        x_label (str): The label for the x-axis.
        bin_count (int): The number of bins for the histogram.
        filename_prefix (str): Prefix for the saved filename.
    """
    if df.empty:
        logger.warning(
            f"DataFrame is empty, skipping feature distribution plot for {feature}."
        )
        return
    if feature not in df.columns:
        logger.error(
            f"Feature column '{feature}' not found in DataFrame. Skipping plot."
        )
        return

    if "event_name" not in df.columns or df["event_name"].nunique() < 2:
        logger.info(
            f"Either 'event_name' column is missing or has less than 2 unique events. Plotting '{feature}' distribution without hue."
        )
        df_filtered = df.dropna(subset=[feature]).copy()
        if df_filtered.empty:
            logger.warning(
                f"No valid data to plot for feature '{feature}' after filtering NaNs. Skipping plot."
            )
            return
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_filtered,
            x=feature,
            kde=True,
            bins=bin_count,
            color="skyblue",  # Correctly uses 'color' when no hue
        )
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel("Count")
        plt.tight_layout()
        save_plot(fig, f"{filename_prefix}{feature}.png")
        return

    df_filtered = df.dropna(subset=[feature, "event_name"]).copy()

    if df_filtered.empty:
        logger.warning(
            f"No valid data to plot for feature '{feature}' after filtering NaNs. Skipping plot."
        )
        return

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_filtered,
        x=feature,
        hue="event_name",
        kde=True,
        bins=bin_count,
        palette="viridis",
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.tight_layout()

    stats_text = ""
    for event in df_filtered["event_name"].unique():
        subset = df_filtered[df_filtered["event_name"] == event][feature]
        if not subset.empty:
            mean_val = subset.mean()
            median_val = subset.median()
            std_val = subset.std()
            stats_text += f"{event}:\n  Mean: {mean_val:.2f}\n  Median: {median_val:.2f}\n  Std: {std_val:.2f}\n"

    if stats_text:
        plt.figtext(
            0.95,
            0.7,
            stats_text,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )
        fig.subplots_adjust(right=0.75)

    save_plot(fig, f"{filename_prefix}{feature}.png")


def plot_sentiment_score_by_time_period(
    df: pd.DataFrame,
    title_suffix: str = "",
    filename: str = "avg_sentiment_by_time_period.png",
) -> None:
    """
    Plots the average compound sentiment score by time period and event.
    Displays quantitative metrics and annotates bar heights.
    """
    required_cols = ["compound_sentiment", "event_name", "time_period"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for sentiment plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data to plot average sentiment after filtering NaNs. Skipping plot."
        )
        return

    fig = plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df_filtered,
        x="time_period",
        y="compound_sentiment",
        hue="event_name",
        palette="viridis",
        order=["Before Event", "During Event", "After Event", "Outside Window"],
        errorbar="sd",
    )
    plt.title(f"Average Compound Sentiment by Time Period and Event {title_suffix}")
    plt.xlabel("Time Period Relative to Event")
    plt.ylabel("Average Compound Sentiment Score")
    plt.legend(title="Event")
    plt.tight_layout()

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

    stats_text = ""
    grouped_stats = (
        df_filtered.groupby(["event_name", "time_period"])["compound_sentiment"]
        .agg(["mean", "std"])
        .unstack(level=0)
    )
    for period in ["Before Event", "During Event", "After Event", "Outside Window"]:
        if period in grouped_stats.index:
            stats_text += f"{period}:\n"
            for event in grouped_stats.columns.levels[1]:
                mean_val = grouped_stats.loc[period, ("mean", event)]
                std_val = grouped_stats.loc[period, ("std", event)]
                if not pd.isna(mean_val):
                    stats_text += f"  {event}: Mean={mean_val:.2f}, Std={std_val:.2f}\n"

    if stats_text:
        plt.figtext(
            0.95,
            0.7,
            stats_text,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )
        fig.subplots_adjust(right=0.75)

    save_plot(fig, filename)


def plot_comment_score_by_time_period(
    df: pd.DataFrame,
    title_suffix: str = "",
    filename: str = "avg_comment_score_by_time_period.png",
) -> None:
    """
    Plots the average comment score by time period and event.
    Displays quantitative metrics and annotates bar heights.
    """
    required_cols = ["comment_score", "event_name", "time_period"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for comment score plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data to plot average comment score after filtering NaNs. Skipping plot."
        )
        return

    fig = plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df_filtered,
        x="time_period",
        y="comment_score",
        hue="event_name",
        palette="viridis",
        order=["Before Event", "During Event", "After Event", "Outside Window"],
        errorbar="sd",
    )
    plt.title(f"Average Comment Score by Time Period and Event {title_suffix}")
    plt.xlabel("Time Period Relative to Event")
    plt.ylabel("Average Comment Score")
    plt.legend(title="Event")
    plt.tight_layout()

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

    stats_text = ""
    grouped_stats = (
        df_filtered.groupby(["event_name", "time_period"])["comment_score"]
        .agg(["mean", "std"])
        .unstack(level=0)
    )
    for period in ["Before Event", "During Event", "After Event", "Outside Window"]:
        if period in grouped_stats.index:
            stats_text += f"{period}:\n"
            for event in grouped_stats.columns.levels[1]:
                mean_val = grouped_stats.loc[period, ("mean", event)]
                std_val = grouped_stats.loc[period, ("std", event)]
                if not pd.isna(mean_val):
                    stats_text += f"  {event}: Mean={mean_val:.2f}, Std={std_val:.2f}\n"

    if stats_text:
        plt.figtext(
            0.95,
            0.7,
            stats_text,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )
        fig.subplots_adjust(right=0.75)

    save_plot(fig, filename)


def plot_pearson_correlation(
    df: pd.DataFrame, filename: str = "pearson_correlation_matrix.png"
) -> None:
    """
    Calculates and visualizes the Pearson correlation matrix for numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        filename (str): The name of the file to save the plot.
    """
    if df.empty:
        logger.warning("DataFrame is empty, skipping Pearson correlation analysis.")
        return

    numerical_cols_from_config = getattr(config, "NUMERICAL_FEATURES", []) + getattr(
        config, "BOOLEAN_FEATURES", []
    )
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

    logger.info("\n--- Calculating Pearson Correlations for Comments Data ---")
    correlation_matrix = df[available_numerical_cols].corr(method="pearson")
    logger.info("Correlation Matrix:\n%s", correlation_matrix)

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Pearson Correlation Matrix of Numerical Features (Comments)")
    plt.tight_layout()
    save_plot(fig, filename)


def plot_engagement_by_post_type(
    df: pd.DataFrame,
    engagement_metric: str,
    title: str,
    y_label: str,
    filename: str,
) -> None:
    """
    Plots the average engagement metric (e.g., comment score, word count)
    by post type and event.

    Args:
        df (pd.DataFrame): The input DataFrame.
        engagement_metric (str): The column name of the engagement metric to plot.
        title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        filename (str): The name of the file to save the plot.
    """
    required_cols = [engagement_metric, "event_name", "post_type"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for post type engagement plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data to plot engagement by post type for '{engagement_metric}' after filtering NaNs. Skipping plot."
        )
        return

    fig = plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df_filtered,
        x="post_type",
        y=engagement_metric,
        hue="event_name",
        palette="viridis",
        errorbar="sd",
    )
    plt.title(title)
    plt.xlabel("Post Type")
    plt.ylabel(y_label)
    plt.legend(title="Event")
    plt.tight_layout()

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

    stats_text = ""
    grouped_stats = (
        df_filtered.groupby(["event_name", "post_type"])[engagement_metric]
        .agg(["mean", "std"])
        .unstack(level=0)
    )
    for post_type in df_filtered["post_type"].unique():
        if post_type in grouped_stats.index:
            stats_text += f"{post_type}:\n"
            for event in grouped_stats.columns.levels[1]:
                mean_val = grouped_stats.loc[post_type, ("mean", event)]
                std_val = grouped_stats.loc[post_type, ("std", event)]
                if not pd.isna(mean_val):
                    stats_text += f"  {event}: Mean={mean_val:.2f}, Std={std_val:.2f}\n"

    if stats_text:
        plt.figtext(
            0.95,
            0.7,
            stats_text,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )
        fig.subplots_adjust(right=0.75)

    save_plot(fig, filename)


def plot_distribution_comparison(
    df: pd.DataFrame,
    feature: str,
    title: str,
    y_label: str,
    filename_prefix: str = "distribution_comparison_",
    plot_type: str = "violin",  # Can be 'box' or 'violin'
) -> None:
    """
    Generates a box plot or violin plot to compare the distribution of a feature
    across different events.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The name of the feature column to plot.
        title (str): The title of the plot.
        y_label (str): The label for the y-axis.
        filename_prefix (str): Prefix for the saved filename.
        plot_type (str): Type of plot to generate ('box' or 'violin').
    """
    required_cols = [feature, "event_name"]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing one or more required columns ({required_cols}) for distribution comparison plot. Skipping."
        )
        return

    df_filtered = df.dropna(subset=required_cols).copy()
    if df_filtered.empty:
        logger.warning(
            f"No valid data to plot distribution comparison for '{feature}' after filtering NaNs. Skipping plot."
        )
        return

    fig = plt.figure(figsize=(12, 7))
    if plot_type == "box":
        sns.boxplot(
            data=df_filtered,
            x="event_name",
            y=feature,
            hue="event_name",
            palette="viridis",
            legend=False,
        )
    elif plot_type == "violin":
        sns.violinplot(
            data=df_filtered,
            x="event_name",
            y=feature,
            hue="event_name",
            palette="viridis",
            inner="quartile",  # Shows quartiles inside the violin
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
    save_plot(fig, f"{filename_prefix}{feature}_{plot_type}.png")


def generate_all_eda_plots(df: pd.DataFrame) -> None:
    """
    Generates a comprehensive set of EDA plots for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data for EDA.
    """
    logger.info("\n--- Generating Exploratory Data Analysis Plots ---")

    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping all EDA plot generation.")
        return

    # Distribution of Key Features (with KDE for smoother shape)
    plot_feature_distribution(
        df,
        "compound_sentiment",
        "Distribution of Compound Sentiment Score by Event",
        "Compound Sentiment Score (-1 to 1)",
        bin_count=30,
        filename_prefix="dist_sentiment_",
    )
    plot_feature_distribution(
        df,
        "comment_score",
        "Distribution of Comment Score by Event",
        "Comment Score",
        bin_count=50,
        filename_prefix="dist_comment_score_",
    )
    plot_feature_distribution(
        df,
        "word_count",
        "Distribution of Comment Word Count by Event",
        "Word Count",
        bin_count=50,
        filename_prefix="dist_word_count_",
    )

    # NEW: Distribution Comparison (Box/Violin Plots) for key features
    logger.info("\n--- Generating Distribution Comparison Plots (Box/Violin) ---")
    plot_distribution_comparison(
        df,
        "compound_sentiment",
        "Distribution of Compound Sentiment Score Across Events",
        "Compound Sentiment Score (-1 to 1)",
        plot_type="violin",
        filename_prefix="dist_comp_sentiment_",
    )
    plot_distribution_comparison(
        df,
        "comment_score",
        "Distribution of Comment Score Across Events",
        "Comment Score",
        plot_type="violin",
        filename_prefix="dist_comp_comment_score_",
    )
    plot_distribution_comparison(
        df,
        "word_count",
        "Distribution of Comment Word Count Across Events",
        "Word Count",
        plot_type="violin",
        filename_prefix="dist_comp_word_count_",
    )
    # You can also generate box plots if preferred:
    # plot_distribution_comparison(df, "compound_sentiment", "Distribution of Compound Sentiment Score Across Events (Box Plot)", "Compound Sentiment Score (-1 to 1)", plot_type="box", filename_prefix="dist_comp_sentiment_")

    # Comparative Analysis of Time Periods
    if "time_period" in df.columns and "event_name" in df.columns:
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
                "comment_distribution_time_periods_table.txt",
                "Comment Distribution Across Time Periods by Event (Counts and Percentages)",
            )

            save_plot(fig, "comment_distribution_time_periods.png")
        else:
            logger.warning(
                "No valid data for 'Comment Distribution Across Time Periods by Event' plot after filtering."
            )
    else:
        logger.warning(
            "Missing 'time_period' or 'event_name' column for 'Comment Distribution Across Time Periods by Event' plot. Skipping."
        )

    # Average Sentiment and Score by Time Period
    df_filtered_sentiment = df.dropna(
        subset=["compound_sentiment", "event_name", "time_period"]
    ).copy()
    if not df_filtered_sentiment.empty:
        grouped_stats_sentiment = (
            df_filtered_sentiment.groupby(["event_name", "time_period"])[
                "compound_sentiment"
            ]
            .agg(["mean", "std"])
            .unstack(level=0)
        )
        _save_dataframe_as_text(
            grouped_stats_sentiment,
            "avg_sentiment_by_time_period_table.txt",
            "Average Compound Sentiment by Time Period and Event (Mean and Std)",
        )
    plot_sentiment_score_by_time_period(
        df, "Combined Events", filename="avg_sentiment_by_time_period.png"
    )

    df_filtered_score = df.dropna(
        subset=["comment_score", "event_name", "time_period"]
    ).copy()
    if not df_filtered_score.empty:
        grouped_stats_score = (
            df_filtered_score.groupby(["event_name", "time_period"])["comment_score"]
            .agg(["mean", "std"])
            .unstack(level=0)
        )
        _save_dataframe_as_text(
            grouped_stats_score,
            "avg_comment_score_by_time_period_table.txt",
            "Average Comment Score by Time Period and Event (Mean and Std)",
        )
    plot_comment_score_by_time_period(
        df, "Combined Events", filename="avg_comment_score_by_time_period.png"
    )

    # NEW: Engagement by Post Type
    if "post_type" in df.columns and "event_name" in df.columns:
        plot_engagement_by_post_type(
            df,
            "comment_score",
            "Average Comment Score by Post Type and Event",
            "Average Comment Score",
            "avg_comment_score_by_post_type.png",
        )
        plot_engagement_by_post_type(
            df,
            "compound_sentiment",
            "Average Compound Sentiment by Post Type and Event",
            "Average Compound Sentiment",
            "avg_sentiment_by_post_type.png",
        )
        plot_engagement_by_post_type(
            df,
            "word_count",
            "Average Word Count by Post Type and Event",
            "Average Word Count",
            "avg_word_count_by_post_type.png",
        )
    else:
        logger.warning(
            "Missing 'post_type' or 'event_name' column for 'Engagement by Post Type' plots. Skipping."
        )

    # Pearson Correlation Matrix
    plot_pearson_correlation(df)

    logger.info("\n--- All EDA Plots Generated ---")
