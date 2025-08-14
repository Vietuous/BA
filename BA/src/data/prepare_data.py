import datetime
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import praw
from dotenv import load_dotenv

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config
from config import POST_EVENT_DAYS, PRE_EVENT_DAYS, enrich_tournament_configs

# Tournament-Zeiträume erweitern
TOURNAMENT_CONFIGS = enrich_tournament_configs(
    config.TOURNAMENT_CONFIGS, PRE_EVENT_DAYS, POST_EVENT_DAYS
)

# Get a logger instance for this module
logger = logging.getLogger("data.pipeline")


# Import custom modules from src/
try:
    from BA.src.data.database_utils import (
        check_db_exists_and_has_data,
        load_data_from_sqlite,
        save_data_to_sqlite,
    )
    from BA.src.data.preprocess import (
        calculate_days_from_event_start,
        categorize_time_period,
        filter_deleted_and_empty_processed_comments,
        initial_clean_dataframe,
    )
    from BA.src.data.reddit_scraper import get_posts_and_comments, get_reddit_instance
    from BA.src.features.feature_engineering import (
        add_event_name,
        calculate_comment_score_per_day,
        calculate_post_title_features,
        categorize_post_type,
        extract_time_features,
    )
    from BA.src.features.text_features import (
        calculate_text_length,
        contains_any_keyword,
        contains_question,
        get_sentiment_scores,
        preprocess_text,
    )
    from BA.src.utils.config_loader import get_dota2_teams, get_keywords
    from BA.src.utils.logging_utils import setup_logger
except ImportError as e:
    logger.error(
        f"Failed to import custom modules. Please check your PYTHONPATH and module paths: {e}"
    )
    sys.exit(1)


def prepare_data() -> pd.DataFrame:
    """
    Orchestrates the entire data preparation pipeline:
    1. Checks for existing processed data in SQLite.
    2. If not found, collects raw data from Reddit.
    3. Performs initial cleaning and preprocessing.
    4. Extracts text features (sentiment, length).
    5. Categorizes comments by time period relative to events.
    6. Performs feature engineering (post title features, keyword presence, ratios, post type).
    7. Combines data from all events.
    8. Saves the final processed DataFrame to an SQLite database.

    Returns:
        pd.DataFrame: The combined and processed DataFrame. Returns an empty DataFrame
                      if the pipeline fails or no data is collected/loaded.
    """
    logger.info("--- Starting Data Preparation Pipeline ---")

    df_combined_cleaned = pd.DataFrame()

    db_name = getattr(config, "DATABASE_NAME", "reddit_data.db")
    table_name = getattr(config, "TABLE_NAME", "comments")
    reddit_subreddit_name = getattr(config, "REDDIT_SUBREDDIT_NAME", "dota2")
    reddit_post_limit = getattr(config, "REDDIT_POST_LIMIT", 1000)
    reddit_comment_limit = getattr(config, "REDDIT_COMMENT_LIMIT", 50)
    reddit_max_posts_to_process = getattr(config, "REDDIT_MAX_POSTS_TO_PROCESS", 15)
    pre_post_window_days = getattr(config, "PRE_POST_WINDOW_DAYS", 7)
    tournament_configs = getattr(config, "TOURNAMENT_CONFIGS", {})

    if check_db_exists_and_has_data():
        logger.info(
            f"\nProcessed data found in '{db_name}'. Loading data directly from database."
        )
        df_comments_loaded = load_data_from_sqlite()
        if not df_comments_loaded.empty:
            df_combined_cleaned = df_comments_loaded
            logger.info(f"Loaded {len(df_combined_cleaned)} rows from database.")
        else:
            logger.warning(
                "Database exists but returned an empty DataFrame. Proceeding with data collection."
            )
    else:
        logger.info(
            f"\nNo processed data found in '{db_name}'. Proceeding with data collection and processing."
        )

        try:
            reddit = get_reddit_instance()
            subreddit = reddit.subreddit(reddit_subreddit_name)
            logger.info(
                f"Reddit instance initialized for subreddit: {reddit_subreddit_name}."
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize Reddit instance or access subreddit: {e}"
            )
            logger.info("--- Data Preparation Pipeline Aborted ---")
            return pd.DataFrame()

        logger.info("\n--- DATENSAMMLUNG ---")
        all_dfs: List[pd.DataFrame] = []
        if not tournament_configs:
            logger.warning(
                "No tournament configurations found in config.TOURNAMENT_CONFIGS. Skipping data collection."
            )

        for event_key, event_params in tournament_configs.items():
            event_name = event_params.get("event_name", event_key)
            query = event_params.get("query")
            start_date = event_params.get("start_date")
            end_date = event_params.get("end_date")

            if not all([query, start_date, end_date]):
                logger.warning(
                    f"Skipping event '{event_name}' due to missing query, start_date, or end_date in config."
                )
                continue

            logger.info(f"\nCollecting data for {event_name} (Query: '{query}')...")
            try:
                comments = get_posts_and_comments(
                    subreddit,
                    query=query,
                    start_date=start_date,
                    end_date=end_date,
                    post_limit=reddit_post_limit,
                    comment_limit=reddit_comment_limit,
                    max_posts_to_process=reddit_max_posts_to_process,
                )
                df = pd.DataFrame(comments)
                if df.empty:
                    logger.warning(
                        f"No comments collected for {event_name}. Skipping further processing for this event."
                    )
                    continue
                logger.info(
                    f"Successfully collected {df.shape[0]} comments for {event_name}."
                )
            except Exception as e:
                logger.error(
                    f"Error collecting data for {event_name}: {e}. Skipping this event."
                )
                continue

            logger.info(f"--- Cleaning and Preprocessing for {event_name} ---")
            df = initial_clean_dataframe(df.copy(), f"{event_name} Comments")
            if df.empty:
                logger.warning(
                    f"DataFrame is empty after initial cleaning for {event_name}. Skipping further processing for this event."
                )
                continue

            if "comment_body" in df.columns:
                df["processed_comment_body"] = df["comment_body"].apply(preprocess_text)
            else:
                logger.warning(
                    f"'comment_body' column not found for {event_name}. Skipping text preprocessing."
                )
                df["processed_comment_body"] = ""

            df = filter_deleted_and_empty_processed_comments(
                df, f"{event_name} Comments"
            )
            if df.empty:
                logger.warning(
                    f"DataFrame is empty after filtering deleted/empty comments for {event_name}. Skipping further processing for this event."
                )
                continue

            logger.info("Performing sentiment analysis and calculating text length...")
            if (
                "processed_comment_body" in df.columns
                and not df["processed_comment_body"].empty
            ):
                df["sentiment_scores"] = df["processed_comment_body"].apply(
                    get_sentiment_scores
                )
                df["neg_sentiment"] = df["sentiment_scores"].apply(lambda x: x["neg"])
                df["neu_sentiment"] = df["sentiment_scores"].apply(lambda x: x["neu"])
                df["pos_sentiment"] = df["sentiment_scores"].apply(lambda x: x["pos"])
                df["compound_sentiment"] = df["sentiment_scores"].apply(
                    lambda x: x["compound"]
                )
            else:
                logger.warning(
                    f"No 'processed_comment_body' for sentiment analysis in {event_name}. Filling with NaNs."
                )
                df["sentiment_scores"] = None
                df["neg_sentiment"] = np.nan
                df["neu_sentiment"] = np.nan
                df["pos_sentiment"] = np.nan
                df["compound_sentiment"] = np.nan

            if "comment_body" in df.columns:
                df[["char_count", "word_count"]] = (
                    df["comment_body"].apply(calculate_text_length).apply(pd.Series)
                )
            else:
                logger.warning(
                    f"No 'comment_body' for text length calculation in {event_name}. Filling with NaNs."
                )
                df["char_count"] = np.nan
                df["word_count"] = np.nan

            # Time Period Marking and Time Difference
            logger.info(
                "Categorizing time periods and calculating days from event start..."
            )

            if "comment_created_utc" in df.columns:

                df["time_period"] = df.apply(
                    lambda row: categorize_time_period(
                        row["comment_created_utc"],
                        event_params["start_date"],
                        event_params["end_date"],
                        event_params[
                            "pre_event_start"
                        ],  # Calculated start date of the pre-event window
                        event_params[
                            "post_event_end"
                        ],  # Calculated end date of the post-event window
                    ),
                    axis=1,
                )

                df["days_from_event_start"] = df.apply(
                    lambda row: calculate_days_from_event_start(
                        row["comment_created_utc"], start_date
                    ),
                    axis=1,
                )
            else:
                logger.warning(
                    f"No 'comment_created_utc' for time period categorization in {event_name}. Filling with NaNs."
                )
                df["time_period"] = "Unknown"
                df["days_from_event_start"] = np.nan

            # Feature Engineering
            logger.info("--- Feature Engineering ---")
            df = calculate_post_title_features(df.copy())
            df["contains_question"] = df["comment_body"].apply(contains_question)
            df["author_karma"] = (
                0  # Placeholder: Implement actual karma fetching if needed
            )

            dota2_teams = get_dota2_teams()
            keywords_data = get_keywords()
            player_keywords = keywords_data.get("player_keywords", [])
            hero_keywords = keywords_data.get("hero_keywords", [])
            tournament_event_keywords = keywords_data.get(
                "tournament_event_keywords", []
            )

            if "comment_body" in df.columns:
                df["contains_team_name"] = df["comment_body"].apply(
                    lambda x: contains_any_keyword(x, dota2_teams)
                )
            else:
                df["contains_team_name"] = False

            if "processed_comment_body" in df.columns:
                df["contains_player_keyword"] = df["processed_comment_body"].apply(
                    lambda x: contains_any_keyword(x, player_keywords)
                )
                df["contains_hero_keyword"] = df["processed_comment_body"].apply(
                    lambda x: contains_any_keyword(x, hero_keywords)
                )
                df["contains_event_keyword"] = df["processed_comment_body"].apply(
                    lambda x: contains_any_keyword(x, tournament_event_keywords)
                )
            else:
                df["contains_player_keyword"] = False
                df["contains_hero_keyword"] = False
                df["contains_event_keyword"] = False

            # Categorize post type
            df = categorize_post_type(
                df.copy()
            )  # Pass a copy to avoid SettingWithCopyWarning

            # Handle potential division by zero for ratios
            df["comment_to_post_score_ratio"] = df.apply(
                lambda row: (
                    row["comment_score"] / (row["post_score"] + 1)
                    if pd.notna(row["post_score"]) and row["post_score"] != 0
                    else row["comment_score"] if pd.notna(row["comment_score"]) else 0
                ),
                axis=1,
            )
            df["comment_score_per_day"] = df.apply(
                calculate_comment_score_per_day, axis=1
            )

            df = add_event_name(df.copy(), event_name)
            df = extract_time_features(df.copy())

            all_dfs.append(df)

        # Data Storage
        if all_dfs:
            logger.info("\n--- Saving processed data to SQLite database ---")
            df_combined_cleaned = pd.concat(all_dfs, ignore_index=True)

            # Remove 'sentiment_scores' column before saving, as SQLite cannot store dictionaries
            if "sentiment_scores" in df_combined_cleaned.columns:
                df_combined_cleaned.drop(columns=["sentiment_scores"], inplace=True)

            save_data_to_sqlite(df_combined_cleaned)
            logger.info("\n--- Data storage in SQLite complete ---")
        else:
            logger.warning(
                "No DataFrames were collected or processed. Skipping SQLite save."
            )

    logger.info("--- Data Preparation Pipeline Completed ---")
    return df_combined_cleaned


if __name__ == "__main__":
    setup_logger("data.pipeline", config.LOG_FILES["data.pipeline"], config.LOG_LEVEL)
    logger.info("Running prepare_data.py as a standalone script...")
    prepared_data = prepare_data()
    if not prepared_data.empty:
        logger.info(
            f"Data preparation successful. Prepared DataFrame shape: {prepared_data.shape}"
        )
    else:
        logger.info("Data preparation resulted in an empty DataFrame.")
