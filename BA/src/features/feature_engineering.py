import datetime
import logging
import os
import re  # Import re for regex operations
import sys
from typing import Any, List, Union

import pandas as pd

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config
from BA.src.features.text_features import (
    contains_any_keyword,  # Import contains_any_keyword (for other uses)
)
from BA.src.utils.config_loader import get_keywords  # Import get_keywords

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def calculate_post_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates 'post_title_length' and 'post_title_word_count' for a DataFrame
    based on the 'post_title' column.
    """
    if "post_title" not in df.columns:
        logger.warning(
            "Column 'post_title' not found. Skipping post title feature calculation."
        )
        df["post_title_length"] = 0
        df["post_title_word_count"] = 0
        return df

    df["post_title"] = df["post_title"].astype(str).fillna("")

    df["post_title_length"] = df["post_title"].apply(len)
    df["post_title_word_count"] = df["post_title"].apply(lambda x: len(x.split()))
    logger.info("Calculated 'post_title_length' and 'post_title_word_count'.")
    return df


def calculate_comment_score_per_day(row: pd.Series) -> Union[float, int]:
    """
    Calculates 'comment_score_per_day' based on comment score and time since post creation.
    If the time difference is zero or negative, returns the original comment score.
    """
    if (
        "comment_created_utc" not in row
        or "post_created_utc" not in row
        or "comment_score" not in row
    ):
        logger.warning(
            "Missing 'comment_created_utc', 'post_created_utc', or 'comment_score' in row. Cannot calculate comment score per day."
        )
        return 0

    comment_time = row["comment_created_utc"]
    post_time = row["post_created_utc"]
    comment_score = row["comment_score"]

    if not isinstance(comment_time, datetime.datetime) or not isinstance(
        post_time, datetime.datetime
    ):
        logger.warning(
            f"Invalid datetime objects for comment_created_utc ({type(comment_time)}) or post_created_utc ({type(post_time)}). Cannot calculate comment score per day."
        )
        return comment_score

    time_since_post_creation = (comment_time - post_time).total_seconds()

    if time_since_post_creation <= 0:
        return comment_score

    days_since_post_creation = time_since_post_creation / (24 * 3600)

    if days_since_post_creation > 0:
        return comment_score / days_since_post_creation
    else:
        return comment_score


def add_event_name(df: pd.DataFrame, event_name: str) -> pd.DataFrame:
    """
    Adds an 'event_name' column to the DataFrame with the specified event name.
    """
    df["event_name"] = event_name
    logger.info(f"Added 'event_name' column with value: {event_name}.")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features ('comment_hour', 'comment_day_of_week')
    from the 'comment_created_utc' column.
    """
    if "comment_created_utc" not in df.columns:
        logger.error(
            "'comment_created_utc' column not found. Cannot extract time features."
        )
        df["comment_hour"] = None
        df["comment_day_of_week"] = None
        return df

    if not pd.api.types.is_datetime64_any_dtype(df["comment_created_utc"]):
        logger.info("Converting 'comment_created_utc' to datetime objects.")
        if pd.api.types.is_numeric_dtype(df["comment_created_utc"]):
            df["comment_created_utc"] = pd.to_datetime(
                df["comment_created_utc"], unit="s", errors="coerce"
            )
        else:
            df["comment_created_utc"] = pd.to_datetime(
                df["comment_created_utc"], errors="coerce"
            )

    df.dropna(subset=["comment_created_utc"], inplace=True)

    if df.empty:
        logger.warning(
            "DataFrame is empty after converting 'comment_created_utc' and dropping NaNs. No time features extracted."
        )
        df["comment_hour"] = None
        df["comment_day_of_week"] = None
        return df

    if df["comment_created_utc"].dt.tz is not None:
        logger.info(
            "Converting 'comment_created_utc' to UTC and removing timezone information."
        )
        df["comment_created_utc"] = (
            df["comment_created_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        )

    df["comment_hour"] = df["comment_created_utc"].dt.hour
    df["comment_day_of_week"] = df["comment_created_utc"].dt.day_name()
    logger.info("Extracted 'comment_hour' and 'comment_day_of_week' features.")
    return df


def categorize_post_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes posts based on keywords in their title and selftext into
    'Player Transfer', 'Tournament Result', 'Ranking Update', or 'Other'.
    Uses a flexible keyword matching approach.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'post_title' and 'selftext' columns.

    Returns:
        pd.DataFrame: The DataFrame with an added 'post_type' column.
    """
    logger.info("Categorizing post types based on keywords...")
    keywords_data = get_keywords()
    post_type_keywords = keywords_data.get("post_type_keywords", {})

    player_transfer_keywords = post_type_keywords.get("player_transfer", [])
    tournament_result_keywords = post_type_keywords.get("tournament_result", [])
    ranking_update_keywords = post_type_keywords.get("ranking_update", [])

    logger.debug(f"Loaded Player Transfer Keywords: {player_transfer_keywords}")
    logger.debug(f"Loaded Tournament Result Keywords: {tournament_result_keywords}")
    logger.debug(f"Loaded Ranking Update Keywords: {ranking_update_keywords}")

    # Ensure columns exist and are string type
    if "post_title" not in df.columns or "selftext" not in df.columns:
        logger.error(
            "Missing 'post_title' or 'selftext' columns for post type categorization. Setting all to 'Other'."
        )
        df["post_type"] = "Other"
        return df

    # Combine title and selftext for keyword checking
    df["combined_post_text"] = (
        df["post_title"].astype(str).fillna("")
        + " "
        + df["selftext"].astype(str).fillna("")
    )
    df["combined_post_text_lower"] = df[
        "combined_post_text"
    ].str.lower()  # Lowercase once for efficiency

    def _flexible_contains_any_keyword(text: str, keyword_list: List[str]) -> bool:
        if not keyword_list or not text:
            return False
        # Use a less strict regex for post type categorization
        # This will match "keyword" in "keywords", "new team" in "a new team has formed"
        # It avoids \b for more flexibility, but still escapes keywords
        # The pattern will be like "keyword1|keyword2|..."
        pattern = "|".join(re.escape(kw) for kw in keyword_list)
        # Use re.search for partial matches within words, but still escape keywords
        # This is the key change for more flexible matching
        return bool(re.search(pattern, text))

    def _classify_post(text_to_check: str) -> str:  # Corrected: takes text directly
        # Prioritize more specific categories if there's overlap
        if _flexible_contains_any_keyword(text_to_check, player_transfer_keywords):
            logger.debug(f"Classified as 'Player Transfer'")  # Removed post_title
            return "Player Transfer"
        if _flexible_contains_any_keyword(text_to_check, tournament_result_keywords):
            logger.debug(f"Classified as 'Tournament Result'")  # Removed post_title
            return "Tournament Result"
        if _flexible_contains_any_keyword(text_to_check, ranking_update_keywords):
            logger.debug(f"Classified as 'Ranking Update'")  # Removed post_title
            return "Ranking Update"
        logger.debug(f"Classified as 'Other'")  # Removed post_title
        return "Other"

    df["post_type"] = df["combined_post_text_lower"].apply(
        _classify_post
    )  # Apply to the lowercased text

    # Drop the temporary combined text columns
    df.drop(columns=["combined_post_text", "combined_post_text_lower"], inplace=True)

    logger.info("Post type categorization complete.")
    post_type_counts = df["post_type"].value_counts()
    logger.info(f"Post type distribution:\n{post_type_counts}")

    # Log if any specific category is empty
    for category in ["Player Transfer", "Tournament Result", "Ranking Update"]:
        if category not in post_type_counts or post_type_counts[category] == 0:
            logger.warning(
                f"No posts categorized as '{category}'. Consider reviewing keywords or data."
            )

    return df
