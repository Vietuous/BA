import datetime
import logging
import os
import sys
from typing import Any, Union

import pandas as pd

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def initial_clean_dataframe(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Performs initial cleaning steps on the DataFrame, including:
    - Ensuring 'comment_body' is string type.
    - Removing duplicate comments based on 'comment_id'.
    - Removing comments with empty or whitespace-only bodies.

    Args:
        df (pd.DataFrame): The input DataFrame containing comment data.
        df_name (str): A name for the DataFrame, used in logging messages.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logger.info(f"--- Initial Cleaning for {df_name} ---")
    initial_rows = df.shape[0]
    logger.info(f"Initial number of rows: {initial_rows}")
    logger.debug(f"DEBUG: Columns received for {df_name}: {df.columns.tolist()}")

    # Check for required columns
    if "comment_id" not in df.columns:
        logger.error(
            f"'comment_id' column not found in {df_name}. Cannot remove duplicates. Skipping initial cleaning."
        )
        return df
    if "comment_body" not in df.columns:
        logger.error(
            f"'comment_body' column not found in {df_name}. Cannot filter empty comments. Skipping initial cleaning."
        )
        return df

    # Ensure 'comment_body' is string type
    df["comment_body"] = df["comment_body"].astype(str)

    # Remove duplicate comments
    df_before_duplicates = df.shape[0]
    df.drop_duplicates(subset=["comment_id"], inplace=True)
    rows_after_duplicates = df.shape[0]
    removed_duplicates = df_before_duplicates - rows_after_duplicates
    logger.info(
        f"Rows after removing duplicate comments: {rows_after_duplicates} (Removed: {removed_duplicates})"
    )

    # Remove comments with empty or whitespace-only bodies
    df_before_empty_body = df.shape[0]
    df = df[
        df["comment_body"].str.strip() != ""
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning
    rows_after_empty_body = df.shape[0]
    removed_empty_body = df_before_empty_body - rows_after_empty_body
    logger.info(
        f"Rows after removing empty/whitespace-only comments: {rows_after_empty_body} (Removed: {removed_empty_body})"
    )

    logger.info(
        f"Final number of rows for {df_name} after initial cleaning: {df.shape[0]}"
    )
    return df


def filter_deleted_and_empty_processed_comments(
    df: pd.DataFrame, df_name: str
) -> pd.DataFrame:
    """
    Filters out comments where the author is '[deleted]' or where the processed comment body is empty.

    Args:
        df (pd.DataFrame): The input DataFrame.
        df_name (str): A name for the DataFrame, used in logging messages.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    logger.info(f"\n--- Filtering Deleted/Empty Comments for {df_name} ---")
    initial_rows = df.shape[0]
    logger.info(f"Initial number of rows: {initial_rows}")

    # Check for required columns
    if "comment_author" not in df.columns:
        logger.error(
            f"'comment_author' column not found in {df_name}. Cannot filter deleted authors. Skipping this filter."
        )
        return df

    # Filter out comments from '[deleted]' authors
    df_filtered = df[df["comment_author"] != "[deleted]"].copy()
    rows_after_deleted_author = df_filtered.shape[0]
    removed_deleted_author = initial_rows - rows_after_deleted_author
    logger.info(
        f"Rows after removing comments from '[deleted]' authors: {rows_after_deleted_author} (Removed: {removed_deleted_author})"
    )

    # Filter out comments with empty processed bodies
    if "processed_comment_body" in df_filtered.columns:
        df_before_empty_processed = df_filtered.shape[0]
        df_filtered["processed_comment_body"] = (
            df_filtered["processed_comment_body"].astype(str).str.strip()
        )
        df_filtered = df_filtered[df_filtered["processed_comment_body"] != ""].copy()
        rows_after_empty_processed = df_filtered.shape[0]
        removed_empty_processed = df_before_empty_processed - rows_after_empty_processed
        logger.info(
            f"Rows after removing empty processed comments: {rows_after_empty_processed} (Removed: {removed_empty_processed})"
        )
    else:
        logger.warning(
            f"Warning: 'processed_comment_body' not found in {df_name}. Skipping filter for empty processed comments."
        )

    logger.info(
        f"Final number of rows for {df_name} after filtering deleted/empty: {df_filtered.shape[0]}"
    )
    return df_filtered


import datetime
import logging

import pandas as pd  # Assuming pandas is used for pd.to_datetime if needed

logger = logging.getLogger(__name__)  # Ensure logger is initialized if not already


def categorize_time_period(
    comment_timestamp: Union[
        int, float, str, datetime.datetime
    ],  # Allow various timestamp types
    event_start_date: datetime.datetime,
    event_end_date: datetime.datetime,
    pre_event_start_date: datetime.datetime,  # New argument: calculated start of pre-event window
    post_event_end_date: datetime.datetime,  # New argument: calculated end of post-event window
) -> str:
    """
    Categorizes comments into 'Before Event', 'During Event', 'After Event',
    or 'Outside Window' periods relative to the event dates.

    Args:
        comment_timestamp (Union[int, float, str, datetime.datetime]): The creation timestamp of the comment.
                                                                        Can be UTC timestamp (int/float),
                                                                        string, or datetime object.
        event_start_date (datetime.datetime): The official start date of the event.
        event_end_date (datetime.datetime): The official end date of the event.
        pre_event_start_date (datetime.datetime): The calculated start date of the 'Before Event' window.
        post_event_end_date (datetime.datetime): The calculated end date of the 'After Event' window.

    Returns:
        str: The categorized time period.
    """
    # Convert comment_timestamp to a datetime object for robust comparison
    if isinstance(comment_timestamp, (int, float)):
        # Assuming UTC timestamp
        comment_datetime = datetime.datetime.utcfromtimestamp(comment_timestamp)
    elif isinstance(comment_timestamp, str):
        # Attempt to parse string to datetime, handle potential errors
        try:
            comment_datetime = datetime.datetime.fromisoformat(comment_timestamp)
        except ValueError:
            # Fallback for other common formats, if needed. Using pandas for flexibility.
            try:
                comment_datetime = pd.to_datetime(comment_timestamp)
            except Exception as e:
                logger.error(
                    f"Could not parse comment_timestamp string '{comment_timestamp}': {e}"
                )
                return "Error: Invalid Timestamp Format"
    elif isinstance(comment_timestamp, datetime.datetime):
        comment_datetime = comment_timestamp
    else:
        logger.error(
            f"Invalid comment_timestamp type provided: {type(comment_timestamp)}"
        )
        return "Error: Invalid Timestamp Type"

    # Ensure all event dates are datetime objects (should be from config)
    if not all(
        isinstance(d, datetime.datetime)
        for d in [
            event_start_date,
            event_end_date,
            pre_event_start_date,
            post_event_end_date,
        ]
    ):
        logger.error(
            f"Invalid event date type provided. event_start_date: {type(event_start_date)}, event_end_date: {type(event_end_date)}, pre_event_start_date: {type(pre_event_start_date)}, post_event_end_date: {type(post_event_end_date)}"
        )
        return "Error: Invalid Event Date Type"

    # Categorize the comment based on its timestamp relative to event windows
    if event_start_date <= comment_datetime <= event_end_date:
        return "During Event"
    elif pre_event_start_date <= comment_datetime < event_start_date:
        return "Before Event"
    elif event_end_date < comment_datetime <= post_event_end_date:
        return "After Event"
    else:
        return "Outside Window"


def calculate_days_from_event_start(
    comment_date: datetime.datetime, event_start: datetime.datetime
) -> Union[int, float, None]:
    """
    Calculates the difference in days from the comment's creation to the event start.
    Positive values mean the comment was made after the event start, negative before.

    Args:
        comment_date (datetime.datetime): The creation date of the comment.
        event_start (datetime.datetime): The start date of the event.

    Returns:
        Union[int, float, None]: The difference in days as an integer, or None if inputs are invalid.
    """
    if not isinstance(comment_date, datetime.datetime) or not isinstance(
        event_start, datetime.datetime
    ):
        logger.error(
            f"Invalid date type provided to calculate_days_from_event_start. comment_date: {type(comment_date)}, event_start: {type(event_start)}"
        )
        return None

    return (comment_date - event_start).days
