import logging
import os
import sqlite3
import sys
from typing import Optional

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

# Use PROJECT_ROOT and DATA_DIR from config.py
# Ensure these attributes exist in config.py
PROJECT_ROOT = getattr(
    config, "PROJECT_ROOT", os.getcwd()
)  # Fallback to current working directory
DATA_BASE_DIR = getattr(
    config, "DATA_DIR", os.path.join(PROJECT_ROOT, "data")
)  # Fallback to 'data' subfolder


def save_data_to_sqlite(df_comments: pd.DataFrame) -> None:
    """
    Saves a single combined DataFrame to an SQLite database.
    The database file path is constructed using config.DATA_DIR and config.DATABASE_NAME.
    The table name is taken from config.TABLE_NAME.

    Args:
        df_comments (pd.DataFrame): The combined DataFrame containing all comments data.
    """
    # Ensure config attributes exist
    if not hasattr(config, "DATABASE_NAME") or not config.DATABASE_NAME:
        logger.error(
            "config.DATABASE_NAME is not defined or is empty. Cannot save data."
        )
        return
    if not hasattr(config, "TABLE_NAME") or not config.TABLE_NAME:
        logger.error("config.TABLE_NAME is not defined or is empty. Cannot save data.")
        return

    # Use DATA_BASE_DIR and then 'processed' subfolder
    db_dir = os.path.join(DATA_BASE_DIR, "processed")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, config.DATABASE_NAME)

    try:
        with sqlite3.connect(db_path) as conn:
            logger.info(f"Attempting to save data to SQLite database: {db_path}")

            df_comments.to_sql(
                config.TABLE_NAME, conn, if_exists="replace", index=False
            )
            logger.info(
                f"Successfully saved '{config.TABLE_NAME}' table with {len(df_comments)} rows to '{db_path}'."
            )

    except sqlite3.Error as e:
        logger.error(f"SQLite error during data saving to '{db_path}': {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data saving to '{db_path}': {e}"
        )


def load_data_from_sqlite() -> pd.DataFrame:
    """
    Loads data from an SQLite database into a Pandas DataFrame.
    The database file path is constructed using config.DATA_DIR and config.DATABASE_NAME.
    The table name is taken from config.TABLE_NAME.

    Returns:
        pd.DataFrame: The comments DataFrame. Returns an empty DataFrame if loading fails.
    """
    # Ensure config attributes exist
    if not hasattr(config, "DATABASE_NAME") or not config.DATABASE_NAME:
        logger.error(
            "config.DATABASE_NAME is not defined or is empty. Cannot load data."
        )
        return pd.DataFrame()
    if not hasattr(config, "TABLE_NAME") or not config.TABLE_NAME:
        logger.error("config.TABLE_NAME is not defined or is empty. Cannot load data.")
        return pd.DataFrame()

    # Use DATA_BASE_DIR and then 'processed' subfolder
    db_dir = os.path.join(DATA_BASE_DIR, "processed")
    db_path = os.path.join(db_dir, config.DATABASE_NAME)

    df_comments = pd.DataFrame()

    if not os.path.exists(db_path):
        logger.warning(
            f"Database file '{db_path}' does not exist. Returning empty DataFrame."
        )
        return df_comments

    try:
        with sqlite3.connect(db_path) as conn:
            logger.info(f"Attempting to load data from SQLite database: {db_path}")

            df_comments = pd.read_sql_query(f"SELECT * FROM {config.TABLE_NAME}", conn)
            logger.info(
                f"Successfully loaded '{config.TABLE_NAME}' table with {len(df_comments)} rows from '{db_path}'."
            )

    except pd.io.sql.DatabaseError as e:  # Specific error for pandas read_sql_query
        logger.error(f"Database error during data loading from '{db_path}': {e}")
    except sqlite3.Error as e:
        logger.error(f"SQLite error during data loading from '{db_path}': {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data loading from '{db_path}': {e}"
        )

    return df_comments


def check_db_exists_and_has_data() -> bool:
    """
    Checks if the SQLite database file exists and if the specified table contains data.
    The database file path is constructed using config.DATA_DIR and config.DATABASE_NAME.
    The table name is taken from config.TABLE_NAME.

    Returns:
        bool: True if the database and table exist and contain data, False otherwise.
    """
    # Ensure config attributes exist
    if not hasattr(config, "DATABASE_NAME") or not config.DATABASE_NAME:
        logger.error(
            "config.DATABASE_NAME is not defined or is empty. Cannot check DB."
        )
        return False
    if not hasattr(config, "TABLE_NAME") or not config.TABLE_NAME:
        logger.error("config.TABLE_NAME is not defined or is empty. Cannot check DB.")
        return False

    # Use DATA_BASE_DIR and then 'processed' subfolder
    db_dir = os.path.join(DATA_BASE_DIR, "processed")
    db_path = os.path.join(db_dir, config.DATABASE_NAME)

    if not os.path.exists(db_path):
        logger.info(f"Database file '{db_path}' does not exist.")
        return False

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{config.TABLE_NAME}';"
            )
            if cursor.fetchone() is None:
                logger.info(
                    f"Table '{config.TABLE_NAME}' does not exist in database '{config.DATABASE_NAME}' at '{db_path}'."
                )
                return False

            # Check if table has data
            cursor.execute(f"SELECT COUNT(*) FROM {config.TABLE_NAME};")
            row_count = cursor.fetchone()[0]
            if row_count > 0:
                logger.info(
                    f"Database '{config.DATABASE_NAME}' at '{db_path}' and table '{config.TABLE_NAME}' exist and contain {row_count} rows."
                )
                return True
            else:
                logger.info(
                    f"Table '{config.TABLE_NAME}' in database '{config.DATABASE_NAME}' at '{db_path}' is empty."
                )
                return False

    except sqlite3.Error as e:
        logger.error(f"SQLite error during DB check for '{db_path}': {e}")
        return False
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during DB check for '{db_path}': {e}"
        )
        return False
