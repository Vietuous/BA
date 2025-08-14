# config.py
import datetime
import logging
import os

# --- Project Structure and Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Database Configuration ---
DATABASE_NAME = "reddit_dota2_analysis.db"
TABLE_NAME = "comments_data"

DATA_DIR = os.path.join(PROJECT_ROOT, "BA", "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
EXTERNAL_DATA_PATH = os.path.join(DATA_DIR, "external")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", DATABASE_NAME)

DATA_CONFIG_DIR = os.path.join(PROJECT_ROOT, "BA", "config")

# --- Reddit Data Collection Parameters ---
REDDIT_SUBREDDIT_NAME = "DotA2"
REDDIT_POST_LIMIT = 1000
REDDIT_COMMENT_LIMIT = 50
REDDIT_MAX_POSTS_TO_PROCESS = 15

# --- Tournament Configurations ---
TOURNAMENT_CONFIGS = {
    "OG_TI8": {
        "query": "OG TI8",
        "start_date": datetime.datetime(2018, 8, 10, 0, 0, 0),
        "end_date": datetime.datetime(2018, 8, 27, 23, 59, 59),
        "event_name": "TI8",
    },
    "TUNDRA_TI11": {
        "query": "Tundra TI11",
        "start_date": datetime.datetime(2022, 10, 15, 0, 0, 0),
        "end_date": datetime.datetime(2022, 10, 31, 23, 59, 59),
        "event_name": "TI11",
    },
    "OG_RM24": {
        "query": "OG",
        "start_date": datetime.datetime(2024, 7, 4, 0, 0, 0),
        "end_date": datetime.datetime(2024, 7, 28, 23, 59, 59),
        "event_name": "OG_RM24",
    },
    "TOPSON_RM24": {
        "query": "Topson",
        "start_date": datetime.datetime(2024, 7, 4, 0, 0, 0),
        "end_date": datetime.datetime(2024, 7, 28, 23, 59, 59),
        "event_name": "Topson_RM24",
    },
}

# --- Data Preprocessing Parameters ---
PRE_EVENT_DAYS = 7
POST_EVENT_DAYS = 5


def enrich_tournament_configs(configs, pre_days, post_days):
    enriched = {}
    for key, value in configs.items():
        start = value["start_date"]
        end = value["end_date"]
        value["pre_event_start"] = start - datetime.timedelta(days=pre_days)
        value["post_event_end"] = end + datetime.timedelta(days=post_days)
        enriched[key] = value
    return enriched


# Wende es auf deine Konfigurationen an:
TOURNAMENT_CONFIGS = enrich_tournament_configs(
    TOURNAMENT_CONFIGS, PRE_EVENT_DAYS, POST_EVENT_DAYS
)


# --- Model Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# TF-IDF Vectorizer Parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)

# XGBoost Hyperparameter Tuning Parameters (RandomizedSearchCV)
XGB_TUNING_N_ITER = 50
XGB_TUNING_CV_FOLDS = 3

# SHAP Plotting Parameters
SHAP_TOP_FEATURES_TO_PLOT = 10

# --- Output Paths for Models, Reports, and Figures ---
MODELS_DIR = os.path.join(PROJECT_ROOT, "BA", "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "BA", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# --- Feature Lists for Model Training ---
CATEGORICAL_FEATURES = [
    "time_period",
    "comment_day_of_week",
    "event_name",
    "link_flair_text",
    "post_type",  # <-- Hinzugefügt
]
NUMERICAL_FEATURES = [
    "compound_sentiment",
    "char_count",
    "word_count",
    "days_from_event_start",
    "comment_hour",
    "author_karma",
    "comment_score_per_day",
    "post_score",
    "post_num_comments",
    "upvote_ratio",
    "post_title_length",
    "post_title_word_count",
    "neg_sentiment",
    "neu_sentiment",
    "pos_sentiment",
]
BOOLEAN_FEATURES = [
    "contains_question",
    "contains_team_name",
    "contains_player_keyword",
    "contains_hero_keyword",
    "contains_event_keyword",
    "is_self",
]
TEXT_FEATURE = "processed_comment_body"
TARGET_VARIABLE = "comment_score"


# --- Logging Configuration ---
LOGS_DIR = os.path.join(PROJECT_ROOT, "BA", "logs")
LOG_FILES = {
    "data.pipeline": os.path.join(LOGS_DIR, "data_pipeline.log"),
    "eda.stats": os.path.join(
        LOGS_DIR, "eda_stats.log"
    ),  # Includes EDA and statistical tests
    "ml.pipeline": os.path.join(LOGS_DIR, "ml_training.log"),
}
LOG_LEVEL = logging.INFO
