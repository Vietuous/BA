import logging
import os
import re
import string
import sys
from typing import Any, Dict, List, Tuple, Union

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- NLTK Data Download Check ---
def check_and_download_nltk_data() -> None:
    """
    Checks if required NLTK data packages are downloaded. If not, attempts to download them.
    Exits the program if critical data cannot be downloaded.
    """
    required_nltk_data = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
        "vader_lexicon": "sentiment/vader_lexicon",
    }

    for data_name, data_path in required_nltk_data.items():
        try:
            nltk.data.find(data_path)
            logger.info(f"NLTK data '{data_name}' already downloaded.")
        except LookupError:
            logger.info(
                f"NLTK data '{data_name}' not found. Attempting to download now..."
            )
            try:
                nltk.download(data_name, quiet=True)
                logger.info(f"NLTK data '{data_name}' downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download NLTK data '{data_name}': {e}")
                logger.error("This data is critical for text processing. Exiting.")
                sys.exit(1)  # Exit if critical data cannot be downloaded


# Perform the check and download when the module is imported
check_and_download_nltk_data()

# Initialize NLTK components
try:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(
        f"Failed to initialize NLTK components: {e}. Ensure NLTK data is correctly downloaded."
    )
    sys.exit(1)


def clean_text(text: Union[str, Any]) -> str:
    """
    Cleans text by converting to lowercase, removing URLs, mentions, hashtags,
    numbers, and punctuation. Handles non-string inputs by converting them to string.

    Args:
        text (Union[str, Any]): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        text = str(text)  # Convert non-string inputs to string

    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove mentions (@username) and hashtags (#)
    text = re.sub(r"\@\w+|\#", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: Union[str, Any]) -> str:
    """
    Applies cleaning, tokenization, stop-word removal, and lemmatization to text.
    Handles non-string inputs gracefully.

    Args:
        text (Union[str, Any]): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:  # If text is empty after cleaning, return empty string
        return ""

    tokens = word_tokenize(cleaned_text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return " ".join(processed_tokens)


def get_sentiment_scores(text: Union[str, Any]) -> Dict[str, float]:
    """
    Returns VADER sentiment polarity scores for a given text.
    Returns a dictionary with 'neg', 'neu', 'pos', and 'compound' scores.
    Handles non-string inputs and empty strings by returning zero scores.

    Args:
        text (Union[str, Any]): The input text for sentiment analysis.

    Returns:
        Dict[str, float]: A dictionary containing the sentiment scores.
    """
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return analyzer.polarity_scores(text)


def calculate_text_length(text: Union[str, Any]) -> Tuple[int, int]:
    """
    Calculates character count and word count for a given text.
    Handles non-string inputs by converting them to string.

    Args:
        text (Union[str, Any]): The input text.

    Returns:
        Tuple[int, int]: A tuple containing (character_count, word_count).
    """
    text_str = str(text)
    char_count = len(text_str)
    word_count = len(text_str.split())
    return char_count, word_count


def contains_any_keyword(text: Union[str, Any], keyword_list: List[str]) -> bool:
    """
    Checks if the text contains any of the keywords from the provided list.
    The check is case-insensitive and matches whole words.
    Handles non-string inputs and empty keyword lists.

    Args:
        text (Union[str, Any]): The input text to search within.
        keyword_list (List[str]): A list of keywords to search for.

    Returns:
        bool: True if any keyword is found, False otherwise.
    """
    if not keyword_list:
        return False
    if not isinstance(text, str):
        text = str(text)

    # Create a regex pattern for whole word matching, escaping special characters in keywords
    pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in keyword_list) + r")\b"
    return bool(re.search(pattern, text.lower()))


def contains_question(text: Union[str, Any]) -> bool:
    """
    Checks if the text contains a question mark.
    Handles non-string inputs.

    Args:
        text (Union[str, Any]): The input text.

    Returns:
        bool: True if a question mark is found, False otherwise.
    """
    return "?" in str(text)
