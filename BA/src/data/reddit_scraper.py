import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Union

import pandas as pd
import praw
from dotenv import load_dotenv
from praw.models import Comment, Submission, Subreddit
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def get_reddit_instance() -> praw.Reddit:
    """
    Initializes and returns a PRAW Reddit instance using credentials from .env file.

    Raises:
        ValueError: If Reddit API credentials (client ID, client secret, user agent)
                    are not found in the .env file.

    Returns:
        praw.Reddit: An authenticated PRAW Reddit instance.
    """
    load_dotenv()  # Load environment variables from .env

    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        missing_vars = [
            name
            for name, value in {
                "REDDIT_CLIENT_ID": REDDIT_CLIENT_ID,
                "REDDIT_CLIENT_SECRET": REDDIT_CLIENT_SECRET,
                "REDDIT_USER_AGENT": REDDIT_USER_AGENT,
            }.items()
            if value is None
        ]
        raise ValueError(
            f"Reddit API credentials not found. Please set {', '.join(missing_vars)} "
            "in your .env file."
        )

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        # username=os.getenv('REDDIT_USERNAME'), # Uncomment if user auth is needed
        # password=os.getenv('REDDIT_PASSWORD')  # Uncomment if user auth is needed
    )
    logger.info("PRAW Reddit instance successfully initialized.")
    return reddit


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(praw.exceptions.APIException)
    | retry_if_exception_type(praw.exceptions.RedditAPIException)
    | retry_if_exception_type(praw.exceptions.ClientException),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,  # Re-raise the last exception after retries are exhausted
)
def _get_subreddit_search_results(
    subreddit_obj: Subreddit, query: str, limit: int, time_filter: str
) -> List[Submission]:
    """
    Helper function to retry subreddit.search calls.

    Args:
        subreddit_obj (praw.models.Subreddit): The PRAW Subreddit object.
        query (str): The search query.
        limit (int): The maximum number of submissions to return.
        time_filter (str): The time period to search within (e.g., 'all', 'year').

    Returns:
        List[praw.models.Submission]: A list of submissions matching the search criteria.
    """
    logger.debug(
        f"Attempting to search subreddit '{subreddit_obj.display_name}' for query '{query}'..."
    )
    return list(subreddit_obj.search(query, limit=limit, time_filter=time_filter))


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(praw.exceptions.APIException)
    | retry_if_exception_type(praw.exceptions.RedditAPIException)
    | retry_if_exception_type(praw.exceptions.ClientException),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)
def _get_subreddit_top_results(
    subreddit_obj: Subreddit, limit: int, time_filter: str
) -> List[Submission]:
    """
    Helper function to retry subreddit.top calls.

    Args:
        subreddit_obj (praw.models.Subreddit): The PRAW Subreddit object.
        limit (int): The maximum number of submissions to return.
        time_filter (str): The time period to search within (e.g., 'all', 'year').

    Returns:
        List[praw.models.Submission]: A list of top submissions.
    """
    logger.debug(
        f"Attempting to get top posts from subreddit '{subreddit_obj.display_name}'..."
    )
    return list(subreddit_obj.top(limit=limit, time_filter=time_filter))


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(praw.exceptions.APIException)
    | retry_if_exception_type(praw.exceptions.RedditAPIException)
    | retry_if_exception_type(praw.exceptions.ClientException),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)
def _get_submission_comments(submission: Submission) -> List[Comment]:
    """
    Helper function to retry submission.comments.list calls.
    Replaces 'MoreComments' objects with actual comments up to a certain depth.

    Args:
        submission (praw.models.Submission): The submission object to fetch comments from.

    Returns:
        List[praw.models.Comment]: A list of comments for the given submission.
    """
    logger.debug(f"Attempting to fetch comments for submission ID: {submission.id}")
    # Fetch comments up to 3 levels deep. Higher limits increase API calls and time.
    submission.comments.replace_more(limit=3)
    return list(submission.comments.list())


def get_posts_and_comments(
    subreddit_obj: Subreddit,
    query: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    post_limit: int = 1000,
    comment_limit: int = 50,
    max_posts_to_process: int = 15,
) -> List[Dict[str, Any]]:
    """
    Fetches posts based on a query within a specific date range and extracts top comments.
    Combines results from search and top listings for better coverage.
    Limits the number of posts from which comments are actually processed.
    Also saves the raw collected comments to a JSON file.

    Args:
        subreddit_obj (praw.models.Subreddit): The PRAW Subreddit object.
        query (str): The search query for posts (e.g., "OG TI8 win").
        start_date (datetime.datetime): The start date for filtering posts (inclusive).
        end_date (datetime.datetime): The end date for filtering posts (inclusive).
        post_limit (int): Maximum number of posts to fetch from Reddit's API for each method.
        comment_limit (int): Maximum number of top-level, most upvoted comments per post.
        max_posts_to_process (int): Maximum number of unique posts to process comments from.
                                    These will be the highest-scoring posts found.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a comment
                              and includes associated post details.
    """
    logger.info(
        f"\nSearching for posts with query: '{query}' from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
    )
    collected_comments: List[Dict[str, Any]] = []

    if not isinstance(subreddit_obj, Subreddit):
        logger.error(
            "Invalid subreddit_obj provided. Must be a praw.models.Subreddit instance."
        )
        return []
    if not isinstance(query, str) or not query.strip():
        logger.error("Query must be a non-empty string. Skipping.")
        return []
    if not isinstance(start_date, datetime.datetime) or not isinstance(
        end_date, datetime.datetime
    ):
        logger.error("start_date and end_date must be datetime objects. Skipping.")
        return []
    if start_date > end_date:
        logger.warning("start_date is after end_date. No posts will be found.")
        return []

    all_submissions_dict: Dict[str, Submission] = (
        {}
    )  # Use a dict to store unique submissions by ID
    try:
        # Attempt to get posts from various sources
        logger.info(f"Fetching posts via search for query: '{query}'...")
        search_results = _get_subreddit_search_results(
            subreddit_obj, query, post_limit, "all"
        )
        logger.info(f"  - Found {len(search_results)} posts via search for '{query}'.")

        logger.info(
            f"Fetching top posts from subreddit: '{subreddit_obj.display_name}'..."
        )
        top_results = _get_subreddit_top_results(subreddit_obj, post_limit, "year")
        logger.info(f"  - Found {len(top_results)} posts via top (time_filter='year').")

        combined_results = search_results + top_results

        for sub in combined_results:
            # Filter by date range immediately
            submission_date = datetime.datetime.fromtimestamp(sub.created_utc)
            if start_date <= submission_date <= end_date:
                all_submissions_dict[sub.id] = sub

    except (
        praw.exceptions.APIException,
        praw.exceptions.RedditAPIException,
        praw.exceptions.ClientException,
    ) as e:
        logger.error(f"Reddit API error during post search for '{query}': {e}")
        return []
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during post search for '{query}': {e}"
        )
        return []

    # Convert dict to list, sort by score, and take only max_posts_to_process
    all_submissions_list = sorted(
        all_submissions_dict.values(), key=lambda sub: sub.score, reverse=True
    )

    # Limit the number of posts we actually process comments from
    submissions_to_process = all_submissions_list[:max_posts_to_process]

    logger.info(
        f"Found {len(all_submissions_dict)} unique submissions for '{query}' within the date range."
    )
    logger.info(
        f"Processing comments from the top {len(submissions_to_process)} highest-scoring submissions (max {max_posts_to_process})..."
    )

    if not submissions_to_process:
        logger.info(
            f"No submissions found for '{query}' within the specified date range. Skipping comment collection."
        )
        return []

    for i, submission in enumerate(submissions_to_process):
        logger.info(
            f"  Processing comments for post {i+1}/{len(submissions_to_process)} (ID: {submission.id}, Score: {submission.score}, Title: '{submission.title[:50]}...')..."
        )
        try:
            all_comments_for_post = _get_submission_comments(submission)

            logger.info(
                f"    Found {len(all_comments_for_post)} comments (including replies) for this post."
            )

            # Filter out MoreComments objects that might remain if limit was too low or due to PRAW behavior
            actual_comments = [
                c for c in all_comments_for_post if isinstance(c, Comment)
            ]

            sorted_comments = sorted(
                actual_comments, key=lambda c: c.score, reverse=True
            )[:comment_limit]

            if sorted_comments:
                for comment in sorted_comments:
                    # Safely get author name, handling deleted authors
                    post_author_name = (
                        submission.author.name if submission.author else "[deleted]"
                    )
                    comment_author_name = (
                        comment.author.name if comment.author else "[deleted]"
                    )

                    collected_comments.append(
                        {
                            "post_id": submission.id,
                            "post_title": submission.title,
                            "post_url": submission.url,
                            "post_author": post_author_name,
                            "post_created_utc": pd.to_datetime(
                                submission.created_utc, unit="s"
                            ),
                            "post_score": submission.score,
                            "post_num_comments": submission.num_comments,
                            "upvote_ratio": submission.upvote_ratio,
                            "is_self": submission.is_self,
                            "selftext": (
                                submission.selftext if submission.is_self else ""
                            ),
                            "link_flair_text": submission.link_flair_text,
                            "permalink": submission.permalink,
                            "comment_id": comment.id,
                            "comment_body": comment.body,
                            "comment_author": comment_author_name,
                            "comment_created_utc": pd.to_datetime(
                                comment.created_utc, unit="s"
                            ),
                            "comment_score": comment.score,
                            "comment_permalink": comment.permalink,
                        }
                    )
        except (
            praw.exceptions.APIException,
            praw.exceptions.RedditAPIException,
            praw.exceptions.ClientException,
        ) as e:
            logger.error(
                f"    Reddit API error fetching comments for post '{submission.id}': {e}"
            )
            continue
        except Exception as e:
            logger.error(
                f"    An unexpected error occurred fetching comments for post '{submission.id}': {e}"
            )
            continue

    logger.info(
        f"Successfully collected {len(collected_comments)} comments for query: '{query}'."
    )

    # Save raw collected comments to JSON file
    if collected_comments:
        # Ensure RAW_DATA_PATH exists from config
        if not hasattr(config, "RAW_DATA_PATH") or not config.RAW_DATA_PATH:
            logger.error(
                "config.RAW_DATA_PATH is not defined or is empty. Cannot save raw data."
            )
            return collected_comments

        os.makedirs(config.RAW_DATA_PATH, exist_ok=True)

        raw_data_filename = f"raw_reddit_comments_{query.replace(' ', '_').replace('/', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        raw_data_filepath = os.path.join(config.RAW_DATA_PATH, raw_data_filename)

        try:
            # Convert datetime objects to string for JSON serialization
            serializable_comments = []
            for comment_dict in collected_comments:
                serializable_comment = comment_dict.copy()
                for key, value in serializable_comment.items():
                    if isinstance(value, datetime.datetime):
                        serializable_comment[key] = value.isoformat()
                serializable_comments.append(serializable_comment)

            with open(raw_data_filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_comments, f, ensure_ascii=False, indent=4)
            logger.info(f"Raw collected comments saved to: {raw_data_filepath}")
        except Exception as e:
            logger.error(
                f"Error saving raw comments to JSON at '{raw_data_filepath}': {e}"
            )
    else:
        logger.info(f"No comments collected for '{query}', skipping raw data save.")

    return collected_comments
