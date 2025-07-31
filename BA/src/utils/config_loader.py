import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Import the centralized configuration
project_root_for_import = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root_for_import not in sys.path:
    sys.path.insert(0, project_root_for_import)

import config

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def load_json_config(
    file_name: str, default_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads a JSON configuration file from the project's data config directory.
    Uses the DATA_CONFIG_DIR defined in config.py.

    Args:
        file_name (str): The name of the JSON file to load (e.g., "teams.json").
        default_key (Optional[str]): An optional key to check if its corresponding list
                                     is empty in the loaded configuration. A warning is logged
                                     if the key exists but the list is empty.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded configuration.
                        Returns an empty dictionary if the file is not found or
                        JSON decoding fails.
    """
    # Safely get DATA_CONFIG_DIR from config, with a fallback
    data_config_dir = getattr(config, "DATA_CONFIG_DIR", None)
    if data_config_dir is None:
        logger.error("config.DATA_CONFIG_DIR is not defined. Cannot load JSON config.")
        return {}

    config_file_path = os.path.join(data_config_dir, file_name)

    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logger.info(f"Successfully loaded JSON config from: {config_file_path}")

        if default_key and isinstance(cfg, dict) and default_key in cfg:
            if not cfg.get(default_key):
                logger.warning(
                    f"'{default_key}' list is empty in {file_name}. Please check the file."
                )
        return cfg
    except FileNotFoundError:
        logger.error(
            f"ERROR: Config file '{file_name}' not found at '{config_file_path}'. "
            "Please ensure the file exists and the path is correct."
        )
        return {}
    except json.JSONDecodeError:
        logger.error(
            f"ERROR: Could not decode JSON from '{config_file_path}'. "
            "Check file format for syntax errors."
        )
        return {}
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading '{config_file_path}': {e}"
        )
        return {}


def get_dota2_teams() -> List[str]:
    """
    Loads and returns the list of Dota 2 teams from 'teams.json'.

    Returns:
        List[str]: A list of Dota 2 team names. Returns an empty list if loading fails.
    """
    cfg = load_json_config("teams.json", "dota2_teams")
    teams = cfg.get("dota2_teams", [])
    if not teams:
        logger.warning("No Dota 2 teams found in 'teams.json'. Returning empty list.")
    return teams


def get_keywords() -> (
    Dict[str, Any]
):  # Changed return type to Any as it can be nested dict
    """
    Loads and returns player, hero, tournament/event, and post type keywords from 'keywords.json'.

    Returns:
        Dict[str, Any]: A dictionary containing lists of different keyword types,
                        including a nested dictionary for 'post_type_keywords'.
                        Returns empty lists/dicts for categories if loading fails or keys are missing.
    """
    cfg = load_json_config("keywords.json")
    keywords_data = {
        "player_keywords": cfg.get("player_keywords", []),
        "hero_keywords": cfg.get("hero_keywords", []),
        "tournament_event_keywords": cfg.get("tournament_event_keywords", []),
        "post_type_keywords": cfg.get("post_type_keywords", {}),  # <-- HIER KORRIGIERT
    }
    if not any(keywords_data.values()):
        logger.warning(
            "No keywords found in 'keywords.json'. All keyword lists are empty."
        )
    return keywords_data
