"""
Project configuration for LLM stylometric fingerprinting.

This module centralizes paths, generation settings, model metadata,
and output schemas. API keys are not stored here. They must be stored
in the .env file.
"""

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
PROMPT_DIR = DATA_DIR / "prompts"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"
ANNOTATION_DIR = DATA_DIR / "annotations"


# ============================================================
# DATA FILE PATHS
# ============================================================

PROMPTS_PATH = PROMPT_DIR / "Prompts.csv"
RAW_OUTPUTS_PATH = RAW_DIR / "model_outputs_raw.csv"
CLEAN_OUTPUTS_PATH = PROCESSED_DIR / "model_outputs_clean.csv"
VALID_OUTPUTS_PATH = PROCESSED_DIR / "valid_model_outputs.csv"
FINAL_FEATURES_PATH = FEATURE_DIR / "final" / "stylometric_features_final.csv"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
SEMANTIC_MAPS_DIR = OUTPUT_DIR / "semantic_maps"
STATISTICAL_TESTS_DIR = OUTPUT_DIR / "statistical_tests"


# ============================================================
# GENERATION SETTINGS
# ============================================================

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 450

TARGET_WORDS = 200
WORD_MIN = 180
WORD_MAX = 220

MAX_API_RETRIES = 5
RETRY_WAIT_SECONDS = 10


# ============================================================
# ENVIRONMENT VARIABLE NAMES
# ============================================================

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {
    "OpenAI": {
        "provider": "OpenAI",
        "model_family": "GPT",
        "model_name_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "api_key_env": OPENAI_API_KEY_ENV,
    },
    "Claude": {
    "provider": "Anthropic",
    "model_family": "Claude",
    "model_name_env": "CLAUDE_MODEL",
    "default_model": "claude-haiku-4-5-20251001",
    "api_key_env": ANTHROPIC_API_KEY_ENV,
    },
    "Gemini": {
    "provider": "Google",
    "model_family": "Gemini",
    "model_name_env": "GEMINI_MODEL",
    "default_model": "gemini-2.5-flash",
    "api_key_env": GOOGLE_API_KEY_ENV,
    },
    "DeepSeek": {
        "provider": "DeepSeek",
        "model_family": "DeepSeek",
        "model_name_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
        "api_key_env": DEEPSEEK_API_KEY_ENV,
    },
    "Mistral": {
        "provider": "Mistral AI",
        "model_family": "Mistral",
        "model_name_env": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
        "api_key_env": MISTRAL_API_KEY_ENV,
    },
}


# ============================================================
# RAW OUTPUT SCHEMA
# ============================================================

RAW_OUTPUT_COLUMNS = [
    "text_id",
    "prompt_id",
    "genre",
    "theme",
    "topic",
    "condition",
    "model_family",
    "model_name",
    "provider",
    "temperature",
    "max_tokens",
    "date_generated",
    "original_prompt_text",
    "controlled_prompt_text",
    "raw_output_text",
    "length_controlled_output_text",
    "output_text",
    "raw_word_count",
    "final_word_count",
    "length_status",
    "within_10_percent_word_limit",
    "generation_attempt",
    "was_rewritten_for_length",
    "api_status",
    "error_message",
    "notes",
]


def create_required_directories() -> None:
    """
    Create required project directories if they do not already exist.
    """
    directories = [
        DATA_DIR,
        PROMPT_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        FEATURE_DIR,
        FEATURE_DIR / "final",
        FEATURE_DIR / "exploratory",
        FEATURE_DIR / "old",
        ANNOTATION_DIR,
        OUTPUT_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        SEMANTIC_MAPS_DIR,
        STATISTICAL_TESTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)