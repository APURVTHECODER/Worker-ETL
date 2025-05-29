# worker_config.py
import os
import logging

# Setup a logger for config loading if needed, or use a generic one
logger_config = logging.getLogger(__name__ + "_config")

# --- Shared Worker Configurations ---
WORKER_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None) # Needed by both
WORKER_GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", 90))
MAX_SAMPLE_SIZE_FOR_AI_CLEANING = 20 # Specific to ai_data_cleaner but can live here

# You can move other WORKER_ constants here if they are shared or might be in future
# For example:
# GCP_PROJECT = os.getenv("GCP_PROJECT")
# GCS_BUCKET = os.getenv("GCS_BUCKET")
# etc.

# Log that config values are loaded (optional)
logger_config.debug(f"WORKER_GEMINI_TIMEOUT set to: {WORKER_GEMINI_TIMEOUT}")
logger_config.debug(f"MAX_SAMPLE_SIZE_FOR_AI_CLEANING set to: {MAX_SAMPLE_SIZE_FOR_AI_CLEANING}")
if not WORKER_GEMINI_API_KEY:
    logger_config.warning("WORKER_GEMINI_API_KEY is not set in worker_config.py (read from environment).")