# worker_config.py
import os
import logging
import time # +++ ADDED +++
import threading # +++ ADDED +++


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
# worker_config.py
import os
import logging
import time # +++ ADDED +++
import threading # +++ ADDED +++

# Setup a logger for config loading if needed, or use a generic one
logger_config = logging.getLogger(__name__ + "_config")

# --- Shared Worker Configurations ---
WORKER_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
WORKER_GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", 90))
MAX_SAMPLE_SIZE_FOR_AI_CLEANING = 20

# +++ START NEW Rate Limiter +++
class APIRateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            self.min_interval_seconds = 0 # Effectively disabled
            logger_config.warning("APIRateLimiter initialized with non-positive requests_per_minute. Rate limiting disabled.")
        else:
            self.min_interval_seconds = 60.0 / requests_per_minute
        self.last_call_time = 0.0
        self._lock = threading.Lock() # Ensure thread safety for last_call_time

    def wait_if_needed(self, logger_instance=None):
        if self.min_interval_seconds <= 0: # If disabled
            return

        with self._lock:
            current_time = time.monotonic()
            elapsed_since_last_call = current_time - self.last_call_time
            wait_duration = self.min_interval_seconds - elapsed_since_last_call

            if wait_duration > 0:
                if logger_instance:
                    logger_instance.debug(f"RateLimiter: Waiting for {wait_duration:.2f} seconds to maintain ~{1/(self.min_interval_seconds/60):.1f} RPM.")
                time.sleep(wait_duration)
            
            self.last_call_time = time.monotonic() # Update timestamp after waiting (or if no wait was needed)
            if logger_instance:
                 log_msg_detail = f"Interval: {self.min_interval_seconds:.2f}s." if self.min_interval_seconds > 0 else "Rate limiting disabled."
                 logger_instance.debug(f"RateLimiter: Proceeding with API call. {log_msg_detail}")

# Instantiate for Gemini Free Tier (15 RPM). Let's use 14 RPM for a small buffer.
# This means roughly one call every 60/14 = ~4.28 seconds.
GEMINI_API_RATE_LIMITER = APIRateLimiter(requests_per_minute=14)
# +++ END NEW Rate Limiter +++

# Log that config values are loaded (optional)
logger_config.debug(f"WORKER_GEMINI_TIMEOUT set to: {WORKER_GEMINI_TIMEOUT}")
logger_config.debug(f"MAX_SAMPLE_SIZE_FOR_AI_CLEANING set to: {MAX_SAMPLE_SIZE_FOR_AI_CLEANING}")
if not WORKER_GEMINI_API_KEY:
    logger_config.warning("WORKER_GEMINI_API_KEY is not set in worker_config.py (read from environment).")