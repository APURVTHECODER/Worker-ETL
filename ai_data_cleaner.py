# ai_data_cleaner.py
import pandas as pd
import json
import re
import logging
from typing import List, Optional, Any
import time # +++ ADDED +++

# Attempt to import google.generativeai
try:
    import google.generativeai as genai
    import google.api_core.exceptions # +++ ADDED for specific exception handling +++
    GEMINI_SDK_AVAILABLE_CLEANER = True
except ImportError:
    GEMINI_SDK_AVAILABLE_CLEANER = False
    # Mock classes for when SDK is not available
    class MockGoogleAPICoreExceptions: # +++ ADDED +++
        ResourceExhausted = type('ResourceExhausted', (Exception,), {})
    google = type('MockGoogleModule', (), {'api_core': type('MockAPICoreModule', (), {'exceptions': MockGoogleAPICoreExceptions})})() # +++ ADDED +++

    class MockGenerativeModel:
        def generate_content(self, *args, **kwargs):
            raise NotImplementedError("Gemini SDK not available.")
    class MockGenAI:
        GenerativeModel = MockGenerativeModel
    genai = MockGenAI()


logger_worker = logging.getLogger(__name__ + "_ai_cleaner")

try:
    # +++ MODIFIED IMPORT: Add GEMINI_API_RATE_LIMITER +++
    from worker_config import WORKER_GEMINI_TIMEOUT, MAX_SAMPLE_SIZE_FOR_AI_CLEANING, GEMINI_API_RATE_LIMITER
    logger_worker.debug("Successfully imported configurations from worker_config.py")
except ImportError as e:
    logger_worker.error(f"CRITICAL: Failed to import from worker_config.py: {e}. Using fallback defaults.")
    WORKER_GEMINI_TIMEOUT = 60
    MAX_SAMPLE_SIZE_FOR_AI_CLEANING = 20
    # Fallback rate limiter (effectively disabled or very permissive if config fails)
    class APIRateLimiter:
        def __init__(self, requests_per_minute: int): self.min_interval_seconds = 0
        def wait_if_needed(self, logger_instance=None): pass
    GEMINI_API_RATE_LIMITER = APIRateLimiter(1000) # Fallback


def _call_gemini_for_cleaning(prompt: str, column_name: str, cleaning_type: str) -> Optional[List[str]]:
    if not GEMINI_SDK_AVAILABLE_CLEANER:
        logger_worker.error(f"Gemini SDK not available. Cannot perform AI cleaning for '{column_name}' ({cleaning_type}).")
        return None
    
    # API Key is configured in etl.py via genai.configure()

    # +++ ADD Rate Limiter Wait +++
    GEMINI_API_RATE_LIMITER.wait_if_needed(logger_worker)

    # The SDK has built-in retries for 429s.
    # We add a loop here mainly for logging and potentially a slightly more aggressive initial wait
    # if the API is consistently overloaded, but primarily rely on SDK's retry for the actual retry logic.
    max_sdk_retries_observed = 2 # How many times we'll let the SDK retry and log it before giving up from this func's perspective
    
    for attempt in range(max_sdk_retries_observed):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger_worker.debug(f"Sending AI cleaning request (attempt {attempt + 1}) for '{column_name}' ({cleaning_type}). Prompt (start): {prompt[:200]}...")
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                ),
                request_options={'timeout': WORKER_GEMINI_TIMEOUT // 2} # Timeout for a single attempt
            )
            
            if not response.parts:
                block_reason_msg = "Response from AI was empty or blocked."
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_msg = f"AI request blocked. Reason: {getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback.block_reason))}"
                logger_worker.error(f"AI cleaning for column '{column_name}' ({cleaning_type}) failed: {block_reason_msg}")
                logger_worker.debug(f"Full AI response object for '{column_name}': {response}")
                return None # Non-retryable error from our perspective here

            generated_json_text = response.text.strip()
            # ... (rest of JSON parsing as before)
            if generated_json_text.startswith("```json"):
               generated_json_text = generated_json_text[len("```json"):].strip()
            if generated_json_text.endswith("```"):
               generated_json_text = generated_json_text[:-len("```")].strip()
            if not generated_json_text:
                logger_worker.warning(f"AI for '{column_name}' ({cleaning_type}) returned empty content after stripping potential markdown.")
                return None
            cleaned_values = json.loads(generated_json_text)
            if not isinstance(cleaned_values, list):
                logger_worker.error(f"AI output for '{column_name}' ({cleaning_type}) was not a JSON list. Got: {type(cleaned_values)}. Raw: {generated_json_text[:200]}")
                return None
            if not all(isinstance(v, (str, type(None))) for v in cleaned_values): # Allow None for nulls
                logger_worker.error(f"AI output list for '{column_name}' ({cleaning_type}) does not contain all strings or nulls. Sample: {str(cleaned_values[:5])[:200]}. Raw: {generated_json_text[:200]}")
                return None
            
            return [str(v) if v is not None else "" for v in cleaned_values] # SUCCESS

        except google.api_core.exceptions.ResourceExhausted as e:
            logger_worker.warning(f"AI cleaning (SDK call attempt {attempt + 1}) for '{column_name}' ({cleaning_type}) hit ResourceExhausted (429): {e}")
            # The SDK's retry mechanism should handle sleeping based on `retry_delay` from the error.
            # If we are here, it means the SDK's retry for this *single* call might have exhausted or
            # this is an outer loop. Let's log and if it's the last attempt of *our* loop, let it fail.
            if attempt < max_sdk_retries_observed - 1:
                # The SDK will retry internally with its own backoff. 
                # Our proactive rate limiter should prevent most of these for *new* calls.
                # If a call still gets 429, the SDK will handle the retry with delay.
                # No explicit sleep here needed *if relying on SDK retry*.
                # However, the log shows the error bubbling up, so SDK retries might have been exhausted for that call.
                # Let's add a small delay to give the API a breather before *our* next conceptual attempt
                # if the error bubbles up to this level.
                delay_match = re.search(r"retry_delay {\s*seconds: (\d+)\s*}", str(e))
                custom_delay = int(delay_match.group(1)) if delay_match else 10 * (attempt + 1)
                logger_worker.info(f"Waiting {custom_delay}s before this function considers retrying the operation for '{column_name}'.")
                time.sleep(custom_delay)
                continue 
            else:
                logger_worker.error(f"AI cleaning for '{column_name}' failed after observing SDK retries due to ResourceExhausted.")
                return None
        except json.JSONDecodeError as e:
            raw_text_for_log = "N/A"
            if 'response' in locals() and hasattr(response, 'text'): raw_text_for_log = response.text[:500]
            logger_worker.error(f"Failed to parse JSON from AI for '{column_name}' ({cleaning_type}): {e}. Raw Response (start): {raw_text_for_log}")
            return None # Not retryable here
        except AttributeError as ae:
            if 'NoneType' in str(ae) and 'default_api_key' in str(ae).lower():
                 logger_worker.critical(f"Gemini API Key not configured. AI cleaning for '{column_name}' failed. Error: {ae}")
            else:
                 logger_worker.error(f"AttributeError during AI call for '{column_name}' ({cleaning_type}): {ae}", exc_info=True)
            return None # Not retryable here
        except Exception as e:
            logger_worker.error(f"Error during AI cleaning call (attempt {attempt+1}) for '{column_name}' ({cleaning_type}): {e}", exc_info=True)
            # For other potentially transient errors, could add a small delay and retry
            if attempt < max_sdk_retries_observed - 1:
                time.sleep(5 * (attempt + 1)) 
                continue
            return None # General failure after retries

    logger_worker.error(f"AI cleaning for '{column_name}' ({cleaning_type}) failed after all attempts in _call_gemini_for_cleaning.")
    return None


# Functions ai_standardize_dates and ai_normalize_text remain structurally the same,
# as they call _call_gemini_for_cleaning which now has the rate limiter.
# ... (ai_standardize_dates and ai_normalize_text as before)
def ai_standardize_dates(column_series: pd.Series, column_name: str) -> pd.Series:
    # ... (uses MAX_SAMPLE_SIZE_FOR_AI_CLEANING) ...
    # (no change to the body of this function needed, only its import of MAX_SAMPLE_SIZE_FOR_AI_CLEANING via worker_config)
    if column_series.empty or column_series.isnull().all():
        logger_worker.debug(f"Skipping AI date standardization for '{column_name}': series is empty or all null.")
        return column_series
    original_values_series = column_series.dropna().astype(str)
    unique_values_to_sample = original_values_series.unique()[:MAX_SAMPLE_SIZE_FOR_AI_CLEANING]
    if len(unique_values_to_sample) == 0:
        logger_worker.debug(f"Skipping AI date standardization for '{column_name}': no unique non-null values to sample.")
        return column_series
    sample_list_str = json.dumps(unique_values_to_sample.tolist())
    prompt = (
        f"You are a data cleaning specialist. Analyze the following sample values from column '{column_name}':\n"
        f"{sample_list_str}\n\n"
        f"Your task is to convert each value to the 'YYYY-MM-DD' date format. Follow these rules strictly:\n"
        f"1. If a value is clearly a date and can be unambiguously converted, provide it in 'YYYY-MM-DD' format.\n"
        f"2. If a value looks like a date but is invalid (e.g., '2023/30/01', 'Feb 30 2023'), or highly ambiguous (e.g. '1/2/3' without context), return the original value *exactly* as it was provided in the input sample.\n"
        f"3. If a value is clearly NOT a date (e.g., 'apple', 'N/A', 'Unknown', a number that isn't a plausible year or Excel date serial), return the original value *exactly* as it was provided.\n"
        f"4. Your response MUST be ONLY a valid JSON list of strings. Each string in the list must correspond to an item in the input sample, in the same order.\n"
        f"5. The JSON list must contain exactly {len(unique_values_to_sample)} string items.\n"
        f"6. Do NOT include explanations, comments, or markdown formatting (like ```json ... ```) outside the JSON list itself.\n\n"
        f"Example Input Sample: [\"01/10/2023\", \"Nov 5, 2022\", \"some text\", \"2023-12-25\", \"2024/15/01\"]\n"
        f"Correct Example Output (JSON list of strings): [\"2023-01-10\", \"2022-11-05\", \"some text\", \"2023-12-25\", \"2024/15/01\"]\n\n"
        f"Now, process the provided sample and return the JSON list of strings:"
    )
    logger_worker.info(f"Attempting AI date standardization for column '{column_name}' (sample size: {len(unique_values_to_sample)})")
    ai_processed_strings = _call_gemini_for_cleaning(prompt, column_name, "standardize_dates")
    if ai_processed_strings and len(ai_processed_strings) == len(unique_values_to_sample):
        value_map = dict(zip(unique_values_to_sample, ai_processed_strings))
        cleaned_series = column_series.copy()
        for i, original_val in enumerate(column_series):
            if pd.isna(original_val): continue
            original_val_str = str(original_val)
            if original_val_str in value_map:
                cleaned_series.iloc[i] = value_map[original_val_str]
        final_series_attempt_datetime = pd.to_datetime(cleaned_series, errors='coerce', dayfirst=False, yearfirst=False)
        reverted_series = final_series_attempt_datetime.copy()
        for i in range(len(reverted_series)):
            if pd.isna(reverted_series.iloc[i]) and not pd.isna(cleaned_series.iloc[i]):
                reverted_series.iloc[i] = cleaned_series.iloc[i]
        logger_worker.info(f"AI successfully processed date standardization for column '{column_name}'.")
        return reverted_series
    else:
        logger_worker.warning(f"AI date standardization failed or returned mismatched data for '{column_name}'. Original series returned.")
        if ai_processed_strings:
            logger_worker.debug(f"AI returned {len(ai_processed_strings)} items, expected {len(unique_values_to_sample)} for '{column_name}'")
        return column_series

def ai_normalize_text(column_series: pd.Series, column_name: str, mode: str = "title_case_trim") -> pd.Series:
    if column_series.empty or column_series.isnull().all():
        logger_worker.debug(f"Skipping AI text normalization for '{column_name}': series is empty or all null.")
        return column_series
    if not (pd.api.types.is_string_dtype(column_series.infer_objects().dtype) or \
            pd.api.types.is_object_dtype(column_series.dtype)):
        logger_worker.debug(f"Skipping AI text normalization for '{column_name}': not primarily string/object type (actual: {column_series.dtype}).")
        return column_series
    original_values_series = column_series.dropna().astype(str)
    unique_values_to_sample = original_values_series.unique()[:MAX_SAMPLE_SIZE_FOR_AI_CLEANING]
    if len(unique_values_to_sample) == 0:
        logger_worker.debug(f"Skipping AI text normalization for '{column_name}': no unique non-null string values to sample.")
        return column_series
    sample_list_str = json.dumps(unique_values_to_sample.tolist())
    normalization_instruction = ""
    example_input_str = ""
    example_output_str = ""
    if mode == "title_case_trim":
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Convert to Title Case (e.g., 'john doe' becomes 'John Doe'). 3. Ensure single spaces between words."
        example_input_str = json.dumps(["  john DOE  ", "  multiple   spaces  ", "ALL CAPS anD miXeD"])
        example_output_str = json.dumps(["John Doe", "Multiple Spaces", "All Caps And Mixed"])
    elif mode == "lower_case_trim":
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Convert to lower case. 3. Ensure single spaces between words."
        example_input_str = json.dumps(["  John DOE  ", "  Multiple   Spaces  ", "ALL CAPS"])
        example_output_str = json.dumps(["john doe", "multiple spaces", "all caps"])
    elif mode == "upper_case_trim":
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Convert to UPPER CASE. 3. Ensure single spaces between words."
        example_input_str = json.dumps(["  John doe  ", "  multiple   spaces  ", "mixed Case"])
        example_output_str = json.dumps(["JOHN DOE", "MULTIPLE SPACES", "MIXED CASE"])
    else:
        logger_worker.warning(f"Unknown text normalization mode '{mode}' for column '{column_name}'. Skipping AI normalization.")
        return column_series
    prompt = (
        f"You are a data cleaning specialist. For the following sample text values from column '{column_name}':\n"
        f"{sample_list_str}\n\n"
        f"Normalize each value according to these rules: {normalization_instruction}\n"
        f"Your response MUST be ONLY a valid JSON list of strings. Each string in the list must correspond to an item in the input sample, in the same order.\n"
        f"The JSON list must contain exactly {len(unique_values_to_sample)} string items.\n"
        f"Do NOT include explanations, comments, or markdown formatting (like ```json ... ```) outside the JSON list itself.\n\n"
        f"Example Input Sample for '{mode}': {example_input_str}\n"
        f"Correct Example Output (JSON list of strings) for '{mode}': {example_output_str}\n\n"
        f"Now, process the provided sample and return the JSON list of strings:"
    )
    logger_worker.info(f"Attempting AI text normalization (mode: {mode}) for column '{column_name}' (sample size: {len(unique_values_to_sample)})")
    ai_processed_strings = _call_gemini_for_cleaning(prompt, column_name, f"normalize_text_{mode}")
    if ai_processed_strings and len(ai_processed_strings) == len(unique_values_to_sample):
        value_map = dict(zip(unique_values_to_sample, ai_processed_strings))
        cleaned_series = column_series.copy()
        for i, original_val in enumerate(column_series):
            if pd.isna(original_val): continue
            original_val_str = str(original_val)
            if original_val_str in value_map:
                 cleaned_series.iloc[i] = value_map[original_val_str]
        logger_worker.info(f"AI successfully normalized text (mode: {mode}) for column '{column_name}'.")
        return cleaned_series
    else:
        logger_worker.warning(f"AI text normalization (mode: {mode}) failed or returned mismatched data for '{column_name}'. Original series returned.")
        if ai_processed_strings:
            logger_worker.debug(f"AI returned {len(ai_processed_strings)} items, expected {len(unique_values_to_sample)} for '{column_name}' (mode {mode})")
        return column_series