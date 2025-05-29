# ai_data_cleaner.py
import pandas as pd
import json
import re
import logging
from typing import List, Optional, Any

# Attempt to import google.generativeai
try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE_CLEANER = True
except ImportError:
    GEMINI_SDK_AVAILABLE_CLEANER = False
    class MockGenerativeModel:
        def generate_content(self, *args, **kwargs):
            raise NotImplementedError("Gemini SDK not available.")
    class MockGenAI:
        GenerativeModel = MockGenerativeModel
    genai = MockGenAI()

logger_worker = logging.getLogger(__name__ + "_ai_cleaner")
# Logging setup remains as is, assuming main etl.py configures it.

# +++ MODIFIED IMPORT +++
try:
    from worker_config import WORKER_GEMINI_TIMEOUT, MAX_SAMPLE_SIZE_FOR_AI_CLEANING
    # You might also want to import WORKER_GEMINI_API_KEY from worker_config if you centralize its check
    # from worker_config import WORKER_GEMINI_API_KEY, GEMINI_SDK_AVAILABLE (if you also move GEMINI_SDK_AVAILABLE to config)
    logger_worker.debug("Successfully imported configurations from worker_config.py")
except ImportError as e:
    logger_worker.error(f"CRITICAL: Failed to import from worker_config.py: {e}. Using fallback defaults.")
    WORKER_GEMINI_TIMEOUT = 60  # Fallback
    MAX_SAMPLE_SIZE_FOR_AI_CLEANING = 20 # Fallback
# +++ END MODIFIED IMPORT +++


def _call_gemini_for_cleaning(prompt: str, column_name: str, cleaning_type: str) -> Optional[List[str]]:
    """
    Helper to call Gemini API for data cleaning tasks.
    Returns a list of processed strings or None on failure.
    """
    if not GEMINI_SDK_AVAILABLE_CLEANER: # This check is local to this file now
        logger_worker.error(f"Gemini SDK not available. Cannot perform AI cleaning for '{column_name}' ({cleaning_type}).")
        return None
    
    # API Key check: This assumes genai.configure() has been called with the API key in etl.py's startup.
    # If WORKER_GEMINI_API_KEY is imported from worker_config, you could add:
    # from worker_config import WORKER_GEMINI_API_KEY
    # if not WORKER_GEMINI_API_KEY:
    #     logger_worker.error("Gemini API Key not configured. Cannot perform AI cleaning.")
    #     return None

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger_worker.debug(f"Sending AI cleaning request for '{column_name}' ({cleaning_type}). Prompt (start): {prompt[:200]}...")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            ),
            # Use the imported WORKER_GEMINI_TIMEOUT
            request_options={'timeout': WORKER_GEMINI_TIMEOUT // 2}
        )
        # ... (rest of _call_gemini_for_cleaning remains the same)
        if not response.parts:
            block_reason_msg = "Response from AI was empty or blocked."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_msg = f"AI request blocked. Reason: {getattr(response.prompt_feedback, 'block_reason_message', str(response.prompt_feedback.block_reason))}"
            logger_worker.error(f"AI cleaning for column '{column_name}' ({cleaning_type}) failed: {block_reason_msg}")
            logger_worker.debug(f"Full AI response object for '{column_name}': {response}")
            return None
        generated_json_text = response.text.strip()
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
        if not all(isinstance(v, (str, type(None))) for v in cleaned_values):
            logger_worker.error(f"AI output list for '{column_name}' ({cleaning_type}) does not contain all strings or nulls. Sample: {str(cleaned_values[:5])[:200]}. Raw: {generated_json_text[:200]}")
            return None
        return [str(v) if v is not None else "" for v in cleaned_values]
    except json.JSONDecodeError as e:
        # Note: if response is not defined due to an early error, response.text will fail.
        # It's better to log the raw text if available, otherwise just the error.
        raw_text_for_log = "N/A"
        if 'response' in locals() and hasattr(response, 'text'):
            raw_text_for_log = response.text[:500]
        logger_worker.error(f"Failed to parse JSON from AI for '{column_name}' ({cleaning_type}): {e}. Raw Response (start): {raw_text_for_log}")
    except AttributeError as ae:
        if 'NoneType' in str(ae) and 'default_api_key' in str(ae).lower():
             logger_worker.critical(f"Gemini API Key not configured (genai.configure() likely not called). AI cleaning for '{column_name}' failed. Error: {ae}")
        else:
             logger_worker.error(f"AttributeError during AI call for '{column_name}' ({cleaning_type}): {ae}", exc_info=True)
    except Exception as e:
        logger_worker.error(f"Error during AI cleaning call for '{column_name}' ({cleaning_type}): {e}", exc_info=True)
    return None


# ... (ai_standardize_dates and ai_normalize_text functions remain the same,
#      they already use MAX_SAMPLE_SIZE_FOR_AI_CLEANING which is now imported from worker_config)
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
    # ... (uses MAX_SAMPLE_SIZE_FOR_AI_CLEANING) ...
    # (no change to the body of this function needed)
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