# ai_data_cleaner.py
import pandas as pd
import json
import re
import logging
from typing import List, Optional, Any
import time

# Attempt to import google.generativeai
try:
    import google.generativeai as genai
    import google.api_core.exceptions # For specific exception handling
    GEMINI_SDK_AVAILABLE_CLEANER = True
except ImportError:
    GEMINI_SDK_AVAILABLE_CLEANER = False
    # Mock classes for when SDK is not available
    class MockGoogleAPICoreExceptions:
        ResourceExhausted = type('ResourceExhausted', (Exception,), {})
    google = type('MockGoogleModule', (), {'api_core': type('MockAPICoreModule', (), {'exceptions': MockGoogleAPICoreExceptions})})()

    class MockGenerativeModel:
        def generate_content(self, *args, **kwargs):
            raise NotImplementedError("Gemini SDK not available.")
    class MockGenAI:
        GenerativeModel = MockGenerativeModel
    genai = MockGenAI()


logger_worker = logging.getLogger(__name__ + "_ai_cleaner")

try:
    from worker_config import WORKER_GEMINI_TIMEOUT, MAX_SAMPLE_SIZE_FOR_AI_CLEANING, GEMINI_API_RATE_LIMITER
    logger_worker.debug("Successfully imported configurations from worker_config.py")
except ImportError as e:
    logger_worker.error(f"CRITICAL: Failed to import from worker_config.py: {e}. Using fallback defaults.")
    WORKER_GEMINI_TIMEOUT = 60
    MAX_SAMPLE_SIZE_FOR_AI_CLEANING = 20
    # Fallback rate limiter
    class APIRateLimiter:
        def __init__(self, requests_per_minute: int): self.min_interval_seconds = 0
        def wait_if_needed(self, logger_instance=None): pass
    GEMINI_API_RATE_LIMITER = APIRateLimiter(1000)


def _call_gemini_for_cleaning(prompt: str, column_name: str, cleaning_type: str) -> Optional[List[str]]:
    if not GEMINI_SDK_AVAILABLE_CLEANER:
        logger_worker.error(f"Gemini SDK not available. Cannot perform AI cleaning for '{column_name}' ({cleaning_type}).")
        return None

    GEMINI_API_RATE_LIMITER.wait_if_needed(logger_worker)

    max_sdk_retries_observed = 2
    for attempt in range(max_sdk_retries_observed):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest') # Ensure this model name is current
            logger_worker.debug(f"Sending AI cleaning request (attempt {attempt + 1}) for '{column_name}' ({cleaning_type}). Prompt (start): {prompt[:200]}...")

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                ),
                request_options={'timeout': WORKER_GEMINI_TIMEOUT // 2}
            )

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

            # Return strings, convert None from JSON to empty string if that's desired, or keep as None.
            # Current logic returns empty string for None. Adjust if None needs to be propagated.
            return [str(v) if v is not None else "" for v in cleaned_values]

        except google.api_core.exceptions.ResourceExhausted as e:
            logger_worker.warning(f"AI cleaning (SDK call attempt {attempt + 1}) for '{column_name}' ({cleaning_type}) hit ResourceExhausted (429): {e}")
            if attempt < max_sdk_retries_observed - 1:
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
            return None
        except AttributeError as ae:
            if 'NoneType' in str(ae) and 'default_api_key' in str(ae).lower():
                 logger_worker.critical(f"Gemini API Key not configured. AI cleaning for '{column_name}' failed. Error: {ae}")
            elif "'MockGenerativeModel' object has no attribute 'types'" in str(ae) and not GEMINI_SDK_AVAILABLE_CLEANER:
                 logger_worker.error(f"Gemini SDK is mocked and 'types' attribute not available. AI cleaning for '{column_name}' failed. Error: {ae}")
            else:
                 logger_worker.error(f"AttributeError during AI call for '{column_name}' ({cleaning_type}): {ae}", exc_info=True)
            return None
        except Exception as e:
            logger_worker.error(f"Error during AI cleaning call (attempt {attempt+1}) for '{column_name}' ({cleaning_type}): {e}", exc_info=True)
            if attempt < max_sdk_retries_observed - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None

    logger_worker.error(f"AI cleaning for '{column_name}' ({cleaning_type}) failed after all attempts in _call_gemini_for_cleaning.")
    return None


def ai_standardize_dates(column_series: pd.Series, column_name: str) -> pd.Series:
    if column_series.empty or column_series.isnull().all():
        logger_worker.debug(f"Skipping AI date standardization for '{column_name}': series is empty or all null.")
        return column_series

    original_values_series = column_series.dropna().astype(str)
    unique_values = original_values_series.unique()
    if len(unique_values) > MAX_SAMPLE_SIZE_FOR_AI_CLEANING:
        sample_indices = pd.Series(unique_values).sample(n=MAX_SAMPLE_SIZE_FOR_AI_CLEANING, random_state=1).index
        unique_values_to_sample = unique_values[sample_indices]
    else:
        unique_values_to_sample = unique_values
    
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
        
        cleaned_series_list = []
        for original_val in column_series:
            if pd.isna(original_val):
                cleaned_series_list.append(None)
            else:
                original_val_str = str(original_val)
                cleaned_series_list.append(value_map.get(original_val_str, original_val_str)) # Fallback to original if not in map

        # Attempt to convert to datetime, then revert non-datetimes back to their AI-processed (or original) string form
        temp_series_for_datetime = pd.Series(cleaned_series_list, index=column_series.index, name=column_series.name, dtype='object')
        final_series_attempt_datetime = pd.to_datetime(temp_series_for_datetime, errors='coerce', dayfirst=False, yearfirst=False)
        
        # Revert NaT values (where conversion failed) back to the string from cleaned_series_list
        # This ensures that non-date strings processed by AI (or originals if not in sample) are kept
        reverted_series_list = []
        for i in range(len(final_series_attempt_datetime)):
            if pd.isna(final_series_attempt_datetime.iloc[i]):
                reverted_series_list.append(cleaned_series_list[i]) # Use the string value
            else:
                reverted_series_list.append(final_series_attempt_datetime.iloc[i]) # Use the datetime object

        final_reverted_series = pd.Series(reverted_series_list, index=column_series.index, name=column_series.name, dtype='object')
        logger_worker.info(f"AI successfully processed date standardization for column '{column_name}'.")
        return final_reverted_series
    else:
        logger_worker.warning(f"AI date standardization failed or returned mismatched data for '{column_name}'. Original series returned.")
        if ai_processed_strings:
            logger_worker.debug(f"AI returned {len(ai_processed_strings)} items, expected {len(unique_values_to_sample)} for '{column_name}'")
        return column_series.copy() # Return a copy to be safe


def ai_normalize_text(column_series: pd.Series, column_name: str, mode: Optional[str] = "title_case_trim") -> pd.Series:
    # +++ Default mode if None is passed (from worker) +++
    if mode is None:
        logger_worker.info(f"No text normalization mode provided for '{column_name}', defaulting to 'title_case_trim'.")
        mode = "title_case_trim"
        
    if column_series.empty or column_series.isnull().all():
        logger_worker.debug(f"Skipping AI text normalization for '{column_name}': series is empty or all null.")
        return column_series.copy() # Return a copy to be safe
    
    # Check if the column is predominantly string/object type
    # This check helps avoid trying to normalize numeric columns that might be misidentified
    if not (pd.api.types.is_string_dtype(column_series.infer_objects().dtype) or \
            pd.api.types.is_object_dtype(column_series.dtype)):
        logger_worker.debug(f"Skipping AI text normalization for '{column_name}': not primarily string/object type (actual inferred: {column_series.infer_objects().dtype}).")
        return column_series.copy()

    original_values_series = column_series.dropna().astype(str)
    unique_values = original_values_series.unique()

    if len(unique_values) > MAX_SAMPLE_SIZE_FOR_AI_CLEANING:
        # Take a random sample of unique values if there are too many
        # Using pandas Series to sample then getting underlying numpy array
        sample_indices = pd.Series(unique_values).sample(n=MAX_SAMPLE_SIZE_FOR_AI_CLEANING, random_state=1).index
        unique_values_to_sample = unique_values[sample_indices]
    else:
        unique_values_to_sample = unique_values
    
    if len(unique_values_to_sample) == 0:
        logger_worker.debug(f"Skipping AI text normalization for '{column_name}': no unique non-null string values to sample.")
        return column_series.copy()

    sample_list_str = json.dumps(unique_values_to_sample.tolist())
    
    normalization_instruction = ""
    example_input_str = ""
    example_output_str = ""

    # +++ Dynamic Prompt based on mode +++
    # Added more robust examples for each case
    if mode == "title_case_trim":
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Convert to Title Case (e.g., 'john doe' becomes 'John Doe', 'McDONALD's' becomes 'McDonald's'). 3. Ensure single spaces between words."
        example_input_str = json.dumps(["  john   DOE  ", "first LAST ", "ALL CAPS anD miXeD", "  leadingAndTrailing  ", "o'malley s.r.l."])
        example_output_str = json.dumps(["John Doe", "First Last", "All Caps And Mixed", "Leadingandtrailing", "O'Malley S.R.L."])
    elif mode == "lower_case_trim":
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Convert to lower case. 3. Ensure single spaces between words."
        example_input_str = json.dumps(["  John   DOE  ", "First LAST", "ALL CAPS", "  leadingAndTrailing  ", "O'MALLEY"])
        example_output_str = json.dumps(["john doe", "first last", "all caps", "leadingandtrailing", "o'malley"])
    elif mode == "upper_case_trim":
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Convert to UPPER CASE. 3. Ensure single spaces between words."
        example_input_str = json.dumps(["  John   doe  ", "first last", "mixed Case", "  leadingAndTrailing  ", "o'malley"])
        example_output_str = json.dumps(["JOHN DOE", "FIRST LAST", "MIXED CASE", "LEADINGANDTRAILING", "O'MALLEY"])
    elif mode == "trim_only":
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Ensure single spaces between internal words. Preserve original casing."
        example_input_str = json.dumps(["  John   Doe  ", "  leading", "trailing  ", "  ALL CAPS  ", "Mixed   Case Example"])
        example_output_str = json.dumps(["John Doe", "leading", "trailing", "ALL CAPS", "Mixed Case Example"])
    elif mode == "trim_preserve_internal_sep":
        normalization_instruction = (
            "1. Trim leading/trailing whitespace ONLY. "
            "2. Preserve ALL internal characters, including underscores (_), hyphens (-), etc., exactly as they are. "
            "3. Preserve original casing exactly as it is. "
            "4. If multiple internal spaces exist between words, consolidate them to a single space, but do not replace other separators like underscores with spaces."
        )
        example_input_str = json.dumps([
            "  Product_1  ", 
            "  product_NAME_two  ", 
            "  Alpha-Beta Company  ", 
            "  Keep   Multiple  Spaces  "
        ])
        example_output_str = json.dumps([
            "Product_1", # Underscore preserved, casing preserved
            "product_NAME_two", # Underscore and casing preserved
            "Alpha-Beta Company", # Hyphen and casing preserved
            "Keep Multiple Spaces" # Internal multiple spaces consolidated
        ])
    # Example for a new mode (you'd add this to frontend options too)
    # elif mode == "remove_special_chars_alpha_num_space_trim":
    #     normalization_instruction = "1. Trim leading/trailing whitespace. 2. Remove all characters except letters (any case), numbers, and single spaces between words. 3. Convert multiple internal spaces to a single space."
    #     example_input_str = json.dumps(["  Product #1! (New)  ", "  $100.00 item  ", "  Keep_This_Okay  ", "  (empty-data)  "])
    #     example_output_str = json.dumps(["Product 1 New", "100 00 item", "Keep This Okay", "empty data"])
    else:
        logger_worker.warning(f"Unknown text normalization mode '{mode}' for column '{column_name}'. Defaulting to 'title_case_trim'.")
        mode = "title_case_trim" # Fallback to a default
        normalization_instruction = "1. Trim leading/trailing whitespace. 2. Convert to Title Case (e.g., 'john doe' becomes 'John Doe'). 3. Ensure single spaces between words."
        example_input_str = json.dumps(["  john DOE  ", "first LAST", "ALL CAPS anD miXeD", "  leadingAndTrailing  "])
        example_output_str = json.dumps(["John Doe", "First Last", "All Caps And Mixed", "Leadingandtrailing"])

    prompt = (
        f"You are a data cleaning specialist. For the following sample text values from column '{column_name}':\n"
        f"{sample_list_str}\n\n"
        f"Your primary goal is to normalize each value according to these rules: {normalization_instruction}\n"
        f"Secondary goal: If a value, after normalization, becomes completely empty or consists only of whitespace, return an empty string \"\" for that item in the output list.\n\n"
        f"Output Rules:\n"
        f"1. Your response MUST be ONLY a valid JSON list of strings.\n"
        f"2. Each string in the list must correspond to an item in the input sample, in the exact same order.\n"
        f"3. The JSON list must contain exactly {len(unique_values_to_sample)} string items.\n"
        f"4. Do NOT include explanations, comments, or markdown formatting (like ```json ... ```) outside the JSON list itself.\n\n"
        f"Example Input Sample for mode '{mode}': {example_input_str}\n"
        f"Correct Example Output (JSON list of strings) for mode '{mode}': {example_output_str}\n\n"
        f"Now, process the provided input sample and return the JSON list of strings:"
    )

    logger_worker.info(f"Attempting AI text normalization (mode: {mode}) for column '{column_name}' (sample size: {len(unique_values_to_sample)})")
    ai_processed_strings = _call_gemini_for_cleaning(prompt, column_name, f"normalize_text_{mode}")

    if ai_processed_strings and len(ai_processed_strings) == len(unique_values_to_sample):
        value_map = dict(zip(unique_values_to_sample, ai_processed_strings))
        
        # Apply the mapping to the entire original column_series (not just unique_values_to_sample)
        cleaned_series_list = []
        for original_val in column_series: # Iterate over the original series to maintain order and all values
            if pd.isna(original_val):
                cleaned_series_list.append(None) # Preserve None/NaN
            else:
                original_val_str = str(original_val)
                if original_val_str in value_map:
                    cleaned_series_list.append(value_map[original_val_str])
                else:
                    # If original_val_str was not in unique_values_to_sample (e.g., it was sampled out),
                    # apply a basic Python-based version of the selected mode as a fallback.
                    # This is a pragmatic approach if AI sample is small.
                    fallback_cleaned_val = original_val_str.strip() # Basic trim always
                    if mode == "title_case_trim": fallback_cleaned_val = fallback_cleaned_val.title()
                    elif mode == "lower_case_trim": fallback_cleaned_val = fallback_cleaned_val.lower()
                    elif mode == "upper_case_trim": fallback_cleaned_val = fallback_cleaned_val.upper()
                    # For "trim_only", strip() is already done.
                    # Add more fallback logic for other modes if implemented
                    cleaned_series_list.append(re.sub(r'\s+', ' ', fallback_cleaned_val).strip()) # Consolidate multiple spaces
        
        final_cleaned_series = pd.Series(cleaned_series_list, index=column_series.index, name=column_series.name, dtype='object')
        
        logger_worker.info(f"AI successfully processed text normalization (mode: {mode}) for column '{column_name}'.")
        return final_cleaned_series
    else:
        logger_worker.warning(f"AI text normalization (mode: {mode}) failed or returned mismatched data for '{column_name}'. Original series returned after basic trim.")
        if ai_processed_strings:
            logger_worker.debug(f"AI returned {len(ai_processed_strings)} items, expected {len(unique_values_to_sample)} for '{column_name}' (mode {mode})")
        # Fallback to just stripping whitespace from the original series if AI fails
        return column_series.astype(str).str.strip().replace(r'^\s*$', '', regex=True) # Ensure completely blank strings become empty