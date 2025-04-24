# ==============================================================================
# SECTION 2: ETL Worker Code
# ==============================================================================
import os
import io
import json
import re
import logging
import pandas as pd
import requests
from dotenv import load_dotenv
from google.cloud import storage, bigquery
from google.cloud.pubsub_v1 import SubscriberClient
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound as GcpNotFound
from google.api_core.exceptions import GoogleAPICallError
# MODIFICATION: Import tempfile Correctly
import tempfile
import traceback

# --- Load Environment Variables ---
load_dotenv()

# --- Worker Logging Setup (Keep as before) ---
log_level_name = os.getenv("LOG_LEVEL", "DEBUG").upper()
log_level = getattr(logging, log_level_name, logging.DEBUG)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(funcName)s - %(message)s')
logging.getLogger('google').setLevel(logging.WARNING); logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING); logging.getLogger('pandas').setLevel(logging.WARNING)
logger_worker = logging.getLogger(__name__ + "_worker")
logger_worker.info(f"ETL Worker script initializing... Log level: {log_level_name}")


# --- Worker Configuration & Validation (Keep as corrected before) ---
# --- Initialize Worker Clients (Keep as before) ---
# --- Worker Helper Functions (Keep as before) ---
# (Keep sanitize_bq_name, get_file_extension, map_pandas_dtype_to_bq)
# ============================ Start of pasted section ============================
# --- Worker Configuration (Corrected) ---
WORKER_GCP_PROJECT = os.getenv("GCP_PROJECT")
WORKER_GCS_BUCKET = os.getenv("GCS_BUCKET")
WORKER_BQ_DATASET = os.getenv("BQ_DATASET")
WORKER_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
WORKER_PUBSUB_SUBSCRIPTION = os.getenv("PUBSUB_SUBSCRIPTION")
WORKER_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json") # Default filename

WORKER_DEFAULT_SCHEMA_STRATEGY = os.getenv("SCHEMA_STRATEGY", "existing_or_gemini")
WORKER_DEFAULT_WRITE_DISPOSITION = os.getenv("BQ_WRITE_DISPOSITION", "WRITE_APPEND")
WORKER_MAX_HEADER_SEARCH_RANGE = int(os.getenv("MAX_HEADER_SEARCH_RANGE", 10))
WORKER_GEMINI_SAMPLE_SIZE = int(os.getenv("GEMINI_SAMPLE_SIZE", 5))
WORKER_ROW_DENSITY_THRESHOLD = float(os.getenv("ROW_DENSITY_THRESHOLD", 0.3))
WORKER_GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", 90))
WORKER_BQ_LOAD_TIMEOUT = int(os.getenv("BQ_LOAD_TIMEOUT_SECONDS", 300))


# --- Input Validation for Worker (Corrected) ---
required_vars = {
    "GCP_PROJECT": WORKER_GCP_PROJECT, "GCS_BUCKET": WORKER_GCS_BUCKET,
    "BQ_DATASET": WORKER_BQ_DATASET, "PUBSUB_SUBSCRIPTION": WORKER_PUBSUB_SUBSCRIPTION }
missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    error_msg = f"Worker Error: Missing required environment variables: {', '.join(missing_vars)}."
    logger_worker.critical(error_msg); raise ValueError(error_msg)
if not os.path.exists(WORKER_CREDENTIALS_PATH):
     error_msg = f"Worker Creds file not found: {WORKER_CREDENTIALS_PATH}"
     logger_worker.critical(error_msg); raise FileNotFoundError(error_msg)
if "gemini" in WORKER_DEFAULT_SCHEMA_STRATEGY and not WORKER_GEMINI_API_KEY:
     logger_worker.warning("SCHEMA_STRATEGY uses Gemini, but GEMINI_API_KEY not set.")

# --- Initialize Worker Clients (Keep as before) ---
try:
    worker_creds = service_account.Credentials.from_service_account_file(WORKER_CREDENTIALS_PATH)
    worker_storage_client = storage.Client(credentials=worker_creds, project=WORKER_GCP_PROJECT)
    worker_bq_client = bigquery.Client(credentials=worker_creds, project=WORKER_GCP_PROJECT)
    worker_subscriber = SubscriberClient(credentials=worker_creds) # Add retry/timeout config?
    logger_worker.info(f"Worker Clients initialized. Project: {WORKER_GCP_PROJECT}, Sub: {WORKER_PUBSUB_SUBSCRIPTION}")
except Exception as e: logger_worker.critical(f"Worker Client Init Failed: {e}", exc_info=True); worker_storage_client = worker_bq_client = worker_subscriber = None; raise

# --- Worker Helper Functions (Keep existing: sanitize_bq_name, get_file_extension, map_pandas_dtype_to_bq) ---
def sanitize_bq_name(name: str) -> str:
    if not isinstance(name, str): name = str(name)
    name = name.strip(); name = re.sub(r'[^\w]', '_', name)
    if name and not re.match(r'^[a-zA-Z_]', name): name = '_' + name
    if not name: name = '_unnamed'; return name[:300]
    return name[:300]

def get_file_extension(object_name: str) -> str: return os.path.splitext(object_name)[1].lower()

def map_pandas_dtype_to_bq(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype): return 'INTEGER'
    if pd.api.types.is_float_dtype(dtype): return 'FLOAT'
    if pd.api.types.is_bool_dtype(dtype) or str(dtype).lower() == 'boolean': return 'BOOLEAN'
    if pd.api.types.is_datetime64_any_dtype(dtype): return 'TIMESTAMP'
    if pd.api.types.is_timedelta64_dtype(dtype): return 'INTERVAL'
    if pd.api.types.is_categorical_dtype(dtype): return 'STRING'
    return 'STRING'
# ============================= End of pasted section =============================


# --- Worker Core Functions ---

def _read_excel_with_fallback(io_source, file_ext, pandas_opts):
    """Internal helper to read excel, trying csv if worksheet error occurs on .xls"""
    is_xls = file_ext == '.xls'
    is_xlsx = file_ext == '.xlsx'
    df = None
    try:
        logger_worker.debug(f"Attempting pd.read_excel (engine=None) opts: {pandas_opts} source type: {type(io_source)}")
        # Ensure stream is reset if possible, may not work for all stream types
        if hasattr(io_source, 'seek') and callable(io_source.seek):
            try: io_source.seek(0); logger_worker.debug("Reset source stream position to 0.")
            except io.UnsupportedOperation: logger_worker.debug("Source stream does not support seek.")

        df = pd.read_excel(io_source, engine=None, **pandas_opts)
        logger_worker.debug(f"pd.read_excel successful. Shape: {df.shape if df is not None else 'None'}")
        return df # Success
    except ValueError as ve:
        if "Worksheet index" in str(ve) or "No sheet named" in str(ve): # Handle multiple pandas versions?
            logger_worker.warning(f"Pandas Worksheet index/name error: {ve}")
            if is_xls:
                logger_worker.info("Worksheet error on .xls. Attempting fallback read as CSV...")
                try:
                    if hasattr(io_source, 'seek') and callable(io_source.seek): io_source.seek(0)
                    # If io_source is a stream, need BytesIO for CSV reader's potential needs
                    elif isinstance(io_source, (io.RawIOBase, io.BufferedIOBase)):
                        logger_worker.debug("Wrapping stream in BytesIO for CSV attempt.")
                        # io_source is already exhausted or unseekable, cannot reread reliably here from stream.
                        # This highlights a limitation. CSV fallback better done from file path if available.
                        logger_worker.warning("Cannot reliably retry CSV from exhausted/unseekable stream. Worksheet error is likely fatal for stream path.")
                        raise ve # Reraise original error

                    io_source_csv = io_source # Assume it's path if not stream or already BytesIO

                    csv_opts = {'dtype': str, 'header': pandas_opts.get('header')} # Keep header opt consistent
                    logger_worker.debug(f"Attempting pd.read_csv on potentially .xls content: {csv_opts}")
                    try:
                        df_csv = pd.read_csv(io_source_csv, sep=None, engine='python', **csv_opts)
                    except (ValueError, pd.errors.ParserError):
                        logger_worker.debug("CSV auto-sep failed, trying comma...")
                        if hasattr(io_source_csv, 'seek') and callable(io_source_csv.seek): io_source_csv.seek(0)
                        df_csv = pd.read_csv(io_source_csv, sep=',', engine='python', **csv_opts)

                    logger_worker.info("Successfully read presumed .xls file content as CSV fallback.")
                    return df_csv # Success using CSV reader
                except Exception as csv_err:
                    logger_worker.error(f"Fallback CSV read failed for .xls source: {csv_err}")
                    raise ve # Reraise original Worksheet error
            else: logger_worker.error(f"Worksheet error on {file_ext} - not .xls. Failing."); raise
        else: logger_worker.error(f"Other pd.read_excel ValueError: {ve}"); raise
    except ImportError as ie:
         if is_xls and 'xlrd' in str(ie).lower(): logger_worker.critical("MISSING: Reading .xls requires `xlrd`. `pip install xlrd`.")
         elif is_xlsx and 'openpyxl' in str(ie).lower(): logger_worker.critical("MISSING: Reading .xlsx requires `openpyxl`. `pip install openpyxl`.")
         else: logger_worker.error(f"ImportError reading Excel: {ie}")
         raise # Reraise import error to stop
    except Exception as e: logger_worker.error(f"General pd.read_excel error: {e}"); raise


def read_data_from_gcs(bucket_name: str, object_name: str) -> pd.DataFrame | None:
    """Reads data from GCS using streaming, with disk fallback and CSV fallback for XLS."""
    if not worker_storage_client: logger_worker.error("Storage client inactive."); raise ConnectionError("Storage client inactive")

    logger_worker.info(f"Reading data from gs://{bucket_name}/{object_name}")
    bucket = worker_storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    processed_df = None; df = None; df_temp_stream = None; df_from_stream = None

    # 1. Check Existence
    try:
        if not blob.exists(): logger_worker.error(f"File not found: {object_name}"); raise FileNotFoundError(f"GCS object not found: {object_name}")
        logger_worker.debug(f"File exists. Size: {blob.size}. Proceeding.")
    except FileNotFoundError: raise
    except Exception as e: logger_worker.error(f"GCS exist check failed: {e}"); return None

    # 2. Attempt Read (Stream first, then file fallback)
    try:
        file_ext = get_file_extension(object_name)
        supported_excel=['.xlsx','.xls']; supported_csv=['.csv']; supported_parquet=['.parquet']
        supported_read=supported_excel+supported_csv+supported_parquet
        read_successful = False

        # --- Attempt 1: Streaming Read ---
        if file_ext in supported_read:
            logger_worker.info(f"Attempting streaming read for {file_ext} file...")
            try:
                with blob.open("rb") as gcs_stream:
                    if file_ext in supported_parquet:
                         df = pd.read_parquet(gcs_stream) # Final DF for parquet
                    elif file_ext in supported_excel + supported_csv:
                        # Stream Header Search Read
                        logger_worker.debug("Phase 1 (Stream): Header Search Read...")
                        # Excel reads try CSV fallback internally via helper
                        if file_ext in supported_excel: df_temp_stream = _read_excel_with_fallback(gcs_stream, file_ext, {'dtype': str, 'header': None})
                        else: df_temp_stream = pd.read_csv(gcs_stream, sep=None, engine='python', dtype=str, header=None)
                        if df_temp_stream is None: raise ValueError("Stream header search read failed.")

                        # Header detection logic
                        detected_header_row=0; best_hdr=-1; max_score=-1; rows_to_chk=min(len(df_temp_stream),WORKER_MAX_HEADER_SEARCH_RANGE);
                        for i in range(rows_to_chk):
                            row_series=df_temp_stream.iloc[i].astype(str);n=row_series[row_series!='None'].notna().sum();d=row_series[row_series.notna()&(row_series!='None')].nunique();ml=max(len(str(c))for c in row_series.tolist() if c) if n>0 else 0;lp=5 if ml>100 else 0;nd=sum(1 for c in row_series.dropna() if pd.to_numeric(str(c),errors='coerce')is not None or re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$',str(c)));t=n-nd;dm=(nd>t*1.5)&n>1;s=(d*1.5)+t-(len(row_series)-n)-lp-(5 if dm else 0);
                            if s>max_score and n>=len(row_series)*WORKER_ROW_DENSITY_THRESHOLD and n>1: max_score=s;best_hdr=i
                        detected_header_row = best_hdr if best_hdr !=-1 else 0; logger_worker.info(f"Header determined at row {detected_header_row}")

                        # Stream Main Read
                        logger_worker.debug(f"Phase 2 (Stream): Main Read (Header {detected_header_row})...")
                        with blob.open("rb") as gcs_stream_main: # Fresh stream essential
                             if file_ext in supported_excel: df = _read_excel_with_fallback(gcs_stream_main, file_ext, {'dtype': str, 'header': detected_header_row})
                             else: # CSV
                                 try: df = pd.read_csv(gcs_stream_main, sep=None, engine='python', dtype=str, header=detected_header_row)
                                 except (ValueError, pd.errors.ParserError):
                                      with blob.open("rb") as gcs_stream_comma: # Fresh again for comma
                                          df = pd.read_csv(gcs_stream_comma, sep=',', engine='python', dtype=str, header=detected_header_row)
                    # End of Excel/CSV stream read section

                if df is None: raise ValueError("Stream read resulted in None DataFrame.")
                logger_worker.info(f"Streaming read successful. Shape before isolation: {df.shape}")
                read_successful = True # Mark stream success

            except Exception as stream_err:
                logger_worker.warning(f"Streaming read failed: {type(stream_err).__name__}: {stream_err}. Attempting disk fallback.", exc_info=False) # Don't need full trace here usually
                # exc_info=True can be very verbose if the error object is large (e.g. some Google API errors)
                if logger_worker.isEnabledFor(logging.DEBUG): # Only log full trace if debugging
                      logger_worker.debug("Stream failure traceback:", exc_info=True)


        # --- Attempt 2: Download-to-Disk Fallback ---
        # Trigger ONLY if stream failed AND format is potentially readable from file
        if not read_successful and file_ext in (supported_excel + supported_csv):
            logger_worker.info(f"Executing download-to-disk fallback for {file_ext} file...")
            # Use TemporaryDirectory to avoid file handle locks
            with tempfile.TemporaryDirectory() as tmpdir:
                # Construct path within the temp directory
                safe_basename = re.sub(r'[^\w.]+', '_', os.path.basename(object_name)) # Sanitize basename slightly
                temp_file_path = os.path.join(tmpdir, safe_basename)
                logger_worker.debug(f"Downloading to temp path: {temp_file_path}")
                try:
                    blob.download_to_filename(temp_file_path) # Download using the path
                    logger_worker.debug(f"Download to temp file complete. Reading from disk...")

                    # Now read from the downloaded file path, applying same logic
                    # Phase 1 File Header Read
                    temp_read_opts_file = {'dtype': str, 'header': None}
                    if file_ext in supported_excel: df_temp_file = _read_excel_with_fallback(temp_file_path, file_ext, temp_read_opts_file)
                    else: df_temp_file = pd.read_csv(temp_file_path, sep=None, engine='python', **temp_read_opts_file)
                    if df_temp_file is None: raise ValueError("File header read returned None (Fallback)")

                    # Header detect logic on df_temp_file
                    detected_header_row_file=0; best_hdr=-1; max_score=-1; rows_to_chk=min(len(df_temp_file),WORKER_MAX_HEADER_SEARCH_RANGE);
                    for i in range(rows_to_chk): # PASTE SCORE LOGIC
                       row_series=df_temp_file.iloc[i].astype(str);n=row_series[row_series!='None'].notna().sum();d=row_series[row_series.notna()&(row_series!='None')].nunique();ml=max(len(str(c))for c in row_series.tolist() if c) if n>0 else 0;lp=5 if ml>100 else 0;nd=sum(1 for c in row_series.dropna() if pd.to_numeric(str(c),errors='coerce')is not None or re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$',str(c)));t=n-nd;dm=(nd>t*1.5)&n>1;s=(d*1.5)+t-(len(row_series)-n)-lp-(5 if dm else 0);
                       if s > max_score and n >= len(row_series) * WORKER_ROW_DENSITY_THRESHOLD and n > 1: max_score = s; best_hdr = i
                    detected_header_row_file = best_hdr if best_hdr!=-1 else 0; logger_worker.info(f"Header (File) @ row {detected_header_row_file}")

                    # Phase 2 File Main Read
                    read_opts_file = {'dtype': str, 'header': detected_header_row_file}
                    if file_ext in supported_excel: df = _read_excel_with_fallback(temp_file_path, file_ext, read_opts_file) # Final DF from file
                    else: # CSV
                       try: df = pd.read_csv(temp_file_path, sep=None, engine='python', **read_opts_file)
                       except (ValueError, pd.errors.ParserError): df = pd.read_csv(temp_file_path, sep=',', engine='python', **read_opts_file)

                    if df is None: raise ValueError("Read from temp file failed.")
                    logger_worker.info(f"Fallback read from temp file successful. Shape: {df.shape}")
                    read_successful = True # Mark success after fallback

                except PermissionError as pe:
                    # Specific handling if temp dir still has issues (unlikely with TemporaryDirectory)
                    logger_worker.error(f"Permission error accessing temporary file '{temp_file_path}': {pe}", exc_info=True)
                    return None # Permission errors often aren't recoverable by retry
                except Exception as file_fallback_err:
                    logger_worker.error(f"Fallback attempt using temp file failed: {file_fallback_err}", exc_info=True)
                    # Keep read_successful as False
                    df = None # Ensure df is None if fallback failed

        # Check final status
        if not read_successful:
             logger_worker.error(f"All read attempts failed for file {object_name}.")
             return None # Return None if neither stream nor fallback worked


        # --- Block Isolation (Now operates on `df` which contains result from either stream or file) ---
        if df is not None and not df.empty:
            logger_worker.debug("Isolating data block from final DataFrame...")
            df_processed=df.copy(); df_processed.dropna(axis=1,how='all',inplace=True); df_processed.dropna(axis=0,how='all',inplace=True)
            if not df_processed.empty:
                df_processed=df_processed.reset_index(drop=True);start_idx,end_idx=-1,-1;min_nn=max(1,int(len(df_processed.columns)*WORKER_ROW_DENSITY_THRESHOLD))
                for i in range(len(df_processed)): #Find start
                    if df_processed.iloc[i].replace(['None','nan','NaN','<NA>'],pd.NA).count()>=min_nn: start_idx=i; break
                if start_idx!=-1: # Find end
                    for i in range(len(df_processed)-1,start_idx-1,-1):
                        if df_processed.iloc[i].replace(['None','nan','NaN','<NA>'],pd.NA).count()>=min_nn: end_idx=i; break
                if start_idx!=-1 and end_idx!=-1 and start_idx<=end_idx:
                     logger_worker.info(f"Slicing block {start_idx}-{end_idx}");processed_df=df_processed.iloc[start_idx:end_idx+1].reset_index(drop=True);logger_worker.info(f"Final isolated block shape: {processed_df.shape}")
                else: logger_worker.warning("No valid data block found."); processed_df=pd.DataFrame()
            else: logger_worker.warning("DF empty after initial processing."); processed_df=pd.DataFrame()
        elif df is None: # Should have been caught earlier, but defensive check
             logger_worker.error("Logic error: DataFrame is None before final block isolation stage.")
             processed_df = None # Explicitly signal failure
        else: # DataFrame is empty
              logger_worker.warning("DataFrame was empty before block isolation stage.")
              processed_df = df # Return the empty frame

    except FileNotFoundError: # GCS file missing
        logger_worker.error(f"FileNotFoundError caught for {object_name}")
        raise # Re-raise for process_object -> ACK
    except ImportError as imp_err: # Missing xlrd etc.
         logger_worker.critical(f"ImportError during read: {imp_err}. Required lib may be missing.", exc_info=True)
         return None # Trigger NACK
    except Exception as e: # Catchall
        logger_worker.error(f"Unexpected error during overall read process for {object_name}: {e}", exc_info=True)
        return None # Trigger NACK

    # Return the final processed DataFrame (can be empty)
    return processed_df


# ============================ Start of pasted remaining functions ============================
# (Paste the *exact* same clean_dataframe, infer_schema_gemini, infer_schema_pandas,
# get_existing_schema, determine_schema, align_dataframe_to_schema, load_to_bq,
# process_object, callback, and if __name__ == "__main__": blocks here as they were
# in the previous fully working code - they handle the None/empty return from read_data)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: logger_worker.warning("Input DF to clean_dataframe empty."); return df
    logger_worker.info("Cleaning isolated data block...")
    first_row_values = df.iloc[0].astype(str)
    num_cols = len(df.columns)
    first_row_non_null = first_row_values.replace(['None','nan','NaN'], pd.NA).count()
    first_row_distinct = first_row_values.replace(['None','nan','NaN'], pd.NA).nunique()
    is_likely_header = (first_row_distinct / max(1, first_row_non_null) > 0.8) and (first_row_non_null / max(1, num_cols) > 0.6) and first_row_distinct > 1
    current_cols_are_generic = all(re.match(r"^(col_|\d+$)", str(col)) for col in df.columns)
    original_cols_before_promote = list(df.columns)
    if is_likely_header and (current_cols_are_generic or first_row_distinct > df.columns.nunique()):
         logger_worker.info("Promoting first row of block to header.")
         df.columns = first_row_values; df = df[1:].reset_index(drop=True)
         if df.empty: logger_worker.warning("DF empty after header promotion."); return df
         logger_worker.debug(f"Cols after promotion: {list(df.columns)}")
    original_columns = df.columns
    sanitized_columns = [sanitize_bq_name(str(col)) for col in original_columns]
    final_columns = []
    counts = {}
    for col in sanitized_columns: # Handle duplicates
        current_count = counts.get(col, 0)
        final_columns.append(f"{col}_{current_count}"[:300] if current_count > 0 else col)
        counts[col] = current_count + 1
    if list(original_columns) != final_columns or list(original_cols_before_promote) != final_columns:
         logger_worker.debug(f"Original cols: {list(original_cols_before_promote)}")
         logger_worker.debug(f"Sanitized cols: {final_columns}")
    df.columns = final_columns
    df.dropna(axis=1, how='all', inplace=True); df.dropna(axis=0, how='all', inplace=True)
    if df.empty: logger_worker.warning("DF empty after drops in cleaning."); return df.reset_index(drop=True)
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=['object', 'string']).columns:
        if col in df_cleaned.columns:
             try: df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
             except Exception as strip_err: logger_worker.warning(f"Strip failed col '{col}': {strip_err}")
    common_nulls = ['', 'none', 'null', 'nan', '<na>', 'nat'];
    for null_val in common_nulls: df_cleaned.replace({null_val: pd.NA, null_val.upper(): pd.NA}, inplace=True)
    logger_worker.info(f"Cleaning complete. Shape: {df_cleaned.shape}")
    return df_cleaned.reset_index(drop=True)

def infer_schema_gemini(df: pd.DataFrame) -> list | None:
    if df.empty: logger_worker.warning("Cannot infer schema from empty DataFrame."); return None
    if not WORKER_GEMINI_API_KEY: logger_worker.error("GEMINI_API_KEY not set."); return None
    sample_df = df.head(WORKER_GEMINI_SAMPLE_SIZE).copy()
    for col in sample_df.columns: # Replace NA with 'null' string
        if sample_df[col].isnull().any(): sample_df[col] = sample_df[col].astype(object).where(sample_df[col].notnull(), 'null')
        sample_df[col] = sample_df[col].astype(str)
    sample = sample_df.to_dict(orient="records")
    if not sample or all(all(v == 'null' or not v for v in row.values()) for row in sample): logger_worker.error("Sample data null/empty after prep for Gemini."); return None
    logger_worker.info(f"Sending sample to Gemini ({len(sample)} rows)...")
    prompt = ("Analyze the following sample data rows (represented as JSON objects) from a table. Suggest appropriate Google BigQuery column names and data types.\nRULES:\n1. The keys in the input JSON objects are the *intended* column names. Use these *exact keys* for the 'name' field in your response JSON. Do not invent new names.\n2. The suggested BigQuery type in the 'type' field MUST be chosen from this list: STRING, INTEGER, FLOAT, NUMERIC, BOOLEAN, TIMESTAMP, DATE, TIME, DATETIME, GEOGRAPHY, JSON, BYTES. Default to STRING if unsure.\n3. Base the data type suggestion ONLY on the sample *values* provided for each key.\n4. Your response MUST be *only* a single valid JSON list of objects. Each object must have exactly two keys: 'name' (string, matching input key) and 'type' (string, one of the allowed BQ types).\n5. Do not include ```json ``` markers or any other text outside the JSON list.\nSAMPLE DATA:\n" + f"{json.dumps(sample)}\n\nJSON Schema:")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={WORKER_GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}}
    headers = {"Content-Type": "application/json"}; result = None
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=WORKER_GEMINI_TIMEOUT)
        response.raise_for_status(); result = response.json()
        if not result or not result.get("candidates"): raise ValueError("Invalid Gemini response: missing candidates.")
        schema_content = result["candidates"][0].get("content",{}).get("parts",[{}])[0].get("text")
        if not schema_content: raise ValueError("Gemini response text empty.")
        try: schema_list = json.loads(schema_content)
        except json.JSONDecodeError: cleaned = re.sub(r"```json\s*([\s\S]*?)\s*```",r"\1",schema_content,flags=re.I).strip(); schema_list=json.loads(cleaned)
        if not isinstance(schema_list, list): raise ValueError("Gemini result not a list.")
        if not schema_list: raise ValueError("Gemini result empty list.")
        validated_schema = []; valid_bq_types={"STRING","BYTES","INTEGER","INT64","FLOAT","FLOAT64","NUMERIC","BIGNUMERIC","BOOLEAN","BOOL","TIMESTAMP","DATE","TIME","DATETIME","GEOGRAPHY","JSON","INTERVAL"}
        input_cols = set(df.columns)
        for item in schema_list:
             name=item.get("name"); type_val=item.get("type")
             if not isinstance(item, dict) or not isinstance(name, str) or not isinstance(type_val, str) or not name or not type_val: raise ValueError(f"Invalid item: {item}")
             if name not in input_cols: logger_worker.warning(f"Gemini name '{name}' mismatch DF.")
             bq_type = type_val.upper();
             if bq_type not in valid_bq_types: logger_worker.warning(f"Invalid BQ type '{type_val}'-> STRING"); bq_type="STRING"
             validated_schema.append({"name": name, "type": bq_type})
        logger_worker.info(f"✅ Gemini schema: {json.dumps(validated_schema)}")
        return validated_schema
    except requests.exceptions.Timeout: logger_worker.error(f"Gemini API timed out ({WORKER_GEMINI_TIMEOUT}s).")
    except requests.exceptions.RequestException as e: logger_worker.error(f"Gemini API request failed: {e}")
    except Exception as e: logger_worker.exception(f"Error in Gemini inference: {e}")
    return None

def infer_schema_pandas(df: pd.DataFrame) -> list:
    if df.empty: logger_worker.warning("Cannot infer pandas schema empty DF."); return []
    logger_worker.info("Inferring schema using pandas dtypes...")
    try:
        df_inferred = df.copy()
        for col in df_inferred.columns:
            original=df_inferred[col]; converted=None
            try: df_inferred[col]=pd.to_numeric(original,errors='raise',downcast='integer'); continue
            except: pass
            try: df_inferred[col]=pd.to_numeric(original,errors='raise'); continue
            except: pass
            try:
                converted_dt=pd.to_datetime(original,errors='raise')
                if not original.astype(str).str.match(r'^\d+$').all(): df_inferred[col]=converted_dt; continue
            except: pass
        df_inferred = df_inferred.infer_objects()
    except Exception as e: logger_worker.warning(f"Pandas inference failed: {e}. Using original."); df_inferred=df
    schema=[{"name": str(col), "type": map_pandas_dtype_to_bq(df_inferred[col].dtype)} for col in df_inferred.columns]
    logger_worker.debug(f"Pandas schema: {schema}")
    return schema

def get_existing_schema(table_id: str) -> list[bigquery.SchemaField] | None:
    if not worker_bq_client: logger_worker.error("BQ Client not init!"); return None
    try: table = worker_bq_client.get_table(table_id); logger_worker.info(f"Found existing schema: {table_id}"); return table.schema
    except GcpNotFound: logger_worker.info(f"Table {table_id} not found."); return None
    except Exception as e: logger_worker.error(f"Error fetching schema {table_id}: {e}"); return None

def determine_schema(df: pd.DataFrame, table_id: str, strategy: str) -> list | None:
    logger_worker.info(f"Determining schema for {table_id} strategy: {strategy}")
    existing=get_existing_schema(table_id); final=None
    if strategy.startswith("existing_or_") and existing: logger_worker.info("Using existing BQ schema."); final=[{"name":f.name,"type":f.field_type,"mode":f.mode} for f in existing]
    if final is None:
        if "gemini" in strategy: logger_worker.info("Trying Gemini..."); final=infer_schema_gemini(df);
        if final is None and "pandas" in strategy.replace("existing_or_",""): logger_worker.info("Using Pandas..."); final=infer_schema_pandas(df);
    if not final: logger_worker.error("Schema determination failed."); return None
    schema_names={i['name'] for i in final}; df_names=set(df.columns); common=schema_names.intersection(df_names)
    if not common: logger_worker.error("Schema mismatch: 0 common cols!"); raise ValueError("Schema mismatch: No common columns.")
    missing=df_names-schema_names; extra=schema_names-df_names
    if missing: logger_worker.warning(f"DF Cols to drop: {sorted(list(missing))}")
    if extra: logger_worker.info(f"Schema Cols to add null: {sorted(list(extra))}")
    for item in final: item.setdefault('mode','NULLABLE')
    return final

def align_dataframe_to_schema(df: pd.DataFrame, schema_list: list) -> pd.DataFrame:
    if df.empty: logger_worker.warning("Align input DF empty."); return df
    logger_worker.info("Aligning DF to schema...")
    target_flds={i['name']: i for i in schema_list}; target_cols_ord=[i['name'] for i in schema_list]; target_cols_set=set(target_cols_ord); current_cols=set(df.columns)
    cols_drop=current_cols-target_cols_set;
    if cols_drop: logger_worker.warning(f"Dropping DF cols: {sorted(list(cols_drop))}"); df=df.drop(columns=list(cols_drop))
    cols_add=target_cols_set-current_cols;
    for col in cols_add: df[col]=pd.NA
    final_ord_cols=[c for c in target_cols_ord if c in df.columns]; df=df[final_ord_cols]
    logger_worker.debug("Attempting type conversions...")
    for col_name in df.columns:
        if col_name not in target_flds: continue
        target_type=target_flds[col_name]['type'].upper(); current_series=df[col_name]
        if current_series.isnull().all(): continue
        converted=None
        try:
            if target_type in ('INTEGER','INT64'): converted = pd.to_numeric(current_series, errors='coerce').astype('Int64')
            elif target_type in ('FLOAT','FLOAT64','NUMERIC','BIGNUMERIC'): converted = pd.to_numeric(current_series, errors='coerce').astype(float)
            elif target_type in ('BOOLEAN','BOOL'):
                bool_map={'true':True,'t':True,'yes':True,'y':True,'1':True,'1.0':True,'false':False,'f':False,'no':False,'n':False,'0':False,'0.0':False,'':pd.NA,'nan':pd.NA,'none':pd.NA,'null':pd.NA}
                if pd.api.types.is_object_dtype(current_series.dtype) or pd.api.types.is_string_dtype(current_series.dtype): converted = current_series.astype(str).str.lower().fillna('').map(bool_map)
                else: converted = current_series
                converted = converted.astype('boolean')
            elif target_type in ('TIMESTAMP','DATETIME','DATE','TIME'):
                dt_series=pd.to_datetime(current_series,errors='coerce')
                if target_type=='DATE': converted=dt_series.dt.date
                elif target_type=='TIME': converted=dt_series.dt.time
                else: converted=dt_series
            elif target_type=='STRING': converted=current_series.astype(str)
            elif target_type=='JSON': converted=current_series.astype(str)
            elif target_type=='BYTES': logger_worker.warning("BYTES default to STRING."); converted=current_series.astype(str)
            if converted is not None: df[col_name] = converted
        except Exception as e:
            logger_worker.warning(f"Conv Error '{col_name}'->{target_type}:{e}. Use STRING.")
            try:
                df[col_name] = df[col_name].astype(str)
            except:
                pass

    logger_worker.info("Alignment/Conversion complete.")
    return df.astype(object).where(pd.notnull(df), None)

def load_to_bq(df: pd.DataFrame, table_id: str, schema_list: list, write_disposition: str):
    if not worker_bq_client: logger_worker.error("BQ Client not init!"); raise ConnectionError("BQ Client not init")
    if df.empty: logger_worker.warning(f"Empty DF skip load: {table_id}"); return
    logger_worker.info(f"Loading {len(df)} rows to {table_id}, disposition: {write_disposition}")
    try: bq_schema = [bigquery.SchemaField(i['name'],i['type'],mode=i.get('mode','NULLABLE')) for i in schema_list]
    except Exception as e: logger_worker.error(f"Bad schema: {e}"); raise ValueError("Bad schema format")
    job_config = bigquery.LoadJobConfig(schema=bq_schema, write_disposition=write_disposition, autodetect=False, source_format=bigquery.SourceFormat.PARQUET)
    try:
        load_job = worker_bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
        logger_worker.info(f"Submitted BQ Load Job: {load_job.job_id}")
        load_job.result(timeout=WORKER_BQ_LOAD_TIMEOUT)
        if load_job.errors: logger_worker.error(f"BQ Job {load_job.job_id} Errors:");[logger_worker.error(f"  - {e['reason']}:{e['message']}") for e in load_job.errors]; raise Exception("BQ load failed.")
        else: logger_worker.info(f"Load OK: {table_id}. Rows: {worker_bq_client.get_table(table_id).num_rows}")
    except Exception as e: logger_worker.error(f"BQ Load Error {table_id}: {e}", exc_info=True); raise

def process_object(object_name: str):
    logger_worker.info(f"--- Starting processing: {object_name} ---")
    try:
        try: df_read = read_data_from_gcs(WORKER_GCS_BUCKET, object_name)
        except FileNotFoundError: logger_worker.warning(f"Not found '{object_name}'. Skip ACK."); return
        if df_read is None: logger_worker.error(f"read_data returned None: {object_name}."); raise ValueError(f"Read fail(None):{object_name}")
        if df_read.empty: logger_worker.warning(f"No data isolated: {object_name}. Skip ACK."); return
        df_cleaned = clean_dataframe(df_read)
        if df_cleaned.empty: logger_worker.warning(f"Empty post-clean: {object_name}. Skip ACK."); return
        base=os.path.splitext(os.path.basename(object_name))[0]; tbl=sanitize_bq_name(base); table_id=f"{WORKER_GCP_PROJECT}.{WORKER_BQ_DATASET}.{tbl}"; logger_worker.info(f"Target BQ table: {table_id}")
        schema = determine_schema(df_cleaned, table_id, WORKER_DEFAULT_SCHEMA_STRATEGY)
        if schema is None: raise ValueError(f"Schema determination failed: {object_name}")
        df_aligned = align_dataframe_to_schema(df_cleaned.copy(), schema)
        if df_aligned.empty and not df_cleaned.empty: raise ValueError("DF emptied on align.")
        elif df_aligned.empty: logger_worker.warning("DF empty pre-load. Skip ACK."); return
        load_to_bq(df_aligned, table_id, schema, WORKER_DEFAULT_WRITE_DISPOSITION)
        logger_worker.info(f"--- Success processing: {object_name} ---")
    except Exception as e: logger_worker.error(f"Error processing {object_name}: {type(e).__name__}-{e}", exc_info=True); raise

def callback(message):
    object_name = None
    
    try:
        object_name = message.data.decode("utf-8")
        logger_worker.info(f"Received msg, processing: {object_name}")
        process_object(object_name)
        message.ack()
        logger_worker.info(f"ACK message: {object_name}")
    except Exception as e: logger_worker.exception(f"CRITICAL error processing {object_name or 'UNK'}: {e}"); message.nack(); logger_worker.info(f"NACK message: {object_name or 'UNK'}")

if __name__ == "__main__":
    if not worker_subscriber or not WORKER_PUBSUB_SUBSCRIPTION: logger_worker.critical("Worker client/sub invalid. Exit."); exit(1)
    worker_sub_path = worker_subscriber.subscription_path(WORKER_GCP_PROJECT, WORKER_PUBSUB_SUBSCRIPTION)
    logger_worker.info(f"Starting Listener: {worker_sub_path}...")
    streaming_pull_future = worker_subscriber.subscribe(worker_sub_path, callback=callback)
    logger_worker.info("Listener started. Waiting...")
    try: streaming_pull_future.result()
    except TimeoutError: logger_worker.warning("Listener timeout.")
    except KeyboardInterrupt: logger_worker.info("Keyboard interrupt...")
    except Exception as e: logger_worker.exception(f"Listener loop error: {e}")
    finally:
        if 'streaming_pull_future' in locals() and streaming_pull_future:
            logger_worker.info("Cancelling listener..."); streaming_pull_future.cancel();
            try: streaming_pull_future.result(timeout=5)
            except Exception: pass
        logger_worker.info("Listener stopped.")
# ============================== End of pasted section ==============================