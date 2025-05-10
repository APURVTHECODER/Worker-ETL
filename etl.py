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
import base64
import atexit
from google.oauth2 import service_account
from dotenv import load_dotenv
from google.cloud import storage, bigquery
from google.cloud.pubsub_v1 import SubscriberClient
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound as GcpNotFound
from google.api_core.exceptions import GoogleAPICallError
import tempfile
import traceback
# +++ MODIFICATION START +++
# Import typing for better type hints
from typing import Dict, Union, List, Optional, Any # Added Optional, Any, List
# +++ MODIFICATION END +++
# --- Multi-Table Detection Imports ---
import numpy as np
try:
    import scipy.ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# For logging thread IDs
import threading

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

# ============================ Start of pasted section ============================
# --- Worker Configuration (Corrected) ---
WORKER_GCP_PROJECT = os.getenv("GCP_PROJECT")
WORKER_GCS_BUCKET = os.getenv("GCS_BUCKET")
WORKER_BQ_DATASET = os.getenv("BQ_DATASET")
WORKER_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
WORKER_PUBSUB_SUBSCRIPTION = os.getenv("PUBSUB_SUBSCRIPTION")
# WORKER_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json") # Default filename

WORKER_DEFAULT_SCHEMA_STRATEGY = os.getenv("SCHEMA_STRATEGY", "existing_or_gemini")
WORKER_DEFAULT_WRITE_DISPOSITION = os.getenv("BQ_WRITE_DISPOSITION", "WRITE_APPEND")
WORKER_MAX_HEADER_SEARCH_RANGE = int(os.getenv("MAX_HEADER_SEARCH_RANGE", 10))
WORKER_GEMINI_SAMPLE_SIZE = int(os.getenv("GEMINI_SAMPLE_SIZE", 5))
WORKER_ROW_DENSITY_THRESHOLD = float(os.getenv("ROW_DENSITY_THRESHOLD", 0.3))
WORKER_GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", 90))
WORKER_BQ_LOAD_TIMEOUT = int(os.getenv("BQ_LOAD_TIMEOUT_SECONDS", 300))

# New Configurable Parameters for Multi-Table Detection (From Code A)
WORKER_MIN_TABLE_ROWS = int(os.getenv("WORKER_MIN_TABLE_ROWS", 3)) # Min rows for a block to be considered a table (incl. header)
WORKER_MIN_TABLE_COLS = int(os.getenv("WORKER_MIN_TABLE_COLS", 2)) # Min cols for a block
WORKER_MIN_HEADER_CONFIDENCE = float(os.getenv("WORKER_MIN_HEADER_CONFIDENCE", 0.6)) # For scoring header likelihood (0-1)
WORKER_BLOCK_DENSITY_THRESHOLD = float(os.getenv("WORKER_BLOCK_DENSITY_THRESHOLD", 0.25)) # Min overall density for a candidate block
# WORKER_ROW_DENSITY_THRESHOLD from old code is likely still present but not used by the primary multi-table path.

# --- Input Validation for Worker (Corrected) ---
required_vars = {
    "GCP_PROJECT": WORKER_GCP_PROJECT, "GCS_BUCKET": WORKER_GCS_BUCKET,
    "BQ_DATASET": WORKER_BQ_DATASET, "PUBSUB_SUBSCRIPTION": WORKER_PUBSUB_SUBSCRIPTION }
missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    error_msg = f"Worker Error: Missing required environment variables: {', '.join(missing_vars)}."
    logger_worker.critical(error_msg); raise ValueError(error_msg)
# if not os.path.exists(WORKER_CREDENTIALS_PATH):
#      error_msg = f"Worker Creds file not found: {WORKER_CREDENTIALS_PATH}"
#      logger_worker.critical(error_msg); raise FileNotFoundError(error_msg)
if "gemini" in WORKER_DEFAULT_SCHEMA_STRATEGY and not WORKER_GEMINI_API_KEY:
     logger_worker.warning("SCHEMA_STRATEGY uses Gemini, but GEMINI_API_KEY not set.")

# --- Initialize Worker Clients (Keep as before) ---
_worker_temp_cred_file_path: Optional[str] = None # For cleanup

# --- Cleanup function for temporary credentials file ---
def _cleanup_worker_temp_file():
    """Removes the temporary credentials file if it was created."""
    global _worker_temp_cred_file_path
    if _worker_temp_cred_file_path and os.path.exists(_worker_temp_cred_file_path):
        try:
            os.remove(_worker_temp_cred_file_path)
            logger_worker.info(f"Cleaned up worker temporary credentials file: {_worker_temp_cred_file_path}")
            _worker_temp_cred_file_path = None
        except Exception as e:
            logger_worker.warning(f"Could not remove worker temp credentials file {_worker_temp_cred_file_path}: {e}")

# --- Initialize Worker Clients ---
worker_storage_client: Optional[storage.Client] = None
worker_bq_client: Optional[bigquery.Client] = None
worker_subscriber: Optional[SubscriberClient] = None

try: # Top Level Try for overall initialization logic
    logger_worker.info("Initializing Worker GCP Clients...")
    gcp_project_id = WORKER_GCP_PROJECT
    # Read the environment variable expecting Base64 content
    encoded_key_content = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # --- Step 1: Prepare Credentials Environment ---
    if encoded_key_content:
        logger_worker.info("Attempting to decode/write Base64 GOOGLE_APPLICATION_CREDENTIALS for worker...")
        try:
            decoded_bytes = base64.b64decode(encoded_key_content)
            key_info = json.loads(decoded_bytes.decode('utf-8'))
            logger_worker.info("Successfully decoded/parsed worker credentials.")

            # Write to temporary file
            try:
                 with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                     json.dump(key_info, temp_file)
                     _worker_temp_cred_file_path = temp_file.name
                 logger_worker.info(f"Worker credentials written to temp file: {_worker_temp_cred_file_path}")
                 # Set environment variable for libraries to find it
                 os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _worker_temp_cred_file_path
                 logger_worker.info(f"Set GOOGLE_APPLICATION_CREDENTIALS env var to: {_worker_temp_cred_file_path}")
                 # Register cleanup only if successful
                 atexit.register(_cleanup_worker_temp_file)

            except Exception as file_err:
                 logger_worker.error(f"Failed to write worker credentials to temp file: {file_err}", exc_info=True)
                 _worker_temp_cred_file_path = None
                 os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None) # Unset if write failed
                 # Let client init fail below if ADC isn't configured on the worker instance

        except Exception as decode_err:
            logger_worker.error(f"Failed to decode/parse worker Base64 credentials: {decode_err}. Check GOOGLE_APPLICATION_CREDENTIALS env var.", exc_info=True)
            _worker_temp_cred_file_path = None
            os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None) # Unset invalid value
            # Let client init fail below if ADC isn't configured

    elif gcp_project_id:
         logger_worker.info("Worker GOOGLE_APPLICATION_CREDENTIALS not set. Relying on Application Default Credentials (ADC)...")
         # Ensure env var is unset if we expect ADC
         os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
    else:
        # Critical failure if no credentials AND no project ID for ADC
        error_msg = "Worker cannot initialize clients: Set GCP_PROJECT env var, or provide valid Base64 credentials in GOOGLE_APPLICATION_CREDENTIALS."
        logger_worker.critical(error_msg)
        raise ValueError(error_msg)


    # --- Step 2: Initialize Clients ---
    # They will use the temp file path set in ENV or ADC
    auth_method = "temp key file" if _worker_temp_cred_file_path else "ADC"
    logger_worker.info(f"Initializing clients using {auth_method}...")

    # Check project ID again, crucial for client initialization
    if not gcp_project_id:
        raise ValueError("Worker cannot initialize clients: GCP_PROJECT is None after credential setup.")

    initialization_failed = False
    try:
        worker_storage_client = storage.Client(project=gcp_project_id)
        logger_worker.info("Storage Client Initialized OK.")
    except Exception as e:
        logger_worker.error(f"Failed to init Storage Client: {e}", exc_info=True)
        initialization_failed = True

    try:
        worker_bq_client = bigquery.Client(project=gcp_project_id)
        logger_worker.info("BigQuery Client Initialized OK.")
    except Exception as e:
        logger_worker.error(f"Failed to init BigQuery Client: {e}", exc_info=True)
        initialization_failed = True

    try:
        worker_subscriber = SubscriberClient() # SubscriberClient usually picks up ADC/ENV well
        logger_worker.info("Pub/Sub Subscriber Client Initialized OK.")
    except Exception as e:
        logger_worker.error(f"Failed to init Pub/Sub Subscriber: {e}", exc_info=True)
        initialization_failed = True

    if initialization_failed:
         logger_worker.critical("One or more worker clients failed initialization.")
         raise RuntimeError("Worker client initialization failed.")
    else:
         logger_worker.info(f"Worker Clients initialized successfully using {auth_method}. Project: {WORKER_GCP_PROJECT}, Sub: {WORKER_PUBSUB_SUBSCRIPTION}")

except Exception as e:
    # Catch any unexpected errors during the entire setup process
    logger_worker.critical(f"Worker setup failed during client initialization: {e}", exc_info=True)
    worker_storage_client = None
    worker_bq_client = None
    worker_subscriber = None
    raise # Re-raise to stop the worker
# --- Worker Helper Functions (Keep existing: sanitize_bq_name, get_file_extension, map_pandas_dtype_to_bq) ---
def sanitize_bq_name(name: str) -> str:
    """Sanitizes a string for use as a BigQuery table or column name."""
    if not isinstance(name, str):
        name = str(name)
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace invalid characters (anything not a letter, number, or underscore) with underscore
    name = re.sub(r'[^\w]', '_', name)
    # Ensure name starts with a letter or underscore
    if name and not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    # Handle empty names after sanitization
    if not name:
        name = '_unnamed'
    # Truncate to BigQuery's maximum length for table/column names (1024, but often limited further in practice, e.g., 300 is safer for combined names)
    # Let's use 300 for safety in combined names. BQ max is 1024.
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




def _read_excel_sheets(io_source, pandas_opts) -> Dict[str, pd.DataFrame]:
    try:
        logger_worker.debug(f"Attempting pd.read_excel (all sheets) opts: {pandas_opts} source type: {type(io_source)}")
        if hasattr(io_source, 'seek') and callable(io_source.seek):
            try: io_source.seek(0); logger_worker.debug("Reset source stream position to 0 for multi-sheet read.")
            except io.UnsupportedOperation: logger_worker.debug("Source stream does not support seek for multi-sheet read.")

        dfs_dict = pd.read_excel(io_source, engine=None, **pandas_opts, sheet_name=None)
        logger_worker.info(f"pd.read_excel read {len(dfs_dict)} sheets successfully.")
        
        processed_dfs_dict = {}
        for sheet_name, df_sheet in dfs_dict.items():
            if df_sheet is not None:
                df_sheet_processed = df_sheet.copy()
                df_sheet_processed.reset_index(drop=True, inplace=True)
                # Ensure columns are simple 0-based integers for consistent processing by _find_all_tables_in_sheet
                df_sheet_processed.columns = range(df_sheet_processed.shape[1])
                processed_dfs_dict[sheet_name] = df_sheet_processed
            else:
                # Handle case where a sheet might be read as None (e.g., completely empty sheet)
                processed_dfs_dict[sheet_name] = pd.DataFrame() 
        return processed_dfs_dict
    except ImportError as ie:
         if 'xlrd' in str(ie).lower(): logger_worker.critical("MISSING dependency: Reading .xls requires `xlrd`. `pip install xlrd`.")
         elif 'openpyxl' in str(ie).lower(): logger_worker.critical("MISSING dependency: Reading .xlsx requires `openpyxl`. `pip install openpyxl`.")
         else: logger_worker.error(f"ImportError reading Excel sheets: {ie}")
         raise
    except Exception as e:
        logger_worker.error(f"General pd.read_excel error reading all sheets: {e}", exc_info=True)
        raise







def _score_table_candidate(
    block_df: pd.DataFrame,
    min_header_confidence: float = WORKER_MIN_HEADER_CONFIDENCE,
    min_data_rows: int = 1
) -> float:
    if block_df.empty or block_df.shape[0] < 1 or block_df.shape[1] < 1:
        return -1.0

    block_df_for_density = block_df.replace(r'^\s*$', pd.NA, regex=True).replace(['None','none','null','NULL','NaN','NAN','<NA>','na','N/A','n/a'], pd.NA)
    actual_data_points = block_df_for_density.notna().sum().sum()
    total_cells = block_df.size
    density = actual_data_points / total_cells if total_cells > 0 else 0

    if density < (WORKER_BLOCK_DENSITY_THRESHOLD / 2): # Hard floor for density
        logger_worker.debug(f"Candidate block rejected due to very low density: {density:.2f}")
        return -1.0

    header_score = 0.0
    has_plausible_header = False
    if block_df.shape[0] > 0: # Must have at least one row to be a header
        first_row_series = block_df.iloc[0].astype(str) # Treat first row as potential header
        first_row_non_null_count = first_row_series.replace(['None','none','null','NULL','NaN','NAN','<NA>','na','N/A','n/a',''], pd.NA).count()
        first_row_distinct_count = first_row_series.nunique()

        if first_row_non_null_count > 0 and \
           (first_row_distinct_count / first_row_non_null_count) >= (min_header_confidence * 0.8) and \
           first_row_distinct_count > 0: # At least one distinct value
            numeric_header_values = pd.to_numeric(first_row_series, errors='coerce').notna().sum()
            if (numeric_header_values / first_row_non_null_count) < 0.4: # Less than 40% numeric
                header_score += 0.4
                has_plausible_header = True
            elif first_row_distinct_count == len(first_row_series): # All unique, strong sign
                header_score += 0.3
                has_plausible_header = True
            else:
                header_score += 0.1 # Some header indication

    data_rows_count = block_df.shape[0] - 1 if has_plausible_header else block_df.shape[0]
    if data_rows_count < min_data_rows:
        logger_worker.debug(f"Candidate block rejected: only {data_rows_count} data rows (min {min_data_rows} required). Header plausible: {has_plausible_header}")
        return -1.0

    score = (density * 0.5) + (header_score * 0.3)
    score += np.log1p(actual_data_points) * 0.01 
    logger_worker.debug(f"Scored candidate (shape {block_df.shape}): Density={density:.2f}, HeaderScr={header_score:.2f}, DataPts={actual_data_points} -> FinalScr={score:.2f}")
    return score




def _find_all_tables_in_sheet(
    original_sheet_df: pd.DataFrame,
    min_rows: int = WORKER_MIN_TABLE_ROWS,
    min_cols: int = WORKER_MIN_TABLE_COLS,
    block_density_threshold: float = WORKER_BLOCK_DENSITY_THRESHOLD,
    header_confidence: float = WORKER_MIN_HEADER_CONFIDENCE
    # Consider passing object_name and sheet_name here for better logging if needed
) -> List[Dict[str, Any]]:
    # This log helps trace entry and SCIPY status for a given sheet processing
    logger_worker.critical(f"LOCAL_FIND_ALL_TABLES_ENTRY_FOR_SHEET: Shape {original_sheet_df.shape} | SCIPY_AVAILABLE: {SCIPY_AVAILABLE} | Thread: {threading.get_ident()}")

    if not SCIPY_AVAILABLE:
        logger_worker.error("Scipy library not found. Multi-table detection per sheet is disabled. Install 'scipy'.")
        if original_sheet_df.empty:
            return []
        
        score = _score_table_candidate(original_sheet_df, header_confidence, max(1, min_rows -1))
        if score > -0.5 and original_sheet_df.shape[0] >= min_rows and original_sheet_df.shape[1] >= min_cols:
             logger_worker.warning("Scipy not available. Treating entire sheet as a single table candidate.")
             df_copy = original_sheet_df.copy()
             df_copy.reset_index(drop=True, inplace=True)
             df_copy.columns = range(df_copy.shape[1])
             return [{
                 'df': df_copy,
                 'id': 'table_1', 
                 'coords_in_sheet': (0, 0, original_sheet_df.shape[0]-1, original_sheet_df.shape[1]-1),
                 'score': score
             }]
        else:
            logger_worker.warning("Scipy not available, and entire sheet does not qualify as a single table. No tables extracted.")
            return []

    if original_sheet_df.empty:
        return []

    logger_worker.debug(f"Starting multi-table search in sheet of shape: {original_sheet_df.shape}")
    processed_df_for_mask = original_sheet_df.replace(r'^\s*$', pd.NA, regex=True).replace(['None','none','null','NULL','NaN','NAN','<NA>','na','N/A','n/a'], pd.NA)
    content_mask = processed_df_for_mask.notna().to_numpy()
    
    # Using 8-connectivity by default. Consider structure=scipy.ndimage.generate_binary_structure(2, 1) for 4-connectivity
    labeled_array, num_features = scipy.ndimage.label(content_mask, structure=np.ones((3,3)))
    logger_worker.debug(f"Found {num_features} initial contiguous regions (features) in sheet.")
    if num_features == 0: return []

    object_slices = scipy.ndimage.find_objects(labeled_array)
    candidate_tables = []
    for i, slc_tuple in enumerate(object_slices):
        if slc_tuple is None: continue
        r_slice, c_slice = slc_tuple
        r1, r2 = r_slice.start, r_slice.stop
        c1, c2 = c_slice.start, c_slice.stop
        num_block_rows = r2 - r1
        num_block_cols = c2 - c1
        if num_block_rows < min_rows or num_block_cols < min_cols:
            logger_worker.debug(f"Region {i+1} too small ({num_block_rows}x{num_block_cols}). Min req: {min_rows}x{min_cols}. Skipping.")
            continue
        
        candidate_df = original_sheet_df.iloc[r1:r2, c1:c2].copy()
        candidate_df_for_density = candidate_df.replace(r'^\s*$', pd.NA, regex=True).replace(['None','none','null','NULL','NaN','NAN','<NA>','na','N/A','n/a'], pd.NA)
        non_na_cells = candidate_df_for_density.notna().sum().sum()
        total_cells = candidate_df.size
        current_block_density = non_na_cells / total_cells if total_cells > 0 else 0

        if current_block_density < block_density_threshold:
            logger_worker.debug(f"Region {i+1} (shape {candidate_df.shape}) below density threshold ({current_block_density:.2f} < {block_density_threshold:.2f}). Skipping.")
            continue
        
        logger_worker.debug(f"Region {i+1} (coords R{r1}-R{r2-1}, C{c1}-C{c2-1}, shape {candidate_df.shape}) passed size and density checks.")
        candidate_tables.append({
            'df_raw': candidate_df,
            'coords_in_sheet': (r1, c1, r2 - 1, c2 - 1), # 0-indexed, inclusive
            'score': 0.0 
        })

    if not candidate_tables:
        logger_worker.info("No candidate regions met size/density criteria after connected components.")
        return []

    for candidate in candidate_tables:
        min_data_rows_for_score = max(1, min_rows -1) if min_rows > 1 else 1
        candidate['score'] = _score_table_candidate(candidate['df_raw'], header_confidence, min_data_rows_for_score)

    valid_scored_candidates = [cand for cand in candidate_tables if cand['score'] > -0.5] # Use -0.5 to catch -1 precisely
    valid_scored_candidates.sort(key=lambda x: x['score'], reverse=True)

    final_tables_details = []
    covered_mask_np = np.full(original_sheet_df.shape, False, dtype=bool)
    table_counter = 0
    for candidate in valid_scored_candidates:
        r1, c1, r2_incl, c2_incl = candidate['coords_in_sheet']
        if np.any(covered_mask_np[r1:r2_incl+1, c1:c2_incl+1]):
            logger_worker.debug(f"Skipping candidate (score {candidate['score']:.2f}) at R{r1}-R{r2_incl}, C{c1}-C{c2_incl} due to overlap.")
            continue

        table_counter += 1
        df_to_process = candidate['df_raw'].copy()
        df_to_process.reset_index(drop=True, inplace=True)
        df_to_process.columns = range(df_to_process.shape[1])

        final_tables_details.append({
            'df': df_to_process,
            'id': f"table_{table_counter}",
            'coords_in_sheet': (r1, c1, r2_incl, c2_incl),
            'score': candidate['score']
        })
        covered_mask_np[r1:r2_incl+1, c1:c2_incl+1] = True
        logger_worker.info(f"Selected table '{'table_'+str(table_counter)}' (score {candidate['score']:.2f}) at R{r1}-R{r2_incl}, C{c1}-C{c2_incl}. Shape {candidate['df_raw'].shape}")

    logger_worker.info(f"Successfully extracted {len(final_tables_details)} non-overlapping tables from sheet.")
    return final_tables_details



def _isolate_data_block(df: pd.DataFrame) -> pd.DataFrame:
    """Isolates the main data block within a DataFrame, removing fully empty rows/cols."""
    if df is None or df.empty:
        logger_worker.debug("Skipping block isolation for empty or None DataFrame.")
        return pd.DataFrame() # Return empty DF

    logger_worker.debug(f"Isolating data block from DataFrame with shape: {df.shape}")
    df_processed = df.copy()
    # Drop fully empty columns first
    df_processed.dropna(axis=1, how='all', inplace=True)
    # Drop fully empty rows
    df_processed.dropna(axis=0, how='all', inplace=True)

    if df_processed.empty:
        logger_worker.warning("DataFrame is empty after dropping all-NA rows/columns.")
        return df_processed.reset_index(drop=True) # Return empty DF

    df_processed = df_processed.reset_index(drop=True)
    start_idx, end_idx = -1, -1
    # Calculate minimum non-null values needed to consider a row as 'data'
    min_non_null = max(1, int(len(df_processed.columns) * WORKER_ROW_DENSITY_THRESHOLD))
    logger_worker.debug(f"Block isolation: min_non_null threshold = {min_non_null} for {len(df_processed.columns)} columns")

    # Find first row meeting the density threshold
    for i in range(len(df_processed)):
        # Count non-null values in the row (treat common null-like strings as NA for count)
        row_non_null_count = df_processed.iloc[i].replace(['None', 'nan', 'NaN', '<NA>', ''], pd.NA).count()
        if row_non_null_count >= min_non_null:
            start_idx = i
            logger_worker.debug(f"Found potential start row {i} with {row_non_null_count} non-null values.")
            break

    # Find last row meeting the density threshold (search backwards from end)
    if start_idx != -1:
        for i in range(len(df_processed) - 1, start_idx - 1, -1):
            row_non_null_count = df_processed.iloc[i].replace(['None', 'nan', 'NaN', '<NA>', ''], pd.NA).count()
            if row_non_null_count >= min_non_null:
                end_idx = i
                logger_worker.debug(f"Found potential end row {i} with {row_non_null_count} non-null values.")
                break

    # Slice the DataFrame if a valid block was found
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        isolated_df = df_processed.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        logger_worker.info(f"Isolated data block from row {start_idx} to {end_idx}. New shape: {isolated_df.shape}")
        return isolated_df
    else:
        logger_worker.warning(f"No valid data block found meeting density threshold (start={start_idx}, end={end_idx}). Returning empty DataFrame.")
        return pd.DataFrame()
# +++ MODIFICATION END +++


# +++ MODIFICATION START +++
# Modified to return Dict[str, pd.DataFrame] for Excel, pd.DataFrame for others, or None
def read_data_from_gcs(bucket_name: str, object_name: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame], None]:
    """
    Reads data from GCS.
    For Excel files (.xls, .xlsx), reads all sheets into a dictionary {sheet_name: DataFrame}.
    For CSV/Parquet, reads into a single DataFrame.
    Uses streaming with disk fallback for Excel/CSV if streaming fails.
    Returns None on critical read errors or if the file is not found.
    """
    if not worker_storage_client:
        logger_worker.error("Storage client inactive.")
        raise ConnectionError("Storage client inactive")

    logger_worker.info(f"Reading data from gs://{bucket_name}/{object_name}")
    bucket = worker_storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    result_data: Union[pd.DataFrame, Dict[str, pd.DataFrame], None] = None # Explicitly define type

    # 1. Check Existence
    try:
        if not blob.exists():
            logger_worker.error(f"File not found in GCS: {object_name}")
            raise FileNotFoundError(f"GCS object not found: {object_name}")
        logger_worker.debug(f"File exists. Size: {blob.size}. Proceeding.")
    except FileNotFoundError:
        raise # Re-raise specific error
    except Exception as e:
        logger_worker.error(f"GCS existence check failed for {object_name}: {e}", exc_info=True)
        return None # Return None on other GCS errors

    # 2. Attempt Read
    file_ext = get_file_extension(object_name)
    supported_excel = ['.xlsx', '.xls']
    supported_csv = ['.csv']
    supported_parquet = ['.parquet']
    read_successful = False

    # --- Define read options (header detection will happen *per sheet* later) ---
    # For initial read, we don't know the header yet, read without promoting
    initial_read_opts = {'dtype': str, 'header': None}

    # --- Attempt 1: Streaming Read ---
    if file_ext in supported_excel + supported_csv + supported_parquet:
        logger_worker.info(f"Attempting streaming read for {file_ext} file...")
        try:
            with blob.open("rb") as gcs_stream:
                if file_ext in supported_parquet:
                    result_data = pd.read_parquet(gcs_stream)
                    logger_worker.info(f"Streaming read successful for Parquet. Shape: {result_data.shape if result_data is not None else 'None'}")
                elif file_ext in supported_excel:
                    # Use the new helper to read all sheets
                    result_data = _read_excel_sheets(gcs_stream, initial_read_opts) # Reads all sheets
                    logger_worker.info(f"Streaming read successful for Excel. Read {len(result_data)} sheets.")
                elif file_ext in supported_csv:
                    # Read CSV without assuming header initially
                    try:
                         result_data = pd.read_csv(gcs_stream, sep=None, engine='python', **initial_read_opts)
                    except (ValueError, pd.errors.ParserError): # Fallback to comma delimiter if auto-detect fails
                        logger_worker.debug("CSV auto-delimiter failed, trying comma...")
                        # Need to reopen the stream to read again
                        with blob.open("rb") as gcs_stream_comma:
                           result_data = pd.read_csv(gcs_stream_comma, sep=',', engine='python', **initial_read_opts)
                    logger_worker.info(f"Streaming read successful for CSV. Shape: {result_data.shape if result_data is not None else 'None'}")

            if result_data is None: # Check if read resulted in None unexpectedly
                 raise ValueError(f"Streaming read for {file_ext} resulted in None unexpectedly.")

            read_successful = True # Mark stream success

        except ImportError as imp_err: # Catch missing libs specifically
             logger_worker.critical(f"ImportError during streaming read: {imp_err}. Required library may be missing.", exc_info=True)
             return None # Cannot recover, return None
        except Exception as stream_err:
            logger_worker.warning(f"Streaming read failed for {object_name}: {type(stream_err).__name__}: {stream_err}. Attempting disk fallback if applicable.", exc_info=False)
            if logger_worker.isEnabledFor(logging.DEBUG):
                logger_worker.debug("Stream failure traceback:", exc_info=True)
            result_data = None # Ensure result is None if stream failed

    # --- Attempt 2: Download-to-Disk Fallback (Only for Excel/CSV if streaming failed) ---
    if not read_successful and file_ext in (supported_excel + supported_csv):
        logger_worker.info(f"Executing download-to-disk fallback for {object_name}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_basename = re.sub(r'[^\w.]+', '_', os.path.basename(object_name))
            temp_file_path = os.path.join(tmpdir, safe_basename)
            logger_worker.debug(f"Downloading to temp path: {temp_file_path}")
            try:
                blob.download_to_filename(temp_file_path)
                logger_worker.debug(f"Download to temp file complete. Reading from disk...")

                if file_ext in supported_excel:
                    # Read all sheets from the downloaded file
                    result_data = _read_excel_sheets(temp_file_path, initial_read_opts)
                    logger_worker.info(f"Disk fallback read successful for Excel. Read {len(result_data)} sheets.")
                elif file_ext in supported_csv:
                    # Read CSV from the downloaded file
                    try:
                        result_data = pd.read_csv(temp_file_path, sep=None, engine='python', **initial_read_opts)
                    except (ValueError, pd.errors.ParserError): # Fallback to comma
                         logger_worker.debug("CSV auto-delimiter failed on disk file, trying comma...")
                         result_data = pd.read_csv(temp_file_path, sep=',', engine='python', **initial_read_opts)
                    logger_worker.info(f"Disk fallback read successful for CSV. Shape: {result_data.shape if result_data is not None else 'None'}")

                if result_data is None:
                     raise ValueError(f"Disk fallback read for {file_ext} resulted in None.")

                read_successful = True # Mark success after fallback

            except ImportError as imp_err: # Catch missing libs specifically
                 logger_worker.critical(f"ImportError during disk fallback read: {imp_err}. Required library may be missing.", exc_info=True)
                 return None # Cannot recover, return None
            except Exception as file_fallback_err:
                logger_worker.error(f"Disk fallback attempt using temp file failed for {object_name}: {file_fallback_err}", exc_info=True)
                result_data = None # Ensure result is None if fallback failed

    # --- Final Check ---
    if not read_successful:
        logger_worker.error(f"All read attempts failed for file {object_name}. Type: {file_ext}")
        return None # Return None if neither stream nor fallback worked

    # --- Return the raw data (DataFrame or Dict[str, DataFrame]) ---
    # Block isolation and header detection will now happen per sheet in process_object
    logger_worker.info(f"Successfully read raw data for {object_name}. Type: {type(result_data)}")
    return result_data
# +++ MODIFICATION END +++


# ============================ Start of pasted remaining functions ============================

# clean_dataframe function remains UNCHANGED
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: logger_worker.warning("Input DF to clean_dataframe empty."); return df
    logger_worker.info("Cleaning isolated data block...")
    # --- Try to detect header row based on heuristics ---
    # Assume first row IS the header if it's distinct enough and not generic numeric/col_N names
    first_row_values = df.iloc[0].astype(str)
    num_cols = len(df.columns)
    first_row_non_null = first_row_values.replace(['None','nan','NaN','NA','<NA>'], pd.NA).count() # Count actual values
    first_row_distinct = first_row_values.replace(['None','nan','NaN','NA','<NA>'], pd.NA).nunique()
    # Check if current column names look like default pandas names (e.g., 0, 1, 2... or unnamed: 0)
    current_cols_are_generic = all(re.match(r"^(Unnamed: \d+|\d+)$", str(col)) for col in df.columns)

    # Heuristic: High distinct ratio, high non-null ratio, distinct > 1, and cols are generic or first row is more distinct than current cols
    is_likely_header = (
        (first_row_distinct / max(1, first_row_non_null)) > 0.8 and
        (first_row_non_null / max(1, num_cols)) > 0.6 and
        first_row_distinct > 1 and
        (current_cols_are_generic or first_row_distinct > df.columns.nunique(dropna=False)) # Promote if first row is *more* distinct
    )

    original_cols_before_promote = list(df.columns) # Store original column names

    if is_likely_header:
         logger_worker.info("Promoting first row of data block to header based on heuristics.")
         df.columns = first_row_values # Set first row as header
         df = df[1:].reset_index(drop=True) # Remove the promoted row
         if df.empty: logger_worker.warning("DataFrame became empty after header promotion."); return df # Return empty if nothing left
         logger_worker.debug(f"Columns after potential promotion: {list(df.columns)}")
    else:
        logger_worker.info("First row not promoted to header based on heuristics.")


    # --- Sanitize column names ---
    original_columns = df.columns
    sanitized_columns = [sanitize_bq_name(str(col)) for col in original_columns]

    # Handle duplicate column names after sanitization
    final_columns = []
    counts = {}
    for col in sanitized_columns:
        current_count = counts.get(col, 0)
        new_col_name = f"{col}_{current_count}" if current_count > 0 else col
        # Ensure the generated name doesn't exceed limit (should be handled by sanitize_bq_name already, but double check)
        final_columns.append(new_col_name[:300])
        counts[col] = current_count + 1

    # Log if column names changed significantly
    if list(original_columns) != final_columns or list(original_cols_before_promote) != final_columns:
        logger_worker.debug(f"Original cols (before promote): {list(original_cols_before_promote)}")
        logger_worker.debug(f"Original cols (before sanitize): {list(original_columns)}")
        logger_worker.debug(f"Final sanitized cols: {final_columns}")

    df.columns = final_columns

    # --- Final NA handling and cleanup ---
    # Drop fully empty columns/rows *again* after potential header promotion and renaming
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)

    if df.empty:
        logger_worker.warning("DataFrame became empty after final NA drops in cleaning.")
        return df.reset_index(drop=True) # Return empty DF

    # Strip whitespace from object/string columns
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=['object', 'string']).columns:
        # Check if column still exists after potential drops
        if col in df_cleaned.columns:
             try:
                 # Ensure Series is treated as string before stripping
                 df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
             except Exception as strip_err:
                 logger_worker.warning(f"Failed to strip whitespace from column '{col}': {strip_err}")
                 # Optionally keep original data or try converting to string differently
                 # df_cleaned[col] = df[col] # Revert if strip fails?

    # Replace common null-like strings with actual NA/None
    # Using pd.NA which is Pandas' preferred null marker
    common_nulls = ['', 'none', 'null', 'nan', '<na>', 'nat'] # Add variations as needed
    for null_val in common_nulls:
        # Case-insensitive replacement for common nulls
        df_cleaned.replace(f'(?i)^{re.escape(null_val)}$', pd.NA, regex=True, inplace=True)


    logger_worker.info(f"Cleaning complete. Final shape: {df_cleaned.shape}")
    return df_cleaned.reset_index(drop=True) # Ensure clean index

# infer_schema_gemini function remains UNCHANGED
def infer_schema_gemini(df: pd.DataFrame) -> list | None:
    if df.empty: logger_worker.warning("Cannot infer schema from empty DataFrame."); return None
    if not WORKER_GEMINI_API_KEY: logger_worker.error("GEMINI_API_KEY not set."); return None
    # Sample data, replacing NA with 'null' string for Gemini prompt
    sample_df = df.head(WORKER_GEMINI_SAMPLE_SIZE).copy()
    for col in sample_df.columns:
        if sample_df[col].isnull().any():
            # Convert to object type before filling NA to avoid dtype issues
            sample_df[col] = sample_df[col].astype(object).where(sample_df[col].notnull(), 'null')
        # Ensure all values are strings for the prompt
        sample_df[col] = sample_df[col].astype(str)

    sample = sample_df.to_dict(orient="records")
    # Check if sample is truly empty after NA replacement
    if not sample or all(all(v == 'null' or not v for v in row.values()) for row in sample):
        logger_worker.error("Sample data appears empty or all null after preparation for Gemini.")
        return None

    logger_worker.info(f"Sending sample to Gemini for schema inference ({len(sample)} rows)...")
    # Updated prompt emphasizing exact name matching and allowed types
    prompt = (
        "Analyze the following sample data rows (represented as JSON objects) from a table. "
        "Suggest appropriate Google BigQuery column names and data types.\n"
        "RULES:\n"
        "1. The keys in the input JSON objects are the *intended* column names. Use these *exact keys* for the 'name' field in your response JSON. Do not invent new names.\n"
        "2. The suggested BigQuery type in the 'type' field MUST be chosen from this list: STRING, INTEGER, FLOAT, NUMERIC, BOOLEAN, TIMESTAMP, DATE, TIME, DATETIME, GEOGRAPHY, JSON, BYTES. Default to STRING if unsure or if the type is ambiguous.\n"
        "3. Base the data type suggestion ONLY on the sample *values* provided for each key.\n"
        "4. Your response MUST be *only* a single valid JSON list of objects. Each object must have exactly two keys: 'name' (string, matching input key) and 'type' (string, one of the allowed BQ types).\n"
        "5. Do not include ```json ``` markers or any other text outside the JSON list.\n"
        "SAMPLE DATA:\n"
        f"{json.dumps(sample)}\n\n"
        "JSON Schema:"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={WORKER_GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json" # Request JSON directly
            }
        }
    headers = {"Content-Type": "application/json"}
    result = None
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=WORKER_GEMINI_TIMEOUT)
        response.raise_for_status() # Check for HTTP errors
        result = response.json()

        if not result or not result.get("candidates"):
            raise ValueError("Invalid Gemini response format: missing 'candidates'.")

        # Extract text content reliably
        try:
            schema_content = result["candidates"][0]["content"]["parts"][0]["text"]
        except (IndexError, KeyError, TypeError) as e:
            raise ValueError(f"Could not extract schema text from Gemini response: {e}. Response: {result}")

        if not schema_content:
            raise ValueError("Gemini response text part is empty.")

        # Parse the JSON string (handle potential markdown)
        try:
            schema_list = json.loads(schema_content)
        except json.JSONDecodeError:
            # Attempt to clean markdown ```json ... ``` if present
            cleaned_content = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", schema_content, flags=re.IGNORECASE).strip()
            try:
                schema_list = json.loads(cleaned_content)
            except json.JSONDecodeError as json_err_cleaned:
                 logger_worker.error(f"Failed to parse Gemini JSON even after cleaning markdown: {json_err_cleaned}")
                 logger_worker.debug(f"Original Gemini Text: {schema_content}")
                 raise ValueError("Gemini response was not valid JSON.") from json_err_cleaned


        # Validate the structure and content of the parsed list
        if not isinstance(schema_list, list):
            raise ValueError(f"Gemini result is not a JSON list. Got: {type(schema_list)}")
        if not schema_list:
            raise ValueError("Gemini returned an empty list.")

        validated_schema = []
        valid_bq_types = {
            "STRING", "BYTES", "INTEGER", "INT64", "FLOAT", "FLOAT64",
            "NUMERIC", "BIGNUMERIC", "BOOLEAN", "BOOL", "TIMESTAMP", "DATE",
            "TIME", "DATETIME", "GEOGRAPHY", "JSON", "INTERVAL"
        }
        input_cols = set(df.columns) # Original columns from the dataframe

        for item in schema_list:
             # Validate item structure
             if not isinstance(item, dict) or 'name' not in item or 'type' not in item:
                 raise ValueError(f"Invalid schema item format from Gemini: {item}")

             name = item.get("name")
             type_val = item.get("type")

             # Validate name and type content
             if not isinstance(name, str) or not isinstance(type_val, str) or not name or not type_val:
                 raise ValueError(f"Invalid name/type in schema item from Gemini: name='{name}', type='{type_val}'")

             # Check if the name from Gemini matches an input column (important!)
             if name not in input_cols:
                 # This is a potential issue, Gemini hallucinated a column or used a sanitized name
                 logger_worker.warning(f"Gemini suggested schema name '{name}' which is not in the original DataFrame columns. Skipping this field.")
                 continue # Skip this field as it doesn't match input

             bq_type = type_val.upper()
             # Validate BQ type and default to STRING if invalid
             if bq_type not in valid_bq_types:
                 logger_worker.warning(f"Gemini suggested invalid BigQuery type '{type_val}' for column '{name}'. Defaulting to STRING.")
                 bq_type = "STRING"

             # Append validated field (name comes from Gemini but must match input)
             validated_schema.append({"name": name, "type": bq_type})

        # Final check: ensure we have at least one valid field matching input
        if not validated_schema:
             raise ValueError("Gemini schema generation failed: No valid schema fields matching input columns were produced.")

        logger_worker.info(f"✅ Successfully validated Gemini schema for {len(validated_schema)} columns.")
        logger_worker.debug(f"Validated Gemini Schema: {json.dumps(validated_schema)}")
        return validated_schema

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        logger_worker.error(f"Gemini API call timed out after {WORKER_GEMINI_TIMEOUT} seconds.")
    except requests.exceptions.RequestException as e:
        # Includes connection errors, HTTP errors (like 4xx, 5xx)
        logger_worker.error(f"Gemini API request failed: {e}")
        if e.response is not None:
            logger_worker.error(f"Gemini Response Status: {e.response.status_code}, Body: {e.response.text[:500]}") # Log response details if available
    except ValueError as e: # Catch our specific validation errors
        logger_worker.error(f"Error processing Gemini response: {e}")
    except Exception as e: # Catch-all for unexpected errors
        logger_worker.exception(f"An unexpected error occurred during Gemini schema inference: {e}") # Use exception to get traceback

    return None # Return None if any error occurred

# infer_schema_pandas function remains UNCHANGED
def infer_schema_pandas(df: pd.DataFrame) -> list:
    if df.empty: logger_worker.warning("Cannot infer pandas schema empty DF."); return []
    logger_worker.info("Inferring schema using pandas dtypes...")
    schema = []
    try:
        # Create a copy to avoid modifying the original DataFrame during inference
        df_inferred = df.copy()

        # Attempt numeric conversion first (integers, then floats)
        for col in df_inferred.columns:
            original_series = df_inferred[col]
            if original_series.isnull().all(): # Skip fully null columns
                continue
            try:
                # Try strict integer conversion (will fail if floats present)
                # Use Int64 to handle potential Pandas NA values
                df_inferred[col] = pd.to_numeric(original_series, errors='raise').astype('Int64')
                logger_worker.debug(f"Column '{col}' inferred as INTEGER by pandas.")
                continue # Success, move to next column
            except (ValueError, TypeError):
                pass # Not strictly integer

            try:
                 # Try float conversion
                 df_inferred[col] = pd.to_numeric(original_series, errors='raise').astype(float)
                 logger_worker.debug(f"Column '{col}' inferred as FLOAT by pandas.")
                 continue # Success
            except (ValueError, TypeError):
                 pass # Not float

            # Attempt datetime conversion (only if not purely numeric-like strings)
            try:
                 # Avoid converting things like ZIP codes ('01234') or IDs ('123456') to dates
                 if not original_series.astype(str).str.match(r'^\d+$').all():
                     # errors='raise' is important here
                     converted_dt = pd.to_datetime(original_series, errors='raise')
                     # Check if conversion actually resulted in datetime objects
                     if pd.api.types.is_datetime64_any_dtype(converted_dt):
                           df_inferred[col] = converted_dt
                           logger_worker.debug(f"Column '{col}' inferred as DATETIME/TIMESTAMP by pandas.")
                           continue # Success
            except (ValueError, TypeError, OverflowError):
                 pass # Not datetime

            # Attempt boolean conversion (more robust check)
            try:
                # Check if unique non-NA values look like booleans
                unique_vals = original_series.dropna().astype(str).str.lower().unique()
                bool_like = {'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0', '1.0', '0.0'}
                if all(val in bool_like for val in unique_vals) and len(unique_vals) > 0:
                     # Map recognized boolean strings/numbers to actual booleans
                     bool_map = {'true': True, 't': True, 'yes': True, 'y': True, '1': True, '1.0': True,
                                 'false': False, 'f': False, 'no': False, 'n': False, '0': False, '0.0': False}
                     # Apply mapping carefully, handle potential NAs
                     df_inferred[col] = original_series.astype(str).str.lower().map(bool_map).astype('boolean') # Use nullable boolean
                     logger_worker.debug(f"Column '{col}' inferred as BOOLEAN by pandas.")
                     continue # Success
            except Exception: # Broad catch during boolean check
                 pass # Not boolean

        # After attempting specific types, use infer_objects() for remaining 'object' columns
        # This can sometimes refine object columns (e.g., to string, boolean if missed)
        df_inferred = df_inferred.infer_objects()

        # Generate the final schema list based on inferred dtypes
        schema = [{"name": str(col), "type": map_pandas_dtype_to_bq(df_inferred[col].dtype)} for col in df_inferred.columns]

    except Exception as e:
        logger_worker.warning(f"Pandas type inference encountered an error: {e}. Falling back to basic dtype mapping.", exc_info=True)
        # Fallback: just use the original dtypes if inference fails badly
        schema = [{"name": str(col), "type": map_pandas_dtype_to_bq(df[col].dtype)} for col in df.columns]

    logger_worker.debug(f"Pandas inferred schema: {schema}")
    return schema

# get_existing_schema function remains UNCHANGED
def get_existing_schema(table_id: str) -> list[bigquery.SchemaField] | None:
    if not worker_bq_client: logger_worker.error("BQ Client not init!"); return None
    try: table = worker_bq_client.get_table(table_id); logger_worker.info(f"Found existing schema: {table_id}"); return table.schema
    except GcpNotFound: logger_worker.info(f"Table {table_id} not found."); return None
    except Exception as e: logger_worker.error(f"Error fetching schema {table_id}: {e}"); return None

# determine_schema function remains UNCHANGED
def determine_schema(df: pd.DataFrame, table_id: str, strategy: str) -> list | None:
    """Determines the BigQuery schema based on the strategy."""
    logger_worker.info(f"Determining schema for table {table_id} using strategy: '{strategy}'")
    existing_schema_bq = None
    final_schema_list = None # Will hold the list of dicts format

    # --- Step 1: Check for Existing Schema if strategy involves it ---
    if strategy.startswith("existing_or_"):
        existing_schema_bq = get_existing_schema(table_id)
        if existing_schema_bq:
            logger_worker.info(f"Using existing BigQuery schema for {table_id}.")
            # Convert BQ SchemaField list to our list of dicts format
            final_schema_list = [{"name": f.name, "type": f.field_type, "mode": f.mode} for f in existing_schema_bq]
        else:
             logger_worker.info(f"Existing schema requested but not found for {table_id}. Proceeding with inference...")

    # --- Step 2: Infer Schema if needed ---
    if final_schema_list is None: # If existing wasn't found or wasn't requested
        inference_strategy = strategy.replace("existing_or_", "") # Get the inference part (e.g., 'gemini', 'pandas', 'gemini_or_pandas')

        if "gemini" in inference_strategy:
            logger_worker.info("Attempting schema inference with Gemini...")
            final_schema_list = infer_schema_gemini(df)
            if final_schema_list:
                 logger_worker.info("Gemini inference successful.")
            else:
                 logger_worker.warning("Gemini inference failed or returned no valid schema.")

        # Fallback to Pandas if Gemini failed (or wasn't primary) and Pandas is allowed
        if final_schema_list is None and "pandas" in inference_strategy:
             logger_worker.info("Attempting schema inference with Pandas...")
             final_schema_list = infer_schema_pandas(df)
             if final_schema_list:
                   logger_worker.info("Pandas inference successful.")
             else:
                   logger_worker.warning("Pandas inference failed or returned no schema.") # Should be rare

    # --- Step 3: Validation and Final Checks ---
    if not final_schema_list:
        logger_worker.error(f"Schema determination ultimately failed for {table_id}. No schema could be determined.")
        return None

    # Ensure schema matches DataFrame columns reasonably well
    schema_column_names = {field['name'] for field in final_schema_list}
    dataframe_column_names = set(df.columns)

    common_columns = schema_column_names.intersection(dataframe_column_names)
    schema_only_columns = schema_column_names - dataframe_column_names
    dataframe_only_columns = dataframe_column_names - schema_column_names

    if not common_columns:
        logger_worker.error(f"Schema mismatch for {table_id}: No common columns found between determined schema and DataFrame!")
        logger_worker.debug(f"Schema columns: {schema_column_names}")
        logger_worker.debug(f"DataFrame columns: {dataframe_column_names}")
        # Decide if this is fatal. Usually, it is.
        raise ValueError(f"Fatal schema mismatch for {table_id}: No common columns.")

    if dataframe_only_columns:
        logger_worker.warning(f"Columns present in DataFrame but not in determined schema for {table_id} (will be dropped): {sorted(list(dataframe_only_columns))}")
    if schema_only_columns:
        logger_worker.info(f"Columns present in schema but not in DataFrame for {table_id} (will be added as NULL): {sorted(list(schema_only_columns))}")
        # Ensure these columns are added to the schema list if using inference (existing schema already has them)
        if not existing_schema_bq: # Add only if we inferred the schema
            for col_name in schema_only_columns:
                  # Add with a default type (STRING, NULLABLE) if not already defined?
                  # This case is less likely if inference is based *on* the DF.
                  # Let align_dataframe handle adding NULL columns based on the schema.
                  pass # Schema list should already be correct based on inference source


    # Ensure all schema fields have a mode (default to NULLABLE)
    for field in final_schema_list:
        field.setdefault('mode', 'NULLABLE')

    logger_worker.info(f"Schema determined successfully for {table_id}.")
    return final_schema_list

# align_dataframe_to_schema function remains UNCHANGED
def align_dataframe_to_schema(df: pd.DataFrame, schema_list: list) -> pd.DataFrame:
    """Aligns DataFrame columns and attempts type conversions based on the target BQ schema."""
    if df.empty:
        logger_worker.warning("Input DataFrame to align_dataframe_to_schema is empty.")
        return df # Return empty DF immediately

    if not schema_list:
        logger_worker.error("Cannot align DataFrame: Schema list is empty.")
        raise ValueError("Schema list cannot be empty for alignment.")

    logger_worker.info(f"Aligning DataFrame (shape: {df.shape}) to target schema...")
    logger_worker.debug(f"Target Schema fields: {[f['name'] for f in schema_list]}")
    logger_worker.debug(f"DataFrame columns before align: {list(df.columns)}")


    target_fields_dict = {field['name']: field for field in schema_list}
    target_columns_ordered = [field['name'] for field in schema_list] # Preserve target order
    target_columns_set = set(target_columns_ordered)
    current_columns_set = set(df.columns)

    # --- Column Alignment ---
    # 1. Identify columns to drop from DataFrame (present in DF but not in target schema)
    columns_to_drop = list(current_columns_set - target_columns_set)
    if columns_to_drop:
        logger_worker.warning(f"Dropping columns from DataFrame not present in target schema: {sorted(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)

    # 2. Identify columns to add to DataFrame (present in target schema but not in DF)
    columns_to_add = list(target_columns_set - current_columns_set)
    if columns_to_add:
        logger_worker.info(f"Adding NULL columns to DataFrame required by target schema: {sorted(columns_to_add)}")
        for col_name in columns_to_add:
            df[col_name] = pd.NA # Use Pandas NA marker

    # 3. Reorder columns to match the target schema order
    # Ensure we only include columns that *actually exist* in the DF after drops/adds
    final_ordered_columns = [col for col in target_columns_ordered if col in df.columns]
    df = df[final_ordered_columns]

    logger_worker.debug(f"DataFrame columns after alignment: {list(df.columns)}")

    # --- Type Conversion ---
    logger_worker.info("Attempting data type conversions based on schema...")
    df_aligned = df.copy() # Work on a copy for conversions

    for col_name in df_aligned.columns:
        if col_name not in target_fields_dict:
            logger_worker.warning(f"Column '{col_name}' found in DataFrame but missing from schema dict during type conversion. Skipping conversion.")
            continue

        target_type = target_fields_dict[col_name]['type'].upper()
        current_series = df_aligned[col_name]

        # Skip conversion if column is entirely null
        if current_series.isnull().all():
            # Ensure it's object type for BQ compatibility if fully null
            df_aligned[col_name] = current_series.astype(object).where(pd.notnull(current_series), None)
            continue

        converted_series = None
        try:
            if target_type in ('INTEGER', 'INT64'):
                # Use Int64 for nullable integers
                converted_series = pd.to_numeric(current_series, errors='coerce').astype('Int64')
            elif target_type in ('FLOAT', 'FLOAT64', 'NUMERIC', 'BIGNUMERIC'):
                # Use float for nullable floats
                converted_series = pd.to_numeric(current_series, errors='coerce').astype(float)
            elif target_type in ('BOOLEAN', 'BOOL'):
                 # Use Pandas nullable boolean type 'boolean'
                bool_map = {'true': True, 't': True, 'yes': True, 'y': True, '1': True, '1.0': True,
                            'false': False, 'f': False, 'no': False, 'n': False, '0': False, '0.0': False,
                            '': pd.NA, 'none': pd.NA, 'null': pd.NA, 'nan': pd.NA} # Add NA mappings

                # Standardize to lower string before mapping, handle potential existing bools/numerics
                standardized_series = current_series.astype(str).str.lower().replace(r'^\s*$', pd.NA, regex=True) # Treat empty strings as NA
                converted_series = standardized_series.map(bool_map).astype('boolean')

            elif target_type in ('TIMESTAMP', 'DATETIME'):
                # errors='coerce' turns unparseable dates into NaT (Pandas NA for datetime)
                 converted_series = pd.to_datetime(current_series, errors='coerce')
                 # Ensure timezone awareness if converting to TIMESTAMP (BQ prefers UTC)
                 # BQ TIMESTAMP expects UTC. BQ DATETIME has no zone.
                 # If original data had timezone, pandas usually preserves it. If not, assume UTC?
                 # Let BQ handle timezone on load if possible, safer than assuming here.
                 # Check if conversion resulted in NaT for originally non-null values
                 # if current_series.notnull().sum() > 0 and converted_series.isnull().all():
                 #      logger_worker.warning(f"Timestamp/Datetime conversion failed for all non-null values in '{col_name}'. Check format.")

            elif target_type == 'DATE':
                 converted_series = pd.to_datetime(current_series, errors='coerce').dt.date
                 # Convert Python date objects to strings for BQ loader compatibility if needed?
                 # BQ load_table_from_dataframe generally handles date objects well.
            elif target_type == 'TIME':
                 # Careful with time conversion, format can vary wildly.
                 # pd.to_datetime might work for some formats like 'HH:MM:SS'
                 converted_series = pd.to_datetime(current_series, errors='coerce').dt.time
                 # Convert Python time objects to strings? BQ loader usually handles time.
            elif target_type == 'STRING':
                # Convert explicitly to string, handle NA
                 converted_series = current_series.astype(str).replace({'<NA>': None, 'nan': None}) # Replace pandas string NA markers
            elif target_type == 'JSON':
                 # Treat as string, BQ will parse on its side if needed
                 converted_series = current_series.astype(str).replace({'<NA>': None, 'nan': None})
            elif target_type == 'BYTES':
                 logger_worker.warning(f"BYTES type conversion not directly implemented for column '{col_name}'. Treating as STRING.")
                 converted_series = current_series.astype(str).replace({'<NA>': None, 'nan': None})
            # Add other types like GEOGRAPHY if needed

            # Apply the conversion if successful
            if converted_series is not None:
                # Check for excessive NaNs introduced by coercion
                original_nulls = current_series.isnull().sum()
                new_nulls = converted_series.isnull().sum()
                if new_nulls > original_nulls + (0.1 * (len(current_series) - original_nulls)) and new_nulls > 5: # Heuristic: >10% new nulls + >5 total
                      logger_worker.warning(f"Type conversion for '{col_name}' to {target_type} introduced significant NULLs ({new_nulls - original_nulls} new). Review data quality or schema.")

                df_aligned[col_name] = converted_series
            else:
                # This case should ideally not happen if all target types are handled above
                logger_worker.debug(f"No specific conversion applied for column '{col_name}' to type {target_type}.")
                # Ensure it's at least object type if no conversion happened
                if not pd.api.types.is_object_dtype(df_aligned[col_name].dtype):
                     df_aligned[col_name] = df_aligned[col_name].astype(object)


        except Exception as e:
            logger_worker.error(f"Failed to convert column '{col_name}' to target type {target_type}: {e}. Leaving as original/object.", exc_info=False)
            # Fallback: Ensure the column is object type to avoid BQ load errors if conversion fails badly
            try:
                 if not pd.api.types.is_object_dtype(df_aligned[col_name].dtype):
                      df_aligned[col_name] = df_aligned[col_name].astype(object)
            except Exception as fallback_e:
                 logger_worker.error(f"Failed to fallback cast column '{col_name}' to object type: {fallback_e}")


    logger_worker.info("Data type conversions attempted.")

    # Final step: Replace Pandas NA/NaT with None for BigQuery compatibility
    # BQ load_table_from_dataframe handles pd.NA correctly now, but explicit None is safer.
    # Important: Do this AFTER all type conversions.
    return df_aligned.astype(object).where(pd.notnull(df_aligned), None)


# load_to_bq function remains UNCHANGED
def load_to_bq(df: pd.DataFrame, table_id: str, schema_list: list, write_disposition: str):
    """Loads the DataFrame to the specified BigQuery table."""
    if not worker_bq_client:
        logger_worker.error("BigQuery Client not initialized. Cannot load data.")
        raise ConnectionError("BigQuery Client not initialized.")

    if df.empty:
        logger_worker.warning(f"DataFrame is empty. Skipping BigQuery load for table: {table_id}")
        return # Nothing to load

    if not schema_list:
         logger_worker.error(f"Schema list is empty for {table_id}. Cannot load data.")
         raise ValueError("Schema list cannot be empty for BQ load.")

    logger_worker.info(f"Starting BigQuery load for {len(df)} rows to {table_id} (Disposition: {write_disposition})")

    # Convert schema list of dicts to BigQuery SchemaField objects
    try:
        bq_schema = [
            bigquery.SchemaField(field['name'], field['type'], mode=field.get('mode', 'NULLABLE'))
            for field in schema_list
        ]
        logger_worker.debug(f"BigQuery SchemaFields prepared for {table_id}: {bq_schema}")
    except KeyError as e:
        logger_worker.error(f"Invalid schema dictionary item: Missing key {e}. Schema: {schema_list}")
        raise ValueError(f"Invalid schema format for BQ load: Missing key {e}") from e
    except Exception as e:
         logger_worker.error(f"Failed to create BigQuery SchemaField list: {e}", exc_info=True)
         raise ValueError("Failed to prepare BigQuery schema from schema list.") from e

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=write_disposition, # e.g., WRITE_APPEND, WRITE_TRUNCATE
        autodetect=False, # We explicitly provide the schema
        source_format=bigquery.SourceFormat.PARQUET # Recommended for loading from DataFrame
        # You might add other options like time_partitioning, clustering_fields etc. here
    )

    try:
        # Initiate the load job
        # Use Parquet format for efficiency when loading from DataFrame
        load_job = worker_bq_client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        logger_worker.info(f"Submitted BigQuery Load Job ID: {load_job.job_id} for table {table_id}")

        # Wait for the job to complete
        load_job.result(timeout=WORKER_BQ_LOAD_TIMEOUT) # Wait for the job to complete, adjust timeout as needed

        # Check for errors
        if load_job.errors:
            logger_worker.error(f"BigQuery Load Job {load_job.job_id} finished with errors for table {table_id}:")
            for error in load_job.errors:
                logger_worker.error(f"  - Reason: {error.get('reason', 'N/A')}, Location: {error.get('location', 'N/A')}, Message: {error.get('message', 'N/A')}")
            # Raise an exception to signal failure
            raise Exception(f"BigQuery load job failed for {table_id}. See logs for details.")
        else:
            # Verify table properties after successful load (optional but good practice)
            destination_table = worker_bq_client.get_table(table_id)
            logger_worker.info(
                f"Load job {load_job.job_id} completed successfully for {table_id}. "
                f"Total rows in table: {destination_table.num_rows}"
            )

    except TimeoutError:
         logger_worker.error(f"BigQuery load job timed out after {WORKER_BQ_LOAD_TIMEOUT} seconds for {table_id}.")
         # Depending on policy, you might want to cancel the job here
         # worker_bq_client.cancel_job(load_job.job_id)
         raise # Re-raise timeout error
    except GoogleAPICallError as e:
         logger_worker.error(f"Google API Call Error during BigQuery load for {table_id}: {e}", exc_info=True)
         raise # Re-raise API error
    except Exception as e:
        # Catch other potential errors during load (e.g., network issues, schema mismatch detected by BQ)
        logger_worker.error(f"An unexpected error occurred during BigQuery load for {table_id}: {e}", exc_info=True)
        raise # Re-raise the exception


# +++ MODIFICATION START +++
# Modified process_object to handle multiple sheets from Excel

def process_object(object_name: str, target_dataset_id: str):
    logger_worker.critical(f"LOCAL_PROCESS_OBJECT_ENTRY_FOR: {object_name} | Thread: {threading.get_ident()}")
    logger_worker.info(f"--- Starting processing GCS object: {object_name} -> Target BQ Dataset: {target_dataset_id} ---")
    all_processing_successful = True
    if not re.match(r"^[a-zA-Z0-9_]+$", target_dataset_id):
         logger_worker.error(f"Invalid target_dataset_id format: '{target_dataset_id}'. Aborting.")
         raise ValueError(f"Invalid target dataset ID format: {target_dataset_id}")

    try:
        logger_worker.critical(f"LOCAL_PROCESS_OBJECT_BEFORE_READ_FOR: {object_name} | Thread: {threading.get_ident()}")
        # logger_worker.info(f"PROCESS_OBJECT_BEFORE_READ: Attempting to read GCS object: {object_name}") # Duplicate, covered by critical
        raw_data_container = read_data_from_gcs(WORKER_GCS_BUCKET, object_name)
        logger_worker.info(f"PROCESS_OBJECT_AFTER_READ: GCS object: {object_name}, raw_data_container type: {type(raw_data_container)}")
        if raw_data_container is None: 
            logger_worker.error(f"PROCESS_OBJECT_ERROR: Failed to read data for {object_name}, raw_data_container is None.")
            raise ValueError(f"Failed to read data from GCS: {object_name}")
        
        sheets_to_process_map: Dict[str, pd.DataFrame] = {}
        if isinstance(raw_data_container, pd.DataFrame):
            sheets_to_process_map["_default_"] = raw_data_container
            logger_worker.info("Read single DataFrame (CSV/Parquet or Excel fallback). Processing as default sheet.")
        elif isinstance(raw_data_container, dict):
            sheets_to_process_map = raw_data_container 
            logger_worker.info(f"Read {len(raw_data_container)} sheets from Excel file.")
        else:
            raise TypeError(f"Unexpected data type from read_data_from_gcs: {type(raw_data_container)}")

        base_filename_sanitized = sanitize_bq_name(os.path.splitext(os.path.basename(object_name))[0])
        overall_processed_table_count = 0

        if not sheets_to_process_map: 
            logger_worker.warning(f"No sheets/data found in {object_name}. Skipping."); 
            return

        logger_worker.info(f"PROCESS_OBJECT_BEFORE_SHEET_LOOP: For {object_name}, sheets to process: {list(sheets_to_process_map.keys()) if sheets_to_process_map else 'None'}")
        for sheet_name, original_raw_sheet_df in sheets_to_process_map.items():
            logger_worker.info(f"PROCESS_OBJECT_SHEET_LOOP_ENTRY: For {object_name}, processing sheet: '{sheet_name}' (Shape: {original_raw_sheet_df.shape if original_raw_sheet_df is not None else 'None'})")
            if original_raw_sheet_df is None or original_raw_sheet_df.empty:
                logger_worker.warning(f"Sheet '{sheet_name}' is empty or None. Skipping sheet.")
                continue
            
            identified_tables_in_sheet = _find_all_tables_in_sheet(
                original_raw_sheet_df,
                min_rows=WORKER_MIN_TABLE_ROWS, min_cols=WORKER_MIN_TABLE_COLS,
                block_density_threshold=WORKER_BLOCK_DENSITY_THRESHOLD,
                header_confidence=WORKER_MIN_HEADER_CONFIDENCE
            )

            if not identified_tables_in_sheet:
                logger_worker.info(f"No distinct tables found in sheet '{sheet_name}'.")
                continue
            logger_worker.info(f"Found {len(identified_tables_in_sheet)} distinct table candidate(s) in sheet '{sheet_name}'.")

            for table_data in identified_tables_in_sheet:
                isolated_df_block = table_data['df'] 
                table_suffix_id = table_data['id']
                table_score = table_data['score']
                logger_worker.info(f"--- Processing extracted table '{table_suffix_id}' from sheet '{sheet_name}' (Score: {table_score:.2f}, Shape: {isolated_df_block.shape}) ---")

                cleaned_df = clean_dataframe(isolated_df_block) # Pass the already isolated block
                if cleaned_df.empty:
                    logger_worker.warning(f"Table '{table_suffix_id}' from sheet '{sheet_name}' is empty after cleaning. Skipping.")
                    all_processing_successful = False; continue

                sanitized_sheet_name_for_bq = sanitize_bq_name(sheet_name)
                final_bq_table_name_parts = [base_filename_sanitized]
                
                if sheet_name != "_default_" or len(sheets_to_process_map) > 1 or (len(identified_tables_in_sheet) > 1 and sheet_name == "_default_"):
                    final_bq_table_name_parts.append(sanitized_sheet_name_for_bq)
                
                if len(identified_tables_in_sheet) > 1:
                    final_bq_table_name_parts.append(sanitize_bq_name(table_suffix_id))
                
                final_bq_table_name = "_".join(part for part in final_bq_table_name_parts if part) 
                final_bq_table_name = final_bq_table_name[:1024]
                bq_table_id_full = f"{WORKER_GCP_PROJECT}.{target_dataset_id}.{final_bq_table_name}"
                logger_worker.info(f"Target BQ table for '{table_suffix_id}' from sheet '{sheet_name}': {bq_table_id_full}")

                try:
                    schema = determine_schema(cleaned_df, bq_table_id_full, WORKER_DEFAULT_SCHEMA_STRATEGY)
                    if schema is None: 
                        logger_worker.error(f"Schema determination failed for {bq_table_id_full}. Skipping table.")
                        all_processing_successful = False; continue
                    
                    aligned_df = align_dataframe_to_schema(cleaned_df.copy(), schema) # Pass copy
                    if aligned_df.empty: 
                        logger_worker.warning(f"DataFrame for {bq_table_id_full} empty after alignment. Skipping load.")
                        continue
                    
                    load_to_bq(aligned_df, bq_table_id_full, schema, WORKER_DEFAULT_WRITE_DISPOSITION)
                    overall_processed_table_count += 1
                except Exception as per_table_error:
                    logger_worker.error(f"--- Error processing table '{table_suffix_id}' ({bq_table_id_full}) from sheet '{sheet_name}': {per_table_error} ---", exc_info=True)
                    all_processing_successful = False

        if overall_processed_table_count > 0 and all_processing_successful: 
            logger_worker.info(f"--- Successfully processed all {overall_processed_table_count} table(s) from GCS: {object_name} -> {target_dataset_id} ---")
        elif overall_processed_table_count > 0 and not all_processing_successful: 
            logger_worker.warning(f"--- Partially processed GCS: {object_name} -> {target_dataset_id}. {overall_processed_table_count} table(s) loaded, errors on others. ---")
        elif overall_processed_table_count == 0 and not sheets_to_process_map: 
            logger_worker.info(f"--- Finished {object_name}: No data/sheets found. ---")
        elif overall_processed_table_count == 0 and all_processing_successful: 
            logger_worker.warning(f"--- Finished {object_name}: No valid tables found/loaded to {target_dataset_id}. ---")
        else: # overall_processed_table_count == 0 and not all_processing_successful
             logger_worker.error(f"--- Failed to process any tables successfully for GCS: {object_name} -> {target_dataset_id}. ---")
             if sheets_to_process_map: # If sheets existed but yielded no tables due to errors
                  raise Exception(f"Failed to process any table successfully for {object_name} despite data being present.")

    except FileNotFoundError: 
        logger_worker.warning(f"GCS object not found: '{object_name}'. Allowing ACK.")
        return
    except ImportError as imp_err: 
        logger_worker.critical(f"PROCESS_OBJECT_IMPORT_ERROR: For {object_name}: {imp_err}. If 'scipy', multi-table detection failed.", exc_info=True)
        raise 
    except Exception as e:
        logger_worker.error(f"PROCESS_OBJECT_CRITICAL_ERROR: For GCS object {object_name}, dataset {target_dataset_id}: {type(e).__name__} - {e}", exc_info=True)
        raise







# +++ MODIFICATION END +++

# callback function remains UNCHANGED
def callback(message):
    object_name = None
    target_dataset_id = None
    raw_message_data = message.data
    
    # Initialize for logging, in case of early decode failure
    object_name_for_log = "PRE_DECODE_UNKNOWN"
    temp_object_name_for_log = "UNKNOWN_OBJECT_IN_CALLBACK" 

    try:
        message_str = raw_message_data.decode("utf-8")
        message_data = json.loads(message_str)
        object_name = message_data.get("object_name")
        
        # Update for logging after successful decode
        object_name_for_log = object_name or "OBJECT_NAME_MISSING_IN_JSON"
        temp_object_name_for_log = object_name_for_log

        target_dataset_id = message_data.get("target_dataset_id")

        # This critical log is now placed *after* attempting to decode object_name
        logger_worker.critical(f"LOCAL_CALLBACK_RECEIVED_MESSAGE_FOR: {object_name_for_log} | Thread: {threading.get_ident()}")
        logger_worker.info(f"CALLBACK_ENTRY: Processing Pub/Sub message for object: {temp_object_name_for_log}, Target Dataset: {target_dataset_id}")


        if not object_name or not target_dataset_id:
            logger_worker.error(f"CALLBACK_ERROR: Invalid Pub/Sub message format for {temp_object_name_for_log}: Missing 'object_name' or 'target_dataset_id'. Data: {message_str}")
            message.nack()
            return

        process_object(object_name, target_dataset_id) # This is where the main work happens
        
        message.ack()
        logger_worker.info(f"CALLBACK_SUCCESS: ACKed message for: {object_name}")

    except json.JSONDecodeError as json_err:
        # object_name_for_log would still be "PRE_DECODE_UNKNOWN" or its initial value if json.loads fails
        logger_worker.error(f"CALLBACK_JSON_ERROR: For initial log name '{object_name_for_log}'. Failed to parse Pub/Sub message JSON: {json_err}. Raw Data: '{raw_message_data.decode('utf-8', errors='replace')}'")
        message.nack()
    except Exception as e:
        # temp_object_name_for_log will hold the best guess for object_name
        logger_worker.exception(f"CALLBACK_CRITICAL_ERROR: For {temp_object_name_for_log}. Error: {e}")
        message.nack()
        logger_worker.info(f"CALLBACK_NACK: NACKed message for: {temp_object_name_for_log} due to critical error.")
    # Note: The critical log "LOCAL_CALLBACK_RECEIVED_MESSAGE_FOR" is now inside the try block,
    # so if json.loads fails, it won't print the decoded object_name. This is acceptable.
    # If you need it outside, you'd have to handle the decode failure before it.






# Main execution block remains UNCHANGED
if __name__ == "__main__":
    if not worker_subscriber or not WORKER_PUBSUB_SUBSCRIPTION:
        logger_worker.critical("Worker Pub/Sub subscriber client or subscription name is not configured. Exiting.")
        exit(1)

    worker_sub_path = worker_subscriber.subscription_path(WORKER_GCP_PROJECT, WORKER_PUBSUB_SUBSCRIPTION)
    logger_worker.info(f"Starting Pub/Sub Listener on subscription: {worker_sub_path}...")

    # Start the subscriber listening for messages
    streaming_pull_future = worker_subscriber.subscribe(worker_sub_path, callback=callback)
    logger_worker.info(f"Listener started successfully for {worker_sub_path}. Waiting for messages...")

    # Keep the main thread alive to allow the subscriber to run in the background
    try:
        # result() blocks indefinitely until the future is cancelled or an error occurs
        streaming_pull_future.result()
    except TimeoutError:
        # This typically won't happen with result() unless a timeout is specified in subscribe() options
        logger_worker.warning("Listener stopped due to timeout (unexpected).")
    except KeyboardInterrupt:
        logger_worker.info("Keyboard interrupt received. Shutting down listener...")
        streaming_pull_future.cancel() # Attempt to gracefully stop the background processing
        streaming_pull_future.result() # Wait for cancellation to complete
    except Exception as e:
        # Catch unexpected errors in the subscriber machinery itself
        logger_worker.exception(f"Pub/Sub Listener loop encountered an unhandled error: {e}")
        streaming_pull_future.cancel()
        try:
             streaming_pull_future.result(timeout=5) # Short timeout for cleanup
        except Exception:
             pass # Ignore errors during cleanup cancellation
    finally:
        # Ensure the future is cancelled if it exists and might still be running
        if 'streaming_pull_future' in locals() and streaming_pull_future and not streaming_pull_future.cancelled():
            logger_worker.info("Ensuring listener cancellation...")
            streaming_pull_future.cancel()
            try:
                streaming_pull_future.result(timeout=5)
            except Exception:
                 pass # Ignore final errors
        logger_worker.info("ETL Worker Listener stopped.")
# ============================== End of pasted section ==============================