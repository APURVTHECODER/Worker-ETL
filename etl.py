import os, io, json, pandas as pd, requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Google Cloud clients and authentication
from google.cloud import storage, bigquery
from google.cloud.pubsub_v1 import SubscriberClient
from google.oauth2 import service_account

# ——— Configuration ———
PROJECT              = os.getenv("GCP_PROJECT")
BUCKET               = os.getenv("GCS_BUCKET")
BIGQUERY_DATASET     = os.getenv("BQ_DATASET")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
SUBSCRIPTION_NAME    = os.getenv("PUBSUB_SUBSCRIPTION")
KEY_PATH             = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "sl-etb-bot.json")

# Authenticate and initialize clients
creds = service_account.Credentials.from_service_account_file(KEY_PATH)
storage_client = storage.Client(credentials=creds, project=PROJECT)
bq_client      = bigquery.Client(credentials=creds, project=PROJECT)
subscriber     = SubscriberClient(credentials=creds)

# ——— Functions ———
def infer_schema(df: pd.DataFrame) -> list:
    # Take sample and log it
    sample = df.head(5).astype(str).to_dict(orient="records")
    print("🔍 Sample sent to Gemini:\n", json.dumps(sample, indent=2))

    # Check if sample is empty or junk
    if not sample or all(all(not v or v.lower() == 'nan' for v in row.values()) for row in sample):
        raise ValueError("❌ Sample data is empty or invalid for schema inference.")

    prompt = (
        "Given these sample rows, suggest SQL column names and BigQuery types:\n"
        f"{sample}\n\n"
        "Return only a JSON list of {{\"name\":..., \"type\":...}}."
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    print("📦 Gemini raw response:\n", json.dumps(result, indent=2))

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        text = text.replace("```json\n", "").replace("\n```", "").strip()
        schema = json.loads(text)
        if not schema:
            raise ValueError("Empty schema returned from Gemini.")
        print("✅ Parsed schema:", schema)
        return schema
    except Exception as e:
        print("❗ Gemini response parsing failed")
        raise ValueError(f"Failed to parse schema from Gemini: {e}")

def get_existing_schema(table_id: str) -> list:
    """Fetch the existing schema of the table from BigQuery."""
    try:
        table = bq_client.get_table(table_id)  # Fetch table metadata
        return table.schema
    except Exception as e:
        print(f"Error fetching schema for table {table_id}: {e}")
        return []

def process_object(object_name: str):
    bucket = storage_client.bucket(BUCKET)
    blob = bucket.blob(object_name)

    if not blob.exists():
        print(f"File {object_name} does not exist in the bucket.")
        return

    data = blob.download_as_bytes()

    # Read the Excel file, skipping metadata rows
    df = pd.read_excel(io.BytesIO(data), header=3)

    # Drop empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Define initial column names
    correct_columns = [
        'Project Name', 'Task Name', 'Assigned to', 
        'Start Date', 'Days Required', 'End Date', 'Progress'
    ]

    # Adjust column names if there are extra columns
    if len(df.columns) > len(correct_columns):
        correct_columns += [f'Unnamed_{i}' for i in range(len(df.columns) - len(correct_columns))]
    df.columns = correct_columns[:len(df.columns)]

    # Clean whitespace from all string values
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert date columns to Python date objects
    date_columns = ['Start Date', 'End Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True).dt.date

    # Convert 'Days Required' to nullable integer (Int64)
    if 'Days Required' in df.columns:
        df['Days Required'] = pd.to_numeric(df['Days Required'], errors='coerce').astype('Int64')

    # Handle the Progress column
    if 'Progress' in df.columns:
        df['Progress'] = df['Progress'].astype(str)
        df['Progress'] = df['Progress'].str.replace('%', '', regex=False).astype(float) / 100.0

    # Infer the schema using Gemini API (assumed to return the schema shown)
    schema = infer_schema(df)

    # Rename DataFrame columns to match schema field names
    field_names = [field['name'] for field in schema]
    df.columns = field_names

    # Load the DataFrame into BigQuery
    table_id = f"{PROJECT}.{BIGQUERY_DATASET}.your_table_name"  # Replace with your table name
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND  # Or WRITE_TRUNCATE to overwrite
    )

    job = bq_client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete
    print(f"Loaded {len(df)} rows into {table_id}")

def callback(msg):
    object_name = msg.data.decode("utf-8")
    try:
        process_object(object_name)
        msg.ack()
    except Exception as e:
        print(f"Error processing {object_name}: {e}")
        msg.nack()

if __name__ == "__main__":
    if not SUBSCRIPTION_NAME:
        raise ValueError("PUBSUB_SUBSCRIPTION not set in environment variables.")
    sub_path = subscriber.subscription_path(PROJECT, SUBSCRIPTION_NAME)
    streaming_pull_future = subscriber.subscribe(sub_path, callback=callback)
    print(f"Listening for messages on {sub_path}...")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        print("Stopped listening for messages.")
