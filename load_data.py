import s3fs
import os
import zipfile

# 1. Setup Environment Variables (Fill these in for your PC)
os.environ.setdefault('AWS_S3_ENDPOINT', 'minio.lab.sspcloud.fr')
os.environ.setdefault('AWS_ACCESS_KEY_ID', '8F2BK7HWJ9O5R78N29DN')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'xJ0PQREn0kGIo9vNpQ3f4VCTZaCbPg8e2t9Y2eM5')
os.environ.setdefault('AWS_SESSION_TOKEN', 'eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiI4RjJCSzdIV0o5TzVSNzhOMjlETiIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzY4MzA5NzU1LCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6InNpbWVvbi50c2FuZ0BlbnNhZS5mciIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJleHAiOjE3Njg5MTQ1NjIsImZhbWlseV9uYW1lIjoiVHNhbmciLCJnaXZlbl9uYW1lIjoiU2ltw6lvbiIsImdyb3VwcyI6WyJVU0VSX09OWVhJQSIsImNvbXBhcmFpc29uLW5pdmVhdS1wcml4Il0sImlhdCI6MTc2ODMwOTc2MiwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDowYjQxYTY5MC0zYWM4LTUzMjktYmEwNS01ZTY3MzU3Yjk5OWIiLCJsb2NhbGUiOiJmciIsIm5hbWUiOiJTaW3DqW9uIFRzYW5nIiwicG9saWN5Ijoic3Rzb25seSIsInByZWZlcnJlZF91c2VybmFtZSI6InNpbTIwMjMiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy1zc3BjbG91ZCJdLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGdyb3VwcyBlbWFpbCIsInNpZCI6IjQ2YjlkNTJiLTg0NzgtMzk3My03YjY1LTQwNDQwYjViNDllNSIsInN1YiI6IjFhYzA5MzI5LWQ2MGYtNDI2OS1hN2NhLWI5ODMyNjUwYmU4OSIsInR5cCI6IkJlYXJlciJ9.FioOMgT4rQwVuaQSIFica44Dspz9Hyncv2QL3748dcVaez93Wi8Oh30YMAqzILZm3HU4YYLMFFUcybR_xXE2tA')

# 2. Clean the Endpoint URL
endpoint = os.environ.get('AWS_S3_ENDPOINT', "minio.lab.sspcloud.fr")
if not endpoint.startswith("http"):
    endpoint_url = f"https://{endpoint}"
else:
    endpoint_url = endpoint

# 3. Initialize Filesystem
# Note: Use anon=False to ensure it uses your credentials
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': endpoint_url},
    key=os.environ.get('AWS_ACCESS_KEY_ID'),
    secret=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    token=os.environ.get('AWS_SESSION_TOKEN')
)

my_bucket = "sim2023"


def safe_download_and_extract(s3_path, extraction_dir):
    # REMOVE 's3://' prefix if it exists for s3fs methods
    clean_path = s3_path.replace("s3://", "")
    local_zip = "temp_data.zip"

    print(f"Downloading {clean_path}...")
    try:
        fs.get(clean_path, local_zip)

        if not os.path.exists(extraction_dir):
            os.makedirs(extraction_dir)

        print(f"Extracting to '{extraction_dir}'...")
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)

        os.remove(local_zip)
        print("Success.")
    except Exception as e:
        print(f"Error: {e}")


# Download Datasets
datasets = [
    "sim2023/intro_model_gen/archive (2).zip"
]

for ds in datasets:
    safe_download_and_extract(ds, "data")
