import os
import sys
from datetime import datetime
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
LOCAL_CSV = os.environ.get("LOCAL_CSV")

if not HF_TOKEN or not HF_DATASET_REPO:
    print("❌ HF_TOKEN veya HF_DATASET_REPO eksik.")
    sys.exit(1)

if not LOCAL_CSV or not os.path.exists(LOCAL_CSV):
    print("❌ LOCAL_CSV yok veya bulunamadı:", LOCAL_CSV)
    sys.exit(1)

# /tmp/data/YYYY-MM-DD/pegos_output.csv -> YYYY-MM-DD
try:
    parts = LOCAL_CSV.strip("/").split("/")
    day_folder = parts[-2]  # .../data/YYYY-MM-DD/pegos_output.csv
except Exception:
    day_folder = datetime.utcnow().strftime("%Y-%m-%d")

print(f"📁 Günlük klasör: {day_folder}")

api = HfApi(token=HF_TOKEN)

# Arşiv adı: blockchain_tweets_YYYY-MM-DD.csv
archive_name = f"data/{day_folder}/blockchain_tweets_{day_folder}.csv"

upload_file(
    path_or_fileobj=LOCAL_CSV,
    path_in_repo=archive_name,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("✅ Uploaded archive:", archive_name)

# Günlük latest
upload_file(
    path_or_fileobj=LOCAL_CSV,
    path_in_repo=f"data/{day_folder}/latest.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("✅ Updated:", f"data/{day_folder}/latest.csv")
