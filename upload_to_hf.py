# =====================================================
# Pegos Upload Script (Daily Folder Upload)
# =====================================================
import os
import sys
from datetime import datetime
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "Caner7/pegos-stream")

if HF_TOKEN is None:
    print("❌ HF_TOKEN missing.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
LOCAL_CSV = os.environ.get("LOCAL_CSV", "/tmp/data/pegos_output.csv")

if not os.path.exists(LOCAL_CSV):
    print(f"❌ CSV not found: {LOCAL_CSV}")
    sys.exit(1)

# Günlük klasör ismi
today = datetime.utcnow().strftime("%Y-%m-%d")
basename = f"data/{today}/blockchain_tweets_{today}.csv"
latest_path = f"data/{today}/latest.csv"

print(f"📁 Günlük klasör: {today}")

# Arşiv dosyası
api.upload_file(
    path_or_fileobj=LOCAL_CSV,
    path_in_repo=basename,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)
print(f"✅ Uploaded archive: {basename}")

# En son veri
api.upload_file(
    path_or_fileobj=LOCAL_CSV,
    path_in_repo=latest_path,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)
print(f"✅ Updated daily latest: {latest_path}")