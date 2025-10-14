import os
import sys
from datetime import datetime
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "Caner7/pegos-stream")

if HF_TOKEN is None:
    print("HF_TOKEN missing.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
LOCAL_CSV = os.environ.get("LOCAL_CSV", "data/output.csv")

if not os.path.exists(LOCAL_CSV):
    print(f"CSV not found: {LOCAL_CSV}")
    sys.exit(1)

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
basename = f"blockchain_tweets_{timestamp}.csv"

# Arşiv dosyası
api.upload_file(
    path_or_fileobj=LOCAL_CSV,
    path_in_repo=f"data/{basename}",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)
print("✅ Uploaded:", f"data/{basename}")

# En son veri
api.upload_file(
    path_or_fileobj=LOCAL_CSV,
    path_in_repo="data/latest.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)
print("✅ Updated: data/latest.csv")
