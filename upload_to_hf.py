# =====================================================
# upload_to_hf.py — Günsüz Pegos CSV yükleyici
# =====================================================
import os, sys
import pandas as pd
from huggingface_hub import upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
LOCAL_CSV = os.getenv("LOCAL_CSV")

if not HF_TOKEN or not HF_DATASET_REPO:
    print("❌ HF_TOKEN veya HF_DATASET_REPO eksik.")
    sys.exit(1)

if not LOCAL_CSV or not os.path.exists(LOCAL_CSV):
    print(f"❌ LOCAL_CSV bulunamadı: {LOCAL_CSV}")
    sys.exit(1)

print("🚀 HF Upload başlatıldı...")

# CSV oku
df = pd.read_csv(LOCAL_CSV, encoding="utf-8", dtype=str)
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

# Geçici kaydet
out = "/tmp/latest.csv"
df.to_csv(out, index=False, encoding="utf-8")

# Hugging Face’e yükle
upload_file(
    path_or_fileobj=out,
    path_in_repo="data/latest.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("✅ Uploaded: data/latest.csv")

# Arşiv kopyası (ham veri yedeği)
upload_file(
    path_or_fileobj=out,
    path_in_repo="data/blockchain_tweets.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("✅ Uploaded: data/blockchain_tweets.csv")
