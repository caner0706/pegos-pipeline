# =====================================================
# upload_to_hf.py
# Günlük CSV'yi Hugging Face Dataset'e yükler (stable schema)
# =====================================================
import os
import sys
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
LOCAL_CSV = os.getenv("LOCAL_CSV")

if not HF_TOKEN or not HF_DATASET_REPO:
    print("❌ HF_TOKEN veya HF_DATASET_REPO eksik.")
    sys.exit(1)

if not LOCAL_CSV or not os.path.exists(LOCAL_CSV):
    print(f"❌ LOCAL_CSV bulunamadı: {LOCAL_CSV}")
    sys.exit(1)

# Günlük klasör belirle
try:
    day_folder = LOCAL_CSV.strip("/").split("/")[-2]
except Exception:
    day_folder = datetime.utcnow().strftime("%Y-%m-%d")

print(f"📁 Günlük klasör: {day_folder}")

# CSV'yi oku ve tek şemaya normalize et
df = pd.read_csv(LOCAL_CSV, encoding="utf-8", dtype=str)
df["time"] = pd.to_datetime(df.get("time"), errors="coerce", utc=True)
df = df.dropna(subset=["time"])
df["day"] = df["time"].dt.strftime("%Y-%m-%d")

REQUIRED_COLS = [
    "keyword","tweet","time","comment","retweet","like","see_count",
    "open","close","diff","direction","day",
    "pred_label","pred_proba","pred_diff","Tahmin",
    "source_day","processing_day"
]
for c in REQUIRED_COLS:
    if c not in df.columns:
        df[c] = pd.NA

df["source_day"] = day_folder
df["processing_day"] = datetime.utcnow().strftime("%Y-%m-%d")
df = df[REQUIRED_COLS]

# Geçersiz satırları filtrele
df.dropna(subset=["tweet"], inplace=True)
if df.empty:
    print("⚠️ CSV boş, upload adımı atlandı.")
    sys.exit(0)

# Dosya kaydet
os.makedirs(f"/tmp/{day_folder}", exist_ok=True)
out_path = f"/tmp/{day_folder}/latest.csv"
df.to_csv(out_path, index=False)

api = HfApi(token=HF_TOKEN)

# Arşiv + Latest olarak yükle
archive_name = f"data/{day_folder}/blockchain_tweets_{day_folder}.csv"
upload_file(path_or_fileobj=out_path,
            path_in_repo=archive_name,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN)
print(f"✅ Uploaded archive: {archive_name}")

upload_file(path_or_fileobj=out_path,
            path_in_repo=f"data/{day_folder}/latest.csv",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN)
print(f"✅ Updated: data/{day_folder}/latest.csv")
