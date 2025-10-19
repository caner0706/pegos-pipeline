# =====================================================
# upload_to_hf.py — Günlük CSV'yi doğru GÜN klasörüne yükle (Rollover Guard)
# =====================================================
import os, sys
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
LOCAL_CSV = os.getenv("LOCAL_CSV")
UTC_TODAY = datetime.utcnow().strftime("%Y-%m-%d")

if not HF_TOKEN or not HF_DATASET_REPO:
    print("❌ HF_TOKEN/HF_DATASET_REPO eksik.")
    sys.exit(1)
if not LOCAL_CSV or not os.path.exists(LOCAL_CSV):
    print(f"❌ LOCAL_CSV bulunamadı: {LOCAL_CSV}")
    sys.exit(1)

# Varsayılan: CSV yolundaki gün klasörü
try:
    path_day = LOCAL_CSV.strip("/").split("/")[-2]
except Exception:
    path_day = UTC_TODAY

# ROLLOVER GUARD: Yol farklı olsa bile DAİMA BUGÜNÜ kullan
day_folder = UTC_TODAY
print(f"📁 Gün klasörü (UTC): {day_folder}")

# CSV oku & min normalize
df = pd.read_csv(LOCAL_CSV, encoding="utf-8", dtype=str)
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
else:
    df["time"] = pd.NaT

# Kaydet (aynı dosyayı hem arşiv hem latest olarak)
os.makedirs(f"/tmp/{day_folder}", exist_ok=True)
out = f"/tmp/{day_folder}/latest.csv"
df.to_csv(out, index=False, encoding="utf-8")

api = HfApi(token=HF_TOKEN)

archive = f"data/{day_folder}/blockchain_tweets_{day_folder}.csv"
upload_file(path_or_fileobj=out, path_in_repo=archive,
            repo_id=HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
print(f"✅ Uploaded archive: {archive}")

upload_file(path_or_fileobj=out, path_in_repo=f"data/{day_folder}/latest.csv",
            repo_id=HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
print(f"✅ Updated: data/{day_folder}/latest.csv")
