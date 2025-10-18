# =====================================================
# Pegos Dataset Cleaning (Daily Folder)
# =====================================================
import os
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
api = HfApi(token=HF_TOKEN)
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

target_file = f"data/{TODAY}/pegos_final_dataset.csv"
print(f"📥 İndiriliyor: {target_file}")

path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=target_file,
    repo_type="dataset",
    token=HF_TOKEN,
)

df = pd.read_csv(path)
print(f"✅ Veri yüklendi ({len(df)} satır)")

# Temizlik
df.drop_duplicates(subset=["tweet", "time"], inplace=True)
df.dropna(subset=["tweet"], inplace=True)
if {"open","close"}.issubset(df.columns):
    df.dropna(subset=["open","close"], inplace=True)

for col in ["comment", "retweet", "like", "see_count", "diff"]:
    if col in df.columns and len(df) > 0:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df = df[df[col].between(q1, q3)]

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df = df[df["time"] >= "2020-01-01"]

# Save & Upload (yalnızca gün klasörü)
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out_path = f"/tmp/{TODAY}/cleaned.csv"
df.to_csv(out_path, index=False)
print(f"💾 Kaydedildi ({len(df)} satır)")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo=f"data/{TODAY}/cleaned.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("🚀 Temizlenmiş veri Hugging Face’e yüklendi.")
