# =====================================================
# Pegos Dataset Cleaning (Only Zero-Engagement Filter)
# =====================================================
import os
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

target = f"data/{TODAY}/pegos_final_dataset.csv"
print("🧽 Cleaning dataset...")
print(f"📥 İndiriliyor: {target}")

p = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=target,
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(p, encoding="utf-8")
print(f"✅ Veri yüklendi ({len(df)} satır)")

if all(c in df.columns for c in ["comment","retweet","like","see_count"]):
    before = len(df)
    df = df[~((df["comment"]==0)&(df["retweet"]==0)&(df["like"]==0)&(df["see_count"]==0))]
    print(f"🧹 Sıfır etkileşimli {before-len(df)} satır temizlendi.")

os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out = f"/tmp/{TODAY}/cleaned.csv"
df.to_csv(out, index=False, encoding="utf-8")
print(f"💾 Kaydedildi ({len(df)} satır)")

upload_file(
    path_or_fileobj=out,
    path_in_repo=f"data/{TODAY}/cleaned.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("🚀 Temizlenmiş veri Hugging Face’e yüklendi.")
