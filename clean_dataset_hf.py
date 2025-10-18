# =====================================================
# Pegos Dataset Cleaning (Only Zero-Engagement Filter)
# =====================================================
import os
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download, upload_file

# === Ortam değişkenleri ===
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

# === Veri indirme ===
target_file = f"data/{TODAY}/pegos_final_dataset.csv"
print(f"📥 İndiriliyor: {target_file}")

path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=target_file,
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(path, encoding="utf-8")
print(f"✅ Veri yüklendi ({len(df)} satır)")

# === 1️⃣ Sadece sıfır etkileşimli satırları temizle ===
if all(col in df.columns for col in ["comment", "retweet", "like", "see_count"]):
    before = len(df)
    df = df[~((df["comment"] == 0) & (df["retweet"] == 0) &
              (df["like"] == 0) & (df["see_count"] == 0))]
    removed = before - len(df)
    print(f"🧹 Sıfır etkileşimli {removed} satır temizlendi.")

# === 2️⃣ Kaydet ve HF'e yükle ===
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out_path = f"/tmp/{TODAY}/cleaned.csv"
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"💾 Kaydedildi ({len(df)} satır)")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo=f"data/{TODAY}/cleaned.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("🚀 Temizlenmiş veri Hugging Face’e yüklendi.")
