# =====================================================
# Pegos Dataset Cleaning (Safe & Minimal Loss Version)
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

# === 1️⃣ Temel temizlik ===
df.drop_duplicates(subset=["tweet", "time"], inplace=True)
df.dropna(subset=["tweet"], inplace=True)
if {"open", "close"}.issubset(df.columns):
    df.dropna(subset=["open", "close"], inplace=True)

# === 2️⃣ Etkileşim filtresi (hepsi 0 olanları sil) ===
if all(col in df.columns for col in ["comment", "retweet", "like", "see_count"]):
    before = len(df)
    df = df[~((df["comment"] == 0) & (df["retweet"] == 0) &
              (df["like"] == 0) & (df["see_count"] == 0))]
    removed = before - len(df)
    print(f"🧹 Sıfır etkileşimli {removed} satır temizlendi.")

# === 3️⃣ Tarih ve zaman doğrulama ===
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df = df[df["time"] >= "2020-01-01"]

# === 4️⃣ Outlier temizliği (devre dışı) ===
# (Veri kaybını önlemek için bu adım atlandı)
# for col in ["comment", "retweet", "like", "see_count", "diff"]:
#     if col in df.columns and len(df) > 0:
#         q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
#         df = df[df[col].between(q1, q3)]

# === 5️⃣ Kaydet ve yükle ===
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
