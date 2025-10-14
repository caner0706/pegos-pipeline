# =====================================================
# Pegos Dataset Cleaning Script (Safe Version) 🧹
# =====================================================
import os
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

api = HfApi(token=HF_TOKEN)

print("📥 Hugging Face'ten merged dataset indiriliyor...")
path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename="data/latest_merged.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)

df = pd.read_csv(path)
print(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

# -----------------------------
# 1️⃣ İlgili sütunları seç
# -----------------------------
columns_needed = [
    "tweet",
    "comment",
    "retweet",
    "like",
    "see_count",
    "time",
    "open",
    "close",
    "diff",
]
df = df[[c for c in columns_needed if c in df.columns]].copy()

df.rename(columns={
    "open": "Açılış Fiyatı (USD)",
    "close": "Kapanış Fiyatı (USD)",
    "diff": "Fark (USD)"
}, inplace=True)

# -----------------------------
# 2️⃣ Yinelenen veriler kaldır
# -----------------------------
before = len(df)
df.drop_duplicates(subset=["tweet", "time"], inplace=True)
print(f"🧩 Yinelenen veriler kaldırıldı: {before - len(df)} satır silindi.")

# -----------------------------
# 3️⃣ Eksik değerleri temizle
# -----------------------------
before = len(df)
df.dropna(subset=["tweet", "Açılış Fiyatı (USD)", "Kapanış Fiyatı (USD)"], inplace=True)
print(f"🚿 Eksik değer temizliği: {before - len(df)} satır silindi.")

# -----------------------------
# 4️⃣ Aykırı değer analizi (soft filter)
# -----------------------------
def soft_outlier_filter(series):
    q1 = series.quantile(0.01)
    q3 = series.quantile(0.99)
    return series.between(q1, q3)

numeric_cols = ["comment", "retweet", "like", "see_count", "Fark (USD)"]
before = len(df)
for col in numeric_cols:
    if col in df.columns:
        mask = soft_outlier_filter(df[col])
        removed = (~mask).sum()
        df = df[mask]
        print(f"⚖️ {col} sütununda {removed} uç değer kaldırıldı.")
print(f"📉 Aykırı değer temizliği sonrası toplam {before - len(df)} satır silindi.")

# -----------------------------
# 5️⃣ Tarih formatı kontrolü
# -----------------------------
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df = df[df["time"] > "2020-01-01"]

# -----------------------------
# 6️⃣ Sonuçları kaydet
# -----------------------------
output_path = "/tmp/pegos_cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print(f"💾 Temizlenmiş veri kaydedildi: {output_path} ({len(df)} satır)")

# -----------------------------
# 7️⃣ Hugging Face'e yükle
# -----------------------------
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo="data/latest_cleaned.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
)
print("🚀 Hugging Face'e yükleme tamamlandı: data/latest_cleaned.csv")
