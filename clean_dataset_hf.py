# =====================================================
# Pegos Dataset Cleaning Script 🧹
# =====================================================
import os
import pandas as pd
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
df = df[[c for c in columns_needed if c in df.columns]]

# Sütun isimlerini Türkçe'ye çevir
df.rename(columns={
    "open": "Açılış Fiyatı (USD)",
    "close": "Kapanış Fiyatı (USD)",
    "diff": "Fark (USD)"
}, inplace=True)

# -----------------------------
# 2️⃣ Yinelenen verileri kaldır
# -----------------------------
before = len(df)
df.drop_duplicates(subset=["tweet", "time"], inplace=True)
after = len(df)
print(f"🧩 Yinelenen veriler kaldırıldı: {before - after} satır silindi.")

# -----------------------------
# 3️⃣ Eksik değerleri temizle
# -----------------------------
missing_before = df.isnull().sum().sum()
df.dropna(subset=["tweet", "Açılış Fiyatı (USD)", "Kapanış Fiyatı (USD)"], inplace=True)
missing_after = df.isnull().sum().sum()
print(f"🚿 Eksik değer temizliği tamamlandı. {missing_before - missing_after} boş alan temizlendi.")

# -----------------------------
# 4️⃣ Aykırı değer analizi (Z-score)
# -----------------------------
import numpy as np
for col in ["comment", "retweet", "like", "see_count", "Fark (USD)"]:
    if col in df.columns:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = (z > 3).sum()
        df = df[z <= 3]
        print(f"⚠️ {col} sütununda {outliers} aykırı değer çıkarıldı.")

# -----------------------------
# 5️⃣ Tarih formatı düzelt
# -----------------------------
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df.dropna(subset=["time"], inplace=True)

# -----------------------------
# 6️⃣ Sonuçları kaydet
# -----------------------------
output_path = "/tmp/pegos_cleaned_dataset.csv"
df.to_csv(output_path, index=False)
print(f"💾 Temizlenmiş veri kaydedildi: {output_path} ({len(df)} satır)")

# Hugging Face'e yükle
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo="data/latest_cleaned.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
)
print("🚀 Hugging Face'e yükleme tamamlandı: data/latest_cleaned.csv")
