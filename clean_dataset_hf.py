# =====================================================
# Pegos Daily Dataset Cleaning Script 🧹
# =====================================================
import os
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

# ------------------------------------------
# Ortam değişkenleri
# ------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
if not HF_TOKEN or not HF_DATASET_REPO:
    raise RuntimeError("❌ HF_TOKEN veya HF_DATASET_REPO tanımlı değil!")

api = HfApi(token=HF_TOKEN)

# ------------------------------------------
# En güncel klasörü bul
# ------------------------------------------
print("📂 Günlük klasörler listeleniyor...")
files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
daily_folders = sorted(
    list({f.split("/")[1] for f in files if f.startswith("data/") and len(f.split("/")) > 2})
)
if not daily_folders:
    raise RuntimeError("❌ Günlük klasör bulunamadı!")
latest_day = daily_folders[-1]
print(f"📅 En güncel klasör: {latest_day}")

merged_file = f"data/{latest_day}/merged.csv"
print(f"📥 Dosya indiriliyor: {merged_file}")

path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=merged_file,
    repo_type="dataset",
    token=HF_TOKEN,
)

df = pd.read_csv(path)
print(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

# ------------------------------------------
# 1️⃣ Kolon seçimi
# ------------------------------------------
columns_needed = [
    "tweet",
    "comment",
    "retweet",
    "like",
    "see_count",
    "time",
    "open",
    "close",
    "diff"
]
df = df[[c for c in columns_needed if c in df.columns]].copy()
df.rename(columns={
    "open": "Açılış Fiyatı (USD)",
    "close": "Kapanış Fiyatı (USD)",
    "diff": "Fark (USD)"
}, inplace=True)

# ------------------------------------------
# 2️⃣ Yinelenen ve eksik veriler
# ------------------------------------------
before = len(df)
df.drop_duplicates(subset=["tweet", "time"], inplace=True)
print(f"🧩 Yinelenenler: {before - len(df)} satır silindi.")

before = len(df)
df.dropna(subset=["tweet", "Açılış Fiyatı (USD)", "Kapanış Fiyatı (USD)"], inplace=True)
print(f"🚿 Eksik değer temizliği: {before - len(df)} satır silindi.")

# ------------------------------------------
# 3️⃣ Aykırı değer analizi (soft filter)
# ------------------------------------------
def soft_outlier_filter(series):
    q1 = series.quantile(0.01)
    q3 = series.quantile(0.99)
    return series.between(q1, q3)

numeric_cols = ["comment", "retweet", "like", "see_count", "Fark (USD)"]
for col in numeric_cols:
    if col in df.columns:
        mask = soft_outlier_filter(df[col])
        removed = (~mask).sum()
        df = df[mask]
        print(f"⚖️ {col}: {removed} uç değer kaldırıldı.")

# ------------------------------------------
# 4️⃣ Tarih kontrolü
# ------------------------------------------
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df = df[df["time"] > "2020-01-01"]

# ------------------------------------------
# 5️⃣ Kaydet ve HF'ye yükle
# ------------------------------------------
output_path = f"/tmp/cleaned_{latest_day}.csv"
df.to_csv(output_path, index=False)
print(f"💾 Temizlenmiş veri kaydedildi: {output_path} ({len(df)} satır)")

remote_path = f"data/{latest_day}/cleaned.csv"
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo=remote_path,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
)
print(f"🚀 Hugging Face'e yükleme tamamlandı: {remote_path}")