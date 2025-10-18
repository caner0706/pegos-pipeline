# =====================================================
# Pegos Dataset Cleaner (Lossless Version - Soft Normalization)
# =====================================================
import os
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

print(f"🧹 Soft cleaning dataset for {TODAY}")

try:
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"data/{TODAY}/pegos_final_dataset.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    df = pd.read_csv(path, encoding="utf-8")
except Exception as e:
    raise RuntimeError(f"❌ Dataset indirilemedi: {e}")

print(f"✅ Veri yüklendi ({len(df)} satır)")

# === 1️⃣ Kolon standardizasyonu ===
drop_cols = [c for c in df.columns if c.endswith(("_x", "_y"))]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# === 2️⃣ Boş verileri doldur (drop yok) ===
text_cols = ["keyword", "tweet"]
num_cols = ["comment", "retweet", "like", "see_count", "open", "close", "diff", "direction"]

for c in text_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown")

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["time"] = df["time"].fillna(datetime.utcnow())

# === 3️⃣ Outlier'ları kırp (truncate) ===
for c in ["comment", "retweet", "like", "see_count", "diff"]:
    if c in df.columns and len(df) > 0:
        low, high = df[c].quantile(0.005), df[c].quantile(0.995)
        df[c] = df[c].clip(lower=low, upper=high)

# === 4️⃣ Boş kolonlar varsa oluştur ===
required = [
    "keyword","tweet","time","comment","retweet","like","see_count",
    "open","close","diff","direction","day",
    "pred_label","pred_proba","pred_diff","Tahmin",
    "source_day","processing_day"
]
for c in required:
    if c not in df.columns:
        df[c] = pd.NA

# === 5️⃣ Kaydet ===
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out = f"/tmp/{TODAY}/cleaned.csv"
df.to_csv(out, index=False, encoding="utf-8")
print(f"💾 Kaydedildi (satır sayısı: {len(df)})")

# === 6️⃣ Yükle ===
try:
    upload_file(
        path_or_fileobj=out,
        path_in_repo=f"data/{TODAY}/cleaned.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print("🚀 Cleaned dataset (lossless) Hugging Face’e yüklendi.")
except Exception as e:
    print(f"⚠️ Upload sırasında hata: {e}")
