# =====================================================
# Pegos Dataset Cleaner (strict numeric zero filter)
# =====================================================
import os
import pandas as pd
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

print("🧽 Dataset temizleniyor...")

# 1️⃣ Veri oku
p = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename="data/daily_raw.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(p, encoding="utf-8")
print(f"✅ Veri yüklendi ({len(df)} satır)")

# 2️⃣ Kolonları zorla numerik tipe çevir
for c in ["comment", "retweet", "like", "see_count"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    else:
        df[c] = 0

# 3️⃣ Temizlik (sadece 0,0,0 olan satırları kaldır)
before = len(df)
df = df[~((df["comment"] == 0) & (df["retweet"] == 0) & (df["like"] == 0))]
after = len(df)
print(f"🧹 {before - after} satır sıfır etkileşimli olarak temizlendi. ({after} satır kaldı)")

# 4️⃣ Kaydet ve yükle
os.makedirs("/tmp/data", exist_ok=True)
out_path = "/tmp/data/cleaned.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo="data/cleaned.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("🚀 Temizlenmiş dataset Hugging Face’e yüklendi.")
