# =====================================================
# Pegos Dataset Cleaner (no-day version)
# =====================================================
import os
import pandas as pd
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

print("🧽 Dataset temizleniyor...")

p = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename="data/pegos_final_dataset.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(p, encoding="utf-8")
print(f"✅ Veri yüklendi ({len(df)} satır)")

if all(c in df.columns for c in ["comment","retweet","like","see_count"]):
    before = len(df)
    df = df[~((df["comment"]==0)&(df["retweet"]==0)&(df["like"]==0)&(df["see_count"]==0))]
    print(f"🧹 {before - len(df)} satır sıfır etkileşimli veri temizlendi.")

out = "/tmp/cleaned.csv"
df.to_csv(out, index=False, encoding="utf-8")

upload_file(
    path_or_fileobj=out,
    path_in_repo="data/cleaned.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("🚀 Temizlenmiş dataset Hugging Face'e yüklendi.")
