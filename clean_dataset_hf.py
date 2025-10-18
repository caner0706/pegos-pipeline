# =====================================================
# Pegos Dataset Cleaner (Stable + UTF-8 Safe)
# =====================================================
import os
import pandas as pd
from datetime import datetime
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

print(f"🧹 Cleaning dataset for {TODAY}")

try:
    path = hf_hub_download(repo_id=HF_DATASET_REPO,
                           filename=f"data/{TODAY}/pegos_final_dataset.csv",
                           repo_type="dataset",
                           token=HF_TOKEN)
    df = pd.read_csv(path, encoding="utf-8")
except Exception as e:
    raise RuntimeError(f"❌ Dataset indirilemedi: {e}")

print(f"✅ Veri yüklendi ({len(df)} satır)")

# Temizlik
drop_cols = [c for c in df.columns if c.endswith(("_x", "_y"))]
df.drop(columns=drop_cols, inplace=True, errors="ignore")
df.dropna(subset=["tweet", "time"], inplace=True)
df.drop_duplicates(subset=["tweet", "time"], inplace=True)

# Aykırı değerleri yumuşat
for c in ["comment", "retweet", "like", "see_count", "diff"]:
    if c in df:
        q1, q99 = df[c].quantile(0.01), df[c].quantile(0.99)
        df = df[df[c].between(q1, q99)]

os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out = f"/tmp/{TODAY}/cleaned.csv"
df.to_csv(out, index=False, encoding="utf-8")
print(f"💾 Kaydedildi ({len(df)} satır)")

upload_file(path_or_fileobj=out,
            path_in_repo=f"data/{TODAY}/cleaned.csv",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN)
print("🚀 Cleaned dataset Hugging Face’e yüklendi.")
