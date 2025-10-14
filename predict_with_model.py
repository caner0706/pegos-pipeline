# =====================================
# Pegos Prediction Module (4-hour loop)
# =====================================
import os
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi, hf_hub_download, upload_file

# 🔐 Ortam değişkenleri (GitHub Secrets)
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "Caner7/pegos-stream")

print("🤖 Running Pegos prediction pipeline...")

# ===================================================
# 1️⃣ Model, scaler ve tokenizer yükleme (try/except)
# ===================================================
try:
    print("📦 Loading model and scaler...")
    model = joblib.load("pegos_lightgbm.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"❌ Model/Scaler load error: {e}")
    print("➡️ Skipping this prediction cycle.")
    exit(0)

try:
    print("🔤 Loading BERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
    bert_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model.to(device)
except Exception as e:
    print(f"⚠️ Failed to load BERT model: {e}")
    exit(0)

# ===================================================
# 2️⃣ Veri indir (Hugging Face Dataset)
# ===================================================
try:
    print("📥 Downloading latest_cleaned.csv from Hugging Face...")
    hf_file = "data/latest_cleaned.csv"
    local_file = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=hf_file,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    df = pd.read_csv(local_file)
    print(f"✅ Loaded {len(df)} rows, {df.shape[1]} columns.")
except Exception as e:
    print(f"⚠️ Failed to download dataset: {e}")
    print("➡️ Skipping this prediction cycle.")
    exit(0)

if df.empty:
    print("⚠️ Dataset is empty — skipping.")
    exit(0)

# ===================================================
# 3️⃣ Embedding çıkarma fonksiyonu
# ===================================================
def get_bert_embeddings(texts, tokenizer, model, batch_size=16, device="cpu"):
    model.to(device)
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(device)
            outputs = model(**tokens)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls_embeds)
    return np.vstack(all_embeds)

# ===================================================
# 4️⃣ Veri hazırlığı (tweet + numerik)
# ===================================================
try:
    df = df.dropna(subset=["tweet"])
    texts = df["tweet"].astype(str).tolist()

    numeric_cols = [
        "comment", "retweet", "like", "see_count"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0  # Eksik sütun varsa doldur

    print("🧠 Generating BERT embeddings...")
    X_text = get_bert_embeddings(texts, tokenizer, bert_model, device=device)
    X_num = scaler.transform(df[numeric_cols].fillna(0))
    X_all = np.hstack([X_text, X_num])
except Exception as e:
    print(f"⚠️ Feature preparation error: {e}")
    exit(0)

# ===================================================
# 5️⃣ Tahmin üretimi
# ===================================================
try:
    print("🤖 Predicting market direction...")
    df["pred_label"] = model.predict(X_all)
    df["pred_proba"] = model.predict_proba(X_all)[:, 1]
    df["prediction"] = df["pred_label"].map({1: "📈 YÜKSELİŞ", 0: "📉 DÜŞÜŞ"})
    print(f"✅ Predictions completed for {len(df)} records.")
except Exception as e:
    print(f"⚠️ Prediction error: {e}")
    exit(0)

# ===================================================
# 6️⃣ Sonuçları kaydet ve Hugging Face’e yükle
# ===================================================
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
output_latest = "/tmp/predictions_latest.csv"
output_time = f"/tmp/predictions_{timestamp}.csv"
df.to_csv(output_latest, index=False)
df.to_csv(output_time, index=False)
print(f"💾 Predictions saved locally to {output_latest}")

try:
    print("🚀 Uploading predictions to Hugging Face...")
    upload_file(
        path_or_fileobj=output_latest,
        path_in_repo="data/predictions_latest.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Upload latest predictions ({timestamp})"
    )
    upload_file(
        path_or_fileobj=output_time,
        path_in_repo=f"data/predictions_{timestamp}.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Upload timestamped predictions ({timestamp})"
    )
    print("✅ Predictions uploaded to HF successfully.")
except Exception as e:
    print(f"⚠️ Upload failed: {e}")

print("🎯 Prediction workflow completed (tolerant, non-blocking).")
