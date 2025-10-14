# =====================================
# Pegos Prediction Module (4-hour loop)
# =====================================
import os
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi

# 🔐 Ortam değişkenleri (GitHub Secrets)
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

# ========================
# 1️⃣ Model ve bileşenleri yükle
# ========================
print("📦 Loading model and tokenizer...")
model = joblib.load("pegos_lightgbm.pkl")
scaler = joblib.load("scaler.pkl")

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
bert_model.eval()

# ========================
# 2️⃣ Veri indir (HF)
# ========================
print("📥 Downloading latest cleaned data from HF...")
api = HfApi(token=HF_TOKEN)
repo = HF_DATASET_REPO
hf_file = "data/latest_cleaned.csv"

local_file = "/tmp/latest_cleaned.csv"
api.hf_hub_download(repo_id=repo, repo_type="dataset", filename=hf_file, local_dir="/tmp", token=HF_TOKEN)

df = pd.read_csv(local_file)
print(f"✅ Loaded {len(df)} rows, {df.shape[1]} columns.")

# ========================
# 3️⃣ Embedding + Tahmin
# ========================
def get_bert_embeddings(texts, tokenizer, model, batch_size=16, device="cpu"):
    model.to(device)
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**tokens)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls_embeds)
    return np.vstack(all_embeds)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Eksik tweetleri at
df = df.dropna(subset=["tweet"])
texts = df["tweet"].astype(str).tolist()

print("🧠 Generating embeddings...")
X_text = get_bert_embeddings(texts, tokenizer, bert_model, device=device)

numeric_cols = ["comment", "retweet", "like", "see_count"]
X_num = scaler.transform(df[numeric_cols].fillna(0))

X_all = np.hstack([X_text, X_num])

# Tahmin
print("🤖 Predicting market direction...")
df["pred_label"] = model.predict(X_all)
df["pred_proba"] = model.predict_proba(X_all)[:, 1]

df["prediction"] = df["pred_label"].map({1: "📈 YÜKSELİŞ", 0: "📉 DÜŞÜŞ"})

# ========================
# 4️⃣ Sonuçları kaydet ve yükle
# ========================
output_file = "/tmp/predictions_latest.csv"
df.to_csv(output_file, index=False)
print(f"💾 Predictions saved locally to {output_file}")

print("🚀 Uploading predictions to HF...")
api.upload_file(
    path_or_fileobj=output_file,
    path_in_repo="data/predictions_latest.csv",
    repo_id=repo,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("✅ Predictions uploaded to HF successfully.")
