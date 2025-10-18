# =====================================================
# Pegos Hybrid Prediction (Daily Folder, Cleaned Input)
# =====================================================
import os
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, upload_file

# Ortam değişkenleri
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

print(f"🤖 Pegos Prediction Pipeline Başladı ({TODAY})")

# =========================
# 🔹 MODEL & TOKENIZER YÜKLE
# =========================
clf_model = joblib.load("pegos_lightgbm.pkl")
reg_model = joblib.load("pegos_regressor.pkl")
scaler = joblib.load("scaler.pkl")

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model.to(device)

# =========================
# 🔹 VERİYİ HUGGING FACE'TEN İNDİR
# =========================
clean_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=f"data/{TODAY}/cleaned.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(clean_path)
print(f"✅ Veri yüklendi ({len(df)} satır)")

# =========================
# 🔹 SAYISAL & METİNSEL VERİ HAZIRLA
# =========================
texts = df["tweet"].astype(str).tolist()
num_cols = ["comment", "retweet", "like", "see_count"]

for col in num_cols:
    if col not in df.columns:
        df[col] = 0

X_num = scaler.transform(df[num_cols].fillna(0))

def get_bert_embeddings(texts):
    bert_model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = bert_model(**tokens)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls_embeds)
    return np.vstack(all_embeds)

X_text = get_bert_embeddings(texts)
X_all = np.hstack([X_text, X_num])

# =========================
# 🔹 TAHMİNLERİ HESAPLA
# =========================
df["pred_label"] = clf_model.predict(X_all)
df["pred_proba"] = clf_model.predict_proba(X_all)[:, 1]
df["pred_diff"] = reg_model.predict(X_all)
df["Tahmin"] = df["pred_label"].map({1: "📈 YÜKSELİŞ", 0: "📉 DÜŞÜŞ"})

# =========================
# 🔹 DOSYA YOLLARI & KAYDETME
# =========================
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

latest_path = f"/tmp/{TODAY}/predictions_latest.csv"
ts_path = f"/tmp/{TODAY}/predictions_{timestamp}.csv"

df.to_csv(latest_path, index=False)
df.to_csv(ts_path, index=False)

# =========================
# 🔹 SADECE GÜNLÜK KLASÖRE YÜKLE
# =========================
for p, dest in [
    (latest_path, f"data/{TODAY}/predictions_latest.csv"),
    (ts_path, f"data/{TODAY}/predictions_{timestamp}.csv"),
]:
    upload_file(
        path_or_fileobj=p,
        path_in_repo=dest,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

print("🚀 Tahmin dosyaları Hugging Face’e yüklendi.")
print("✅ Hybrid model predictions completed and uploaded to HF (pred_label, pred_diff, predicted_price).")
