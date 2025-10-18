# =====================================================
# Pegos Hybrid Prediction (Safe for empty datasets)
# =====================================================
import os
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
TODAY = datetime.utcnow().strftime("%Y-%m-%d")

print(f"🤖 Pegos Prediction Pipeline Başladı ({TODAY})")

# === Modelleri yükle ===
clf_model = joblib.load("pegos_lightgbm.pkl")
reg_model = joblib.load("pegos_regressor.pkl")
scaler = joblib.load("scaler.pkl")

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model.to(device)
bert_model.eval()

# === Günlük veri yükle ===
try:
    clean_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"data/{TODAY}/cleaned.csv",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    df = pd.read_csv(clean_path)
except Exception as e:
    raise RuntimeError(f"❌ Günlük veri indirilemedi: {e}")

print(f"✅ Veri yüklendi ({len(df)} satır)")

# === Boş veri kontrolü ===
if df.empty:
    print("⚠️ Boş veri tespit edildi. Tahmin adımı atlanıyor...")
    df = pd.DataFrame(columns=[
        "keyword", "tweet", "time", "comment", "retweet", "like", "see_count",
        "pred_label", "pred_proba", "pred_diff", "Tahmin"
    ])

else:
    # === Sayısal özellikler ===
    num_cols = ["comment", "retweet", "like", "see_count"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
    X_num = scaler.transform(df[num_cols].fillna(0))

    # === BERT gömme (embedding) ===
    def get_bert_embeddings(texts, bs=16):
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                toks = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)
                out = bert_model(**toks).last_hidden_state[:, 0, :].cpu().numpy()
                embs.append(out)
        return np.vstack(embs)

    texts = df["tweet"].astype(str).tolist()
    X_text = get_bert_embeddings(texts)
    X_all = np.hstack([X_text, X_num])

    # === Tahminler ===
    df["pred_label"] = clf_model.predict(X_all)
    df["pred_proba"] = clf_model.predict_proba(X_all)[:, 1]
    df["pred_diff"] = reg_model.predict(X_all)
    df["Tahmin"] = df["pred_label"].map({1: "📈 YÜKSELİŞ", 0: "📉 DÜŞÜŞ"})

# === Kaydet & Yükle ===
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
latest_path = f"/tmp/{TODAY}/predictions_latest.csv"
ts_path = f"/tmp/{TODAY}/predictions_{timestamp}.csv"

df.to_csv(latest_path, index=False)
df.to_csv(ts_path, index=False)

for src, dst in [
    (latest_path, f"data/{TODAY}/predictions_latest.csv"),
    (ts_path, f"data/{TODAY}/predictions_{timestamp}.csv"),
]:
    upload_file(
        path_or_fileobj=src,
        path_in_repo=dst,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

print("🚀 Tahmin dosyaları Hugging Face’e başarıyla yüklendi.")
