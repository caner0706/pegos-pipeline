# =====================================================
# Pegos Hybrid Prediction (Training-schema output + Predictions)
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

# Modeller
clf = joblib.load("pegos_lightgbm.pkl")
reg = joblib.load("pegos_regressor.pkl")
scaler = joblib.load("scaler.pkl")

tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
device = "cuda" if torch.cuda.is_available() else "cpu"
bert.to(device).eval()

# Veri (bugün)
p = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=f"data/{TODAY}/cleaned.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(p, encoding="utf-8")
print(f"✅ Veri yüklendi ({len(df)} satır)")

if df.empty:
    print("⚠️ Boş veri — tahmin atlandı.")
else:
    # sayısal
    for c in ["comment","retweet","like","see_count"]:
        if c not in df.columns:
            df[c] = 0
    X_num = scaler.transform(df[["comment","retweet","like","see_count"]].fillna(0))

    # text -> bert
    def embed(texts, bs=16):
        embs=[]
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = [str(t) for t in texts[i:i+bs]]
                tks = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                out = bert(**tks).last_hidden_state[:,0,:].cpu().numpy()
                embs.append(out)
        return np.vstack(embs) if embs else np.empty((0, bert.config.hidden_size))

    X_text = embed(df["tweet"].tolist())
    X = np.hstack([X_text, X_num]) if len(df) else np.empty((0, X_num.shape[1] + X_text.shape[1]))

    # tahmin
    df["pred_label"] = clf.predict(X) if len(df) else []
    df["pred_proba"] = clf.predict_proba(X)[:,1] if len(df) else []
    df["pred_diff"]  = reg.predict(X) if len(df) else []
    df["Tahmin"]     = df["pred_label"].map({1:"📈 YÜKSELİŞ", 0:"📉 DÜŞÜŞ"})

# ÇIKIŞ ŞEMASI: eğitim şeması + tahmin kolonu
# Eğitim şeması kolonları (adlar birebir):
base_cols = [
    "tweet","comment","retweet","like","see_count","time",
    "Açılış Fiyatı (USD)","Kapanış Fiyatı (USD)","Fark (USD)","target"
]
for c in base_cols:
    if c not in df.columns:
        df[c] = pd.NA

out_df = df[base_cols + ["pred_label","pred_proba","pred_diff","Tahmin"]].copy()

# Kaydet & Yükle
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
latest = f"/tmp/{TODAY}/predictions_latest.csv"
tsf    = f"/tmp/{TODAY}/predictions_{ts}.csv"
out_df.to_csv(latest, index=False, encoding="utf-8")
out_df.to_csv(tsf,    index=False, encoding="utf-8")

for src, dst in [
    (latest, f"data/{TODAY}/predictions_latest.csv"),
    (tsf,    f"data/{TODAY}/predictions_{ts}.csv"),
]:
    upload_file(
        path_or_fileobj=src,
        path_in_repo=dst,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

print("🚀 Tahmin dosyaları Hugging Face’e yüklendi.")
