# =====================================================
# Pegos Prediction (only latest batch)
# =====================================================
import os
import joblib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

print("🤖 Pegos Prediction (yalnızca yeni veriler) başlatıldı")

# Yalnızca son batch verisi
p = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename="data/latest.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(p, encoding="utf-8")
print(f"✅ Yeni batch veri yüklendi ({len(df)} satır)")

if df.empty:
    print("⚠️ Yeni veri yok, çıkılıyor.")
    exit()

# Model dosyaları
clf = joblib.load("pegos_lightgbm.pkl")
reg = joblib.load("pegos_regressor.pkl")
scaler = joblib.load("scaler.pkl")

# BERT modeli
tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
device = "cuda" if torch.cuda.is_available() else "cpu"
bert.to(device).eval()

# Sayısal veriler
for c in ["comment","retweet","like","see_count"]:
    if c not in df.columns:
        df[c] = 0
X_num = scaler.transform(df[["comment","retweet","like","see_count"]].fillna(0))

# Metin embedding
def embed(texts, bs=16):
    embs=[]
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = [str(t) for t in texts[i:i+bs]]
            tks = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = bert(**tks).last_hidden_state[:,0,:].cpu().numpy()
            embs.append(out)
    return np.vstack(embs)

X_text = embed(df["tweet"].tolist())
X = np.hstack([X_text, X_num])

# Tahmin
df["pred_label"] = clf.predict(X)
df["pred_proba"] = clf.predict_proba(X)[:,1]
df["pred_diff"]  = reg.predict(X)
df["Tahmin"]     = df["pred_label"].map({1:"📈 YÜKSELİŞ", 0:"📉 DÜŞÜŞ"})

# Kaydet
os.makedirs("/tmp/data", exist_ok=True)
out_path = "/tmp/data/predict.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo="data/predict.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("🚀 predict.csv (sadece yeni veriler) Hugging Face’e yüklendi.")
