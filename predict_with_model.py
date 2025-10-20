# =====================================================
# Pegos Prediction (Full Detail – All Columns)
# =====================================================
import os
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, upload_file

# === Ortam değişkenleri ===
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

print("🤖 Pegos Prediction (tüm sütunlu detaylı çıktı) başlatıldı")

# === 1️⃣ Yeni batch verisini indir (latest.csv) ===
p = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename="data/cleaned.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)
df = pd.read_csv(p, encoding="utf-8-sig")
print(f"✅ Yeni batch veri yüklendi ({len(df)} satır)")

if df.empty:
    print("⚠️ Yeni veri yok, çıkılıyor.")
    exit()

# === 2️⃣ Model dosyaları ===
clf = joblib.load("pegos_lightgbm.pkl")
reg = joblib.load("pegos_regressor.pkl")
scaler = joblib.load("scaler.pkl")

# === 3️⃣ BERT modeli ===
tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
bert = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
device = "cuda" if torch.cuda.is_available() else "cpu"
bert.to(device).eval()

# === 4️⃣ Sayısal kolonları düzenle ===
for c in ["comment", "retweet", "like", "see_count"]:
    if c not in df.columns:
        df[c] = 0
df[["comment", "retweet", "like", "see_count"]] = df[["comment", "retweet", "like", "see_count"]].fillna(0)
X_num = scaler.transform(df[["comment", "retweet", "like", "see_count"]])

# === 5️⃣ Metin embedding işlemi ===
def embed(texts, bs=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = [str(t) for t in texts[i:i+bs]]
            tks = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = bert(**tks).last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(out)
    return np.vstack(embs)

X_text = embed(df["tweet"].tolist())
X = np.hstack([X_text, X_num])

# === 6️⃣ Model tahminleri ===
df["pred_label"] = clf.predict(X)                     # 1 = yükseliş, 0 = düşüş
df["pred_proba"] = clf.predict_proba(X)[:, 1]         # güven olasılığı
df["pred_diff"] = reg.predict(X)                      # fiyat farkı (oransal)
df["AI_Model_Tahmini (%)"] = (df["pred_diff"] * 100).round(2)

# === 7️⃣ Yön ve güven sütunları ===
df["AI_Model_Yonu"] = np.where(
    df["pred_diff"] > 0,
    "📈 Artış Bekleniyor",
    np.where(df["pred_diff"] < 0, "📉 Düşüş Bekleniyor", "⚖️ Değişim Yok")
)
df["Tahmin"] = df["pred_label"].map({1: "📈 YÜKSELİŞ", 0: "📉 DÜŞÜŞ"})
df["Güven (%)"] = (df["pred_proba"] * 100).round(1)

# === 8️⃣ İşlem günü ===
df["prediction_day"] = datetime.utcnow().strftime("%Y-%m-%d")

# === 9️⃣ Sütun sıralamasını düzenle ===
ordered_cols = [
    "tweet", "comment", "retweet", "like", "see_count",
    "pred_label", "pred_proba", "pred_diff",
    "AI_Model_Tahmini (%)", "AI_Model_Yonu",
    "Tahmin", "Güven (%)", "prediction_day"
]
df = df[ordered_cols]

# === 🔟 Kaydet & Yükle ===
os.makedirs("/tmp/data", exist_ok=True)
out_path = "/tmp/data/predict.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo="data/predict.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"🚀 predict.csv Hugging Face’e yüklendi ({len(df)} satır)")
print(f"📅 prediction_day: {df['prediction_day'].iloc[0]}")
