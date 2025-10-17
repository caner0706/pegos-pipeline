# =====================================================
# Pegos Daily Hybrid Prediction Module 🚀
# =====================================================
import os
import joblib
import torch
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, upload_file, HfApi

# ------------------------------------------
# Ortam değişkenleri
# ------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "Caner7/pegos-stream")
if not HF_TOKEN or not HF_DATASET_REPO:
    raise RuntimeError("❌ HF_TOKEN veya HF_DATASET_REPO eksik!")

api = HfApi(token=HF_TOKEN)

print("🤖 Pegos hybrid prediction pipeline başlatıldı...")

# ===================================================
# 1️⃣ Günlük klasörü bul (en güncel gün)
# ===================================================
print("📂 Günlük klasörler listeleniyor...")
files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
daily_folders = sorted(
    list({f.split("/")[1] for f in files if f.startswith("data/") and len(f.split("/")) > 2})
)
if not daily_folders:
    raise RuntimeError("❌ Günlük klasör bulunamadı!")
latest_day = daily_folders[-1]
print(f"📅 En güncel veri klasörü: {latest_day}")

# ===================================================
# 2️⃣ Model, scaler ve tokenizer yükle
# ===================================================
try:
    print("📦 Modeller yükleniyor...")
    clf_model = joblib.load("pegos_lightgbm.pkl")
    reg_model = joblib.load("pegos_regressor.pkl")
    scaler = joblib.load("scaler.pkl")
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model.to(device)
    bert_model.eval()
    print("✅ Model yükleme tamamlandı.")
except Exception as e:
    raise RuntimeError(f"❌ Model yüklenemedi: {e}")

# ===================================================
# 3️⃣ Günlük cleaned.csv dosyasını indir
# ===================================================
try:
    clean_file = f"data/{latest_day}/cleaned.csv"
    print(f"📥 HF'den indiriliyor: {clean_file}")
    local_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=clean_file,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    df = pd.read_csv(local_path)
    print(f"✅ {len(df)} satır yüklendi.")
except Exception as e:
    raise RuntimeError(f"❌ Veri indirilemedi: {e}")

if df.empty:
    raise RuntimeError("⚠️ Veri boş geldi, tahmin yapılmadı.")

# ===================================================
# 4️⃣ BERT embedding fonksiyonu
# ===================================================
def get_bert_embeddings(texts, tokenizer, model, batch_size=16, device="cpu"):
    model.to(device)
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**tokens)
            cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls_embeds)
    return np.vstack(all_embeds)

# ===================================================
# 5️⃣ Özellikleri hazırla
# ===================================================
try:
    df = df.dropna(subset=["tweet"])
    texts = df["tweet"].astype(str).tolist()

    expected_cols = ["comment", "retweet", "like", "see_count"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    X_num = scaler.transform(df[expected_cols].fillna(0))

    print("🧠 BERT embedding'leri üretiliyor...")
    X_text = get_bert_embeddings(texts, tokenizer, bert_model, device=device)
    X_all = np.hstack([X_text, X_num])
except Exception as e:
    raise RuntimeError(f"⚠️ Özellik hazırlama hatası: {e}")

# ===================================================
# 6️⃣ Tahmin üretimi
# ===================================================
try:
    print("📈 Tahminler üretiliyor...")
    df["pred_label"] = clf_model.predict(X_all)
    df["pred_proba"] = clf_model.predict_proba(X_all)[:, 1]
    df["pred_diff"] = reg_model.predict(X_all)
    df["prediction"] = df["pred_label"].map({1: "📈 YÜKSELİŞ", 0: "📉 DÜŞÜŞ"})

    # Güncel BTC fiyatı
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
        current_price = resp.json()["bitcoin"]["usd"]
    except Exception:
        current_price = np.nan

    df["current_price"] = current_price
    df["predicted_price"] = current_price + df["pred_diff"]

    print(f"✅ Tahmin tamamlandı: {len(df)} kayıt.")
except Exception as e:
    raise RuntimeError(f"⚠️ Tahmin hatası: {e}")

# ===================================================
# 7️⃣ Dosyaları kaydet
# ===================================================
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
output_dir = f"/tmp/data/{latest_day}"
os.makedirs(output_dir, exist_ok=True)

output_latest = f"{output_dir}/predictions_latest.csv"
output_time = f"{output_dir}/predictions_{timestamp}.csv"

df.to_csv(output_latest, index=False)
df.to_csv(output_time, index=False)
print(f"💾 Kaydedildi: {output_latest}")

# ===================================================
# 8️⃣ Hugging Face'e yükle
# ===================================================
try:
    print("🚀 Hugging Face'e yükleniyor...")

    upload_file(
        path_or_fileobj=output_latest,
        path_in_repo=f"data/{latest_day}/predictions_latest.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Upload latest predictions for {latest_day}"
    )

    upload_file(
        path_or_fileobj=output_time,
        path_in_repo=f"data/{latest_day}/predictions_{timestamp}.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Upload timestamped predictions for {latest_day}"
    )

    print(f"✅ Tahmin dosyaları Hugging Face'e yüklendi: data/{latest_day}/")
except Exception as e:
    raise RuntimeError(f"⚠️ Upload hatası: {e}")