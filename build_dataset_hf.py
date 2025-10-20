# =====================================================
# Pegos Dataset Builder (No-day folder, cumulative + daily dataset)
# =====================================================
import os
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

print("📂 Pegos dataset builder (günsüz, kümülatif + günlük) başlatıldı.")

# 1️⃣ Eski dataset (varsa)
try:
    p = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="data/pegos_final_dataset.csv",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    existing_df = pd.read_csv(p)
    print(f"🔁 Eski dataset bulundu: {len(existing_df)} satır")
except Exception:
    existing_df = pd.DataFrame()
    print("ℹ️ Eski dataset yok, sıfırdan başlıyor.")

# 2️⃣ Yeni tweet datası (latest)
try:
    p = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="data/latest.csv",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    new_df = pd.read_csv(p)
    print(f"✅ Yeni veri bulundu: {len(new_df)} satır")
except Exception:
    new_df = pd.DataFrame(columns=["tweet", "comment", "retweet", "like", "see_count", "time"])
    print("⚠️ Yeni tweet verisi bulunamadı / boş.")

# 3️⃣ BTC fiyatı (CoinGecko)
def get_btc_ohlc():
    try:
        day = datetime.utcnow().date()
        start_ts = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
        cg = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={start_ts}&to={end_ts}"
        r = requests.get(cg, timeout=15)
        if r.status_code == 200:
            prices = r.json().get("prices", [])
            if prices:
                o, c = prices[0][1], prices[-1][1]
                return o, c
    except Exception:
        pass
    return None, None

open_usd, close_usd = get_btc_ohlc()
diff_usd = (close_usd - open_usd) if open_usd and close_usd else None

# 4️⃣ Şemayı normalize et
def normalize(df):
    if df.empty:
        return df
    for c in ["comment", "retweet", "like", "see_count"]:
        if c not in df.columns:
            df[c] = 0
    df["Açılış Fiyatı (USD)"] = open_usd
    df["Kapanış Fiyatı (USD)"] = close_usd
    df["Fark (USD)"] = diff_usd
    df["target"] = pd.NA
    return df

new_df = normalize(new_df)
existing_df = normalize(existing_df)

# 5️⃣ Birleştir (eski + yeni → kümülatif dataset)
combined = pd.concat([existing_df, new_df], ignore_index=True)
combined.drop_duplicates(subset=["tweet", "time"], inplace=True)

# 6️⃣ Kaydet ve yükle (pegos_final_dataset.csv)
os.makedirs("/tmp/data", exist_ok=True)
out_path = "/tmp/data/pegos_final_dataset.csv"
combined.to_csv(out_path, index=False, encoding="utf-8")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo="data/pegos_final_dataset.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print("🚀 Güncel dataset Hugging Face'e yüklendi (pegos_final_dataset.csv).")

# 7️⃣ Yeni: Günlük ham veri dosyasını (daily_raw.csv) oluştur ve yükle
if not new_df.empty:
    daily_path = "/tmp/data/daily_raw.csv"
    new_df.to_csv(daily_path, index=False, encoding="utf-8")

    upload_file(
        path_or_fileobj=daily_path,
        path_in_repo="data/daily_raw.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("📤 daily_raw.csv (sadece son batch) Hugging Face’e yüklendi.")
else:
    print("⚠️ Yeni veri bulunamadığı için daily_raw.csv oluşturulmadı.")
