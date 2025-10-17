# =====================================================
# Pegos Daily Dataset Builder (tweets + BTC merge)
# =====================================================
import os
import time
import pandas as pd
import requests
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download

# ------------------------------------------
# Ortam değişkenleri
# ------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
if not HF_TOKEN or not HF_DATASET_REPO:
    raise RuntimeError("❌ HF_TOKEN veya HF_DATASET_REPO tanımlı değil!")

api = HfApi(token=HF_TOKEN)

# ------------------------------------------
# Günlük veriyi bul (en yeni gün klasörü)
# ------------------------------------------
print("📂 Hugging Face'ten günlük klasörler listeleniyor...")
files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
daily_folders = sorted(
    list({f.split("/")[1] for f in files if f.startswith("data/") and len(f.split("/")) > 2})
)
if not daily_folders:
    raise RuntimeError("❌ Günlük klasör bulunamadı!")

latest_day = daily_folders[-1]
print(f"📅 En güncel veri klasörü: {latest_day}")

latest_file = f"data/{latest_day}/latest.csv"
print(f"📥 Günlük veri indiriliyor: {latest_file}")

# Hugging Face'ten indir
local_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    filename=latest_file,
    repo_type="dataset",
    token=HF_TOKEN,
)

df = pd.read_csv(local_path)
print(f"✅ {len(df)} tweet yüklendi.")

# ------------------------------------------
# BTC fiyatlarını CoinGecko’dan al
# ------------------------------------------
def get_btc_prices(day):
    base = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    start = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end = start + 86400 - 1
    url = f"{base}?vs_currency=usd&from={start}&to={end}"
    r = requests.get(url, headers={"User-Agent": "Pegos-Dataset-Builder/1.0"})
    if r.status_code != 200:
        print(f"⚠️ CoinGecko hatası: {r.status_code}")
        return None, None
    data = r.json().get("prices", [])
    if not data:
        return None, None
    data.sort(key=lambda x: x[0])
    return data[0][1], data[-1][1]

print(f"💰 BTC verisi alınıyor: {latest_day}")
open_price, close_price = get_btc_prices(latest_day)
btc_df = pd.DataFrame([{
    "day": latest_day,
    "open": open_price,
    "close": close_price,
    "diff": (close_price - open_price) if (open_price and close_price) else None,
    "direction": int((close_price or 0) > (open_price or 0))
}])
print("✅ BTC fiyatları alındı.")

# ------------------------------------------
# Tweetlerle birleştir
# ------------------------------------------
df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
df["day"] = df["time"].dt.date.astype(str)
btc_df["day"] = btc_df["day"].astype(str)

merged = df.merge(btc_df, on="day", how="left")
print(f"🔗 Birleştirme tamamlandı: {len(merged)} satır")

# ------------------------------------------
# Kaydet ve Hugging Face'e yükle
# ------------------------------------------
output_path = f"/tmp/merged_{latest_day}.csv"
merged.to_csv(output_path, index=False)
print(f"💾 Kaydedildi: {output_path}")

# HF yükleme
remote_path = f"data/{latest_day}/merged.csv"
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo=remote_path,
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
)
print(f"🚀 Yüklendi: {remote_path}")