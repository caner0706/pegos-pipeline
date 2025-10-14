# build_dataset_hf.py
import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

api = HfApi(token=HF_TOKEN)

# 1️⃣ Hugging Face'ten CSV'leri indir
print("📥 HF dataset indiriliyor...")
files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
csv_files = [f for f in files if f.endswith(".csv") and "blockchain_tweets_" in f]

if not csv_files:
    raise RuntimeError("❌ HF üzerinde blockchain CSV dosyası bulunamadı.")

local_paths = []
for f in csv_files:
    path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=f, repo_type="dataset", token=HF_TOKEN)
    local_paths.append(path)
    print(f"✅ {f} indirildi.")

# 2️⃣ CSV'leri birleştir
dfs = [pd.read_csv(p) for p in local_paths]
merged = pd.concat(dfs, ignore_index=True)
merged.drop_duplicates(subset=["tweet", "time"], inplace=True)
merged["time"] = pd.to_datetime(merged["time"], errors="coerce", utc=True)
merged["day"] = merged["time"].dt.date
print(f"✅ {len(merged)} tweet birleştirildi.")

# 3️⃣ Günleri al
unique_days = merged["day"].dropna().drop_duplicates().sort_values().tolist()

# 4️⃣ BTC fiyatlarını CoinGecko'dan al
def get_btc_prices(day):
    base = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    start = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end = int(datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc).timestamp())
    url = f"{base}?vs_currency=usd&from={start}&to={end}"
    r = requests.get(url)
    if r.status_code != 200:
        return None, None
    data = r.json().get("prices", [])
    if not data:
        return None, None
    data.sort(key=lambda x: x[0])
    return data[0][1], data[-1][1]

btc_rows = []
for day in unique_days:
    op, cl = get_btc_prices(day)
    btc_rows.append({"day": day, "open": op, "close": cl})
    time.sleep(1)

btc_df = pd.DataFrame(btc_rows)
btc_df["diff"] = btc_df["close"] - btc_df["open"]
btc_df["direction"] = (btc_df["diff"] > 0).astype(int)
print(f"💰 BTC verisi {len(btc_df)} gün için alındı.")

# 5️⃣ Tweetlerle birleştir
merged["day"] = pd.to_datetime(merged["day"])
btc_df["day"] = pd.to_datetime(btc_df["day"])
final_df = merged.merge(btc_df, on="day", how="left")

# 6️⃣ HF'ye yükle
output_path = "/tmp/pegos_final_dataset.csv"
final_df.to_csv(output_path, index=False)
print(f"💾 Kaydedildi: {output_path} ({len(final_df)} satır)")

basename = f"merged_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo=f"data/{basename}",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo="data/latest_merged.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
)
print("🚀 HF'ye yükleme tamamlandı.")
