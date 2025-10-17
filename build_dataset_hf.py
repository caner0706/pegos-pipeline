# =====================================================
# Pegos Build Dataset (Daily Folder Compatible)
# =====================================================
import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, hf_hub_download

print("📂 Hugging Face'ten günlük klasörler listeleniyor...")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

api = HfApi(token=HF_TOKEN)

# 1️⃣ Günlük klasörü bul
base_dir = "/tmp/data"
if not os.path.exists(base_dir):
    raise RuntimeError("❌ Ana klasör /tmp/data bulunamadı!")

daily_folders = sorted(
    [os.path.join(base_dir, d) for d in os.listdir(base_dir)
     if os.path.isdir(os.path.join(base_dir, d))],
    reverse=True
)

if not daily_folders:
    # fallback olarak eski path kontrolü
    fallback = "/tmp/pegos_output.csv"
    if os.path.exists(fallback):
        print("⚠️ Günlük klasör bulunamadı, eski CSV kullanılıyor.")
        daily_folders = [os.path.dirname(fallback)]
    else:
        raise RuntimeError("❌ Günlük klasör bulunamadı!")

latest_folder = daily_folders[0]
csv_path = os.path.join(latest_folder, "pegos_output.csv")

if not os.path.exists(csv_path):
    raise RuntimeError(f"❌ CSV bulunamadı: {csv_path}")

print(f"✅ Güncel CSV bulundu: {csv_path}")

# 2️⃣ CSV oku
df = pd.read_csv(csv_path)
print(f"✅ {len(df)} tweet yüklendi. BTC verisiyle birleştiriliyor...")

# 3️⃣ BTC fiyatlarını CoinGecko'dan al
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

df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
df["day"] = df["time"].dt.date
unique_days = df["day"].dropna().drop_duplicates().sort_values().tolist()

btc_rows = []
for day in unique_days:
    op, cl = get_btc_prices(day)
    btc_rows.append({"day": day, "open": op, "close": cl})
    time.sleep(1)

btc_df = pd.DataFrame(btc_rows)
btc_df["diff"] = btc_df["close"] - btc_df["open"]
btc_df["direction"] = (btc_df["diff"] > 0).astype(int)

final_df = df.merge(btc_df, on="day", how="left")

# 4️⃣ HF'ye yükle
output_path = "/tmp/pegos_final_dataset.csv"
final_df.to_csv(output_path, index=False)
print(f"💾 Kaydedildi: {output_path} ({len(final_df)} satır)")

basename = f"merged_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo=f"data/{basename}",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo="data/latest_merged.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset"
)
print("🚀 HF'ye yükleme tamamlandı.")
