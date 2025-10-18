# =====================================================
# Pegos Dataset Builder (Append + Daily Folder + Fallback)
# =====================================================
import os
import time
import pandas as pd
import requests
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
api = HfApi(token=HF_TOKEN)

TODAY = datetime.utcnow().strftime("%Y-%m-%d")
print(f"📂 Günlük klasör: {TODAY}")

# 1) Mevcut günün final dosyası varsa indir
merged_hf = f"data/{TODAY}/pegos_final_dataset.csv"
existing_df = pd.DataFrame()
try:
    print(f"📥 HF üzerinde {merged_hf} aranıyor...")
    local_existing = hf_hub_download(
        repo_id=HF_DATASET_REPO, filename=merged_hf,
        repo_type="dataset", token=HF_TOKEN
    )
    existing_df = pd.read_csv(local_existing)
    print(f"🔁 Mevcut veri: {len(existing_df)} satır")
except Exception:
    print("ℹ️ Mevcut günlük dataset yok, yeni oluşturulacak.")

# 2) Yeni tweet dosyası: ÖNCE günlük latest.csv, yoksa o güne ait arşiv
new_df = pd.DataFrame()
daily_latest = f"data/{TODAY}/latest.csv"
try_first = None
try:
    print(f"📥 Günün latest.csv indiriliyor: {daily_latest}")
    p = hf_hub_download(
        repo_id=HF_DATASET_REPO, filename=daily_latest,
        repo_type="dataset", token=HF_TOKEN
    )
    try_first = p
except Exception:
    print("ℹ️ Günlük latest.csv bulunamadı, arşiv dosyasına bakılıyor...")
    # Arşiv adı sabit formatta
    alt = f"data/{TODAY}/blockchain_tweets_{TODAY}.csv"
    try:
        p = hf_hub_download(
            repo_id=HF_DATASET_REPO, filename=alt,
            repo_type="dataset", token=HF_TOKEN
        )
        try_first = p
        print(f"📥 Arşiv bulundu: {alt}")
    except Exception:
        pass

if try_first is None:
    raise RuntimeError("❌ HF üzerinde tweet CSV yok (günlük latest veya arşiv).")

new_df = pd.read_csv(try_first)
if new_df.empty:
    print("⚠️ Yeni tweet verisi boş geldi (yine de pipeline devam).")

# normalize & gün kolonu
new_df["time"] = pd.to_datetime(new_df.get("time"), errors="coerce", utc=True)
new_df["day"] = new_df["time"].dt.date

# 3) Append + unique
combined = pd.concat([existing_df, new_df], ignore_index=True)
combined.drop_duplicates(subset=["tweet", "time"], inplace=True)
print(f"📊 Birleştirilmiş toplam: {len(combined)} satır")

# 4) BTC fiyatları
def get_btc_prices(day):
    base = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    start = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end = int(datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc).timestamp())
    url = f"{base}?vs_currency=usd&from={start}&to={end}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return None, None
    data = r.json().get("prices", [])
    if not data:
        return None, None
    data.sort(key=lambda x: x[0])
    return data[0][1], data[-1][1]

unique_days = combined["day"].dropna().drop_duplicates().sort_values().tolist()
btc_rows = []
for day in unique_days:
    op, cl = get_btc_prices(day)
    btc_rows.append({"day": day, "open": op, "close": cl})
    time.sleep(0.25)

btc_df = pd.DataFrame(btc_rows)
btc_df["diff"] = btc_df["close"] - btc_df["open"]
btc_df["direction"] = (btc_df["diff"] > 0).astype(int)

combined["day"] = pd.to_datetime(combined["day"])
btc_df["day"] = pd.to_datetime(btc_df["day"])
final_df = combined.merge(btc_df, on="day", how="left")

# 5) Save & Upload (yalnızca gün klasörü)
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out_path = f"/tmp/{TODAY}/pegos_final_dataset.csv"
final_df.to_csv(out_path, index=False)
print(f"💾 Kaydedildi: {out_path} ({len(final_df)} satır)")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo=f"data/{TODAY}/pegos_final_dataset.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message=f"Append merged dataset for {TODAY}",
)
print("🚀 Günlük dataset Hugging Face’e başarıyla yüklendi.")
