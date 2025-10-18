# =====================================================
# Pegos Dataset Builder (Append + Daily Folder)
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

# 🔹 Günlük klasör ismi
TODAY = datetime.utcnow().strftime("%Y-%m-%d")
print(f"📂 Günlük klasör: {TODAY}")

# =====================================================
# 1️⃣ HF üzerinde mevcut günlük dosya varsa indir
# =====================================================
merged_path_hf = f"data/{TODAY}/pegos_final_dataset.csv"
local_existing = None
existing_df = pd.DataFrame()

try:
    print(f"📥 HF üzerinde {merged_path_hf} aranıyor...")
    local_existing = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=merged_path_hf,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    existing_df = pd.read_csv(local_existing)
    print(f"🔁 Mevcut veri bulundu: {len(existing_df)} satır")
except Exception:
    print("ℹ️ Mevcut veri bulunamadı, yeni dosya oluşturulacak.")

# =====================================================
# 2️⃣ Yeni tweet CSV’lerini indir
# =====================================================
print("📥 Yeni tweet dosyaları indiriliyor...")
files = api.list_repo_files(repo_id=HF_DATASET_REPO, repo_type="dataset")
tweet_files = [f for f in files if f.endswith(".csv") and "blockchain_tweets_" in f]

if not tweet_files:
    raise RuntimeError("❌ HF üzerinde tweet CSV bulunamadı!")

dfs = []
for f in tweet_files:
    p = hf_hub_download(
        repo_id=HF_DATASET_REPO, filename=f, repo_type="dataset", token=HF_TOKEN
    )
    dfs.append(pd.read_csv(p))
print(f"✅ {len(dfs)} dosya indirildi")

new_df = pd.concat(dfs, ignore_index=True)
new_df["time"] = pd.to_datetime(new_df["time"], errors="coerce", utc=True)
new_df["day"] = new_df["time"].dt.date
print(f"🆕 Yeni tweet sayısı: {len(new_df)}")

# =====================================================
# 3️⃣ Eski + Yeni birleştir
# =====================================================
combined = pd.concat([existing_df, new_df], ignore_index=True)
combined.drop_duplicates(subset=["tweet", "time"], inplace=True)
print(f"📊 Birleştirilmiş toplam: {len(combined)} satır")

# =====================================================
# 4️⃣ BTC fiyatlarını al
# =====================================================
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

unique_days = combined["day"].dropna().drop_duplicates().sort_values().tolist()
btc_rows = []
for day in unique_days:
    op, cl = get_btc_prices(day)
    btc_rows.append({"day": day, "open": op, "close": cl})
    time.sleep(0.5)

btc_df = pd.DataFrame(btc_rows)
btc_df["diff"] = btc_df["close"] - btc_df["open"]
btc_df["direction"] = (btc_df["diff"] > 0).astype(int)

combined["day"] = pd.to_datetime(combined["day"])
btc_df["day"] = pd.to_datetime(btc_df["day"])
final_df = combined.merge(btc_df, on="day", how="left")

# =====================================================
# 5️⃣ Kaydet ve Hugging Face’e yükle
# =====================================================
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
