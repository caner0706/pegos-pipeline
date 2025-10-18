# =====================================================
# Pegos Dataset Builder (Stable + Binance Fallback + UTF-8 Safe)
# =====================================================
import os
import time
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from huggingface_hub import hf_hub_download, upload_file

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

TODAY = datetime.utcnow().strftime("%Y-%m-%d")
print(f"📂 Günlük klasör: {TODAY}")

# === 1️⃣ Mevcut final dosyası (varsa) ===
existing_df = pd.DataFrame()
try:
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"data/{TODAY}/pegos_final_dataset.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    existing_df = pd.read_csv(path)
    print(f"🔁 Mevcut veri bulundu: {len(existing_df)} satır")
except Exception:
    print("ℹ️ Mevcut final dataset yok, yeni oluşturulacak.")

# === 2️⃣ Yeni tweet verisini al (latest.csv öncelikli) ===
new_df = pd.DataFrame()
for name in [f"data/{TODAY}/latest.csv", f"data/{TODAY}/blockchain_tweets_{TODAY}.csv"]:
    try:
        path = hf_hub_download(repo_id=HF_DATASET_REPO, filename=name, repo_type="dataset", token=HF_TOKEN)
        new_df = pd.read_csv(path, encoding="utf-8")
        print(f"✅ Veri bulundu: {name}")
        break
    except Exception:
        continue

if new_df.empty:
    print("⚠️ Yeni tweet verisi boş veya bulunamadı.")
    new_df = pd.DataFrame(columns=existing_df.columns if not existing_df.empty else [])

# === 3️⃣ Normalize ===
for df in [existing_df, new_df]:
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["day"] = df["time"].dt.strftime("%Y-%m-%d")

combined = pd.concat([existing_df, new_df], ignore_index=True)
combined.drop_duplicates(subset=["tweet", "time"], inplace=True)
combined["day"] = combined["day"].astype(str)

# === 4️⃣ BTC fiyatlarını al (CoinGecko + Binance fallback) ===
def get_btc(day_str: str):
    """Günlük BTC fiyatı: önce CoinGecko, sonra Binance fallback"""
    try:
        day = datetime.strptime(day_str, "%Y-%m-%d").date()
        start_ts = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())

        # 🟢 1. CoinGecko API
        cg_url = (
            f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
            f"?vs_currency=usd&from={start_ts}&to={end_ts}"
        )
        r = requests.get(cg_url, timeout=15)
        if r.status_code == 200:
            data = r.json().get("prices", [])
            if data:
                data.sort(key=lambda x: x[0])
                open_p, close_p = data[0][1], data[-1][1]
                return open_p, close_p

        # 🟡 2. Binance fallback
        start_ms = start_ts * 1000
        end_ms = end_ts * 1000
        binance_url = (
            f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"
            f"&startTime={start_ms}&endTime={end_ms}"
        )
        res = requests.get(binance_url, timeout=10)
        if res.status_code == 200 and len(res.json()) > 0:
            o, c = float(res.json()[0][1]), float(res.json()[0][4])
            return o, c

        print(f"⚠️ BTC fiyatı bulunamadı ({day_str})")
        return None, None

    except Exception as e:
        print(f"⚠️ get_btc hata ({day_str}): {e}")
        return None, None


# === 5️⃣ Günlük fiyat verilerini uygula ===
unique_days = sorted(combined["day"].dropna().unique())
btc_info = {}

for day in unique_days:
    op, cl = get_btc(day)
    if op and cl:
        diff = cl - op
        direction = int(diff > 0)
    else:
        diff, direction = None, None
    btc_info[day] = {"open": op, "close": cl, "diff": diff, "direction": direction}
    time.sleep(0.2)

for col in ["open", "close", "diff", "direction"]:
    combined[col] = combined["day"].map(lambda d: btc_info.get(d, {}).get(col))

# === 6️⃣ Kaydet ve HF'e yükle ===
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out = f"/tmp/{TODAY}/pegos_final_dataset.csv"
combined.to_csv(out, index=False, encoding="utf-8")
print(f"💾 Kaydedildi: {out} ({len(combined)} satır)")

try:
    upload_file(
        path_or_fileobj=out,
        path_in_repo=f"data/{TODAY}/pegos_final_dataset.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Append dataset for {TODAY} (with BTC fallback)"
    )
    print("🚀 Dataset Hugging Face’e başarıyla yüklendi.")
except Exception as e:
    print(f"⚠️ Upload hatası: {e}")
