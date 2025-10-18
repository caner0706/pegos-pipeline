# =====================================================
# Pegos Dataset Builder (Stable + No _x/_y + UTF-8 Safe)
# =====================================================
import os
import time
import pandas as pd
import requests
from datetime import datetime, timezone
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

# === 4️⃣ BTC fiyatı ekle (sadece map, merge yok) ===
def get_btc(day_str):
    try:
        d = datetime.strptime(day_str, "%Y-%m-%d").date()
        start = int(datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end = int(datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc).timestamp())
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={start}&to={end}"
        r = requests.get(url, timeout=15)
        data = r.json().get("prices", [])
        if not data:
            return None, None
        data.sort(key=lambda x: x[0])
        return data[0][1], data[-1][1]
    except Exception:
        return None, None

unique_days = sorted(combined["day"].dropna().unique())
btc_info = {}
for day in unique_days:
    op, cl = get_btc(day)
    btc_info[day] = {"open": op, "close": cl, "diff": (cl or 0) - (op or 0) if op and cl else None,
                     "direction": int((cl or 0) > (op or 0)) if op and cl else None}
    time.sleep(0.2)

for c in ["open", "close", "diff", "direction"]:
    combined[c] = combined["day"].map(lambda d: btc_info.get(d, {}).get(c))

# === 5️⃣ Kaydet ===
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out = f"/tmp/{TODAY}/pegos_final_dataset.csv"
combined.to_csv(out, index=False, encoding="utf-8")
print(f"💾 Kaydedildi: {out} ({len(combined)} satır)")

upload_file(
    path_or_fileobj=out,
    path_in_repo=f"data/{TODAY}/pegos_final_dataset.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN
)
print("🚀 Dataset HF'e yüklendi.")
