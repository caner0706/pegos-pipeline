# =====================================================
# Pegos Dataset Builder (Append + Daily Folder + Fallback + Stable)
# =====================================================
import os
import time
import pandas as pd
import requests
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download, upload_file

# === ENV AYARLARI ===
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
api = HfApi(token=HF_TOKEN)

TODAY = datetime.utcnow().strftime("%Y-%m-%d")
print(f"📂 Günlük klasör: {TODAY}")

# =====================================================
# 1️⃣ Mevcut günün final dosyası varsa indir
# =====================================================
merged_hf = f"data/{TODAY}/pegos_final_dataset.csv"
existing_df = pd.DataFrame()
try:
    print(f"📥 HF üzerinde {merged_hf} aranıyor...")
    local_existing = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=merged_hf,
        repo_type="dataset",
        token=HF_TOKEN
    )
    existing_df = pd.read_csv(local_existing)
    print(f"🔁 Mevcut veri bulundu: {len(existing_df)} satır")
except Exception:
    print("ℹ️ Mevcut günlük dataset yok, yeni oluşturulacak.")

# =====================================================
# 2️⃣ Yeni tweet dosyasını bul (önce latest.csv, yoksa arşiv)
# =====================================================
new_df = pd.DataFrame()
candidate_files = [
    f"data/{TODAY}/latest.csv",
    f"data/{TODAY}/blockchain_tweets_{TODAY}.csv"
]

found_path = None
for file in candidate_files:
    try:
        print(f"📥 Kontrol ediliyor: {file}")
        found_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=file,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"✅ Dosya bulundu: {file}")
        break
    except Exception:
        continue

if not found_path:
    raise RuntimeError("❌ HF üzerinde tweet CSV bulunamadı (ne latest ne arşiv).")

new_df = pd.read_csv(found_path)
if new_df.empty:
    print("⚠️ Yeni tweet verisi boş geldi, pipeline devam ediyor...")

# normalize & gün kolonu
new_df["time"] = pd.to_datetime(new_df.get("time"), errors="coerce", utc=True)
new_df["day"] = new_df["time"].dt.strftime("%Y-%m-%d")

# =====================================================
# 3️⃣ Append + unique
# =====================================================
combined = pd.concat([existing_df, new_df], ignore_index=True)
combined.drop_duplicates(subset=["tweet", "time"], inplace=True)
print(f"📊 Birleştirilmiş toplam: {len(combined)} satır")

# =====================================================
# 4️⃣ BTC fiyatlarını al (her gün için open/close)
# =====================================================
def get_btc_prices(day_str: str):
    try:
        day = datetime.strptime(day_str, "%Y-%m-%d").date()
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
    except Exception:
        return None, None

# tür karışıklığını önlemek için day -> str
combined["day"] = combined["day"].astype(str)
unique_days = sorted(combined["day"].dropna().unique())

btc_rows = []
for day in unique_days:
    op, cl = get_btc_prices(day)
    btc_rows.append({"day": day, "open": op, "close": cl})
    time.sleep(0.3)

btc_df = pd.DataFrame(btc_rows)
btc_df["diff"] = btc_df["close"] - btc_df["open"]
btc_df["direction"] = (btc_df["diff"] > 0).astype(int)

final_df = combined.merge(btc_df, on="day", how="left")

# =====================================================
# 5️⃣ Kaydet & Hugging Face'e yükle
# =====================================================
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out_path = f"/tmp/{TODAY}/pegos_final_dataset.csv"
final_df.to_csv(out_path, index=False)
print(f"💾 Kaydedildi: {out_path} ({len(final_df)} satır)")

try:
    upload_file(
        path_or_fileobj=out_path,
        path_in_repo=f"data/{TODAY}/pegos_final_dataset.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=f"Append merged dataset for {TODAY}"
    )
    print("🚀 Günlük dataset Hugging Face’e başarıyla yüklendi.")
except Exception as e:
    print(f"⚠️ Upload sırasında hata oluştu: {e}")
