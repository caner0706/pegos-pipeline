# =====================================================
# Pegos Dataset Builder (Today-only, Stable Schema, BTC Fallback)
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

# 1) Bugünkü final dosyası (varsa)
existing_df = pd.DataFrame()
try:
    p = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"data/{TODAY}/pegos_final_dataset.csv",
        repo_type="dataset",
        token=HF_TOKEN
    )
    existing_df = pd.read_csv(p, encoding="utf-8")
    print(f"🔁 Mevcut veri bulundu: {len(existing_df)} satır")
except Exception:
    print("ℹ️ Mevcut günlük dataset yok, yeni oluşturulacak.")

# 2) Yeni tweet CSV (önce latest, yoksa arşiv)
new_df = pd.DataFrame()
for name in [f"data/{TODAY}/latest.csv", f"data/{TODAY}/blockchain_tweets_{TODAY}.csv"]:
    try:
        p = hf_hub_download(repo_id=HF_DATASET_REPO, filename=name, repo_type="dataset", token=HF_TOKEN)
        new_df = pd.read_csv(p, encoding="utf-8")
        print(f"✅ Veri bulundu: {name}")
        break
    except Exception:
        continue

if new_df.empty:
    print("⚠️ Yeni tweet verisi bulunamadı / boş.")
    new_df = pd.DataFrame(columns=["tweet","comment","retweet","like","see_count","time"])

# 3) Normalize — time parse et, AMA tüm satırları bugünün dosyasında bugüne yaz
if "time" in new_df.columns:
    new_df["time"] = pd.to_datetime(new_df["time"], errors="coerce", utc=True)
else:
    new_df["time"] = pd.NaT

# ŞEMAYI SABİTLE (eğitim şemasına uygun + target boş)
for c in ["tweet","comment","retweet","like","see_count","time"]:
    if c not in new_df.columns:
        new_df[c] = pd.NA

# Eski veri içinden sadece bugüne aitleri tut (başka gün kalmasın)
if not existing_df.empty:
    if "time" in existing_df.columns:
        existing_df["time"] = pd.to_datetime(existing_df["time"], errors="coerce", utc=True)
    # Eğitim şemasına oturt
    for c in ["tweet","comment","retweet","like","see_count","time",
              "Açılış Fiyatı (USD)","Kapanış Fiyatı (USD)","Fark (USD)","target"]:
        if c not in existing_df.columns:
            existing_df[c] = pd.NA

# Tüm yeni satırlar için "bugünün klasörü = bugünün günü" kuralı
# (tweet eski tarihli de olsa bugünün datası sayıyoruz)
# -> BTC de bugünün açılış/kapanışı olacak
new_df["_processing_day"] = TODAY

# 4) BTC fiyatı — CoinGecko ➜ Binance fallback ➜ önceki kapanış
def get_btc_ohlc_for_day(day_str: str, prev_close=None):
    try:
        day = datetime.strptime(day_str, "%Y-%m-%d").date()
        start_ts = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        end_ts   = int(datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())

        # 4.1 CoinGecko
        cg = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={start_ts}&to={end_ts}"
        r = requests.get(cg, timeout=15)
        if r.status_code == 200:
            prices = r.json().get("prices", [])
            if prices:
                prices.sort(key=lambda x: x[0])
                o, c = prices[0][1], prices[-1][1]
                return o, c

        # 4.2 Binance fallback
        b = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start_ts*1000}&endTime={end_ts*1000}"
        rr = requests.get(b, timeout=10)
        if rr.status_code == 200 and len(rr.json()) > 0:
            o, c = float(rr.json()[0][1]), float(rr.json()[0][4])
            return o, c

        print(f"⚠️ BTC fiyatı alınamadı ({day_str})")
        if prev_close is not None:
            # open=close=prev_close ile doldur
            return prev_close, prev_close
        return None, None
    except Exception as e:
        print(f"⚠️ BTC hata ({day_str}): {e}")
        if prev_close is not None:
            return prev_close, prev_close
        return None, None

# Bugün için OHLC çek (tek gün)
prev_close_known = None
o, c = get_btc_ohlc_for_day(TODAY, prev_close=prev_close_known)
open_usd, close_usd = o, c
diff_usd = (close_usd - open_usd) if (open_usd is not None and close_usd is not None) else None

# 5) Eğitim şemasına uygun kolonları kur & değerle
def to_training_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["tweet"]  = df["tweet"].astype(str)
    out["comment"] = pd.to_numeric(df["comment"], errors="coerce").fillna(0).astype(int)
    out["retweet"] = pd.to_numeric(df["retweet"], errors="coerce").fillna(0).astype(int)
    out["like"]    = pd.to_numeric(df["like"], errors="coerce").fillna(0).astype(int)
    out["see_count"] = pd.to_numeric(df["see_count"], errors="coerce").fillna(0).astype(int)
    out["time"]   = pd.to_datetime(df["time"], errors="coerce", utc=True)

    # BTC kolonlarını sabit adlarla doldur
    out["Açılış Fiyatı (USD)"]  = open_usd
    out["Kapanış Fiyatı (USD)"] = close_usd
    out["Fark (USD)"]           = diff_usd

    # target bilinmiyor -> boş
    out["target"] = pd.NA
    return out

existing_t = to_training_schema(existing_df) if not existing_df.empty else pd.DataFrame(columns=[
    "tweet","comment","retweet","like","see_count","time",
    "Açılış Fiyatı (USD)","Kapanış Fiyatı (USD)","Fark (USD)","target"
])
new_t = to_training_schema(new_df)

# Sadece bugünün verisini barındır (eski günler bu dosyada tutulmasın)
# Not: existing tarafında da olsa, bugüne ait olmayanları at
def is_today(ts):
    try:
        return (pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d") == TODAY)
    except Exception:
        return True  # zaman yoksa bugüne say

existing_t = existing_t[existing_t["time"].apply(is_today)] if not existing_t.empty else existing_t

combined = pd.concat([existing_t, new_t], ignore_index=True)
combined.drop_duplicates(subset=["tweet","time"], inplace=True)

# 6) Kaydet & Yükle
os.makedirs(f"/tmp/{TODAY}", exist_ok=True)
out_path = f"/tmp/{TODAY}/pegos_final_dataset.csv"
combined.to_csv(out_path, index=False, encoding="utf-8")
print(f"💾 Kaydedildi: {out_path} ({len(combined)} satır)")

upload_file(
    path_or_fileobj=out_path,
    path_in_repo=f"data/{TODAY}/pegos_final_dataset.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message=f"Append merged dataset for {TODAY} (today-only, stable schema)"
)
print("🚀 Dataset Hugging Face’e başarıyla yüklendi.")
