# 🧠 Pegos AI – Hybrid Blockchain Market Intelligence System

![Build](https://img.shields.io/github/actions/workflow/status/Caner7/pegos-pipeline/schedule.yml?label=Pipeline&logo=github)
![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue?logo=huggingface)
![API](https://img.shields.io/badge/HuggingFace-Spaces-green?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)

> 🚀 **Real-time AI-driven blockchain sentiment & price prediction system**  
> Predicts market direction and delta every 4 hours using NLP, regression & automation.

---

## 📘 Overview

**Pegos AI** is a **hybrid FinTech & AI intelligence system** that continuously collects and analyzes blockchain-related tweets, applying **BERT embeddings** and **LightGBM** models to forecast **market sentiment**, **trend direction**, and **expected price changes**.  

This repository includes:  
- 🧩 Automated GitHub Actions pipeline (data → clean → predict → upload)  
- ☁️ Hugging Face Dataset & API integrations  
- 🧠 AI inference with BERT & LightGBM  
- 🕒 Fully autonomous 4-hour cycle  

---

## 🧭 Architecture

```text
GitHub Actions (every 4 hours)
     │
     ├── data.ipynb              # Twitter scraping & cleaning
     ├── clean_dataset_hf.py     # Data cleaning & merge
     ├── predict_with_model.py   # BERT + LightGBM predictions
     │
     ▼
Hugging Face Datasets (Caner7/pegos-stream)
     │
     └── data/
          ├── 2025-10-18/
          │   ├── cleaned.csv
          │   ├── predictions_latest.csv
          │   └── predictions_20251018_XXXX.csv
          └── 2025-10-19/
              ├── cleaned.csv
              ├── predictions_latest.csv
              └── predictions_20251019_XXXX.csv

Hugging Face Spaces (Pegos API)
     └── FastAPI app serving JSON endpoints
```

---

## 🔗 Live Resources

| Component | Platform | Link |
|-----------|----------|------|
| 📊 Dataset | Hugging Face Datasets | [Caner7/pegos-stream](https://huggingface.co/datasets/Caner7/pegos-stream) |
| 🧩 API | Hugging Face Spaces | [Pegos Data API](https://caner7-pegos-data.hf.space) |
| 🧱 Dashboard | Hugging Face Spaces| [Caner7/pegos-dashboard](https://huggingface.co/spaces/Caner7/pegos_dashboard) |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10 |
| Framework | FastAPI, Uvicorn |
| AI Models | Hugging Face Transformers (BERT), LightGBM |
| Data Tools | Pandas, NumPy, Scikit-learn |
| Automation | GitHub Actions (cron: every 4h) |
| Deployment | Docker + Hugging Face Spaces |
| Storage | Hugging Face Datasets |
| Security | OAuth2 HF Token Authentication |

---

## 🧠 AI Workflow

1. 🧹 **Data Scraping & Cleaning** – Tweets collected & normalized
2. 🧠 **Embedding Generation** – BERT (dbmdz/bert-base-turkish-cased) → sentence vectors
3. ⚙️ **Feature Fusion** – Merge embeddings + numeric features (likes, retweets, views)
4. 📈 **Prediction Layer** – LightGBM Classifier + Regressor
5. 💾 **Storage & Sync** – Save under `/data/<date>/` and upload to HF
6. 🌐 **API Delivery** – Served live via FastAPI on Hugging Face Spaces

---

## ⏰ Schedule (UTC)

| Trigger | Turkey Time | Description |
|---------|-------------|-------------|
| 06:00 | 09:00 | Morning Batch |
| 10:00 | 13:00 | Midday Update |
| 14:00 | 17:00 | Afternoon Refresh |
| 18:00 | 21:00 | Evening Update |

**Cron:** `"0 */4 * * *"`

---

## 📁 Repository Structure

```
pegos-pipeline/
│
├── .github/workflows/
│   └── schedule.yml
│
├── data.ipynb
├── clean_dataset_hf.py
├── predict_with_model.py
│
├── app/
│   ├── main.py
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py
│   │   ├── cache.py
│   │   ├── hf_client.py
│   ├── routers/
│   │   ├── base.py
│   │   ├── latest.py
│   │   ├── history.py
│   │   ├── signal.py
│   │   ├── predictions.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🌐 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | System status check |
| `/latest` | Retrieve last predictions |
| `/signal` | Aggregated sentiment (UP / DOWN / NEUTRAL) |
| `/history` | Historical predictions with pagination |
| `/predictions` | Full AI prediction table |

---

## 📊 Example Response

```json
{
  "count": 150,
  "updated_at": "2025-10-18T09:00:00Z",
  "items": [
    {
      "tweet": "BTC yükseliş sinyali veriyor!",
      "pred_label": 1,
      "pred_proba": 0.93,
      "pred_diff": 324.5,
      "Tahmin": "📈 YÜKSELİŞ",
      "processing_day": "2025-10-18"
    }
  ]
}
```

---

## 📋 Example CSV Output

| tweet | pred_label | pred_proba | pred_diff | Tahmin | processing_day |
|-------|------------|------------|-----------|---------|----------------|
| "BTC rekor tazeliyor!" | 1 | 0.94 | +327.4 | 📈 YÜKSELİŞ | 2025-10-18 |
| "ETH satış baskısı altında" | 0 | 0.88 | −152.7 | 📉 DÜŞÜŞ | 2025-10-18 |

---

## 🧩 Key Highlights

✅ Automated daily data processing  
✅ Real-time AI predictions every 4 hours  
✅ Hybrid text + numeric modeling  
✅ Dynamic day-based folder structure  
✅ Historical data archive (30+ days)  
✅ Hugging Face Dataset & Spaces integration  
✅ Lightweight caching for scalability  

---

## 💡 Developer Notes

- 🧩 The pipeline is 100% autonomous (no manual triggers)
- ⚙️ Each script (data → clean → predict → upload) runs sequentially
- ☁️ HF dataset refresh = API auto-update
- 🚀 Cached responses for faster API load
- 🔁 Models retrain independently

---

## ⚡ Local Development

```bash
git clone https://github.com/Caner7/pegos-pipeline.git
cd pegos-pipeline
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open → http://localhost:8000/docs

---

## 🧾 License

**MIT License**

Developed by **Caner Giden**  
📧 canergiden.dev@gmail.com  
💼 [LinkedIn](linkedin.com/in/caner-giden)  
🧠 [Hugging Face](https://huggingface.co/Caner7)

---

## 🌟 Future Enhancements

- 🔮 Reinforcement learning for adaptive trends
- 📊 Real-time market feeds (CoinGecko / Binance)
- 🧮 Federated learning (privacy-safe)
- 🌐 Multilingual NLP (EN + TR)
- 📈 Interactive dashboards (Streamlit / Plotly)

---

## 🏁 Summary

**Pegos AI** merges automation, AI, and blockchain data in a single decentralized ecosystem.  
From data collection to model prediction and live API delivery — everything runs autonomously.

🧩 **Data → Model → Prediction → API → Insight — 100% Autonomous.**

---

✨ **Pegos AI** - The future of blockchain intelligence, today! 🚀
