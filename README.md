# Mandi Intelligence

Where and when should a Gujarat farmer sell Onion, Tomato or Potato? This project forecasts mandi prices for the next 7 days and picks the plan with the highest **net profit** after transport, storage and spoilage.

See [FINAL_DOCUMENTATION.md](FINAL_DOCUMENTATION.md) for how it works, and [reports/MODEL_REPORT.md](reports/MODEL_REPORT.md) for held-out accuracy.

## Layout

```
api/ml_api.py              FastAPI app (/recommend, /health, /mandis, /locations, /model_report, ...)
ml_arbitrage/
  data_loader.py           raw AGMARKNET CSV → clean daily price panel
  features.py              leakage-free multi-horizon features
  forecaster.py            per-crop XGBoost, train / evaluate / save / forecast
  arbitrage_engine.py      transport, storage and spoilage economics → recommendation
  distance_calculator.py   road distances between Gujarat towns
  service.py               loads data + models, caches forecasts (used by API and CLI)
scripts/
  fetch_agmarknet.py       download live Gujarat prices from data.gov.in
  train.py                 retrain models and write reports/MODEL_REPORT.md
  recommend.py             command-line recommendation
models/                    trained models (*.ubj) + metadata/metrics (*.meta.json)
dataset/commodity_price.csv  bundled training data (2021–2024)
tests/                     pytest suite
frontend/                  Nuxt UI (calls POST /recommend on port 8000)
```

## Quick start

```bash
pip install -r requirements.txt
uvicorn api.ml_api:app --port 8000        # run from this directory
cd frontend && npm install && npm run dev # UI
```

```bash
curl -X POST localhost:8000/recommend -H "Content-Type: application/json" \
     -d '{"crop": "Onion", "quantity": 2000, "farmer_location": "Gandhinagar"}'
```

## Using live AGMARKNET data

```bash
export DATA_GOV_IN_API_KEY=<your free key from data.gov.in>   # optional
python scripts/fetch_agmarknet.py --start 2023-01-01
MANDI_DATASET=dataset/agmarknet_gujarat.csv python scripts/train.py
MANDI_DATASET=dataset/agmarknet_gujarat.csv uvicorn api.ml_api:app --port 8000
```

The public sample key returns only 10 records per request at about 0.5 requests/s, so a full history takes hours. A personal key may allow larger pages and a higher `--rate`. The download is resumable (it keeps a checkpoint in `dataset/raw/`) and fetches the newest days first. The API retrains automatically at startup if the saved models were trained on a different dataset.
