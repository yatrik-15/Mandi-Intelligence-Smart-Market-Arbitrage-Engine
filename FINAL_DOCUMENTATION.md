# Mandi Intelligence: Final Project Documentation (v4)

## 1. Project Overview

**Mandi Intelligence** (part of the BeejRakshak ecosystem) is a decision-support engine that tells a Gujarat farmer **where and when to sell** Onion, Tomato or Potato for the highest **net profit**. It forecasts prices at every mandi for the next 7 days and then subtracts real-world costs:

- **Transport**: fuel (runtime-adjustable petrol price), driver wages, tolls, maintenance and loading/unloading. The vehicle is chosen by load (Mini Truck < 7.5 q, LCV < 25 q, Medium < 60 q, Heavy), with multiple trips when needed.
- **Storage**: ₹50 per quintal per day while waiting.
- **Spoilage**: exponential decay, `qty × e^(−λ·days)` (λ = 0.030 Onion, 0.055 Tomato, 0.020 Potato).

Net Profit ≠ Gross Earnings. The mandi with the highest price is often not the most profitable one once these costs are counted.

---

## 2. What changed in v4 (and why)

| Area | v3.0 | v4 |
| :-- | :-- | :-- |
| **Training features** | Rolling averages, EMA, std and momentum included *today's* price, which is the value being predicted. | Every feature uses only information available at the forecast origin (verified by a test). |
| **Forecast horizon** | Trained to predict the same day, then fed back into itself 7 times. | Direct forecasts 1–14 days ahead; one model per crop, pooled across mandis. |
| **Evaluation** | Early stopping and reported metrics used the same 20 % split. | Separate train / validation / test periods by date, with 14-day gaps; compared against naive baselines. |
| **Uncertainty** | Interval from the model's errors on data it was trained on (too narrow). | 80 % interval from held-out validation errors, per horizon. |
| **Profit maths** | Spoilage was subtracted twice; the profit range was 100× too narrow (quintals vs kg). | Spoilage counted once; range in rupees for the kilograms actually sold. |
| **Waiting decision** | Ranked purely on point forecasts. | Ranked on risk-adjusted profit; waiting is recommended only when the gain outweighs forecast uncertainty. |
| **Outlier handling** | Global per-crop IQR filter (would delete genuine price spikes). | Removes only isolated typos (> 2.5× the local median). |
| **API startup** | Retrained 21 models on every start. | Loads saved models (retrains only when the data changes). |
| **`/recommend`** | Re-ran all forecasts on every request. | Forecasts cached per crop; a request is pure arithmetic. |
| **Unknown locations** | Silently assumed 150 km to every mandi. | Rejected with HTTP 422 and the list of supported towns. |

---

## 3. Data Pipeline

**Source.** AGMARKNET daily modal prices (Directorate of Marketing & Inspection), Gujarat, Onion / Tomato / Potato.

- `dataset/commodity_price.csv` is the bundled training set: 7 mandis, 2021-01-01 → 2024-12-31. Every series has exactly one report every day, which is unusual for real mandi data, so treat this file as illustrative.
- `scripts/fetch_agmarknet.py` downloads the **live official data** from the data.gov.in API (*Variety-wise Daily Market Prices Data of Commodity*), which is updated daily. A test pull from mid-Sept to mid-Oct 2026 returned 34 active Gujarat mandis. The public key is slow (10 records/request), so set `DATA_GOV_IN_API_KEY` to a free personal key. Train on the result with `MANDI_DATASET=dataset/agmarknet_gujarat.csv`.

**Cleaning** (`ml_arbitrage/data_loader.py`): canonical mandi names (*"Ahmedabad(Chimanbhai Patal Market Vasana) APMC"* → *Ahmedabad*), median across varieties and sub-yards per day, ₹/quintal → ₹/kg, bounds ₹1–₹250/kg, and isolated-outlier removal.

**Features** (`ml_arbitrage/features.py`). The target is `log(p[t+h] / p[t])`. All inputs are trailing:
- Price level and deviation from 7/14/30/90/365-day averages
- Returns over 1/3/7/14/30 days, and volatility over 14 and 60 days
- Reporting density and days since the previous report (real mandis skip days)
- Premium over the Gujarat-wide median price and the state-wide trend
- Seasonality: day-of-year, and what prices did over the same window last year
- Horizon `h`

---

## 4. Model

One **XGBoost** regressor per crop (pseudo-Huber loss, hist trees), pooled across all mandis. Each mandi's own price level is factored out by predicting relative change, so the model learns from every mandi at once and still works for mandis that report irregularly. This was compared against separate per-mandi models and other loss functions on the validation period, and it was as good or better in every case. Models are stored in XGBoost's native `.ubj` format in `models/`.

---

## 5. Model Accuracy (honest, held-out)

Test period: 2024-03-14 → 2024-12-30. These dates were never used for training or early stopping. The figures are averages over the three crops; per-crop tables are in `reports/MODEL_REPORT.md`.

| Horizon | v4 MAE | v4 MAPE | v4 R² | Persistence MAE | v3.0 MAE (re-tested honestly) |
| :-- | --: | --: | --: | --: | --: |
| 1 day  | **₹0.82/kg** | 4.6 % | 0.955 | ₹1.15 | ₹1.03 |
| 3 days | **₹0.82/kg** | 4.6 % | 0.954 | ₹1.14 | ₹1.01 |
| 7 days | **₹0.82/kg** | 4.6 % | 0.953 | ₹1.22 | ₹1.14 |
| 14 days | **₹0.81/kg** | 4.6 % | 0.952 | ₹1.49 | — |

- **vs. v3.0:** 20 % lower error at 1 day and 28 % lower at 7 days, measured on the same test period.
- **vs. naive "price stays the same":** 28–34 % lower error at 1–7 days, and about 45 % lower at 14 days.
- **80 % intervals:** actual coverage was 80 % for Onion, 89 % for Potato and 74 % for Tomato. Tomato's test period was more volatile than its validation period, so its intervals are slightly too narrow.

> [!IMPORTANT]
> **About the old numbers.** v3.0 documented MAE ₹0.29/kg, MAPE 1.6 % and R² 0.993. Those came from leakage: the model's inputs included averages of the very price it was predicting. Re-tested the way the API actually used it (forecasting from past data only), v3.0 achieved ₹1.03/kg at 1 day and ₹1.14/kg at 7 days. The v4 figures above are measured the honest way.

The model's error stays flat from 1 to 14 days while the baselines get worse. That suggests the bundled dataset is a smooth trend plus random day-to-day noise. The model tracks the trend, and the remaining ~₹0.8/kg is noise that no model can predict. Expect different (probably larger) errors on live AGMARKNET data.

---

## 6. Performance

Measured with the same benchmark (FastAPI TestClient, 30 mixed-crop requests):

| | v3.0 | v4 |
| :-- | --: | --: |
| API startup | 10.8 s (retrains 21 models) | **1.4 s** (loads 3 saved models) |
| `/recommend` median latency | 624 ms | **6.6 ms** |
| Full retrain (`scripts/train.py`) | ~10 s, 21 models | ~47 s, 3 models incl. evaluation |

---

## 7. Running it

```bash
pip install -r requirements.txt
python scripts/train.py                  # retrain + write reports/MODEL_REPORT.md
uvicorn api.ml_api:app --port 8000       # API (frontend expects port 8000)
python scripts/recommend.py              # interactive CLI
python -m pytest tests -q                # 15 tests: leakage, profit maths, API contract
```

The `/recommend` request and response format is unchanged, so the Nuxt frontend works as before. `alternatives` now lists the best plan at each *other* mandi (no duplicates) and adds a `days_to_wait` field. There is also a new `GET /locations` endpoint that lists the supported towns.
