# DATASET_GUIDE.md — Mandi Intelligence v3.0
## Recommended Datasets for Training & Testing

---

### 🥇 TIER 1 — Production (Best Accuracy)

#### 1. AGMARKNET (Government of India — Primary Source)
| Field        | Detail |
|---|---|
| **URL**      | https://agmarknet.gov.in/SearchCmmMkt.aspx |
| **Auth**     | Free, no login required |
| **Format**   | CSV download per State/Commodity/Date range |
| **Coverage** | 3,000+ mandis, 300+ commodities, daily prices |
| **Fields**   | `Arrival_Date, State, District, Market, Commodity, Variety, Min_Price, Max_Price, Modal_Price` |
| **Best for** | Production system — authoritative government data |

**Download steps:**
1. Select State → Gujarat
2. Select Commodity → Onion / Tomato / Potato / Wheat / Cotton
3. Set date range (use max available — 3 years+)
4. Export CSV → place at `dataset/commodity_price.csv`

---

#### 2. eNAM API (National Agriculture Market — Real-Time)
| Field        | Detail |
|---|---|
| **URL**      | https://enam.gov.in/web/dashboard/trade-data |
| **Auth**     | API key (free registration) |
| **Format**   | JSON REST API |
| **Coverage** | Live prices, ~15 min delay during market hours |
| **Best for** | Real-time recommendations; integrate via `requests` |

```python
import requests
# Example API call (replace YOUR_API_KEY)
resp = requests.get(
    "https://enam.gov.in/web/Ajax_ctrl/trade_data_list",
    params={"start_date": "01-01-2024", "end_date": "31-12-2024",
            "commodity": "Onion", "state": "Gujarat"},
    headers={"api-key": "YOUR_API_KEY"}
)
df = pd.DataFrame(resp.json()["data"])
```

---

### 🥈 TIER 2 — Development & Hackathons

#### 3. Kaggle — Daily Wholesale Commodity Prices India Mandis
| Field        | Detail |
|---|---|
| **URL**      | https://www.kaggle.com/datasets/ramkrijal/agriculture-vegetables-fruits-time-series-prices |
| **Auth**     | Free Kaggle account |
| **Format**   | `commodity_price.csv` (~2,733 rows, multiple states) |
| **Coverage** | 2021–2025, Gujarat subset ~200–400 rows |
| **Fields**   | Same as AGMARKNET format |
| **Best for** | This project's demo — already supported by `data_loader.py` |

**Download:** `kaggle datasets download ramkrijal/agriculture-vegetables-fruits-time-series-prices`

---

#### 4. data.gov.in — Open Government Data
| Field        | Detail |
|---|---|
| **URL**      | https://data.gov.in/catalog/wholesale-price-index-horticulture-commodities |
| **Auth**     | Free |
| **Format**   | CSV / XLSX |
| **Coverage** | Monthly WPI for horticulture commodities, all-India |
| **Best for** | Long-term trend baseline for Prophet Stage 1A |

---

#### 5. Gujarat APMC Portal
| Field        | Detail |
|---|---|
| **URL**      | https://agri.gujarat.gov.in/ |
| **Auth**     | Free |
| **Format**   | PDF / HTML tables (scraping required) |
| **Coverage** | 20+ Gujarat APMCs, daily arrivals + prices |
| **Best for** | Ground truth for specific Gujarat mandis |

---

### 🥉 TIER 3 — Supplementary

#### 6. IMD Weather Data (for weather covariate in TFT)
| Field | Detail |
|---|---|
| **URL** | https://mausam.imd.gov.in/imd_latest/contents/all-india-telecast-bulletin.php |
| **Format** | JSON API / CSV |
| **Use** | Weather condition as a categorical feature → improves perishability prediction |

---

### 📁 Expected File Layout

```
mandi_intelligence/
└── dataset/
    ├── commodity_price.csv          ← Primary (Kaggle / AGMARKNET)
    ├── enam_live.json               ← Optional real-time (eNAM API)
    └── weather_gujarat.csv          ← Optional (IMD)
```

---

### ✅ Minimum Viable Dataset (for testing without internet)

The system auto-generates a statistically realistic 2-year synthetic panel
when no real data is found. Call it explicitly:

```python
from ml_arbitrage.data_loader import MandiDataLoader
loader = MandiDataLoader()
df = loader.generate_rich_synthetic_data(
    n_days=730,
    mandis=["Ahmedabad", "Rajkot", "Surat", "Mehsana", "Anand"],
    crops=["Onion", "Tomato", "Potato"],
)
```

**Synthetic data characteristics:**
- 2 years × 8 mandis × 5 crops ≈ 27,700 rows
- Additive trend + Rabi/Kharif seasonality + Gaussian noise
- ~3 price shocks per crop-mandi-year (simulates drought / flood events)
- ~5 % missing dates (tests imputation robustness)

---

### 🔧 Minimum Records Required per Mandi-Crop Pair

| Records | Outcome |
|---|---|
| < 30 | Skipped (model not trained; spatial arbitrage only) |
| 30–100 | Model trained; moderate accuracy (MAPE ~10–20 %) |
| 100–365 | Good accuracy (MAPE ~5–10 %) |
| 365+ | Best accuracy (MAPE < 5 %); full temporal arbitrage |

---

### 📈 Expected Accuracy by Dataset

| Dataset | Records/pair | Expected MAPE | R² |
|---|---|---|---|
| Synthetic (2yr) | ~670 | 4–8 % | 0.75–0.90 |
| Kaggle CSV (Gujarat) | 10–50 | 15–25 % | 0.30–0.60 |
| AGMARKNET (3yr) | 500–1000 | 3–7 % | 0.80–0.95 |
| eNAM API (live) | 1000+ | 2–5 % | 0.85–0.97 |
