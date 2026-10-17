# Model Report

Dataset: `commodity_price.csv` — 30,681 mandi-days, 7 mandis, 2021-01-01 → 2024-12-31. Training took 47s.

All numbers are on the held-out **test** period (the last 20% of dates), which the model never saw during training or early stopping. *Skill* is the MAE reduction versus persistence ("the price stays where it is today").

## Onion

Test origins 2024-03-14 → 2024-12-30 (27,979 forecasts), 1146 trees, 80% interval coverage 80%.

| Horizon / method | MAE | RMSE | MAPE | R² | Skill vs persistence |
| :-- | --: | --: | --: | --: | --: |
| **1-day — XGBoost** | ₹0.83 | ₹1.05 | 3.74% | 0.965 | +30% |
| 1-day — 7-day average | ₹0.92 | ₹1.15 | 4.15% | 0.957 | +22% |
| 1-day — persistence | ₹1.18 | ₹1.48 | 5.29% | 0.930 | +0% |
| **3-day — XGBoost** | ₹0.83 | ₹1.05 | 3.74% | 0.964 | +30% |
| 3-day — 7-day average | ₹0.98 | ₹1.22 | 4.43% | 0.951 | +17% |
| 3-day — persistence | ₹1.18 | ₹1.47 | 5.35% | 0.930 | +0% |
| **7-day — XGBoost** | ₹0.83 | ₹1.05 | 3.77% | 0.964 | +34% |
| 7-day — 7-day average | ₹1.16 | ₹1.43 | 5.27% | 0.932 | +7% |
| 7-day — persistence | ₹1.26 | ₹1.57 | 5.69% | 0.918 | +0% |
| **14-day — XGBoost** | ₹0.82 | ₹1.04 | 3.77% | 0.963 | +46% |
| 14-day — 7-day average | ₹1.58 | ₹1.90 | 7.17% | 0.875 | -4% |
| 14-day — persistence | ₹1.52 | ₹1.89 | 6.96% | 0.877 | +0% |

## Tomato

Test origins 2024-03-14 → 2024-12-30 (27,979 forecasts), 948 trees, 80% interval coverage 74%.

| Horizon / method | MAE | RMSE | MAPE | R² | Skill vs persistence |
| :-- | --: | --: | --: | --: | --: |
| **1-day — XGBoost** | ₹0.99 | ₹1.46 | 7.24% | 0.966 | +28% |
| 1-day — 7-day average | ₹1.10 | ₹1.62 | 7.98% | 0.958 | +20% |
| 1-day — persistence | ₹1.38 | ₹2.05 | 9.94% | 0.933 | +0% |
| **3-day — XGBoost** | ₹0.99 | ₹1.46 | 7.16% | 0.966 | +26% |
| 3-day — 7-day average | ₹1.16 | ₹1.70 | 8.49% | 0.954 | +13% |
| 3-day — persistence | ₹1.34 | ₹2.01 | 9.70% | 0.936 | +0% |
| **7-day — XGBoost** | ₹1.00 | ₹1.47 | 7.23% | 0.966 | +32% |
| 7-day — 7-day average | ₹1.37 | ₹1.98 | 10.12% | 0.939 | +7% |
| 7-day — persistence | ₹1.47 | ₹2.18 | 10.64% | 0.926 | +0% |
| **14-day — XGBoost** | ₹0.99 | ₹1.46 | 7.06% | 0.967 | +45% |
| 14-day — 7-day average | ₹1.83 | ₹2.57 | 13.82% | 0.899 | -1% |
| 14-day — persistence | ₹1.81 | ₹2.63 | 13.29% | 0.894 | +0% |

## Potato

Test origins 2024-03-14 → 2024-12-30 (27,979 forecasts), 276 trees, 80% interval coverage 89%.

| Horizon / method | MAE | RMSE | MAPE | R² | Skill vs persistence |
| :-- | --: | --: | --: | --: | --: |
| **1-day — XGBoost** | ₹0.64 | ₹0.80 | 2.96% | 0.933 | +28% |
| 1-day — 7-day average | ₹0.70 | ₹0.87 | 3.24% | 0.920 | +21% |
| 1-day — persistence | ₹0.89 | ₹1.11 | 4.12% | 0.870 | +0% |
| **3-day — XGBoost** | ₹0.64 | ₹0.80 | 2.95% | 0.932 | +28% |
| 3-day — 7-day average | ₹0.74 | ₹0.92 | 3.43% | 0.910 | +16% |
| 3-day — persistence | ₹0.88 | ₹1.10 | 4.10% | 0.870 | +0% |
| **7-day — XGBoost** | ₹0.63 | ₹0.80 | 2.93% | 0.930 | +32% |
| 7-day — 7-day average | ₹0.85 | ₹1.06 | 3.99% | 0.876 | +9% |
| 7-day — persistence | ₹0.94 | ₹1.16 | 4.34% | 0.852 | +0% |
| **14-day — XGBoost** | ₹0.63 | ₹0.80 | 2.89% | 0.926 | +45% |
| 14-day — 7-day average | ₹1.17 | ₹1.40 | 5.45% | 0.772 | -2% |
| 14-day — persistence | ₹1.14 | ₹1.40 | 5.30% | 0.770 | +0% |
