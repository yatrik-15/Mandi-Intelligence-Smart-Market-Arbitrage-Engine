"""
Mandi Intelligence — price forecasting and net-profit arbitrage for Gujarat mandis.

    data_loader         raw AGMARKNET CSV → clean daily panel
    features            leakage-free multi-horizon features
    forecaster          per-crop pooled XGBoost (1–14 day horizons)
    arbitrage_engine    transport / storage / spoilage economics → recommendation
    distance_calculator road distances between Gujarat towns
    service             wires the above together for the API and CLI
"""

__version__ = "4.0.0"
