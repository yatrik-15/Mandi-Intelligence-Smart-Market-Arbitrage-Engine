"""
ML-Powered Mandi Arbitrage Engine v3.0
=======================================
Four-Stage Ensemble Pipeline: Prophet → TFT → ArbitrageMath → CatBoost

Stages:
  1A — Prophet  : Macro trend decomposition + gap imputation
  1B — TFT      : Multi-horizon price forecast (t+3, t+7)
  2  — Math     : Net Profit computation per logistics scenario
  3  — CatBoost : Optimal action classification
"""

__version__ = "3.0.0"
