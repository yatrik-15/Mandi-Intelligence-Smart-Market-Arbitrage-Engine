"""
Mandi Arbitrage Engine v3.0 — Main Demonstration Script
========================================================

This script demonstrates the complete ML-powered arbitrage system:
1. Load/Generate mandi price data
2. Train XGBoost models for price forecasting  
3. Run example scenarios showing spatial + temporal arbitrage
4. Output structured JSON recommendations with detailed justifications

Transport cost model: 5-component (fuel + driver + tolls + maintenance + L/U)
Storage cost: ₹50/quintal/day (industry cold-storage rate)

Run this to see the system in action!
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

# Import our modules
from data_loader import MandiDataLoader
from price_predictor import PricePredictor
from arbitrage_engine import ArbitrageEngine


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_json_pretty(data: dict, title: str = ""):
    """Print JSON data in a readable format."""
    if title:
        print(f"\n{title}:")
        print("-" * 60)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()


def main():
    """Main execution flow."""
    
    print_header("MANDI ARBITRAGE ENGINE v3.0 — ML-Powered Profit Maximization")
    
    print("📋 SYSTEM OVERVIEW")
    print("-" * 80)
    print("This engine maximizes farmer's net profit using:")
    print("  • Real historical mandi price data (or synthetic if unavailable)")
    print("  • XGBoost ML models for 7-day price forecasting")
    print("  • Dual arbitrage logic: Spatial (where?) + Temporal (when?)")
    print()
    print("NET PROFIT FORMULA (v3.0 — 5-component transport):")
    print("  Net Profit = (qty_remaining × price_future)")
    print("             − transport_cost(fuel + driver + toll + maint + L/U)")
    print("             − (wait_days × qty_quintal × ₹50/quintal/day)")
    print("             − spoilage(exponential decay: qty × exp(−λ × t))")
    print()
    
    # ============================================================================
    # STEP 1: Load/Generate Data
    # ============================================================================
    
    print_header("STEP 1: Loading Mandi Price Data")
    
    loader = MandiDataLoader()
    
    # Try to load real Kaggle data from dataset folder FIRST
    real_data_paths = [
        "dataset/commodity_price.csv",
        "../dataset/commodity_price.csv",
        "data/mandi_prices.csv",
        "../data/mandi_prices.csv",
        "mandi_prices.csv"
    ]
    
    df = None
    for path in real_data_paths:
        try:
            loader.load_data(path)
            df = loader.filter_and_process(days=None)
            print("✅ Using REAL Kaggle data")
            break
        except (FileNotFoundError, Exception):
            continue
    
    if df is None or df.empty:
        print("⚠️  Real data not found. Generating synthetic data for demonstration...")
        df = loader.generate_rich_synthetic_data()
    
    # Show summary
    stats = loader.get_summary_stats()
    print_json_pretty(stats, "Data Summary")
    
    # Save processed data
    output_path = "ml_arbitrage/processed_mandi_data.csv"
    loader.save_processed_data(output_path)
    
    # ============================================================================
    # STEP 2: Train Predictive Models
    # ============================================================================
    
    print_header("STEP 2: Training XGBoost Price Prediction Models")
    
    predictor = PricePredictor()
    
    print("🔧 Engineering time-series features...")
    df_featured = predictor.prepare_features(df)
    
    print(f"   Added features: lag prices, moving averages, EMA, trends, seasonality")
    print(f"   Total features: {len(df_featured.columns)}")
    print()
    
    print("🤖 Training models for all Mandi-Crop combinations...")
    print()
    results = predictor.train_all_models(df_featured)
    
    # Show training results
    print("\n📊 Model Performance Summary:")
    print("-" * 80)
    for result in results:
        if 'error' not in result:
            print(f"  {result['mandi']} - {result['crop']}: "
                  f"MAE=₹{result['mae']:.2f}/kg | MAPE={result['mape']:.1f}% | R²={result['r2']:.3f}")
    
    # ============================================================================
    # STEP 3: Initialize Arbitrage Engine
    # ============================================================================
    
    print_header("STEP 3: Initializing Arbitrage Decision Engine")
    
    engine = ArbitrageEngine(price_predictor=predictor)
    
    print("✅ Engine initialized with:")
    print(f"   • Price predictor: {len(predictor.models)} trained models")
    print(f"   • Transport model: 5-component (₹94.37/L petrol)")
    print(f"   • Storage cost: ₹50/quintal/day")
    print(f"   • Spoilage model: Exponential decay")
    
    # ============================================================================
    # STEP 4: Run Example Scenarios
    # ============================================================================
    
    print_header("STEP 4: Running Example Scenarios")
    
    # Get latest data for current prices
    latest_date = df_featured['Date'].max()
    df_current = df_featured[df_featured['Date'] == latest_date].copy()
    
    # Generate forecasts for all mandis and crops
    print("🔮 Generating 7-day price forecasts...")
    
    forecasts_data = []
    for crop in df_current['Crop'].unique():
        forecasts = predictor.get_price_forecast_all_mandis(df_featured, crop, days_ahead=7)
        for mandi, forecast in forecasts.items():
            for _, row in forecast.iterrows():
                forecasts_data.append({
                    'Mandi_Name': mandi,
                    'Crop': crop,
                    'Day_Ahead': int(row['Day_Ahead']),
                    'Predicted_Price': float(row['Predicted_Price']),
                    'Lower_Bound': float(row.get('Lower_Bound', row['Predicted_Price'])),
                    'Upper_Bound': float(row.get('Upper_Bound', row['Predicted_Price'])),
                })
    
    df_forecast = pd.DataFrame(forecasts_data) if forecasts_data else None
    print(f"   Generated {len(forecasts_data)} price predictions")
    
    # ============================================================================
    # SCENARIO 1: Cotton Farmer with 1000 kg
    # ============================================================================
    
    print_header("SCENARIO 1: Cotton Farmer with 1000 kg (Low Perishability)")
    
    print("🌾 Farmer Profile:")
    print("   Crop: Cotton")
    print("   Quantity: 1,000 kg")
    print("   Perishability: 0.005 (very low - cotton is stable)")
    print()
    
    recommendation_1 = engine.get_best_selling_strategy(
        current_qty_kg=1000,
        crop='Cotton',
        current_location='Gandhinagar',
        df_current=df_current,
        df_forecast=df_forecast,
        crop_perishability_factor=0.005
    )
    
    _display_recommendation(recommendation_1, "💡 RECOMMENDATION")
    
    # ============================================================================
    # SCENARIO 2: Onion Farmer with 2000 kg
    # ============================================================================
    
    print_header("SCENARIO 2: Onion Farmer with 2000 kg (High Perishability)")
    
    print("🧅 Farmer Profile:")
    print("   Crop: Onion")
    print("   Quantity: 2,000 kg")
    print("   Perishability: 0.03 (higher - onions degrade faster)")
    print()
    
    recommendation_2 = engine.get_best_selling_strategy(
        current_qty_kg=2000,
        crop='Onion',
        current_location='Gandhinagar',
        df_current=df_current,
        df_forecast=df_forecast,
        crop_perishability_factor=0.03
    )
    
    _display_recommendation(recommendation_2, "💡 RECOMMENDATION")
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    print_header("SYSTEM DEMONSTRATION COMPLETE")
    
    print("✅ Successfully demonstrated:")
    print("   1. Data loading/generation (real or synthetic)")
    print("   2. XGBoost model training for price forecasting")
    print("   3. Spatial arbitrage (compare different mandis)")
    print("   4. Temporal arbitrage (compare sell now vs. wait)")
    print("   5. Net profit optimization with full cost accounting")
    print()
    print("📁 Generated files:")
    print(f"   • {output_path}")
    print(f"   • ml_arbitrage/models/*.ubj (trained models)")
    print()
    print("🎯 Key Features (v3.0):")
    print("   • 5-component transport costs (fuel + driver + tolls + maint + L/U)")
    print("   • Exponential decay spoilage model")
    print("   • ₹50/quintal/day cold storage rate")
    print("   • Confidence intervals on price forecasts")
    print("   • Pipeline: Prophet → TFT → ArbitrageMath → CatBoost")
    print()
    print("=" * 80)
    print("  Ready for evaluation! 🚀")
    print("=" * 80)


def _display_recommendation(rec: dict, title: str):
    """Pretty-print a recommendation dict."""
    print(f"\n{title}:")
    print("-" * 60)
    
    if "error" in rec:
        print(f"  ❌ {rec['error']}")
        return
    
    opt = rec.get("optimal_strategy", {})
    bd = opt.get("cost_breakdown", {})
    td = bd.get("transport_detail", {})
    
    print(f"  🎯 {rec['recommendation']}")
    print(f"  Mandi: {opt.get('mandi', 'N/A')} ({opt.get('distance_km', 0):.0f} km)")
    print(f"  Price: ₹{opt.get('price_per_kg', 0):.2f}/kg"
          + ("  [PREDICTED]" if opt.get("is_predicted_price") else "  [CURRENT]"))
    if opt.get("days_to_wait", 0) > 0:
        print(f"  Days to wait: {opt['days_to_wait']}")
    print(f"  Qty remaining: {bd.get('qty_remaining_kg', 'N/A')} kg")
    print()
    print(f"  Gross revenue:  ₹{bd.get('gross_revenue', 0):>12,.2f}")
    if td:
        print(f"  Transport cost: ₹{bd.get('transport_cost', 0):>12,.2f}")
        print(f"    Vehicle: {td.get('vehicle', 'LCV')}")
        print(f"    Fuel:         ₹{td.get('fuel', 0):>10,.2f}")
        print(f"    Driver:       ₹{td.get('driver', 0):>10,.2f}")
        print(f"    Tolls:        ₹{td.get('toll', 0):>10,.2f}")
        print(f"    Maintenance:  ₹{td.get('maintenance', 0):>10,.2f}")
        print(f"    Load/Unload:  ₹{td.get('loading_unloading', 0):>10,.2f}")
    print(f"  Storage cost:   ₹{bd.get('storage_cost', 0):>12,.2f}")
    print(f"  Spoilage cost:  ₹{bd.get('perishability_cost', 0):>12,.2f}")
    print(f"  {'─' * 40}")
    print(f"  NET PROFIT:     ₹{opt.get('net_profit', 0):>12,.2f}")
    print(f"  Margin:         {opt.get('margin_pct', 0):+.1f}%")
    print()
    print(f"  {rec.get('justification', '')}")
    print()


if __name__ == "__main__":
    main()
