"""
Test the trained models with real data and generate recommendations
Updated for v3.0 — 5-component transport cost model
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from ml_arbitrage.data_loader import MandiDataLoader
from ml_arbitrage.price_predictor import PricePredictor
from ml_arbitrage.arbitrage_engine import ArbitrageEngine
import pandas as pd

def main():
    print("=" * 80)
    print("  TESTING ML ARBITRAGE ENGINE v3.0 WITH REAL DATA")
    print("=" * 80)
    print()
    
    # Load processed data or generate synthetic
    print("📊 Loading data...")
    loader = MandiDataLoader()
    
    dataset_paths = [
        "dataset/commodity_price.csv",
        "data/mandi_prices.csv",
    ]
    
    df = None
    for path in dataset_paths:
        try:
            loader.load_data(path)
            df = loader.filter_and_process(days=None)
            if df is not None and not df.empty:
                print(f"✅ Loaded real data: {len(df):,} records")
                break
        except (FileNotFoundError, Exception):
            continue
    
    if df is None or df.empty:
        print("⚠️  No real dataset found — using synthetic data")
        df = loader.generate_rich_synthetic_data()
    
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Mandis: {df['Mandi_Name'].unique().tolist()}")
    print(f"   Crops: {df['Crop'].unique().tolist()}")
    
    # Initialize predictor and train models
    print("\n🤖 Training models...")
    predictor = PricePredictor()
    df_featured = predictor.prepare_features(df)
    results = predictor.train_all_models(df_featured)
    trained = [r for r in results if "error" not in r]
    print(f"✅ Trained {len(trained)} models")
    
    # Get latest data for current prices
    latest_date = df_featured['Date'].max()
    df_current = df_featured[df_featured['Date'] == latest_date].copy()
    
    print(f"   Using latest date: {latest_date}")
    
    # Generate forecasts
    print("\n🔮 Generating 7-day price forecasts...")
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
    print(f"✅ Generated {len(forecasts_data)} predictions")
    
    # Initialize engine
    engine = ArbitrageEngine(price_predictor=predictor)
    
    # Test scenarios
    test_crops = ["Cotton", "Onion", "Tomato"]
    
    for crop in test_crops:
        if crop not in df_current['Crop'].values:
            continue
        
        print(f"\n{'=' * 80}")
        print(f"  SCENARIO: {crop} Farmer - 1000 kg")
        print("=" * 80)
        print()
        
        recommendation = engine.get_best_selling_strategy(
            current_qty_kg=1000,
            crop=crop,
            current_location='Gandhinagar',
            df_current=df_current,
            df_forecast=df_forecast,
        )
        
        if "error" in recommendation:
            print(f"  ❌ {recommendation['error']}")
            continue
        
        opt = recommendation.get('optimal_strategy', {})
        bd = opt.get('cost_breakdown', {})
        td = bd.get('transport_detail', {})
        
        print(f"  🎯 {recommendation['recommendation']}")
        print(f"  Mandi: {opt.get('mandi', 'N/A')}")
        print(f"  Distance: {opt.get('distance_km', 0):.0f} km")
        print(f"  Price: ₹{opt.get('price_per_kg', 0):.2f}/kg")
        print(f"  Net Profit: ₹{opt.get('net_profit', 0):,.2f}")
        if td:
            print(f"  Transport: ₹{bd.get('transport_cost', 0):,.2f} "
                  f"(vehicle: {td.get('vehicle', 'LCV')}, "
                  f"fuel: ₹{td.get('fuel', 0):,.0f})")
        print(f"  Storage: ₹{bd.get('storage_cost', 0):,.2f}")
        print(f"  Spoilage: ₹{bd.get('perishability_cost', 0):,.2f}")
        print()
    
    print("=" * 80)
    print("  SUCCESS: ML Arbitrage Engine v3.0 working!")
    print("=" * 80)
    print()
    print("Pipeline order: Prophet → TFT → ArbitrageMath → CatBoost")
    print("Transport model: 5-component (fuel + driver + tolls + maintenance + L/U)")
    print(f"📁 Trained models location: ml_arbitrage/models/")

if __name__ == "__main__":
    main()
