"""
Train ML Arbitrage Engine with NEW Combined Dataset
=====================================================

Combines the original commodity_price.csv (historical Gujarat data) with the
new_dataset.csv (latest multi-state APMC prices from 31-Mar-2026) to retrain
XGBoost price prediction models.

The new dataset adds:
  - Fresh price data points for Gujarat mandis (Rajpipla, Damnagar, Gondal, etc.)
  - Cross-state reference prices for Onion, Tomato, Potato from other states
  - Broader commodity coverage for potential future expansion

Strategy:
  1. Load both CSV files
  2. Standardise column names to a common schema
  3. Concatenate into a unified dataset
  4. Run the existing data-loader pipeline (filter Gujarat + target crops)
  5. Retrain XGBoost models with the enriched dataset
  6. Compare old vs new model metrics
"""

import sys
import os
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np

from ml_arbitrage.data_loader import MandiDataLoader
from ml_arbitrage.price_predictor import PricePredictor


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names across different CSV formats."""
    df.columns = (df.columns
                  .str.strip()
                  .str.replace("_x0020_", "_", regex=False)
                  .str.replace(r"\s+", "_", regex=True))
    return df


def load_old_model_metrics() -> dict:
    """Load existing model metrics from .meta.json files for comparison."""
    import pathlib
    models_dir = pathlib.Path("ml_arbitrage/models")
    metrics = {}
    for meta_file in models_dir.glob("*.meta.json"):
        try:
            data = json.loads(meta_file.read_text())
            key = f"{data['mandi']}_{data['crop']}"
            metrics[key] = {
                "mae": data.get("mae", 0),
                "rmse": data.get("rmse", 0),
                "mape": data.get("mape", 0),
                "r2": data.get("r2", 0),
            }
        except Exception:
            continue
    return metrics


def main():
    start_time = time.time()
    
    print("=" * 80)
    print("  TRAINING WITH COMBINED DATASET (Old + New)")
    print("  " + "=" * 76)
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # ── Step 1: Load old model metrics for comparison ──────────────────────
    print("📋 Step 1: Loading old model metrics for comparison...")
    old_metrics = load_old_model_metrics()
    if old_metrics:
        print(f"   Found {len(old_metrics)} existing model metrics")
    else:
        print("   No existing model metrics found (first training)")
    print()

    # ── Step 2: Load and combine datasets ──────────────────────────────────
    print("📊 Step 2: Loading datasets...")
    
    old_path = "dataset/commodity_price.csv"
    new_path = "dataset/new_dataset.csv"
    
    # Load old dataset
    print(f"   Loading old dataset: {old_path}")
    df_old = pd.read_csv(old_path, encoding="utf-8", low_memory=False)
    df_old = standardise_columns(df_old)
    print(f"   ✅ Old dataset: {len(df_old):,} records")
    print(f"      Columns: {list(df_old.columns)}")
    print(f"      States: {df_old['State'].unique().tolist()[:5]}...")
    print()
    
    # Load new dataset
    print(f"   Loading new dataset: {new_path}")
    df_new = pd.read_csv(new_path, encoding="utf-8", low_memory=False)
    df_new = standardise_columns(df_new)
    print(f"   ✅ New dataset: {len(df_new):,} records")
    print(f"      Columns: {list(df_new.columns)}")
    
    # Show new dataset summary
    if "State" in df_new.columns:
        states = df_new["State"].value_counts()
        print(f"      States breakdown:")
        for state, count in states.head(10).items():
            marker = "🟢" if "Gujarat" in str(state) else "  "
            print(f"         {marker} {state}: {count} records")
    
    if "Commodity" in df_new.columns:
        commodities = df_new["Commodity"].value_counts()
        target_crops = ["Onion", "Tomato", "Potato", "Wheat", "Cotton"]
        print(f"      Target commodities in new data:")
        for crop in target_crops:
            count = df_new[df_new["Commodity"].str.contains(crop, case=False, na=False)].shape[0]
            if count > 0:
                print(f"         ✅ {crop}: {count} records")
            else:
                print(f"         ⚪ {crop}: 0 records")
    print()
    
    # ── Step 3: Combine datasets ──────────────────────────────────────────
    print("🔗 Step 3: Combining datasets...")
    
    # Ensure column names match between old and new
    # Old: State, District, Market, Commodity, Variety, Grade, Arrival_Date, Min_Price, Max_Price, Modal_Price
    # New: State, District, Market, Commodity, Variety, Grade, Arrival_Date, Min_Price, Max_Price, Modal_Price
    
    common_cols = list(set(df_old.columns) & set(df_new.columns))
    print(f"   Common columns: {common_cols}")
    
    if len(common_cols) >= 4:  # Need at least date, market, commodity, price
        df_combined = pd.concat([df_old[common_cols], df_new[common_cols]], 
                                ignore_index=True)
    else:
        # Fall back to combining all columns (fill missing with NaN)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    
    print(f"   ✅ Combined dataset: {len(df_combined):,} records")
    
    # Remove exact duplicates
    before_dedup = len(df_combined)
    df_combined = df_combined.drop_duplicates()
    after_dedup = len(df_combined)
    if before_dedup > after_dedup:
        print(f"   🧹 Removed {before_dedup - after_dedup} duplicate rows")
    
    # Save combined dataset
    combined_path = "dataset/combined_dataset.csv"
    df_combined.to_csv(combined_path, index=False)
    print(f"   💾 Saved combined dataset → {combined_path}")
    print()

    # ── Step 4: Run data loader pipeline ──────────────────────────────────
    print("🔧 Step 4: Processing through data pipeline...")
    
    loader = MandiDataLoader()
    loader.load_data(combined_path)
    
    try:
        df_processed = loader.filter_and_process(days=None)  # Use ALL data
    except ValueError as e:
        print(f"   ⚠️  Pipeline warning: {e}")
        print("   Attempting with relaxed filters...")
        # Try with just Gujarat, but if no Gujarat data, expand
        try:
            df_processed = loader.filter_and_process(days=None, state_filter="Gujarat")
        except ValueError:
            print("   ⚠️  No Gujarat data found, trying all states...")
            df_processed = loader.filter_and_process(days=None, state_filter=None)
    
    if len(df_processed) == 0:
        print("   ❌ No data after processing. Falling back to synthetic data...")
        df_processed = loader.generate_rich_synthetic_data(n_days=730)
    
    # Show summary
    stats = loader.get_summary_stats()
    print("\n📈 Data Summary:")
    print(f"   Total records      : {stats.get('total_records', 0):,}")
    date_info = stats.get("date_range", {})
    print(f"   Date range         : {date_info.get('start', '?')} → {date_info.get('end', '?')} ({date_info.get('days', 0)} days)")
    print(f"   Mandis             : {stats.get('mandis', [])}")
    print(f"   Crops              : {stats.get('crops', [])}")
    
    if "records_per_mandi_crop" in stats:
        mc_counts = stats["records_per_mandi_crop"]
        print(f"   Mandi-Crop pairs   : {len(mc_counts)}")
        for (mandi, crop), count in sorted(mc_counts.items()):
            trainable = "✅" if count >= 30 else "⚠️ "
            print(f"      {trainable} {mandi} × {crop}: {count} records")
    
    # Save processed data
    output_path = "ml_arbitrage/processed_mandi_data_combined.csv"
    loader.save_processed_data(output_path)
    print(f"\n   💾 Saved processed data → {output_path}")
    print()

    # ── Step 5: Train XGBoost models ──────────────────────────────────────
    print("=" * 80)
    print("  TRAINING XGBOOST MODELS")
    print("=" * 80)
    print()
    
    predictor = PricePredictor()
    
    # Prepare features
    print("🔧 Engineering time-series features...")
    df_featured = predictor.prepare_features(df_processed)
    print(f"   Features: {len(df_featured.columns)} columns")
    print(f"   Records:  {len(df_featured):,}")
    print()
    
    # Train all models
    print("🤖 Training XGBoost models for all Mandi-Crop combinations...")
    print("-" * 80)
    results = predictor.train_all_models(df_featured)
    print()

    # ── Step 6: Results comparison ────────────────────────────────────────
    print("=" * 80)
    print("  TRAINING RESULTS & COMPARISON")
    print("=" * 80)
    print()
    
    successful = [r for r in results if isinstance(r, dict) and "error" not in r]
    failed = [r for r in results if isinstance(r, dict) and "error" in r]
    
    print(f"✅ Successfully trained: {len(successful)} models")
    if failed:
        print(f"⚠️  Skipped/Failed:      {len(failed)} models")
        for f in failed:
            print(f"   - {f.get('mandi', '?')} × {f.get('crop', '?')}: {f.get('error', 'unknown')}")
    print()
    
    # Detailed comparison table
    print("📊 Model Performance (New vs Old):")
    print("-" * 100)
    print(f"{'Mandi':<14} {'Crop':<8} {'MAE ₹/kg':>10} {'RMSE ₹/kg':>11} {'MAPE %':>8} {'R²':>8} {'Δ MAE':>10} {'Status':>10}")
    print("-" * 100)
    
    improvements = 0
    regressions = 0
    
    for result in sorted(successful, key=lambda x: (x["mandi"], x["crop"])):
        key = f"{result['mandi']}_{result['crop']}"
        old = old_metrics.get(key, {})
        
        mae = result.get("mae", 0)
        rmse = result.get("rmse", 0)
        mape = result.get("mape", 0)
        r2 = result.get("r2", 0)
        
        # Calculate improvement
        if old and "mae" in old:
            delta_mae = mae - old["mae"]
            if delta_mae < -0.01:
                status = "📈 Better"
                improvements += 1
            elif delta_mae > 0.01:
                status = "📉 Worse"
                regressions += 1
            else:
                status = "➡️  Same"
            delta_str = f"{delta_mae:+.4f}"
        else:
            status = "🆕 New"
            delta_str = "N/A"
        
        print(f"  {result['mandi']:<12} {result['crop']:<8} "
              f"₹{mae:>8.4f} ₹{rmse:>9.4f} {mape:>7.2f}% {r2:>7.4f} "
              f"{delta_str:>10} {status}")
    
    print("-" * 100)
    print()
    
    # Summary statistics
    if successful:
        avg_mae = np.mean([r["mae"] for r in successful])
        avg_rmse = np.mean([r["rmse"] for r in successful])
        avg_mape = np.mean([r["mape"] for r in successful])
        avg_r2 = np.mean([r.get("r2", 0) for r in successful])
        
        print("📉 Average Metrics:")
        print(f"   MAE:  ₹{avg_mae:.4f}/kg")
        print(f"   RMSE: ₹{avg_rmse:.4f}/kg")
        print(f"   MAPE: {avg_mape:.2f}%")
        print(f"   R²:   {avg_r2:.4f}")
        print()
    
    if old_metrics:
        print(f"📊 Improvement Summary:")
        print(f"   📈 Improved: {improvements} models")
        print(f"   📉 Regressed: {regressions} models")
        print(f"   ➡️  Same:     {len(successful) - improvements - regressions} models")
    print()

    # ── Step 7: Generate sample predictions ───────────────────────────────
    print("=" * 80)
    print("  SAMPLE 7-DAY FORECASTS")
    print("=" * 80)
    print()
    
    for result in successful[:3]:  # Show first 3 models
        mandi, crop = result["mandi"], result["crop"]
        try:
            forecast = predictor.predict_future_price(df_featured, mandi, crop, days_ahead=7)
            if not forecast.empty:
                print(f"  📈 {mandi} - {crop} (7-day forecast):")
                for _, row in forecast.iterrows():
                    print(f"      Day {row['Day_Ahead']}: ₹{row['Predicted_Price']:.2f}/kg "
                          f"[₹{row['Lower_Bound']:.2f} – ₹{row['Upper_Bound']:.2f}]")
                print()
        except Exception as e:
            print(f"  ⚠️  Could not forecast {mandi}-{crop}: {e}")
    
    # ── Final Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("=" * 80)
    print("  TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Models saved:     ml_arbitrage/models/")
    print(f"  Combined data:    {combined_path}")
    print(f"  Processed data:   {output_path}")
    print(f"  Time elapsed:     {elapsed:.1f} seconds")
    print()
    print("  Next steps:")
    print("  → Run: python get_recommendation.py")
    print("  → Or:  python ml_arbitrage/main.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
