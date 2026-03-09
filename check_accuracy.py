import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ml_arbitrage.data_loader import MandiDataLoader
from ml_arbitrage.price_predictor import PricePredictor

loader = MandiDataLoader()
loader.load_data('dataset/commodity_price.csv')
df = loader.filter_and_process(days=None)

predictor = PricePredictor()
df_feat = predictor.prepare_features(df)
results = predictor.train_all_models(df_feat)

print()
print("=" * 60)
print("MODEL ACCURACY REPORT")
print("=" * 60)
for r in results:
    accuracy_pct = 100 - r['mape']
    print(f"{r['mandi']}-{r['crop']}: "
          f"MAE=Rs{r['mae']:.2f}/kg | "
          f"MAPE={r['mape']:.1f}% | "
          f"R2={r['r2']:.3f} | "
          f"Accuracy~{accuracy_pct:.1f}%")

if results:
    avg_mape = sum(r['mape'] for r in results) / len(results)
    avg_r2   = sum(r['r2']   for r in results) / len(results)
    avg_mae  = sum(r['mae']  for r in results) / len(results)
    print("-" * 60)
    print(f"AVERAGE ACROSS {len(results)} MODELS:")
    print(f"  MAE  : Rs{avg_mae:.2f} / kg")
    print(f"  MAPE : {avg_mape:.1f}%")
    print(f"  R2   : {avg_r2:.3f}")
    print(f"  Overall Accuracy : ~{100 - avg_mape:.1f}%")
