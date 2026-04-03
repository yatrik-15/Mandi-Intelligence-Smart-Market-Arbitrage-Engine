"""Quick dependency check script."""
import sys
print("Python:", sys.version)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch: NOT INSTALLED")
except Exception as e:
    print("PyTorch: installed but error:", e)

try:
    import pytorch_lightning as pl
    print("PyTorch Lightning:", pl.__version__)
except ImportError:
    print("PyTorch Lightning: NOT INSTALLED")

try:
    import pytorch_forecasting
    print("pytorch-forecasting:", pytorch_forecasting.__version__)
except ImportError:
    print("pytorch-forecasting: NOT INSTALLED")

try:
    import prophet
    print("Prophet:", prophet.__version__)
except ImportError:
    print("Prophet: NOT INSTALLED")

try:
    import catboost
    print("CatBoost:", catboost.__version__)
except ImportError:
    print("CatBoost: NOT INSTALLED")

try:
    import xgboost
    print("XGBoost:", xgboost.__version__)
except ImportError:
    print("XGBoost: NOT INSTALLED")
