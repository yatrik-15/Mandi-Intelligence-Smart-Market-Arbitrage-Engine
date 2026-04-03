"""Final check: can we create TFT from dataset and train?"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping

# Check MRO
for base in TemporalFusionTransformer.__mro__:
    print(f"  {base.__module__}.{base.__name__}")

# Create minimal test dataset
np.random.seed(42)
N = 200
data = pd.DataFrame({
    "time_idx": list(range(N)),
    "group_id": "test_group",
    "target": np.random.randn(N).cumsum() + 20,
})

ds = TimeSeriesDataSet(
    data, time_idx="time_idx", target="target",
    group_ids=["group_id"],
    max_encoder_length=30, max_prediction_length=7,
    time_varying_unknown_reals=["target"],
    target_normalizer=GroupNormalizer(groups=["group_id"]),
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
)

loader = ds.to_dataloader(train=True, batch_size=32, num_workers=0)

tft = TemporalFusionTransformer.from_dataset(
    ds, learning_rate=0.01, hidden_size=8,
    attention_head_size=1, dropout=0.1,
    hidden_continuous_size=4, lstm_layers=1,
    output_size=7, loss=QuantileLoss(),
)
print(f"TFT created: {type(tft)}")
print(f"Is LightningModule? {isinstance(tft, pl.LightningModule)}")

trainer = pl.Trainer(max_epochs=2, accelerator="cpu", enable_progress_bar=False, logger=False)
trainer.fit(tft, train_dataloaders=loader)
print("Training OK!")
