# models/

Trained XGBoost artifacts for daily load prediction: one model per load metric.
The app loads native XGBoost `Booster` JSON files at runtime, plus small pickle
metadata files for feature ordering and inverse transforms.

## Artifact contract

Deployable artifacts live under `models/xgboost/<target>/`. Each target must
have the full artifact triad:

| File | Description |
|---|---|
| `models/xgboost/{target}/model.json` | Native XGBoost `Booster` model used by the Streamlit app |
| `models/xgboost/{target}/feature_cols.pkl` | Ordered list of 45 feature columns used at fit and inference time |
| `models/xgboost/{target}/transform.pkl` | Transform metadata, either `{"type": "none", ...}` or `{"type": "log1p", "inverse": "expm1"}` |

Current targets:

| Target | Meaning | Transform | Notes |
|---|---|---|---|
| `acc_total` | High-intensity acceleration count | none | Tweedie objective |
| `total_distance` | Daily running distance, metres | log1p | Apply `np.expm1` after prediction |
| `vel_total` | High-speed running distance, metres | none | Raw MSE objective |

## Regenerating models

```bash
# 1. Build data/processed/model_data.parquet from the raw CSV
jupyter nbconvert --to notebook --execute notebooks/data_pipeline.ipynb

# 2. Train and save all three model artifact triads
python train_models.py
```

The pipeline reads `data/raw/data_acute_vs_chronic.csv` and can extract it from
the committed `data/data_acute_vs_chronic.zip` archive if needed.

## Loading a model at inference

```python
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

MODELS_DIR = Path("models") / "xgboost"
TARGET = "acc_total"  # or "total_distance" / "vel_total"
ARTIFACT_DIR = MODELS_DIR / TARGET

model = xgb.Booster()
model.load_model(str(ARTIFACT_DIR / "model.json"))

with (ARTIFACT_DIR / "feature_cols.pkl").open("rb") as file:
    feature_cols = pickle.load(file)

with (ARTIFACT_DIR / "transform.pkl").open("rb") as file:
    transform = pickle.load(file)

dmatrix = xgb.DMatrix(X_new[feature_cols])
preds = model.predict(dmatrix)

if transform["type"] == "log1p":
    preds = np.expm1(preds)

preds = np.clip(preds, 0, None)
```
