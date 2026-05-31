# models/

Trained XGBoost artifacts for daily load prediction — one model per load metric.

## Artifact format

Each target has one file under `models/xgboost/{target}/`:

| File | Content |
|---|---|
| `bundle.joblib` | joblib-serialised dict: `model` (XGBRegressor), `scaler` (MinMaxScaler), `feature_cols` (list[str]), `ewma_spans` (dict) |

## Current targets

| Directory | Target | Unit |
|---|---|---|
| `xgboost/total_distance/` | Daily running distance | metres |
| `xgboost/accelerations/` | High-intensity acceleration count | count |
| `xgboost/sprint_distance/` | High-speed running distance | metres |

## Loading a bundle at inference

```python
import joblib
import numpy as np
from pathlib import Path

bundle       = joblib.load(Path("models/xgboost/total_distance/bundle.joblib"))
model        = bundle["model"]
scaler       = bundle["scaler"]
feature_cols = bundle["feature_cols"]

X_scaled = scaler.transform(X_new[feature_cols])
pred     = np.clip(np.expm1(model.predict(X_scaled)), 0, None)
```

The app loads and validates all three bundles via `src/real_madrid_acwr/modeling/artifacts.py`.

## Regenerating models

```bash
pip install -e ".[dev]"
python train_models.py
```

`train_models.py` runs the full pipeline: data extraction → cleaning → outlier treatment →
daily aggregation → feature engineering → RandomizedSearchCV (50 iters, 10-fold KFold) →
saves `bundle.joblib` for each target and `data/processed/daily.parquet`.
