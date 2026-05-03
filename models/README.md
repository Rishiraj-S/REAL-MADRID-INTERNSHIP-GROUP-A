# models/

Trained XGBoost models for daily load prediction — one model per load metric.
All artifacts are auto-saved here when the corresponding notebook is run.

---

## Artifacts

### acc_total (high-intensity acceleration count)

| File | Description |
|---|---|
| `acc_total_model.pkl` | Fitted `Pipeline(MinMaxScaler → XGBRegressor)`, Tweedie loss |
| `acc_total_feature_cols.pkl` | List of 52 feature column names used at fit time |
| `acc_total_transform.pkl` | `{"type": "none", "loss": "tweedie"}` — no inverse transform needed |
| `acc_total_learning_curve.png` | Bias-variance diagnostic (train vs. CV MAE vs. dataset size) |

**Test results:** MAE ≈ 3.63 · RMSE ≈ 5.00 · R² ≈ 0.38

---

### total_distance (daily running distance, metres)

| File | Description |
|---|---|
| `total_distance_model.pkl` | Fitted `Pipeline(MinMaxScaler → XGBRegressor)`, log-MSE |
| `total_distance_feature_cols.pkl` | List of 52 feature column names |
| `total_distance_transform.pkl` | `{"type": "log1p", "inverse": "expm1"}` — apply `np.expm1` at inference |
| `total_distance_learning_curve.png` | Bias-variance diagnostic |

**Test results:** MAE ≈ 800 m · R² ≈ 0.42

---

### vel_total (high-speed running distance, metres)

| File | Description |
|---|---|
| `vel_total_model.pkl` | Winner of full vs. SHAP-reduced feature set comparison |
| `vel_total_feature_cols.pkl` | Feature column list (may be smaller than 52 if SHAP-reduced won) |
| `vel_total_transform.pkl` | `{"type": "none", "loss": "mse"}` — predictions are in original units |
| `vel_total_learning_curve.png` | Bias-variance diagnostic |
| `vel_total_shap_cumulative.png` | SHAP cumulative importance plot (feature selection diagnostic) |

---

## Regenerating models

```
1. Run notebooks/data_pipeline.ipynb   →  writes data/processed/model_data.parquet
2. Run notebooks/acc_total.ipynb       →  writes acc_total_*.pkl + acc_total_*.png
3. Run notebooks/total_distance.ipynb  →  writes total_distance_*.pkl + *.png
4. Run notebooks/vel_total.ipynb       →  writes vel_total_*.pkl + *.png
```

Steps 2–4 are independent of each other and can run in any order after step 1.

---

## Loading a model at inference

```python
import pickle, numpy as np
from pathlib import Path

MODELS_DIR = Path("models")
TARGET = "acc_total"   # or "total_distance" / "vel_total"

model       = pickle.load(open(MODELS_DIR / f"{TARGET}_model.pkl",        "rb"))
feature_cols = pickle.load(open(MODELS_DIR / f"{TARGET}_feature_cols.pkl", "rb"))
transform   = pickle.load(open(MODELS_DIR / f"{TARGET}_transform.pkl",     "rb"))

preds = model.predict(X_new[feature_cols])

# Apply inverse transform if needed (only total_distance uses log1p)
if transform["type"] == "log1p":
    preds = np.expm1(preds)

preds = np.clip(preds, 0, None)   # predictions are physically bounded at 0
```
