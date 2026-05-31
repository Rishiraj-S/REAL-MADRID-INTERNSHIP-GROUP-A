# notebooks/

EDA and model exploration notebooks. These are **not** part of the production pipeline — run `python train_models.py` instead.

## Files

| File | Purpose |
|---|---|
| `datapipeline.py` | Shared preprocessing module imported by all three notebooks |
| `total_distance.ipynb` | EDA + XGBoost training for `total_distance` |
| `accelerations.ipynb` | EDA + XGBoost training for `accelerations` |
| `sprint_distance.ipynb` | EDA + XGBoost training for `sprint_distance` |

## Shared pipeline (`datapipeline.py`)

`run_pipeline(target)` chains: load → clean → IQR-cap outliers → aggregate to player-day → spine fill rest days → random 80/20 split → day-of-week OHE → log1p transform → MinMaxScale.

Returns a dict with `X_tr_np`, `y_tr_np`, `test_base`, `scaler`, `feature_cols`, `daily`.

## Notebook structure

Each notebook follows the same steps:

1. `run_pipeline(TARGET)` — full preprocessing in one call
2. Target distribution plot (raw vs log1p)
3. XGBoost training — RandomizedSearchCV (50 iters, 10-fold KFold)
4. Learning curve
5. Test-set evaluation (RMSE, MAE, R²)
6. SHAP feature importance (bar, beeswarm, waterfall)

## Feature set

Cross-sectional only — no lags, no EWMA, no microcycle stats:

- `n_periods`, `n_exercise_types`
- `height`, `weight`, `age`
- `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`
- `dow_0 … dow_6` (day-of-week OHE)
- `total_distance` (covariate for `accelerations` and `sprint_distance` only)

## Running the notebooks

```bash
pip install -e ".[notebooks]"
jupyter notebook
```

Notebooks must be run from the **repository root** so `datapipeline.py` imports resolve correctly.
