# Notebooks

Three notebooks, each a full ML pipeline for one GPS load target. All share the same architecture — only the target column and a leakage-prevention drop differ.

---

## Notebooks

| Notebook | Target | Raw Column |
|---|---|---|
| `total_distance.ipynb` | Daily GPS distance (m) | `total_distance` |
| `acceleration_model.ipynb` | Daily acceleration events (count) | `acc_band7plus_total_effort_count` → `accelerations` |
| `sprint_distance_model.ipynb` | Daily high-velocity distance (m) | `velocity_band6plus7_total_distance` → `sprint_distance` |

---

## Shared Pipeline

All three notebooks execute the same steps in the same order.

### 1 · Data Loading
- Read `data/raw/data_acute_vs_chronic.zip` (CSV inside)
- One row per training period; multiple periods per player per day

### 2 · Cleaning
- Rename `velocity_band6plus7_total_distance` → `sprint_distance` and `acc_band7plus_total_effort_count` → `accelerations`
- Parse `exercise_type` from `period_name`; parse and strip timezone from datetime columns
- Compute player age at session date: `(session_date − DOB) / 365.25`
- Drop rows with missing static metadata (height, weight, position, date of birth)

### 3 · Outlier Treatment
- **Player 94884:** Replace match `total_distance` ≥ 20,000 m with the player's within-player median match distance
- **Trialists:** Drop entire records for players with `weight = 200` kg (placeholder value)

### 4 · Daily Aggregation
- One-hot encode `exercise_type` per period (`has_MATCH`, `has_TAC`, `has_G`, `has_BP`, `has_TEC`)
- Group by `(player_id, date)` — sum distances and counts, keep static columns, merge session-type flags

### 5 · Rest Day Fill
- Build continuous date spine `[min_date, max_date]` per player
- Left-join spine → introduces rest-day rows; fill load columns with `0`, forward/backward-fill static columns

### 6 · Chronological Train / Test Split
- Sort unique dates; cut at 80th-percentile date
- No row shuffling — temporal order preserved

### 7 · Feature Engineering
All load-history features use `shift(1)` so today's load never appears in today's features (leakage prevention).

| Feature group | Details |
|---|---|
| Day of week | `dt.dayofweek` (0 = Monday) |
| Microcycle (weekly) | Cumulative load sum, load std dev, monotony (`mean/std`), strain (`sum × monotony`) — all lagged by 1 week |
| EWMA loads | `acute_load` (7-day EWMA), `chronic_load` (28-day EWMA), `training_stress_balance` (chronic − acute), `acwr` (acute / chronic) |
| Lag features | `shift(1/3/5/7/14)` of daily total distance |
| Rolling averages | 3 / 7 / 14-day rolling mean of total distance (shifted 1 day) |

### 8 · Target Transform & Scaling
- Apply `log1p()` to the target to normalise right-skewed distributions
- Fit `MinMaxScaler` on train features only; transform both splits (prevents test leakage)
- `acceleration_model` and `sprint_distance_model` also drop `total_distance` from features before fitting (same-session leakage — distance and the other two targets are measured in the same period)

### 9 · Hyperparameter Tuning
- `RandomizedSearchCV`, 25 iterations, date-blocked expanding-window CV (4 folds, splits on date boundaries — panel-safe)
- Three models per notebook: **LightGBM**, **XGBoost**, **Random Forest**
- Scored on `neg_mean_squared_error` against the log1p target

### 10 · Evaluation
- Train / validation / test RMSE, MAE, R² reported for all three models
- Learning curves, Q-Q plots, SHAP importance bars and beeswarms

### 11 · Feature Selection
- Consensus SHAP ranking averaged across all three models
- Top-5 features selected; all three models retrained and compared against full-feature baseline

### 12 · Forecast Bundle
Saved to `models/notebook_experiments/<target>/forecast_bundle.joblib`:

```
{
  "model":        best XGBoost estimator,
  "scaler":       fitted MinMaxScaler,
  "feature_cols": list of feature column names,
  "ewma_spans":   {"acute": 7, "chronic": 28}
}
```

### 13 · Recursive Forecast
Given a coach's session schedule (player, date, session types):

1. Concatenate real history (≥ 90 days) with planned future rows
2. Run `_add_features()` on the combined frame
3. Scale with the bundle's scaler
4. Predict; clip to ≥ 0; force rest-day rows to 0
5. Write prediction back into the working frame (feeds next day's lag/EWMA features)
6. After all dates: compute ACWR on the full real + predicted series

### 14 · Validation
- **Skew test:** Single recursive step must match direct `add_features → scale → predict` path (max diff < 1e-6)
- **15-day backtest:** Recursive forecast on 15 held-out known dates; compare RMSE / MAE on load and ACWR; plot predicted vs actual

---

## Outputs per Notebook

| File | Location |
|---|---|
| Model comparison table | `models/notebook_experiments/<target>/model_comparison.csv` |
| Trained estimators | `models/notebook_experiments/<target>/{lightgbm,xgboost,random_forest}_best.joblib` |
| Forecast bundle | `models/notebook_experiments/<target>/forecast_bundle.joblib` |

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `log1p` target transform | Right-skewed GPS distributions; improves residual normality |
| `shift(1)` on all load history | Prevents today's load from appearing in today's features |
| Drop `total_distance` in acceleration and sprint notebooks | Same-session leakage — targets are sub-components of total distance |
| Date-blocked CV | Respects temporal structure; all players stay in the same fold |
| Scaler fit on train only | Prevents test-set information leaking into feature scaling |
| EWMA windows: 7 / 28 days | Standard sports science convention for acute / chronic workload ratio |
| Rest days forced to 0 | Model output overridden; no session = no load regardless of prediction |
