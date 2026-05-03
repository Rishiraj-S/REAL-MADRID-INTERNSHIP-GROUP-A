# notebooks/

Four Jupyter notebooks. Run `data_pipeline.ipynb` first; the three model notebooks
are independent of each other but all depend on its output.

---

## Execution order

```
data_pipeline.ipynb
       │
       └── data/processed/model_data.parquet
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
acc_total  total_distance  vel_total
```

---

## Notebooks

### `data_pipeline.ipynb` — Data Engineering & EDA

End-to-end pipeline from raw CSV to feature-engineered Parquet.

| Section | Output |
|---|---|
| 0. Setup & Data Loading | raw `df` |
| 1. Data Cleaning | cleaned `df` (3,802 rows, 28 players) |
| 2. Outlier Treatment | player 94884 outlier fixed; trialists dropped |
| 3. Daily Aggregation | `daily` (2,103 rows) |
| 4. Modeling Input | feature engineering → `model_data` |
| Persist | `data/processed/model_data.parquet` |

**Run this notebook first.** All model notebooks load from its output.

---

### `acc_total.ipynb` — Acceleration Count Model

Predicts `acc_total` (daily high-intensity acceleration count, Band 7+).

| Section | Description |
|---|---|
| Setup | Imports, constants, REPO_ROOT resolver |
| 1. Load Preprocessed Data | Reads `model_data.parquet`; adds player one-hots |
| 2. Target Distribution Diagnostic | Skewness, zero-inflation check |
| 3. Train / Test Split | Random 80/20 |
| 4. Model Pipelines — Tweedie | `MinMaxScaler → XGBRegressor(objective="reg:tweedie")` |
| 5. Hyperparameter Tuning | `RandomizedSearchCV` n_iter=100, 5-fold CV, elastic net |
| 6. SHAP Interpretability | Feature importance via `TreeExplainer` |
| 7. Learning Curve | Bias-variance diagnostic |
| 8. Model Persistence | Saves to `models/` |
| 9. Final Metrics | MAE, RMSE, R² on train and test |

**Loss:** Tweedie — chosen for count-like data with 3.1% zeros and right skew (skew ≈ 1.55).

---

### `total_distance.ipynb` — Running Distance Model

Predicts `total_distance` (daily total running distance, metres).

| Section | Description |
|---|---|
| Setup | Imports, constants, REPO_ROOT resolver |
| 1. Load Preprocessed Data | Reads `model_data.parquet`; adds player one-hots |
| 2. Target Skewness & Log Transform | `log1p` applied; `expm1` at inference |
| 3. Train / Test Split | Random 80/20 |
| 4. Model Pipelines | `MinMaxScaler → XGBRegressor(objective="reg:squarederror")` on log target |
| 5. Hyperparameter Tuning | `RandomizedSearchCV` n_iter=100, 5-fold CV, elastic net |
| 6. SHAP Interpretability | Feature importance |
| 7. Learning Curve | Bias-variance diagnostic |
| 8. Model Persistence | Saves to `models/` |
| 9. Final Metrics | MAE, RMSE, R² in original units (metres) |

**Loss:** log-MSE — `total_distance` is right-skewed (skew ≈ 0.9); log transform normalises it.

---

### `vel_total.ipynb` — High-Speed Running Model

Predicts `vel_total` (daily high-speed running distance, metres, velocity bands 6+7).

Split into two parts due to the target's heavy zero-inflation (~30% zeros).

**Part A — Target Distribution Analysis**

| Section | Description |
|---|---|
| A.0 Setup | Imports |
| A.1 Load Preprocessed Data | Reads `model_data.parquet` |
| A.2 Distribution Statistics | Skewness, percentiles, zero count |
| A.3 Visualisations | Histogram + Q-Q plots (raw and log1p) |
| A.4 Modeling Decision Flags | Automated guidance on loss / transform choice |

**Part B — Model Training**

| Section | Description |
|---|---|
| B.0 Setup | Imports, TARGET, SHAP threshold constant |
| B.1 Load Preprocessed Data | Reads `model_data.parquet`; adds player one-hots |
| B.2 Train / Test Split | Random 80/20 |
| B.3 Pipelines & Helpers | `MinMaxScaler → XGBRegressor(objective="reg:squarederror")` |
| B.4 Round 1 — Full Features | Tune on all 52 features |
| B.5 SHAP Feature Selection | Retain features covering 90% cumulative SHAP importance |
| B.6 Round 2 — Reduced Features | Re-tune on SHAP-selected subset |
| B.7 Model Comparison | Pick winner by test MAE |
| B.8 Learning Curve | Bias-variance diagnostic for winner |
| B.9 Model Persistence | Saves to `models/` |
| B.10 Final Metrics | MAE, RMSE, R² |

**Loss:** Raw MSE — chosen after distribution analysis. Two-round SHAP feature selection used to combat the sparse zero-heavy signal.

---

## Path resolution

Each notebook uses a `REPO_ROOT` resolver in its Setup cell:

```python
REPO_ROOT  = _repo_root()   # git rev-parse or walk-up fallback
DATA_DIR   = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
```

This works both locally (running from `notebooks/`) and in Databricks (where CWD is the repo root).
