# Technical Documentation — Real Madrid ACWR Prediction Tool

This document is the technical reference for the deployed ACWR forecasting system. It explains:

- the operational problem the tool solves,
- the meaning and limitations of the raw data,
- the cleaning and feature-engineering pipeline,
- the mathematical and statistical design of the predictive models,
- the conversion from predicted loads to forecast ACWR trajectories.

Production behaviour is defined by the Python code under `src/` and the deployable artifacts under `models/xgboost/`. Notebook material is cited only when it explains why a deployed modeling choice was made.

## Table of contents

- [1. Problem description](#1-problem-description)
- [2. Data description](#2-data-description)
- [3. Cleaning and cohort definition](#3-cleaning-and-cohort-definition)
- [4. Data processing pipeline](#4-data-processing-pipeline)
- [5. Full-calendar reconstruction for ACWR](#5-full-calendar-reconstruction-for-acwr)
- [6. Feature vector used by the deployed models](#6-feature-vector-used-by-the-deployed-models)
- [7. Model architecture](#7-model-architecture)
- [8. Target-specific formulations](#8-target-specific-formulations)
- [9. Training procedure](#9-training-procedure)
- [10. Model artifacts and serving contract](#10-model-artifacts-and-serving-contract)
- [11. ACWR methodology](#11-acwr-methodology)
- [12. Forecast-generation algorithm](#12-forecast-generation-algorithm)
- [13. End-to-end system flow](#13-end-to-end-system-flow)
- [14. File-level mapping of responsibilities](#14-file-level-mapping-of-responsibilities)
- [15. Assumptions and limitations](#15-assumptions-and-limitations)
- [16. Practical interpretation](#16-practical-interpretation)
- [17. Reproducing the deployed artifacts](#17-reproducing-the-deployed-artifacts)
- [18. Summary](#18-summary)

## Notation

Throughout the document:

- `p` denotes a player,
- `t` denotes a calendar day,
- `m` denotes a target metric,
- `x[p, t]` denotes the feature vector for player `p` on day `t`,
- `y[p, t, m]` denotes the observed load,
- `ŷ[p, t, m]` denotes the predicted load.

---

## 1. Problem description

### 1.1 Operational goal

The tool is designed for football fitness coaches planning the next 15 days of training.
The coaches do **not** directly enter expected numeric load values such as metres or acceleration counts. Instead, they schedule **session events** on an interactive calendar, tagging each event with one or more session types:

- `G` — game-based / small-sided game,
- `TAC` — tactical,
- `BP` — set pieces,
- `TEC` — technical,
- `MATCH` — official match,
- or no event on a day (rest).

From this plan, the system predicts for each squad player:

1. the future daily external load for three metrics,
2. the future acute and chronic workload states,
3. the resulting ACWR trajectory,
4. the final risk zone at the end of the forecast horizon.

### 1.2 Formal problem statement

For each player `p`, for each day `t` in a 15-day horizon, and for each metric `m` in

- `total_distance`,
- `accelerations`,
- `sprint_distance`,

we want to estimate a future load

`ŷ[p, t, m] = f_m(x[p, t])`

where `x[p, t]` is a cross-sectional feature vector composed of:

- static player attributes (height, weight, age),
- session-type indicators for the planned day,
- session activity counts (n_periods, n_exercise_types),
- calendar features (day of week, one-hot encoded),
- cross-metric load features where applicable.

The predicted loads are then stitched to the player's historical load time series and converted into a forecast ACWR:

`ACWR[p, t, m] = Acute[p, t, m] / Chronic[p, t, m]`

where acute and chronic are exponentially weighted moving averages (EWMA) over 7 and 28 days respectively.

### 1.3 Why this matters

The underlying coaching question is not "what was the load yesterday?" but:

> If we plan a specific sequence of sessions over the next two weeks, how will each player's load-risk profile evolve?

This is therefore a **counterfactual forecasting** problem driven by a planned session template, not a passive reporting dashboard.

---

## 2. Data description

### 2.1 Raw source

The bootstrap source archive is:

- `data/data_acute_vs_chronic.zip`

It is extracted locally to:

- `data/raw/data_acute_vs_chronic.csv`

The raw CSV contains one row per **training period** (drill/block inside a session), not one row per player-day.

### 2.2 Observed dataset characteristics

- date range: **2024-07-16 to 2025-06-26**,
- raw players before cleaning: **35**,
- final cleaned cohort: **28** players,
- `daily.parquet` shape after full-calendar reconstruction: **6,310 rows × 16 columns**.

### 2.3 Core raw fields

| Field | Meaning |
|---|---|
| `player_id` | Player identifier |
| `position_name_en` | Categorical position label |
| `height`, `weight`, `date_of_birth` | Anthropometric and demographic metadata |
| `period_start_time` | Timestamp of the training period |
| `period_name` | Drill label whose prefix encodes session type |
| `activity_id` | Session / activity identifier |
| `is_official_match` | Match flag |
| `total_distance` | Total running distance (metres) |
| `acc_band7plus_total_effort_count` | High-intensity acceleration count |
| `velocity_band6plus7_total_distance` | High-speed running distance (metres) |

### 2.4 Semantics of `period_name`

The field `period_name` has the form `{CATEGORY} {DRILL_ID}` such as `G 1960`, `TAC 0133`, `BP 2351`. The prefix is interpreted as the session family:

| Prefix | Meaning |
|---|---|
| `G` | Game-based / small-sided game |
| `TAC` | Tactical |
| `BP` | Set pieces |
| `TEC` | Technical |
| `NaN` with official-match flag | Match |

Rows with `period_name = NaN` are treated as official matches and mapped to `exercise_type = 'MATCH'`.

### 2.5 Three target load metrics

The system forecasts three different external-load targets independently:

| Deployed target | Raw source column | Meaning |
|---|---|---|
| `total_distance` | `total_distance` | Aerobic / volume load (metres) |
| `accelerations` | `acc_band7plus_total_effort_count` | High-intensity acceleration count |
| `sprint_distance` | `velocity_band6plus7_total_distance` | High-speed running distance (metres) |

They are intentionally **not combined** into a single score because they capture different physiological stresses.

---

## 3. Cleaning and cohort definition

### 3.1 Column renaming

The pipeline renames raw columns at ingestion:

```
acc_band7plus_total_effort_count    →  accelerations
velocity_band6plus7_total_distance  →  sprint_distance
```

All subsequent code uses these names exclusively.

### 3.2 Type normalisation

- `period_start_time` → `datetime64[ns]`, timezone stripped, normalised to midnight
- `date_of_birth` → `datetime64[ns]`, timezone stripped
- `player_id` → categorical
- `is_official_match` → null-filled with 0, then dropped after deriving `exercise_type`

### 3.3 Age calculation

Age is computed continuously at the row date:

```python
age = (period_start_time - date_of_birth).dt.days / 365.25
```

### 3.4 Player exclusions

Players are excluded for one or more of the following reasons:

- implausible placeholder metadata (`weight == 200`),
- missing all core player metadata (height, weight, position, date of birth).

The final deployed cohort contains **28 players**.

### 3.5 Row-level outlier treatment

The pipeline applies an **IQR cap (Q3 + 3×IQR)** per player per exercise type to `total_distance` (always) and the training target (when different from `total_distance`). This replaces any extreme period-level values with the player's own within-group upper fence, preserving the row while dampening GPS outliers.

Separately, players with `weight == 200` (trialist placeholders) are dropped entirely.

---

## 4. Data processing pipeline

### 4.1 Aggregation target: player-day

The period-level table is aggregated to one row per `(player_id, date)`:

- load metrics are **summed** over all periods that day,
- session composition is preserved as per-type binary flags,
- static player metadata is carried forward via `first`.

The resulting daily columns are:

| Column | Derivation |
|---|---|
| `total_distance` | sum of `total_distance` across periods |
| `accelerations` | sum of `accelerations` across periods |
| `sprint_distance` | sum of `sprint_distance` across periods |
| `n_periods` | count of `activity_id` rows |
| `n_exercise_types` | nunique of `exercise_type` |
| `height`, `weight`, `age`, `position` | first value for the player-day |

### 4.2 Session composition encoding

From each player-day's set of exercise types, the pipeline derives binary flags using one-hot encoding of `exercise_type`:

- `has_G`
- `has_TAC`
- `has_BP`
- `has_TEC`
- `has_MATCH`

If a session type does not appear in the data, its column is added with all zeros.

### 4.3 Full calendar spine reconstruction

After aggregating active days, the pipeline builds a **continuous daily grid** per player — one row per calendar day from each player's first to last active date. Gaps (rest days) are filled with:

- `0` for all load metrics and session flags,
- `0` for `n_periods` and `n_exercise_types`,
- forward-then-backward fill for `height`, `weight`, `age`, `position`.

This spine reconstruction ensures rest days contribute to EWMA decay correctly in the ACWR computation.

### 4.4 Persisted daily table

The full-calendar daily frame is saved at:

- `data/processed/daily.parquet` — **6,310 rows × 16 columns**, 28 players, dates 2024-07-16 to 2025-06-26

This is the **primary artifact** consumed at both training and inference time.

### 4.5 Day-of-week feature engineering

Feature engineering adds a single calendar feature:

```python
d["day_of_week"] = d["date"].dt.dayofweek   # 0 = Monday … 6 = Sunday
```

This is then one-hot encoded with fixed categories to produce `dow_0 … dow_6`, ensuring a consistent column schema regardless of which days appear in a given split:

```python
dow     = pd.Categorical(df["day_of_week"], categories=range(7))
dummies = pd.get_dummies(dow, prefix="dow").astype(int)
```

---

## 5. Full-calendar reconstruction for ACWR

### 5.1 Why rest-day filling is mandatory

ACWR is defined on a continuous daily timeline where rest days contribute zeros and influence EWMA decay. If missing days were skipped rather than filled with zeros:

- chronic load would decay too slowly,
- acute load would remain artificially elevated,
- the ACWR ratio would not reflect actual calendar exposure.

### 5.2 Current ACWR state used by the dashboard

At application load time, `load_player_data()` (`src/app/loaders.py`) reads `daily.parquet` and for each player and metric:

1. extracts the complete daily load series from the full-calendar grid,
2. runs `compute_acwr(loads)` — the EWMA-ACWR computation,
3. drops warmup `NaN` values,
4. takes the last valid ACWR value,
5. maps that value to an operational risk zone.

The dashboard and the forecast page share the same ACWR engine. The forecast simply appends predicted loads before re-running the same computation.

---

## 6. Feature vector used by the deployed models

### 6.1 Feature set

The production models use a **cross-sectional** feature set — no lag, EWMA, or microcycle features. All features are known before the session starts:

| Feature group | Features |
|---|---|
| **Session / activity** | `n_periods`, `n_exercise_types`, `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH` |
| **Player anthropometrics** | `height`, `weight`, `age` |
| **Calendar** | `dow_0`, `dow_1`, `dow_2`, `dow_3`, `dow_4`, `dow_5`, `dow_6` (day-of-week OHE) |
| **Cross-metric load** | `total_distance` (for `accelerations` and `sprint_distance` models only) |

### 6.2 Per-target feature counts

`scale_train()` drops the target column, `player_id`, `position`, `date`, and `total_distance` (when not the target) before selecting `feature_cols` from all remaining numeric columns.

| Target | Feature count | Includes `total_distance`? |
|---|---|---|
| `total_distance` | **17** | No (is the target) |
| `accelerations` | **18** | Yes (cross-metric covariate, set to 0 at inference) |
| `sprint_distance` | **18** | Yes (cross-metric covariate, set to 0 at inference) |

The exact ordered feature list is persisted inside `bundle.joblib` as `feature_cols` and is used to align the inference matrix at serving time.

### 6.3 What is intentionally absent

The deployed feature set does **not** include:

- lag features (`load_lag_*`) — not applicable to a cross-sectional approach,
- EWMA load states (`acute_load`, `chronic_load`) — these are computed downstream for ACWR only, not as model inputs,
- microcycle statistics (`monotony`, `strain`) — removed for simplicity and reduced overfitting risk,
- player identity encodings (`pid_*`) — not needed when features are session-level,
- position one-hot encodings — `position` is dropped as a non-numeric string column.

---

## 7. Model architecture

### 7.1 One model per target

The system trains **three independent XGBoost regressors** — one each for `total_distance`, `accelerations`, and `sprint_distance`. This treats each load component as a distinct physiological response to session composition.

### 7.2 Functional form

For a given target `m`, the model is a boosted additive tree ensemble:

`f_m(x) = Σ_{k=1..K} η · b_k(x)`

where `b_k` is the k-th regression tree, `K` is the number of boosting rounds, and `η` is the learning rate. XGBoost minimizes a regularized objective:

`L = Σ_i l(y_i, ŷ_i) + Σ_k Ω(b_k)`

The deployed training configuration uses:

- `objective = "reg:squarederror"` (on the log1p-transformed target),
- `tree_method = "hist"`,
- `random_state = 42`,
- `n_jobs = -1`.

### 7.3 Why tree boosting fits this problem

Gradient-boosted trees naturally handle:

- sparse binary indicators (`has_MATCH`, `has_G`, etc.),
- non-linear interactions (e.g. a match on a Monday vs a Friday),
- mixed feature scales (age in years, height in cm, binary flags).

---

## 8. Target-specific formulations

### 8.1 Uniform log1p transform

All three targets are right-skewed and non-negative. The training pipeline applies the same transform to every target:

```python
train[target] = np.log1p(train[target])
test[target]  = np.log1p(test[target])
```

At inference time, predictions are inverted and clipped:

```python
pred = np.clip(np.expm1(model.predict(X_scaled)), 0, None)
```

This approach is consistent across all three targets; there are no per-target objective differences.

### 8.2 Feature scaling

Before fitting, all feature columns are scaled with `MinMaxScaler`:

```python
scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols]  = scaler.transform(test[feature_cols])
```

The fitted scaler is saved in `bundle.joblib` alongside the model and is applied identically at inference time.

---

## 9. Training procedure

### 9.1 Train/test split

The dataset is split randomly at the row level:

```python
train_base, test_base = train_test_split(daily, test_size=0.2, random_state=42)
```

This matches the notebook experimental setup. An 80/20 random split is used consistently across all three targets.

### 9.2 Cross-validation

Within the training set, hyperparameter search uses **10-fold KFold** (shuffled):

```python
cv10 = KFold(n_splits=10, shuffle=True, random_state=42)
```

### 9.3 Hyperparameter search

`RandomizedSearchCV` is used with **50 iterations** per target over a wide continuous search space:

| Parameter | Distribution |
|---|---|
| `n_estimators` | randint(200, 3000) |
| `max_depth` | randint(2, 16) |
| `max_leaves` | randint(0, 1024) |
| `min_child_weight` | loguniform(0.1, 500) |
| `gamma` | loguniform(1e-6, 100) |
| `subsample` | uniform(0.3, 0.7) |
| `colsample_bytree` | uniform(0.3, 0.7) |
| `colsample_bylevel` | uniform(0.3, 0.7) |
| `colsample_bynode` | uniform(0.3, 0.7) |
| `learning_rate` | loguniform(1e-3, 5e-1) |
| `reg_alpha` | loguniform(1e-8, 100) |
| `reg_lambda` | loguniform(1e-8, 100) |
| `grow_policy` | ["depthwise", "lossguide"] |
| `max_bin` | randint(64, 1024) |
| `max_delta_step` | randint(0, 10) |

Scoring: `neg_mean_squared_error` on the log1p-transformed target. `error_score=np.nan` silently skips numerically unstable combinations. The best estimator is refitted on the full training set before evaluation.

### 9.4 Test-set evaluation

After fitting, the model is evaluated on the held-out test set in the original (non-log) space:

```python
y_pred = np.clip(np.expm1(model.predict(X_te)), 0, None)
y_true = np.expm1(y_te)
```

Metrics reported: RMSE, MAE, R², and CV RMSE (on the log1p scale).

---

## 10. Model artifacts and serving contract

### 10.1 Artifact format

Each target produces one file under `models/xgboost/{target}/`:

| File | Content |
|---|---|
| `bundle.joblib` | joblib-serialised dict: `model` (XGBRegressor), `scaler` (MinMaxScaler), `feature_cols` (list[str]), `ewma_spans` (dict) |

The `ewma_spans` dict is `{"acute": 7, "chronic": 28}` for all three targets (stored for documentation; not used during inference since EWMA is computed downstream on the ACWR series).

### 10.2 Artifact validation at serving time

Before any inference, `src/real_madrid_acwr/modeling/artifacts.py` validates each bundle against a strict contract. For every target, the loader checks:

- the bundle file exists and can be loaded,
- all four keys (`model`, `scaler`, `feature_cols`, `ewma_spans`) are present,
- `feature_cols` is a non-empty list of strings with no duplicates,
- `ewma_spans` is a dict with exactly the keys `"acute"` and `"chronic"`.

Validation failures raise `ArtifactLoadError` with a full error list before the Streamlit app renders any content.

### 10.3 Feature alignment at inference

The serving code reads `feature_cols` from the bundle and uses it to extract and order the inference matrix:

```python
featurized = encode_dow(add_features(plan_frame))
for col in feature_cols:
    if col not in featurized.columns:
        featurized[col] = 0.0
X_raw    = featurized[feature_cols].values
X_scaled = scaler.transform(X_raw)
pred     = np.clip(np.expm1(model.predict(X_scaled)), 0, None)
```

Any feature column missing from the inference frame is filled with `0.0` before extraction.

---

## 11. ACWR methodology

### 11.1 Acute and chronic EWMAs

For a player's daily load series `load[t]`, the system computes two uncoupled exponentially weighted moving averages.

**Acute load** (α = 2/(7+1) = 0.25):

`acute[t] = α_a · load[t] + (1 − α_a) · acute[t−1]`

**Chronic load** (α = 2/(28+1) ≈ 0.069):

`chronic[t] = α_c · load[t] + (1 − α_c) · chronic[t−1]`

Both are initialised at zero and computed from the same raw load series independently (uncoupled formulation).

### 11.2 ACWR ratio

`ACWR[t] = acute[t] / chronic[t]`

Safeguards:

- if `chronic[t] == 0`, ACWR is set to `NaN`,
- the first 28 days are masked as warmup (chronic not yet stable).

### 11.3 Why EWMA instead of rolling averages

EWMA gives exponentially higher weight to recent sessions and handles rest-day decay more smoothly than simple rolling averages. This matches the Williams et al. (2017) formulation cited in sport-science literature.

### 11.4 Risk zone classification

| Zone | ACWR range | Interpretation |
|---|---|---|
| `undertraining` | < 0.8 | Insufficient load stimulus |
| `optimal` | 0.8 ≤ ACWR < 1.3 | Safe training range |
| `caution` | 1.3 ≤ ACWR < 1.5 | Elevated risk — monitor closely |
| `danger` | ≥ 1.5 | High injury risk — reduce load |

If the value is unavailable after warmup masking, the app reports `unknown`. These thresholds are heuristic sport-science conventions, not a medical diagnosis.

---

## 12. Forecast-generation algorithm

The forecast engine lives in `src/app/forecasting.py`.

### 12.1 Inputs

The forecasting function receives a 15-element list of daily plan dicts, each produced by collapsing the coach's calendar events via `build_plan_days_from_events()`:

```python
{
    "is_rest": False,
    "G": True,
    "TAC": False,
    "BP": True,
    "TEC": False,
    "MATCH": False,
}
```

### 12.2 Plan frame construction

For each target, a plan DataFrame is built with one row per (player, plan day):

```python
row = {
    "player_id":        pid,
    "date":             last_active + timedelta(days=d),
    "n_periods":        0 if is_rest else 1,
    "n_exercise_types": count of active session types,
    "height":           player profile value,
    "weight":           player profile value,
    "age":              player profile value,
    "has_G": ..., "has_TAC": ..., ...   # from plan flags
}
```

### 12.3 Direct single-pass inference

All 15 plan days × all 28 players are predicted in a **single model call** (`_direct_forecast`):

```python
featurized = encode_dow(add_features(plan_frame))
for col in feature_cols:
    if col not in featurized.columns:
        featurized[col] = 0.0
X_raw    = featurized[feature_cols].values
X_scaled = scaler.transform(X_raw)
pred     = np.clip(np.expm1(model.predict(X_scaled)), 0, None)
pred[featurized["n_periods"].values == 0] = 0.0   # rest days → 0
```

There is no day-by-day loop and no feedback of predictions into subsequent days. Because the feature set is cross-sectional (session flags, anthropometrics, day of week), each day's prediction is fully independent.

### 12.4 From predicted loads to forecast ACWR

After all 15 daily loads are predicted for a player and metric:

1. Retrieve the historical load series from `player_data[pid]["grid"]`.
2. Append the 15 predicted loads.
3. Call `compute_acwr_with_forecast(hist_loads, fore_loads)`.
4. Slice the last 60 historical days plus the 15 forecast days for charting.
5. Extract `day15_acwr` (the last valid ACWR in the forecast segment) and classify its zone.

The stitched computation:

```
load*[1:H+15] = [h[1], ..., h[H], f[1], ..., f[15]]
```

runs the same EWMA recursion over the concatenated series. Historical momentum carries naturally into the forecast region without any manual state-seeding.

**Key modeling idea:** the predictive model forecasts **load**, not ACWR directly. ACWR is a deterministic downstream transformation of the load path.

---

## 13. End-to-end system flow

```
data/raw/data_acute_vs_chronic.csv  (extracted from ZIP if needed)
         │
         ▼  python train_models.py
         │    load_data()          — read CSV / extract ZIP
         │    clean_data()         — rename columns, parse dates, drop missing
         │    treat_outliers()     — IQR-cap per player per exercise type
         │    build_full_daily()   — aggregate all 3 metrics, fill rest-day spine
         │    → save daily.parquet (6,310 rows × 16 cols, all 28 players)
         │
         │    For each target (accelerations, sprint_distance, total_distance):
         │      run_pipeline(target)  — data → clean → outliers → aggregate → spine → split → scale
         │      RandomizedSearchCV   — 50-iter search, 10-fold KFold
         │      → save bundle.joblib — model, scaler, feature_cols, ewma_spans
         │
         ▼  streamlit run main.py
         │    loaders.load_models()       — @st.cache_resource: loads 3 bundles, validates contract
         │    loaders.load_player_data()  — @st.cache_resource: reads daily.parquet,
         │                                  computes current ACWR per player × metric
         │
         ▼  Dashboard page
         │    shows current ACWR per player × 3 metrics, risk zone KPIs
         │
         ▼  Planning & Forecast page
              coach adds events to FullCalendar
              events → build_plan_days_from_events() → 15-day bool flag list
              build_forecast(plan_days)
                → _build_plan_frame()    — one row per (player × day)
                → _direct_forecast()    — single-pass XGBoost inference for all rows
                → compute_acwr_with_forecast() — EWMA on stitched history + predicted loads
              render per-player ACWR charts + day-15 summary table
```

Compact mathematical view:

`planned sessions → session flags → feature vectors → predicted loads → EWMA acute/chronic → ACWR → risk zone`

---

## 14. File-level mapping of responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Streamlit entry point, page dispatch, nav state management |
| `src/app/loaders.py` | Load model bundles and player data; compute current ACWR |
| `src/app/forecasting.py` | Direct load prediction and ACWR stitching |
| `src/app/planning.py` | Event model helpers, plan → daily session flags, stale-detection signature |
| `src/app/pages.py` | All Streamlit page renderers (dashboard, planner, sidebar) |
| `src/app/charts.py` | Plotly ACWR chart builder |
| `src/app/constants.py` | Domain constants, color palettes, full ENG/ESP translation table |
| `src/app/i18n.py` | `t()` translation lookup, date formatting helpers |
| `src/app/styles.py` | CSS injection (Real Madrid brand styling) |
| `src/real_madrid_acwr/acwr.py` | Pure EWMA-ACWR computation and zone classification |
| `src/real_madrid_acwr/config.py` | Shared Path constants |
| `src/real_madrid_acwr/modeling/artifacts.py` | Bundle loader with contract validation |
| `src/real_madrid_acwr/modeling/datapipeline.py` | Shared preprocessing pipeline (used by app and training) |
| `src/real_madrid_acwr/modeling/train.py` | Compatibility shim → `training/train.py` |
| `src/real_madrid_acwr/modeling/training/train.py` | Orchestrator: saves daily.parquet, calls all three model trainers |
| `src/real_madrid_acwr/modeling/training/acceleration_model_train.py` | XGBoost training for `accelerations` |
| `src/real_madrid_acwr/modeling/training/sprint_distance_model_train.py` | XGBoost training for `sprint_distance` |
| `src/real_madrid_acwr/modeling/training/total_distance_model_train.py` | XGBoost training for `total_distance` |
| `train_models.py` | CLI entry point — calls `training/train.py:main()` |
| `data_decisions.md` | Cleaning and methodology rationale |

---

## 15. Assumptions and limitations

### 15.1 Data limitations

- Only one season (2024–25) is available. Long-run trends, seasonal effects, and inter-season recovery are not captured.
- There is no session duration, RPE (rating of perceived exertion), or wellness data.
- Rest days and tracking gaps are observationally indistinguishable in the raw data.

### 15.2 Modeling limitations

- The feature set is cross-sectional: session flags, anthropometrics, and day of week. The model has no access to a player's recent load history at inference time.
- For the `accelerations` and `sprint_distance` models, `total_distance` is included as a cross-metric feature but is set to `0.0` at inference (since it is not predicted at the same time). This differs from training where all three metrics were observed together.
- The models do not encode player identity. Systematic load differences between players must be captured indirectly through anthropometrics and session type responses.
- `sprint_distance` is the noisiest and sparsest of the three targets and is expected to have lower predictive accuracy.

### 15.3 ACWR limitations

- ACWR is a workload proxy, not an injury diagnosis.
- Risk zone thresholds (0.8, 1.3, 1.5) are heuristic sport-science conventions. Their clinical validity in football-specific contexts is debated.
- The model predicts load and propagates it into ACWR; it does not predict injury probability directly.

---

## 16. Practical interpretation

The system operates as a two-stage engine:

**Stage 1 — Session-to-load model**

Given a player's physical profile (height, weight, age) and a planned session composition (session type flags, day of week), estimate how much external load that session day is likely to generate. The model is trained on historical player-day observations and learns the expected load response to different session configurations.

**Stage 2 — Load-to-ACWR simulator**

Given the player's full historical load series and the 15 predicted future loads, propagate the EWMA state forward and determine whether the player enters a safe, caution, or danger zone at day 15.

This separation keeps the logic transparent:

- the predictive model handles **session composition → expected load**,
- the ACWR equations handle **physiological workload accumulation over time**.

Coaches can interrogate both layers independently: if a player's forecasted ACWR looks alarming, they can inspect which planned sessions drive the predicted load spike and modify the schedule before the sessions occur.

---

## 17. Reproducing the deployed artifacts

From a clean checkout:

```bash
# 1. Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install runtime + dev dependencies
pip install -e ".[dev]"

# 3. Train models and produce daily.parquet
#    (extracts the raw CSV from data/data_acute_vs_chronic.zip automatically)
python train_models.py

# 4. Launch the app
streamlit run main.py
```

`train_models.py` is a self-contained CLI that handles data extraction, preprocessing, daily-grid construction, feature engineering, model training, and artifact saving. No notebook execution is required for production use. The notebooks under `notebooks/` are for EDA and model exploration only.

---

## 18. Summary

In concrete terms, this project does the following:

1. Converts period-level GPS training logs into a continuous player-day grid with zero-load rest days.
2. Applies IQR-based outlier capping per player per session type.
3. Engineers a cross-sectional feature set: session composition flags, player anthropometrics, and day-of-week (one-hot).
4. Trains three independent XGBoost regressors — one per load metric — using random 80/20 splits and 10-fold KFold CV with 50-iteration randomised hyperparameter search.
5. Saves each model, its MinMaxScaler, and its ordered feature list as a single `bundle.joblib` artifact.
6. At inference time, builds a 15-day plan frame (one row per player × day), featurises all rows at once, and predicts all loads in a single model call.
7. Propagates the combined load series (history + predictions) through the 7-day vs 28-day EWMA ACWR model.
8. Presents the resulting risk trajectory and day-15 zone to coaches in an interactive Streamlit application.

The core system is a **cross-sectional load prediction + ACWR simulation pipeline** — the model responds to session plan inputs, and ACWR converts those load predictions into a clinically interpretable risk metric.
