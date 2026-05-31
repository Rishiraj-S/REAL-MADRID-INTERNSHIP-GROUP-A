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

where `x[p, t]` is a feature vector composed of:

- static player attributes (height, weight, age),
- session-type indicators for the planned day,
- calendar features (day of week),
- time-series features derived from the player's load history (lags, rolling means, EWMA states, microcycle statistics),
- the other two load metrics where available.

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

A continuous age feature preserves more information than integer-rounded age and ensures the model can use age at the exact point of each session.

### 3.4 Player exclusions

Players are excluded for one or more of the following reasons:

- implausible placeholder metadata (e.g. `weight == 200`),
- missing all core player metadata (height, weight, position, date of birth),
- preseason-only presence.

The final deployed cohort contains **28 players**.

### 3.5 Row-level outlier treatment

One extreme outlier is explicitly corrected:

- player `94884`, `total_distance ≥ 20,000` metres on a single match period.

The correction strategy:

```python
median_94884 = df[
    (df["player_id"] == 94884) &
    (df["exercise_type"] == "MATCH") &
    (df["total_distance"] <= 20000)
]["total_distance"].median()

df.loc[(df["player_id"] == 94884) & (df["total_distance"] >= 20000), "total_distance"] = median_94884
```

This preserves the player-day event while avoiding contamination of both the predictive model and downstream EWMA calculations.

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

If a session type does not appear in the data, its column is added with all zeros. If a player-day contains both `TAC` and `BP` work, then `has_TAC = 1`, `has_BP = 1`.

### 4.3 Full calendar spine reconstruction

After aggregating active days, the pipeline builds a **continuous daily grid** per player — one row per calendar day from each player's first to last active date. Gaps (rest days) are filled with:

- `0` for all load metrics and session flags,
- `0` for `n_periods` and `n_exercise_types`,
- forward-then-backward fill for `height`, `weight`, `age`, `position`.

This spine reconstruction ensures rest days contribute to EWMA decay correctly. Without it, chronic load would decay too slowly and ACWR would be systematically overstated.

### 4.4 Persisted daily table

The full-calendar daily frame is saved at:

- `data/processed/daily.parquet` — **6,310 rows × 16 columns**, 28 players, dates 2024-07-16 to 2025-06-26

This is the **primary artifact** consumed at both training and inference time.

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

### 6.1 Feature engineering

Feature engineering is applied inside `_add_features(daily_df, target)` in `src/real_madrid_acwr/modeling/train.py`. All lag and rolling features are computed with a **one-day shift** (`shift(1)`) to prevent target leakage.

| Feature group | Features |
|---|---|
| **Calendar** | `day_of_week` |
| **Session / activity** | `n_periods`, `n_exercise_types`, `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH` |
| **Player anthropometrics** | `height`, `weight`, `age` |
| **EWMA states** | `acute_load` (span=7, shifted), `chronic_load` (span=28, shifted), `training_stress_balance` (chronic − acute), `acwr` (acute / chronic) |
| **Microcycle statistics** | `microcycle_load_sum`, `microcycle_load_std_dev`, `monotony` (mean/std), `strain` (sum × monotony) — computed per player per ISO week, shifted |
| **Lag features** | `load_lag_1`, `load_lag_3`, `load_lag_5`, `load_lag_7`, `load_lag_14` |
| **Rolling means** | `load_ma_3`, `load_ma_7`, `load_ma_14` |
| **Cross-metric loads** | Other load metrics where not the target (see §6.2) |

All NaN values in numeric columns are filled with `0` after feature construction.

### 6.2 Per-target feature counts

Each model drops its own target column plus `player_id`, `position`, and `date` before selecting `feature_cols`. Additionally, `total_distance` is dropped from the feature set of the `accelerations` and `sprint_distance` models to avoid a dominant shortcut feature.

| Target | Feature count | Cross-metric features included |
|---|---|---|
| `total_distance` | **29** | `accelerations`, `sprint_distance` |
| `accelerations` | **28** | `sprint_distance` |
| `sprint_distance` | **28** | `accelerations` |

The exact ordered feature list is persisted inside `bundle.joblib` as `feature_cols` and is used to align the inference matrix at serving time.

### 6.3 What is intentionally absent

The deployed feature set does **not** include:

- player identity one-hot encodings (`pid_*`) — the time-series features (lags, EWMA) carry sufficient player-specific signal,
- position one-hot encodings — `position` is dropped as a non-numeric column; session-type flags and EWMA states encode positional load differences implicitly,
- calendar distance features such as `days_since_last_activity` or `days_since_last_match` — these were used in notebook experiments but are not part of the production artifact.

---

## 7. Model architecture

### 7.1 One model per target

The system trains **three independent XGBoost regressors** — one each for `total_distance`, `accelerations`, and `sprint_distance`. A single global model per target is preferred over hierarchical or player-stratified approaches because the time-series features (lags, EWMA states) already capture player-specific load patterns without explicit identity encoding.

### 7.2 Functional form

For a given target `m`, the model is a boosted additive tree ensemble:

`f_m(x) = Σ_{k=1..K} η · b_k(x)`

where `b_k` is the k-th regression tree, `K` is the number of boosting rounds, and `η` is the learning rate. XGBoost minimizes a regularized objective:

`L = Σ_i l(y_i, ŷ_i) + Σ_k Ω(b_k)`

where `l` is the loss function and `Ω` is the tree-complexity regulariser. The deployed training configuration uses:

- `objective = "reg:squarederror"` (after log1p transform — see §8),
- `tree_method = "hist"`,
- `random_state = 42`,
- `n_jobs = -1`.

### 7.3 Why tree boosting is appropriate here

Gradient-boosted trees handle the mix of continuous features (age, EWMA states), sparse binary indicators (session flags), and non-linear interactions (e.g. a match following consecutive rest days) without manual interaction engineering.

---

## 8. Target-specific formulations

### 8.1 Uniform log1p transform

All three targets are right-skewed and non-negative. The training pipeline applies the same transform to every target:

```python
train[target] = np.log1p(train[target])
test[target]  = np.log1p(test[target])
```

The model minimises squared error in log-space. At inference time, predictions are inverted and clipped:

```python
pred = np.clip(np.expm1(model.predict(X_scaled)), 0, None)
```

The `max(·, 0)` clipping guarantees non-negative output. This approach is consistent across all three targets; there are no per-target objective differences in the deployed system.

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

### 9.1 Chronological train/test split

The dataset is split by date, not by row index or random sampling:

```python
all_dates = np.sort(daily["date"].unique())
cutoff    = all_dates[int(len(all_dates) * 0.8) - 1]
train_base = daily[daily["date"] <= cutoff]
test_base  = daily[daily["date"] >  cutoff]
```

The first 80% of calendar dates form the training set; the last 20% form the held-out test set. This respects the temporal ordering of the data and prevents future-leaking into training.

### 9.2 Date-blocked cross-validation

Within the training set, hyperparameter search uses a custom date-blocked CV scheme:

```python
def _date_blocked_cv(dates, n_splits=4):
    unique_dates = np.sort(pd.unique(dates))
    blocks = np.array_split(unique_dates, n_splits + 1)
    for i in range(1, n_splits + 1):
        train_dates = np.concatenate(blocks[:i])
        val_dates   = blocks[i]
        ...
```

Each fold expands the training window by one block and validates on the next block — an expanding-window walk-forward scheme. This prevents any data-leakage within cross-validation.

### 9.3 Hyperparameter search

`RandomizedSearchCV` is used with 25 iterations per target over the following search space:

| Parameter | Options |
|---|---|
| `n_estimators` | 200, 400, 600, 800 |
| `learning_rate` | 0.01, 0.02, 0.05, 0.1 |
| `max_depth` | 3, 4, 5, 6, 8 |
| `min_child_weight` | 1, 3, 5, 10 |
| `subsample` | 0.6, 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.5, 0.6, 0.7, 1.0 |
| `gamma` | 0, 0.1, 0.3, 0.5 |
| `reg_lambda` | 0.1, 0.5, 1.0, 5.0 |

Scoring: `neg_mean_squared_error`. The best estimator is refitted on the full training set before evaluation.

### 9.4 Test-set evaluation

After fitting, the model is evaluated on the held-out test set in the original (non-log) space:

```python
y_pred = np.clip(np.expm1(model.predict(X_te)), 0, None)
y_true = np.expm1(y_te)
mae    = mean_absolute_error(y_true, y_pred)
r2     = r2_score(y_true, y_pred)
```

---

## 10. Model artifacts and serving contract

### 10.1 Artifact format

Each target produces one file under `models/xgboost/{target}/`:

| File | Content |
|---|---|
| `bundle.joblib` | `joblib`-serialised dict: `model` (XGBRegressor), `scaler` (MinMaxScaler), `feature_cols` (list[str]), `ewma_spans` (dict) |

The `ewma_spans` dict is `{"acute": 7, "chronic": 28}` for all three targets.

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
X_raw    = day_rows[feature_cols].values
X_scaled = scaler.transform(X_raw)
pred     = np.clip(np.expm1(model.predict(X_scaled)), 0, None)
```

Any feature column missing from the inference frame is filled with `0.0` before extraction. This handles the case where the plan frame lacks certain columns because no history exists for a particular session type.

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

The forecasting loop receives a 15-element list of daily plan dicts, each produced by collapsing the coach's calendar events via `build_plan_days_from_events()`:

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
for d, day in enumerate(plan_days, start=1):
    date = last_active + pd.Timedelta(days=d)
    is_rest = day.get("is_rest", False)
    row = {
        "player_id": pid,
        "date": date,
        "n_periods": 0 if is_rest else 1,
        "n_exercise_types": count of active session types,
        "height": ..., "weight": ..., "age": ..., "position": ...,
        target: np.nan,          # to be predicted
        other_metric: 0.0,       # cross-metric features
        "has_G": ..., "has_TAC": ..., ...
    }
```

The target column is `NaN` in the plan frame; cross-metric load columns are set to `0.0` for plan days.

### 12.3 Recursive one-step-ahead inference

The core loop (`_recursive_forecast`) iterates over plan dates in order:

```python
for current_date in sorted(plan_work["date"].unique()):
    combined   = pd.concat([history, plan_work], ignore_index=True)
    featurized = _add_features(combined, target)   # recompute all features on full frame

    day_rows   = featurized[featurized["date"] == current_date]
    X_raw      = day_rows[feature_cols].values
    X_scaled   = scaler.transform(X_raw)
    pred       = np.clip(np.expm1(model.predict(X_scaled)), 0, None)

    # Rest days always produce 0 load
    pred[day_rows["n_periods"].values == 0] = 0.0

    # Write predictions back into plan frame for the next iteration
    plan_work.loc[date == current_date & player == pid, target] = val
```

**Why recursive?** Lag features (`load_lag_1`, `load_lag_3`, …) and EWMA states (`acute_load`, `chronic_load`) for day `d` depend on the target load on day `d−1` (and earlier). Feeding predictions back into the frame allows these features to propagate load shocks forward across the 15-day window. A batch (non-recursive) approach would leave all future-day lag features as `NaN` or stale.

`_add_features` is called on the **combined** (history + plan so far) frame on every iteration, so all time-series features are recomputed with the most recent predicted values. This is the most expensive part of inference: O(15 × n_players × feature_engineering_cost).

### 12.4 From predicted loads to forecast ACWR

After all 15 daily loads are predicted for a player and metric:

1. Retrieve the historical load series from the full player grid.
2. Append the 15 predicted loads.
3. Call `compute_acwr_with_forecast(hist_loads, fore_loads)`.
4. Slice the last 60 historical days plus the 15 forecast days for charting.
5. Extract `day15_acwr` (the last valid ACWR in the forecast segment) and classify its zone.

The stitched computation:

```
load*[1:H+15] = [h[1], ..., h[H], f[1], ..., f[15]]
```

runs the same EWMA recursion over the concatenated series. The EWMA state carries historical momentum naturally into the forecast region — no manual state-seeding is required.

**Key modeling idea:** the predictive model forecasts **load**, not ACWR directly. ACWR is a deterministic downstream transformation of the load path. This makes the system interpretable: coaches can see both the predicted load trajectory and the resulting ACWR, and understand why a particular ACWR zone is reached.

---

## 13. End-to-end system flow

```
data/raw/data_acute_vs_chronic.csv  (extracted from ZIP if needed)
         │
         ▼  python train_models.py
         │    _preprocess()       — rename columns, clean outliers, strip trialists
         │    _build_daily()      — aggregate to player-day, fill rest days with zero
         │    _add_features()     — lags, MAs, EWMA states, microcycle stats
         │    RandomizedSearchCV  — 25-iter hyperparameter search, date-blocked CV
         │    save bundle.joblib  — model, scaler, feature_cols, ewma_spans (× 3 targets)
         │    save daily.parquet  — 6,310 rows, full calendar grid for all 28 players
         │
         ▼  streamlit run main.py
         │    loaders.load_models()       — @st.cache_resource: loads 3 bundles, validates contract
         │    loaders.load_player_data()  — @st.cache_resource: reads daily.parquet,
         │                                  rebuilds per-player grids, computes current ACWR
         │
         ▼  Dashboard page
         │    shows current ACWR per player × 3 metrics, risk zone KPIs
         │
         ▼  Planning & Forecast page
         │    coach adds events to FullCalendar
         │    events → build_plan_days_from_events() → 15-day bool flag list
         │    build_forecast(plan_days)
         │      → _build_plan_frame()        — one row per (player, day)
         │      → _recursive_forecast()      — day-by-day XGBoost inference
         │      → compute_acwr_with_forecast() — EWMA on stitched series
         │    render per-player ACWR charts + day-15 summary table
```

A compact mathematical view:

`planned sessions → session flags → feature vectors → predicted loads → EWMA acute/chronic → ACWR → risk zone`

---

## 14. File-level mapping of responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Streamlit entry point, page dispatch, nav state management |
| `src/app/loaders.py` | Load model bundles and player data; compute current ACWR |
| `src/app/forecasting.py` | Recursive load forecasting and ACWR stitching |
| `src/app/planning.py` | Event model helpers, plan → daily session flags, stale-detection signature |
| `src/app/pages.py` | All Streamlit page renderers (dashboard, planner, sidebar) |
| `src/app/charts.py` | Plotly ACWR chart builder |
| `src/app/constants.py` | Domain constants, color palettes, full ENG/ESP translation table |
| `src/app/i18n.py` | `t()` translation lookup, date formatting helpers |
| `src/app/styles.py` | CSS injection (Real Madrid brand styling) |
| `src/real_madrid_acwr/acwr.py` | Pure EWMA-ACWR computation and zone classification |
| `src/real_madrid_acwr/config.py` | Shared Path constants |
| `src/real_madrid_acwr/modeling/train.py` | Full training pipeline: preprocess → daily build → feature engineering → CV → XGBoost → save |
| `src/real_madrid_acwr/modeling/artifacts.py` | Bundle loader with contract validation |
| `train_models.py` | CLI wrapper — calls `train.py:main()` |
| `data_decisions.md` | Cleaning and methodology rationale |

---

## 15. Assumptions and limitations

### 15.1 Data limitations

- Only one season (2024–25) is available. Long-run trends, seasonal effects, and inter-season recovery are not captured.
- There is no session duration, RPE (rating of perceived exertion), or wellness data.
- Rest days and tracking gaps are observationally indistinguishable in the raw data.
- Cross-metric load columns are set to `0.0` for forecast days where the other targets have not yet been predicted. This differs from the training distribution where all three metrics are observed simultaneously.

### 15.2 Modeling limitations

- Hyperparameters are selected via 25-iteration randomised search with date-blocked CV. The search space is coarse and a more exhaustive search could improve performance.
- The models do not encode player identity explicitly. If a player's load profile is systematically unlike others in the training set, the EWMA lag features may not fully compensate.
- `sprint_distance` is the noisiest and sparsest of the three targets and is expected to have lower predictive accuracy than `total_distance`.

### 15.3 Forecast limitations

- The recursive forecasting loop recomputes features from the combined history+plan frame on every day. Cross-metric loads for plan days are set to `0.0`, which can underestimate features like `load_lag_1` derived from the other two metrics on active plan days.
- Forecast quality degrades the further ahead the horizon extends, as prediction errors compound through lag features.

### 15.4 ACWR limitations

- ACWR is a workload proxy, not an injury diagnosis.
- Risk zone thresholds (0.8, 1.3, 1.5) are heuristic sport-science conventions. Their clinical validity in football-specific contexts is debated.
- The model predicts load and propagates it into ACWR; it does not predict injury probability directly.

---

## 16. Practical interpretation

The system operates as a two-stage engine:

**Stage 1 — Session-to-load model**

Given a player's recent load history (via lag/EWMA features) and a planned session composition (via session type flags), estimate how much external load that session day is likely to generate.

**Stage 2 — Load-to-ACWR simulator**

Given the player's full historical load series and the 15 predicted future loads, propagate the EWMA state forward and determine whether the player enters a safe, caution, or danger zone at day 15.

This separation is important because it keeps the logic transparent:

- the predictive model handles **behavioural/physical response to a planned session**,
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

`train_models.py` is a self-contained CLI that handles data extraction, preprocessing, daily-grid construction, feature engineering, model training, and artifact saving. No notebook execution is required for production use. The notebooks under `notebooks/` are for EDA and exploration only.

---

## 18. Summary

In concrete terms, this project does the following:

1. Converts period-level GPS training logs into a continuous player-day grid with zero-load rest days.
2. Engineers time-series features (lags, rolling means, EWMA states, microcycle statistics) for each player-day.
3. Trains three independent XGBoost regressors — one per load metric — using chronological train/test splits and date-blocked CV.
4. Saves each model, its MinMaxScaler, and its ordered feature list as a single `bundle.joblib` artifact.
5. At inference time, builds a 15-day plan frame, recursively predicts daily loads by feeding each day's prediction back as lag input for the next day, and stitches the predicted loads onto each player's historical series.
6. Propagates the combined load series through the 7-day vs 28-day EWMA ACWR model.
7. Presents the resulting risk trajectory and day-15 zone to coaches in an interactive Streamlit application.

The core technical object is not just an ACWR calculator; it is a **recursive load forecasting + ACWR simulation system** where the predictive model and the physiological accumulation model are distinct, composable layers.
