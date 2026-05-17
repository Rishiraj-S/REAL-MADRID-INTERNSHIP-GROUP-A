# Technical Documentation — Real Madrid ACWR Prediction Tool

This document is the technical reference for the deployed ACWR forecasting system. It explains:

- the operational problem the tool solves,
- the meaning and limitations of the raw data,
- the cleaning and feature-engineering pipeline,
- the mathematical and statistical design of the predictive models,
- the conversion from predicted loads to forecast ACWR trajectories.

Production behavior is defined by the Python code under `src/` and the deployable artifacts under `models/xgboost/`. Notebook material is cited only when it explains why a deployed modeling choice was made.

## Table of contents

- [1. Problem description](#1-problem-description)
- [2. Data description](#2-data-description)
- [3. Cleaning and cohort definition](#3-cleaning-and-cohort-definition)
- [4. Data processing pipeline](#4-data-processing-pipeline)
- [5. Full-calendar reconstruction for ACWR](#5-full-calendar-reconstruction-for-acwr)
- [6. Feature vector used by the deployed models](#6-feature-vector-used-by-the-deployed-models)
- [7. Model architecture](#7-model-architecture)
- [8. Target-specific formulations](#8-target-specific-formulations)
- [9. Deployed hyperparameters](#9-deployed-hyperparameters)
- [10. Training and serving procedure](#10-training-and-serving-procedure)
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
The coaches do **not** directly enter expected numeric load values such as metres or acceleration counts. Instead, they enter a **session composition plan** day by day:

- `G` — game-based / small-sided game,
- `TAC` — tactical,
- `BP` — set pieces,
- `TEC` — technical,
- `MATCH` — official match,
- `REST` — no activity.

From this plan, the system predicts for each squad player:

1. the future daily external load for three metrics,
2. the future acute and chronic workload states,
3. the resulting ACWR trajectory,
4. the final risk zone at the end of the forecast horizon.

### 1.2 Formal problem statement

For each player `p`, for each day `t` in a 15-day horizon, and for each metric `m` in

- `total_distance`,
- `acc_total`,
- `vel_total`,

we want to estimate a future load

`ŷ[p, t, m] = f_m(x[p, t])`

where `x[p, t]` is a feature vector composed of:

- static player attributes,
- session-type indicators for the planned day,
- calendar/history features,
- a player identity one-hot encoding.

The predicted loads are then stitched to the player’s historical load time series and converted into a forecast ACWR:

`ACWR[p, t, m] = Acute[p, t, m] / Chronic[p, t, m]`

where acute and chronic are exponentially weighted moving averages (EWMA) over 7 and 28 days respectively.

### 1.3 Why this matters

The underlying coaching question is not “what was the load yesterday?” but:

> If we plan a specific sequence of sessions over the next two weeks, how will each player’s load-risk profile evolve?

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

From the project decision log:

- initial shape: **3,903 rows × 13 columns**,
- date range: **2024-07-16 to 2025-06-26**,
- raw players: **35**,
- final cleaned cohort: **28** players,
- season length covered: **345 days**.

### 2.3 Core raw fields

Important columns in the raw export include:

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

The field `period_name` has the form `{CATEGORY} {DRILL_ID}` such as:

- `G 1960`
- `TAC 0133`
- `BP 2351`

The prefix is interpreted as the session family:

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

| Deployed target | Raw source | Meaning |
|---|---|---|
| `total_distance` | `total_distance` | Aerobic / volume load |
| `acc_total` | `acc_band7plus_total_effort_count` | High-intensity acceleration count |
| `vel_total` | `velocity_band6plus7_total_distance` | High-speed running distance |

They are intentionally **not combined** into a single score because they capture different physiological stresses.

---

## 3. Cleaning and cohort definition

### 3.1 Type normalization

The pipeline applies the following type conversions:

- `period_start_time` → `datetime64[ns]`
- `date_of_birth` → `datetime64[ns]`
- `player_id` → categorical-like identifier
- `is_official_match` → boolean after filling nulls with 0

A `date` field is derived from `period_start_time` at calendar-day granularity.

### 3.2 Age calculation

Age is computed continuously at the row date, not at a fixed season reference date:

```python
age = (period_start_time - date_of_birth).dt.days / 365.25
```

This matters because the model predicts future day-level loads. A continuous age feature preserves more information than integer-rounded age.

### 3.3 Player exclusions

Several players are excluded before modeling because of one or more of the following:

- implausible or placeholder metadata,
- preseason-only presence,
- too few observations to support stable ACWR or load modeling,
- missing all core player metadata.

The final deployed cohort contains **28 players**.

### 3.4 Row-level outlier treatment

One extreme outlier is explicitly corrected:

- player `94884`, match on `2025-02-15`, `total_distance = 32,299.89`

This value is considered physiologically implausible and inconsistent with the player’s other match metrics. The correction strategy is:

- replace the outlier with that player’s **median non-outlier match distance**,
- keep the row,
- keep the other two metrics unchanged.

This preserves the player-day event in the time series while avoiding contamination of both the predictive model and downstream EWMA calculations.

---

## 4. Data processing pipeline

### 4.1 Aggregation target: player-day

ACWR is defined on a daily timeline, so the period-level table is first aggregated to one row per:

- `(player_id, date)`

At this stage:

- load metrics are **summed** over all periods that day,
- session composition is preserved as a set of session types,
- static player metadata is carried forward.

The renamed daily target columns are:

- `total_distance`
- `acc_total`
- `vel_total`

The resulting active-day table contains **2,103 rows**.

### 4.2 Session composition encoding

The daily aggregation retains the set of exercise categories completed on that day. From this set, the modeling pipeline derives binary flags:

- `has_G`
- `has_TAC`
- `has_BP`
- `has_TEC`
- `has_MATCH`

and also:

- `n_session_types = |exercise_types|`

So if a player-day contains both tactical and set-piece work, then:

- `has_TAC = 1`
- `has_BP = 1`
- `n_session_types = 2`

This is how session design enters the model.

### 4.3 Position encoding

Player position is one-hot encoded into:

- `pos_central_back`
- `pos_central_midfielder`
- `pos_forward`
- `pos_full_back`
- `pos_winger`

These features capture systematic biomechanical and tactical differences in expected load.

### 4.4 Calendar feature

The pipeline encodes each player-day as time since the season anchor date:

- `SEASON_START_DATE = 2024-07-15`

and defines:

`days_since_start = (period_start_time - SEASON_START_DATE).days`

This provides the model with season-phase context.

### 4.5 Activity-history features

Two recency variables are computed on the active-day table, using only **past** information.

### Days since last activity

For player `p` and active day `t_i`:

`days_since_last_activity[p, i] = t_i - t_{i-1}`

where `t_{i-1}` is the previous active day for that same player.

### Days since last match

For player `p` and active day `t_i`:

`days_since_last_match[p, i] = t_i - max{s < t_i : day s included a match}`

If no prior event exists, the value is filled with a lookback cap.

### Capping rule

Both history features are clipped to a maximum of **21 days**:

`feature = min(feature, 21)`

and missing first values are also filled with `21`.

This cap is important because the forecast code uses the same convention; it avoids extrapolating too far beyond the observed training distribution.

### 4.6 Persisted modeling table

The final feature-engineered active-day dataset is stored at:

- `data/processed/model_data.parquet`

Notebook output indicates the deployed modeling table has shape approximately:

- **2,103 rows × 21 columns** before adding player identity one-hots.

Those 21 columns consist of:

- `player_id`
- 3 targets
- 17 base predictor columns

---

## 5. Full-calendar reconstruction for ACWR

A subtle but important design point is that `model_data.parquet` contains **active days only**, because it is optimized for supervised learning of session-day loads.

However, ACWR requires a **continuous daily series**, where rest days contribute zeros and therefore influence both acute and chronic EWMA decay.

### 5.1 Reconstructed daily grid at runtime

At application load time, `load_player_data()` reconstructs a full per-player calendar grid from:

- each player’s first active day,
- to that player’s last active day,
- at daily frequency.

Missing dates are filled with:

- `0.0` for all three load metrics,
- `0` for `has_MATCH`.

This produces the correct input for EWMA-based ACWR.

### 5.2 Why rest-day filling is mandatory

If missing days were skipped rather than filled with zeros:

- chronic load would decay too slowly,
- acute load would remain artificially elevated,
- the ACWR ratio would not reflect actual calendar exposure.

The system therefore distinguishes between:

- **modeling table**: active days only,
- **ACWR computation grid**: full daily calendar with zero-load rest days.

### 5.3 Current ACWR state used by the dashboard

`load_player_data()` does more than rebuild the daily grid. For each player and each metric, it also:

1. computes ACWR on the reconstructed historical series,
2. drops warmup `NaN` values,
3. keeps the last valid ACWR value,
4. maps that value to an operational risk zone.

So the dashboard and the forecast page share the same ACWR engine; the forecast simply appends predicted loads before re-running the same recursion.

---

## 6. Feature vector used by the deployed models

### 6.1 Deployed feature contract

The production model artifacts enforce a feature vector of exactly **45 columns**:

- **17 base features**,
- **28 player identity one-hot columns** `pid_*`.

The 17 base features are:

1. `height`
2. `weight`
3. `age`
4. `has_G`
5. `has_TAC`
6. `has_BP`
7. `has_TEC`
8. `has_MATCH`
9. `n_session_types`
10. `pos_central_back`
11. `pos_central_midfielder`
12. `pos_forward`
13. `pos_full_back`
14. `pos_winger`
15. `days_since_start`
16. `days_since_last_activity`
17. `days_since_last_match`

The remaining 28 columns are of the form:

- `pid_<player_id>`

and exactly one of them is `1` for a given player.

### 6.2 Why include player one-hots?

The one-hot player identity columns allow a single global model to learn player-specific baselines without switching to a more complex hierarchical or mixed-effects model.

Operationally, this lets the model capture patterns such as:

- some players systematically produce more high-speed distance in the same session type,
- some players have lower or higher typical match loads,
- some players have persistent anthropometric or role effects not fully explained by position alone.

### 6.3 Historical note on 52 features

Some notebook-era documentation mentions:

- 17 base features + 35 player one-hots = 52 features.

That reflects an earlier wider cohort before final cleaning and deployment constraints were fixed. The **deployed** artifact contract is:

- 45 total features,
- 28 `pid_*` columns.

This is validated in `src/real_madrid_acwr/modeling/artifacts.py` and in `tests/test_model_artifacts.py`.

---

## 7. Model architecture

### 7.1 One model per target

The system trains **three independent XGBoost regressors**, one for each target:

- `acc_total`
- `total_distance`
- `vel_total`

This design is intentional. The three outputs are not treated as a single multivariate target because the relationships between session design and each load metric differ materially.

### 7.2 Functional form

For a given target `m`, the model is a boosted additive tree ensemble:

`f_m(x) = Σ_{k=1..K} η · b_k(x)`

where:

- `b_k` is the `k`-th regression tree,
- `K` is the number of boosting rounds / trees,
- `η` is the learning rate.

In XGBoost terms, the model minimizes a regularized objective of the form:

`L = Σ_i l(y_i, ŷ_i) + Σ_k Ω(b_k)`

where:

- `l` is the target-specific loss,
- `Ω` is the tree-complexity regularizer controlled by depth, child-weight, gamma, and L1/L2 penalties.

The deployed training code uses:

- `tree_method = "hist"`
- `random_state = 42`
- `n_jobs = -1`

### 7.3 Why tree boosting is appropriate here

Gradient-boosted trees are a good fit because the problem mixes:

- continuous features (`age`, `height`, `days_since_start`),
- sparse binary indicators (`has_MATCH`, `pid_*`),
- non-linear effects,
- feature interactions.

Examples of interactions the model can learn naturally include:

- a tactical day affecting a winger differently from a central back,
- a match after a long rest producing different expected load from a match in congested scheduling,
- mixed session templates (`G` + `TAC`) behaving differently from either alone.

---

## 8. Target-specific formulations

### 8.1 `total_distance`: log-transformed squared error

#### Statistical issue

`total_distance` is a strictly non-negative continuous variable with a right-skewed distribution.
A raw squared-error model would overweight unusually large distances.

#### Transformation

The training target is transformed as:

`z = log(1 + y)`

where `y = total_distance`.

The model learns:

`f_distance(x) ≈ z`

using XGBoost with objective:

- `reg:squarederror`

So the training loss is approximately:

`l(y, ŷ) = (log(1 + y) - ŷ)^2`

where `ŷ` is the model prediction in log-space.

#### Inverse transform at inference

Predictions are converted back to metres by:

`ŷ_distance = max(exp(ŷ_log) - 1, 0)`

The `max(·, 0)` clipping guarantees non-negative output.

#### Why this helps

The log transform compresses the upper tail and makes optimization more stable, while still returning predictions in the original physical unit after inversion.

### 8.2 `acc_total`: Tweedie regression

#### Statistical issue

`acc_total` is count-like, non-negative, and right-skewed, with a small but meaningful mass at zero.

#### Objective

The deployed model uses:

- `objective = "reg:tweedie"`
- `tweedie_variance_power = 1.9`

For Tweedie models with power parameter `p` in `(1, 2)`, the family interpolates between Poisson-like and Gamma-like behavior and is suitable for non-negative data with a point mass near zero.

At a high level, XGBoost optimizes the Tweedie negative log-likelihood under a log link, i.e. a mean parameter of the form:

`μ(x) = exp(f(x))`

and, up to constants independent of the model, the loss can be written as:

`l(y, μ) = - y * μ^(1-p) / (1-p) + μ^(2-p) / (2-p)`

with `p = 1.9` here.

#### Why this helps

This allows the model to handle:

- non-negativity,
- skewness,
- limited zero inflation,

without splitting the task into separate classification and regression stages.

### 8.3 `vel_total`: raw squared error

#### Statistical issue

`vel_total` is sparse and noisier than the other targets, with roughly 30% zeros in notebook analysis.

#### Objective

The deployed model uses:

- `objective = "reg:squarederror"`

on the raw target:

`l(y, ŷ) = (y - ŷ)^2`

Predictions are clipped at zero after inference:

`ŷ_vel = max(ŷ_raw, 0)`

#### Why no log transform?

Notebook experimentation indicated that for this target, raw MSE combined with tuned tree regularization gave better deployed behavior than a Tweedie or log-space alternative after feature selection analysis.

---

## 9. Deployed hyperparameters

The package training module `src/real_madrid_acwr/modeling/train.py` uses the following best parameter sets.

| Target | Objective | Key hyperparameters |
|---|---|---|
| `acc_total` | `reg:tweedie`, `p=1.9` | `n_estimators=1600`, `learning_rate=0.005`, `max_depth=4`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.7`, `colsample_bylevel=0.6`, `gamma=0.3`, `reg_alpha=1`, `reg_lambda=5` |
| `total_distance` | `reg:squarederror` on `log1p(y)` | `n_estimators=600`, `learning_rate=0.02`, `max_depth=8`, `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.5`, `colsample_bylevel=0.6`, `gamma=0.5`, `reg_alpha=0.01`, `reg_lambda=0.5` |
| `vel_total` | `reg:squarederror` | `n_estimators=200`, `learning_rate=0.07`, `max_depth=3`, `min_child_weight=2`, `subsample=1.0`, `colsample_bytree=1.0`, `colsample_bylevel=0.8`, `gamma=0.3`, `reg_alpha=10`, `reg_lambda=5` |

These parameters were selected in the modeling notebooks using randomized search and then hard-coded into the production training module.

---

## 10. Training and serving procedure

### 10.1 Split strategy

Training uses a random shuffled split:

- `test_size = 0.20`
- `random_state = 42`
- `shuffle = True`

So the supervised learning problem is evaluated as an i.i.d.-style tabular regression task over player-days.

#### Why not a chronological split?

The project comments indicate that a naive time split is awkward here because players have overlapping, non-identical timelines. The deployed implementation chooses a random split to preserve sample size and player coverage.

This should be interpreted carefully: the supervised model is learning a **session-to-load mapping**, while the final ACWR forecast is a downstream simulation built on top of that mapping.

### 10.2 Model fitting and evaluation

For each target:

1. build `X` from the 45 feature columns,
2. build `y` from the target column,
3. apply `log1p` only for `total_distance`,
4. fit the XGBoost regressor,
5. predict on the test set,
6. invert the transform if needed,
7. clip predictions to zero,
8. compute:
   - MAE,
   - R².

Notebook summaries and the project README report approximate held-out performance around:

| Target | Test MAE | Test R² |
|---|---|---|
| `total_distance` | ≈ 800 m | ≈ 0.43 |
| `acc_total` | ≈ 3.63 efforts | ≈ 0.38 |
| `vel_total` | ≈ 18.3 m | ≈ 0.16 |

The relatively lower `vel_total` performance is consistent with the sparser, noisier nature of that target.

### 10.3 Artifact persistence

Each target produces three files under `models/xgboost/{target}/`:

| File | Purpose |
|---|---|
| `model.json` | Native XGBoost `Booster` |
| `feature_cols.pkl` | Ordered inference feature schema |
| `transform.pkl` | Transform metadata (`none` or `log1p`) |

The application loads the native `Booster` directly, rather than an sklearn pipeline pickle, to reduce portability and binary-compatibility problems.

### 10.4 Artifact validation at serving time

Before inference, `src/real_madrid_acwr/modeling/artifacts.py` validates each target directory against a strict contract. For every target, the loader checks that:

- the full artifact triad exists,
- `model.json` can be loaded as a valid XGBoost `Booster`,
- `feature_cols.pkl` is a non-empty list of strings,
- the feature list has exactly 45 entries,
- there are no duplicate feature names,
- all 17 base features are present,
- there are exactly 28 `pid_*` columns,
- `transform.pkl` matches one of the supported transform dictionaries.

This matters because the forecast loop depends on exact schema alignment. A mismatch between trained feature order and serving-time feature construction would silently corrupt predictions if it were not validated early.

---

## 11. ACWR methodology

### 11.1 Acute and chronic EWMAs

For a player’s daily load series `load[t]`, the system computes two uncoupled exponentially weighted moving averages.

#### Acute load

`acute[t] = λ_a * load[t] + (1 - λ_a) * acute[t-1]`

with:

`λ_a = 2 / (7 + 1) = 0.25`

#### Chronic load

`chronic[t] = λ_c * load[t] + (1 - λ_c) * chronic[t-1]`

with:

`λ_c = 2 / (28 + 1) ≈ 0.0689655`

The initialization convention is equivalent to starting from zero and letting the EWMA recurse forward.

### 11.2 ACWR ratio

The ratio is:

`ACWR[t] = acute[t] / chronic[t]`

with safeguards:

- if `chronic[t] == 0`, ACWR is set to `NaN`,
- the first 28 days are masked as warmup.

### 11.3 Why EWMA instead of rolling averages?

EWMA gives higher weight to recent sessions and handles rest-day decay more smoothly than simple rolling averages. This is the method explicitly implemented in `src/real_madrid_acwr/acwr.py`.

### 11.4 Risk zone classification

The app maps ACWR values into four operational bands:

| Zone | Range |
|---|---|
| `undertraining` | `< 0.8` |
| `optimal` | `0.8 <= ACWR < 1.3` |
| `caution` | `1.3 <= ACWR < 1.5` |
| `danger` | `>= 1.5` |

If the value is unavailable after warmup masking, the app reports the zone as `unknown`.

These bands are used as decision-support thresholds, not as a medical diagnosis.

---

## 12. Forecast-generation algorithm

The forecast engine lives in `src/app/forecasting.py`.

### 12.1 Inputs

The forecasting loop receives a 15-day plan of daily booleans such as:

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

### 12.2 Per-player state at forecast start

For each player, the app loads:

- the reconstructed historical daily grid,
- the player profile (height, weight, age, position one-hots),
- the last active day,
- the last match day,
- the current `days_since_start` value.

### 12.3 Day-by-day recursive feature construction

For each future day `d = 1, ..., 15`, the model computes dynamic recency features.

### Days since last activity

If `prev_active_d` is the last forecast-step index that contained activity, then:

`dsla[d] = min(d - prev_active_d, 21)`

Initially `prev_active_d = 0`, meaning the count is relative to the forecast window start and the player’s pre-horizon history.

### Days since last match

Similarly:

`dslm[d] = min(d - prev_match_d, 21)`

where `prev_match_d` is initialized using the gap between the player’s most recent active date and most recent match date.

### Rest-day behavior

If `is_rest = True` on day `d`:

- all three predicted loads are forced to `0.0`,
- the activity counters are **not** reset,
- the sequence moves to the next day.

This preserves the effect of consecutive rest days on future recency features.

### 12.4 Feature vector assembly at inference time

For each target model and each player-day, the app builds a dictionary with:

- static player features,
- session flags for the planned day,
- dynamic day-history features,
- all 28 `pid_*` columns.

The resulting row is then reordered to exactly match `feature_cols.pkl` before being passed to `xgboost.DMatrix(...)`.

Formally, for player `p`, day `d`, target `m`:

`x[p, d] = concat(profile[p], session_flags[d], recency[p, d], pid_one_hot[p])`

and prediction is:

`ŷ[p, d, m] = g_m(x[p, d])`

followed by target-specific post-processing:

- `exp(·) - 1` inversion for `total_distance`,
- zero clipping for all targets.

### 12.5 From predicted loads to forecast ACWR

After the 15 daily loads are predicted for a player and a metric:

1. take the historical load vector from the full daily grid,
2. append the 15 predicted loads,
3. run `compute_acwr_with_forecast(...)`,
4. keep the forecast segment and its final day-15 ACWR.

Formally, if historical loads are `h[1:H]` and forecast loads are `f[1:15]`, then the stitched sequence is:

`load*[1:H+15] = [h[1], ..., h[H], f[1], ..., f[15]]`

and the same EWMA recursion is run over this combined series.

This is the key modeling idea: **the predictive model forecasts load, not ACWR directly**. ACWR is a deterministic downstream transformation of the load path.

---

## 13. End-to-end system flow

The deployed pipeline is:

1. raw CSV / ZIP bootstrap,
2. notebook data pipeline creates `data/processed/model_data.parquet`,
3. package training code fits three XGBoost models,
4. artifacts are saved under `models/xgboost/`,
5. the Streamlit app loads artifacts and player data,
6. coaches define a future 15-day session plan,
7. the app predicts future daily loads,
8. the app computes forecast ACWR trajectories and risk zones.

A compact mathematical view is:

`planned sessions -> feature vectors -> predicted loads -> EWMA acute/chronic -> ACWR -> risk zone`

---

## 14. File-level mapping of responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Streamlit entry point |
| `src/app/loaders.py` | Load artifacts, rebuild player grids, compute current ACWR |
| `src/app/forecasting.py` | Roll-forward prediction of daily loads and future ACWR |
| `src/real_madrid_acwr/acwr.py` | EWMA and ACWR computations |
| `src/real_madrid_acwr/modeling/train.py` | Production training and artifact generation |
| `src/real_madrid_acwr/modeling/artifacts.py` | Artifact validation and loading contract |
| `notebooks/data_pipeline.ipynb` | Raw-to-processed data engineering notebook |
| `data_decisions.md` | Cleaning and methodology rationale |

---

## 15. Assumptions and limitations

The current system should be interpreted with the following constraints in mind.

### 15.1 Data limitations

- Only one season is available.
- There is no session duration, sRPE, or wellness data.
- Rest days and tracking gaps can be observationally indistinguishable.
- Some players have sparse activity histories.

### 15.2 Modeling limitations

- The supervised train/test split is random, not chronological.
- Player identity is encoded by one-hot features rather than a hierarchical model.
- `vel_total` remains a comparatively noisy target.
- Forecast quality depends on the assumption that future session composition behaves similarly to historical session composition.

### 15.3 ACWR limitations

- ACWR is a workload proxy, not an injury diagnosis.
- Risk zones are heuristic thresholds from sports-science convention.
- The model predicts load and propagates it into ACWR; it does not predict injury probability directly.

---

## 16. Practical interpretation

The system should be read as a two-stage engine:

### Stage 1 — Session-to-load model

Given a player profile and a planned session type, estimate how much external load that day is likely to generate.

### Stage 2 — Load-to-ACWR simulator

Given the player’s historical loads and the predicted next 15 loads, propagate the EWMA state forward and inspect whether the player enters a safe, caution, or danger zone.

This separation is important because it makes the logic transparent:

- the model handles **behavioral/physical response to a planned session**,
- the ACWR equations handle **physiological workload accumulation over time**.

---

## 17. Reproducing the deployed artifacts

From the repository conventions, the minimal regeneration flow is:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[notebooks]"
jupyter nbconvert --to notebook --execute notebooks/data_pipeline.ipynb
python train_models.py
```

Then run the application with:

```bash
streamlit run main.py
```

---

## 18. Summary

In concrete terms, this project does the following:

1. converts period-level GPS training logs into player-day features,
2. learns how planned session composition maps to expected external load,
3. predicts the next 15 daily loads for each player,
4. propagates those loads through a 7-day vs 28-day EWMA ACWR model,
5. presents the resulting risk trajectory to coaches in an interactive app.

So the core technical object is not just an ACWR calculator; it is a **load forecasting + ACWR simulation system**.

