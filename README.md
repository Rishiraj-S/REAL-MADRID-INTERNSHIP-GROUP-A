# Prediction of Acute vs Chronic Workload Ratio for Players

**Client:** Departamento de Data del Club (Club Data Department)  
**Team:** Group A — trAIn Labs  
**Date:** March 2026  
**License:** Apache License 2.0

---

## Overview

This repository contains the end-to-end solution developed by Group A for the Real Madrid internship project. The goal is to design, build, and deliver a **prediction and visualisation tool for the Acute vs Chronic Workload Ratio (ACWR)** for football players — a key indicator for assessing injury risk and optimising athletic performance.

The ACWR compares an athlete's recent workload (acute, ~7 days) against their longer-term workload (chronic, ~28 days). Values outside a safe range signal elevated injury risk. This tool enables fitness coaches and technical staff to make data-driven training decisions without writing code or querying databases directly.

For a deeper production-oriented explanation of the problem, data pipeline, modeling stack, forecasting logic, and ACWR mathematics, see [`TECHNICAL_DOCUMENTATION.md`](TECHNICAL_DOCUMENTATION.md).

> Reference: [ACWR definition — Science for Sport](https://www.scienceforsport.com/acutechronic-workload-ratio)

---

## Problem Statement

The Club's technical staff and fitness coaches need answers to questions such as:

- What will the ACWR of our athletes be if we run a high-intensity training session tomorrow?
- Which athletes are most likely to have a dangerously high ACWR based on their training history?

Currently, training is planned based on experience and recent metric values alone. This project bridges that gap with a predictive, interactive solution.

---

## Solution Architecture

### 1. Data

Input data is a single CSV containing one row per training period (drill/block within a session) over the 2024–25 season:

| Field | Description |
|---|---|
| `player_id`, `position_name_en`, `height`, `weight`, `date_of_birth` | Player identity and anthropometrics |
| `period_start_time`, `period_name`, `activity_id` | Session timing and drill identification |
| `is_official_match` | Flag distinguishing matches from training |
| `total_distance` | External load: aerobic volume (metres) |
| `acc_band7plus_total_effort_count` | External load: high-intensity accelerations (count) |
| `velocity_band6plus7_total_distance` | External load: high-speed running (metres) |

Training categories extracted from `period_name` prefix:

| Code | Description |
|---|---|
| **G** | Game-based / Small-Sided Game |
| **TAC** | Tactical |
| **BP** | Set Pieces |
| **TEC** | Technical |
| **MATCH** | Official Match |

### 2. Data Pipeline (`src/real_madrid_acwr/modeling/datapipeline.py`)

| Step | Function | Output |
|---|---|---|
| Load | `load_data()` | Raw period-level DataFrame |
| Clean | `clean_data()` | Renamed columns, parsed dates, age computed |
| Outlier treatment | `treat_outliers()` | IQR-cap (Q3 + 3×IQR) per player per exercise type |
| Daily aggregation | `aggregate_daily()` | One row per player-day; session-type flags |
| Spine fill | `spine_fill()` | Rest-day zero rows inserted for contiguous dates |
| Feature engineering | `add_features()` + `encode_dow()` | `day_of_week` → `dow_0…dow_6` (OHE) |
| Scaling | `scale_train()` | `log1p(target)` + `MinMaxScaler` (fit on train only) |
| Full daily save | `build_full_daily()` | `data/processed/daily.parquet` (all 3 metrics) |

### 3. Load Prediction Models

Three independent XGBoost models — one per load metric — trained via `src/real_madrid_acwr/modeling/training/`.

| Target | Unit | Test MAE | Test R² |
|---|---|---|---|
| `total_distance` | metres | ~339 | ~0.78 |
| `accelerations` | count | ~1.6 | ~0.73 |
| `sprint_distance` | metres | ~7.0 | ~0.31 |

**Feature vector (17–18 features per model):**
- Session composition: `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`, `n_periods`, `n_exercise_types`
- Player profile: `height`, `weight`, `age`
- Calendar: `dow_0 … dow_6` (day-of-week one-hot)
- Cross-metric: `total_distance` (covariate for `accelerations` and `sprint_distance` only)

**Training setup:** random 80/20 split · 10-fold KFold CV · `RandomizedSearchCV` (50 iterations) · wide continuous hyperparameter distributions · `log1p` target transform · `MinMaxScaler`.

Each model is saved as `models/xgboost/{target}/bundle.joblib` containing `model`, `scaler`, `feature_cols`, and `ewma_spans`.

### 4. ACWR Utilities (`src/real_madrid_acwr/acwr.py`)

| Function | Description |
|---|---|
| `compute_acwr(daily_loads)` | EWMA-ACWR for a single player's load series |
| `compute_acwr_with_forecast(hist, fore)` | Stitches historical + forecast loads, returns full ACWR series |
| `classify_acwr_zone(value)` | Maps ACWR value to risk zone string |

EWMA parameters: Acute α = 0.25 (7-day) · Chronic α ≈ 0.069 (28-day) · Warmup: first 28 days masked as NaN.

### 5. Interactive Application (`main.py`)

A Streamlit application with two pages:

| Page | Description |
|---|---|
| **Dashboard** | Current ACWR status for all 28 squad players across all three load metrics, with risk zone flags and KPI summary cards |
| **Planning & Forecast** | Coaches schedule sessions on an interactive FullCalendar; "Run Forecast" predicts 15-day ACWR trajectories for all players and renders per-player charts + a day-15 summary table |

**Forecast flow:**
1. Build plan frame: one row per (player × day) with session flags + player profile
2. Apply `encode_dow(add_features(plan_frame))` → adds `dow_0…dow_6`
3. Predict all rows in a **single XGBoost call** — no recursion, no day-by-day loop
4. Stitch predictions onto historical load series → compute ACWR via EWMA

---

## ACWR Risk Zones

| Zone | ACWR Range | Interpretation |
|---|---|---|
| Undertraining | < 0.8 | Insufficient load stimulus |
| Optimal | 0.8 – 1.3 | Safe training range |
| Caution | 1.3 – 1.5 | Elevated risk — monitor closely |
| Danger | ≥ 1.5 | High injury risk — reduce load |

---

## Running the Project

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install runtime + dev dependencies
pip install -e ".[dev]"

# 3. Train all three models and produce daily.parquet
python train_models.py

# 4. Launch the Streamlit app
streamlit run main.py
```

### Notebook exploration (optional)

```bash
pip install -e ".[notebooks]"
jupyter notebook
```

Notebooks under `notebooks/` are for EDA only — production training uses `train_models.py`.

### Quality checks

```bash
make quality   # ruff + mypy + pytest
```

---

## Repository Structure

```
.
├── main.py                                    # Streamlit app entry point
├── train_models.py                            # CLI: trains all models + saves artifacts
├── pyproject.toml                             # Package metadata, deps, tool config
├── Makefile                                   # Common local commands
├── data_decisions.md                          # Cleaning & methodology decision log
│
├── data/
│   ├── data_acute_vs_chronic.zip              # Bootstrap archive (committed)
│   ├── raw/data_acute_vs_chronic.csv          # Extracted raw data (gitignored)
│   └── processed/daily.parquet               # Player-day grid — produced by train_models.py
│
├── models/xgboost/
│   ├── total_distance/bundle.joblib
│   ├── accelerations/bundle.joblib
│   └── sprint_distance/bundle.joblib
│
├── notebooks/
│   ├── datapipeline.py                        # Shared preprocessing module
│   ├── total_distance.ipynb
│   ├── accelerations.ipynb
│   └── sprint_distance.ipynb
│
├── src/
│   ├── app/                                   # Streamlit UI layer
│   │   ├── charts.py                          # Plotly ACWR chart builder
│   │   ├── constants.py                       # Domain constants + ENG/ESP translations
│   │   ├── forecasting.py                     # Load prediction + ACWR stitching
│   │   ├── i18n.py                            # Translation + date-format helpers
│   │   ├── loaders.py                         # Cached model + player data loaders
│   │   ├── pages.py                           # Dashboard + planner page renderers
│   │   ├── planning.py                        # Calendar event model + plan helpers
│   │   └── styles.py                          # CSS injection
│   │
│   └── real_madrid_acwr/                      # Core ML package
│       ├── acwr.py                            # EWMA-ACWR computation
│       ├── config.py                          # Shared Path constants
│       └── modeling/
│           ├── artifacts.py                   # Bundle loader + contract validation (app-facing)
│           ├── datapipeline.py                # Shared preprocessing (app + training)
│           ├── train.py                       # Compatibility shim → training/train.py
│           └── training/
│               ├── train.py                   # Orchestrator: daily.parquet + all 3 models
│               ├── acceleration_model_train.py
│               ├── sprint_distance_model_train.py
│               └── total_distance_model_train.py
│
├── static/img/                                # App image assets (logos)
├── tests/                                     # Pytest suite
└── references/                                # Reference documents and briefs
```

---

## Project Status

- [x] Data exploration and cleaning complete
- [x] Daily aggregation and full-calendar grid (6,310 rows · 28 players)
- [x] Feature engineering pipeline (cross-sectional: session flags, anthropometrics, DOW OHE)
- [x] Load prediction models trained (XGBoost · 50-iter RandomizedSearchCV · 10-fold KFold)
- [x] EWMA-ACWR computed for all players × all load metrics with warmup masking
- [x] 15-day ACWR simulation via direct single-pass inference
- [x] Interactive Streamlit application — Dashboard + Planning & Forecast
- [x] Professional UI with Real Madrid branding
- [ ] Final presentation

---

## License

This project is licensed under the **Apache License, Version 2.0**.
