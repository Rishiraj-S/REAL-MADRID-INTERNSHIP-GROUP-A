# Prediction of Acute vs Chronic Workload Ratio for Players

**Client:** Departamento de Data del Club (Club Data Department)  
**Team:** Group A — trAIn Labs  
**Date:** March 2026  
**License:** Apache License 2.0

---

## Overview

This repository contains the end-to-end solution developed by Group A for the Real Madrid internship project. The goal is to design, build, and deliver a **prediction and visualisation tool for the Acute vs Chronic Workload Ratio (ACWR)** for football players — a key indicator for assessing injury risk and optimising athletic performance.

The ACWR compares an athlete's recent workload (acute, ~7 days) against their longer-term workload (chronic, ~28 days). Values outside a safe range signal elevated injury risk. This tool enables fitness coaches and technical staff to make data-driven training decisions without writing code or querying databases directly.

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

### 2. Data Pipeline (`notebooks/data_pipeline.ipynb`)

| Step | Output |
|---|---|
| Raw load & type coercion | Cleaned `df` (3,802 rows, 28 players) |
| Outlier treatment | Player 94884 `total_distance` replaced with player median; `weight=200` players dropped |
| Daily aggregation | `daily` — one row per `(player_id, date)`, loads summed |
| Feature engineering | Session flags (`has_G`, `has_TAC`, …), position one-hots, calendar features, activity history |
| Persist | `data/processed/model_data.parquet` |

### 3. Load Prediction Models (`src/real_madrid_acwr/modeling/train.py`)

Three independent XGBoost models — one per load metric — trained via the package
training module. The root `train_models.py` file remains as a compatibility
wrapper, so `python train_models.py` still works after the project is installed.
Hyperparameters were found via RandomizedSearchCV (100 iterations, 5-fold CV)
in the model notebooks.

| Target | Loss | Test MAE | Test R² | Notes |
|---|---|---|---|---|
| `total_distance` | log-MSE | ≈ 800 m | ≈ 0.43 | log1p transform; right-skewed distribution |
| `acc_total` | Tweedie (p=1.9) | ≈ 3.63 efforts | ≈ 0.38 | Count data; Tweedie handles zero-inflation |
| `vel_total` | Raw MSE | ≈ 18.3 m | ≈ 0.16 | SHAP feature selection applied (Round 2) |

**Feature vector (45 features):**
- 17 base features: anthropometrics, session type flags, position one-hots, calendar/activity history
- 28 `pid_*` player one-hot columns (one per squad member)

Models are saved as **XGBoost native JSON** under `models/xgboost/{target}/model.json` — not sklearn Pipeline pickle — to avoid scipy binary incompatibilities across conda environments.

### 4. ACWR Utilities (`src/real_madrid_acwr/acwr.py`)

Core EWMA computation library:

| Function | Description |
|---|---|
| `compute_acwr(daily_loads)` | EWMA-ACWR for a single player's load series |
| `compute_acwr_with_forecast(hist, fore)` | Stitches historical + forecast loads, returns full ACWR series with `is_forecast` flag |
| `classify_acwr_zone(value)` | Maps ACWR value to risk zone string |

**EWMA parameters:**
- Acute: α = 2/(7+1) = 0.250
- Chronic: α = 2/(28+1) ≈ 0.069
- Warmup mask: first 28 days masked as NaN (chronic not yet stable)

### 5. Interactive Application (`app.py`)

A Streamlit single-file application with three pages:

| Page | Description |
|---|---|
| **Dashboard** | Current ACWR status for all 28 squad players across all three load metrics, with risk zone flags and stat summary cards |
| **Session Planner** | Coaches select session types (G / TAC / BP / TEC / MATCH / REST) for each of 15 upcoming days via an interactive grid editor |
| **Forecast Results** | 15-day ACWR trajectory per player with three stacked Plotly charts (one per metric), a session calendar view, injury risk alerts, and a day-15 summary table |

**Key functions:**

| Function | Description |
|---|---|
| `load_models()` | `@st.cache_resource` — loads all three XGBoost Boosters from JSON |
| `load_player_data()` | `@st.cache_resource` — builds complete daily calendar grids and computes current ACWR per player |
| `build_forecast(plan_days)` | Roll-forward 15-day inference loop — predicts loads then feeds them into EWMA to compute future ACWR |
| `build_acwr_chart(mdata, meta)` | Returns Plotly figure with zone bands, threshold lines, and dual historical/forecast traces |

**Navigation** uses `st.radio` with a `_pending_nav` intermediary to allow programmatic page switching (e.g. after forecast completes) without hitting Streamlit's widget ownership lock.

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

Install runtime and training dependencies:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -e .
```

For development, install the package with its dev extra:

```bash
python -m pip install -e ".[dev]"
```

```bash
# Step 1 — Run data pipeline once.
# The notebook reads data/raw/data_acute_vs_chronic.csv,
# extracting it from data/data_acute_vs_chronic.zip if needed.
jupyter nbconvert --to notebook --execute notebooks/data_pipeline.ipynb

# Step 2 — Train all three XGBoost models
make train

# Step 3 — Launch the Streamlit app
make run

# Alternative — using a specific conda environment
conda run -n <env> streamlit run main.py
```

### Dependencies

Dependencies are declared in `pyproject.toml`, which is the single dependency
source for the project.

Notebook exploration dependencies are available as a separate optional extra
because they are heavier and are not needed for CI:

```bash
python -m pip install -e ".[notebooks]"
```

### Quality Checks

The repository uses `pytest`, `ruff`, and `mypy`, configured in `pyproject.toml`.

```bash
make quality
```

GitHub Actions runs the same checks on pull requests and pushes to `main`.

---

## Repository Structure

```
.
├── app.py                              # Streamlit application (3 pages)
├── train_models.py                     # Compatibility wrapper for model training
├── Makefile                            # Common local commands
├── pyproject.toml                      # Project metadata and tool configuration
├── data_decisions.md                   # Cleaning & methodology decision log
├── tests/                              # Pytest suite for core contracts
│
├── data/
│   ├── README.md                       # Data directory conventions
│   ├── raw/                            # Immutable local raw data
│   ├── external/                       # Local third-party data
│   ├── interim/                        # Intermediate generated data
│   └── processed/
│       └── model_data.parquet          # Feature-engineered pipeline output
│
├── models/
│   └── xgboost/
│       └── {target}/
│           ├── model.json              # XGBoost Booster — native JSON format
│           ├── feature_cols.pkl        # Ordered list of 45 feature names
│           └── transform.pkl           # Inverse-transform metadata
│
├── notebooks/
│   ├── data_pipeline.ipynb             # Data engineering → model_data.parquet
│   ├── acc_total.ipynb                 # EDA + model exploration (accelerations)
│   ├── total_distance.ipynb            # EDA + model exploration (running distance)
│   └── vel_total.ipynb                 # EDA + model (high-speed running; 2-round SHAP)
│
├── references/
│   └── Tema1.prediction_acute_chronic.v1.english.pdf
│
├── reports/
│   └── figures/
│       ├── acc_total/
│       ├── total_distance/
│       └── vel_total/
│
├── src/
│   └── real_madrid_acwr/
│       ├── acwr.py                     # EWMA-ACWR computation utilities
│       ├── data/                       # Dataset-building code namespace
│       ├── features/                   # Feature engineering code namespace
│       ├── modeling/
│       │   └── train.py                # Model training implementation
│       └── visualization/              # Visualization code namespace
│
├── utils/
│   └── acwr.py                         # Compatibility wrapper for older imports
│
└── static/
    └── img/
        ├── Real-Madrid-CF-v2002.svg    # Club logo (sidebar)
        └── trAIn_labs.png              # Team logo (sidebar footer)
```

---

## Key Data Facts

- **Granularity shift:** raw data is one row per *drill period*; pipeline aggregates to one row per *player-day*
- **28 players** tracked across the 2024/25 season; **18** have ≥ 50 active days (used for modelling)
- **Three load metrics** are independent: Pearson r ≈ 0.48 (distance↔velocity), 0.22 (velocity↔acc), 0.19 (distance↔acc)
- **Feature count:** 45 = 17 base + 28 player one-hots
- **model_data.parquet** columns: `player_id`, `height`, `weight`, `age`, `total_distance`, `acc_total`, `vel_total`, `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`, `n_session_types`, `pos_*` (5), `days_since_start`, `days_since_last_activity`, `days_since_last_match`

---

## Project Status

- [x] Data exploration and cleaning complete (see `data_decisions.md`)
- [x] Daily aggregation and full-calendar grid built (2,103 active rows; 28 players)
- [x] Feature engineering pipeline complete (`model_data.parquet`)
- [x] EWMA-ACWR computed for all players × all load metrics, with warmup masking
- [x] Load prediction models trained and validated (`acc_total`, `total_distance`, `vel_total`)
- [x] SHAP interpretability analysis complete for all three models
- [x] 15-day roll-forward ACWR simulation (`build_forecast` in `app.py`)
- [x] Interactive Streamlit application (`app.py`) — Dashboard, Planner, Forecast Results
- [x] Professional UI with Real Madrid branding (RM official colours, club logo, team logo)
- [ ] Final documentation and presentation

---

## Project Timeline (8 Weeks)

| Week | Focus |
|---|---|
| 1–2 | Data exploration, approach definition, work plan submission |
| 3–4 | Data preparation pipeline and initial model development |
| 5–6 | Model validation, iteration, and visualisation design |
| 7 | App finalisation, documentation, and pre-delivery review |
| 8 | Final presentation and live demo delivery |

---

## Mentoring Sessions

| Session | Timing | Duration | Focus |
|---|---|---|---|
| 1 | End of Week 2 | 45 min | Validate approach and data understanding |
| 2 | End of Week 4 | 60 min | Review pipeline and model validation |
| 3 | End of Week 6 | 60 min | Feedback on visualisation and what-if design |
| 4 | End of Week 7 | 45 min | Pre-delivery review and presentation rehearsal |

---

## License

This project is licensed under the **Apache License, Version 2.0**.
