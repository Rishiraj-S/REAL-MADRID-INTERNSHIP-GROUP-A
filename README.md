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

### 2. Data Pipeline

`python train_models.py` handles the full pipeline end-to-end:

| Step | Output |
|---|---|
| Raw load & type coercion | Cleaned frame (28 players) |
| Outlier treatment | Player 94884 `total_distance` capped at player median; `weight=200` trialists dropped |
| Daily aggregation | One row per `(player_id, date)`, loads summed, rest days filled with zero |
| Feature engineering | EWMA lags, rolling MAs, microcycle stats, session type flags, calendar features |
| Persist | `data/processed/daily.parquet` |

### 3. Load Prediction Models

Three independent XGBoost models — one per load metric — trained via `src/real_madrid_acwr/modeling/train.py`. `train_models.py` at the root is the CLI entry point.

| Target | Transform | Notes |
|---|---|---|
| `total_distance` | log1p | Right-skewed aerobic volume |
| `accelerations` | log1p | High-intensity effort count |
| `sprint_distance` | log1p | High-speed running distance |

**Feature engineering (per target):**
- Lagged loads: t-1, t-3, t-5, t-7, t-14
- Rolling means: 3, 7, 14 days
- Shifted EWMA acute (span 7) and chronic (span 28)
- Microcycle cumsum, monotony, strain, training stress balance
- Session type flags: `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`
- `n_periods`, `n_exercise_types`, `day_of_week`, anthropometrics

Training uses date-blocked 4-fold CV (no data leakage) with `RandomizedSearchCV` over XGBoost hyperparameters (25 iterations).

Model artifacts saved as `models/xgboost/{target}/bundle.joblib` — a `joblib`-serialised dict containing `model`, `scaler`, `feature_cols`, and `ewma_spans`.

### 4. ACWR Utilities (`src/real_madrid_acwr/acwr.py`)

Core EWMA computation library:

| Function | Description |
|---|---|
| `compute_acwr(daily_loads)` | EWMA-ACWR for a single player's load series |
| `compute_acwr_with_forecast(hist, fore)` | Stitches historical + forecast loads; returns full ACWR series with `is_forecast` flag |
| `classify_acwr_zone(value)` | Maps ACWR value to risk zone string |

**EWMA parameters:**
- Acute: α = 2/(7+1) = 0.250
- Chronic: α = 2/(28+1) ≈ 0.069
- Warmup mask: first 28 days masked as NaN (chronic not yet stable)

### 5. Interactive Application (`main.py`)

A Streamlit application with two pages:

| Page | Description |
|---|---|
| **Dashboard** | Current ACWR status for all 28 squad players across all three load metrics, with risk zone flags and KPI summary cards |
| **Planning & Forecast** | Coaches plan the next 15 days on an interactive FullCalendar, then run a squad-wide recursive ACWR forecast. Results show per-player ACWR trajectories, an injury risk alert banner, and a day-15 summary table |

**Key functions:**

| Function | File | Description |
|---|---|---|
| `load_models()` | `app/loaders.py` | `@st.cache_resource` — loads all three XGBoost bundles |
| `load_player_data()` | `app/loaders.py` | `@st.cache_resource` — builds daily grids and computes current ACWR per player |
| `build_forecast(plan_days)` | `app/forecasting.py` | Recursive 15-day inference loop — predicts loads then computes future ACWR |
| `build_acwr_chart(mdata, meta)` | `app/charts.py` | Plotly figure with zone bands, threshold lines, and dual historical/forecast traces |

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

### Requirements

- Python 3.11 or higher
- `pip` (standard) or `conda`

### Step 1 — Clone and set up a virtual environment

```bash
git clone <repo-url>
cd REAL-MADRID-INTERNSHIP-GROUP-A

# Using venv (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Using conda
conda create -n acwr python=3.11
conda activate acwr
```

### Step 2 — Install dependencies

```bash
# Runtime dependencies only (run the app and train models)
pip install -e .

# With development tools (linting, type checking, tests)
pip install -e ".[dev]"

# With notebook dependencies (EDA notebooks)
pip install -e ".[notebooks]"

# Or use Make
make install        # runtime only
make install-dev    # runtime + dev tools
```

### Step 3 — Train the models

This step reads the raw data, builds the daily load grid, trains three XGBoost models, and saves all artifacts. Run it once before starting the app.

```bash
python train_models.py
# or
make train
```

This produces:
- `data/processed/daily.parquet` — player load history used at inference time
- `models/xgboost/total_distance/bundle.joblib`
- `models/xgboost/accelerations/bundle.joblib`
- `models/xgboost/sprint_distance/bundle.joblib`

> The raw CSV (`data/raw/data_acute_vs_chronic.csv`) is extracted automatically from the committed ZIP archive if not already present.

### Step 4 — Launch the app

```bash
streamlit run main.py
# or
make run
```

The app opens at `http://localhost:8501` by default.

### Quality checks

```bash
make quality        # runs ruff + mypy + pytest

# Individual steps
make lint           # ruff check
make typecheck      # mypy
make test           # pytest
```

---

## Repository Structure

```
.
├── main.py                                  # Streamlit app entry point
├── train_models.py                          # CLI wrapper for model training
├── Makefile                                 # Common local commands
├── pyproject.toml                           # Package metadata, dependencies, tool config
├── data_decisions.md                        # Cleaning & methodology decision log
├── AGENTS.md                                # Coding agent guidance
├── TECHNICAL_DOCUMENTATION.md              # Deep-dive production documentation
│
├── src/
│   ├── app/                                 # Streamlit UI layer
│   │   ├── constants.py                     # Domain constants: targets, session types,
│   │   │                                    #   color palettes, ENG+ESP translation table
│   │   ├── styles.py                        # CSS injection (Real Madrid brand styling)
│   │   ├── i18n.py                          # t() translation lookup, fmt_date_* helpers
│   │   ├── loaders.py                       # @st.cache_resource: daily.parquet + model bundles
│   │   ├── planning.py                      # Event model helpers, plan → daily flags, signature
│   │   ├── forecasting.py                   # Recursive XGBoost inference + ACWR stitching
│   │   ├── charts.py                        # Plotly ACWR chart builder
│   │   └── pages.py                         # All Streamlit page renderers:
│   │                                        #   page_dashboard(), page_planner(), render_sidebar()
│   │
│   └── real_madrid_acwr/                    # Core domain / ML layer (no Streamlit dependency)
│       ├── config.py                        # Shared Path constants (PROJECT_ROOT, DATA_DIR, etc.)
│       ├── acwr.py                          # Pure EWMA-ACWR math and zone classification
│       └── modeling/
│           ├── train.py                     # Full training pipeline (preprocess → build_daily
│           │                                #   → add_features → date-blocked CV → XGBoost → save)
│           └── artifacts.py                 # Typed bundle loader with contract validation
│
├── data/
│   ├── raw/
│   │   ├── data_acute_vs_chronic.zip        # Bootstrap archive (committed to git)
│   │   └── data_acute_vs_chronic.csv        # Extracted raw data (gitignored)
│   ├── processed/
│   │   ├── daily.parquet                    # Daily load grid per player (produced by train_models.py)
│   │   └── model_data.parquet               # Legacy pipeline output (notebooks only)
│   ├── external/                            # Third-party data (empty placeholder)
│   └── interim/                             # Intermediate generated data (empty placeholder)
│
├── models/
│   ├── xgboost/
│   │   ├── total_distance/
│   │   │   └── bundle.joblib                # {model, scaler, feature_cols, ewma_spans}
│   │   ├── accelerations/
│   │   │   └── bundle.joblib
│   │   └── sprint_distance/
│   │       └── bundle.joblib
│   └── notebook_experiments/               # Legacy notebook model artifacts (not used by app)
│       ├── acc_total/
│       ├── total_distance/
│       └── vel_total/
│
├── notebooks/                               # EDA + exploration only — not used by the app
│   ├── acceleration_model.ipynb
│   ├── sprint_distance_model.ipynb
│   └── total_distance.ipynb
│
├── tests/
│   ├── test_acwr.py                         # EWMA-ACWR math contracts
│   ├── test_forecasting.py                  # Forecast pipeline contracts
│   ├── test_model_artifacts.py              # Bundle loader validation
│   ├── test_pages.py                        # Page renderer smoke tests
│   ├── test_planning.py                     # Planning helper contracts
│   └── test_project_contracts.py           # Cross-module integration contracts
│
├── static/
│   └── img/
│       ├── Real-Madrid-CF-v2002.svg         # Club logo (favicon + sidebar)
│       └── trAIn_labs.png                   # Team credit logo (sidebar footer)
│
├── references/
│   ├── Tema1.prediction_acute_chronic.v1.english.pdf
│   └── project_explanation_handout.pdf
│
└── reports/
    └── figures/
        ├── acc_total/
        ├── total_distance/
        └── vel_total/
```

---

## Key Data Facts

- **Granularity shift:** raw data is one row per *drill period*; pipeline aggregates to one row per *player-day*
- **28 players** tracked across the 2024/25 season
- **Three load metrics** are modelled independently: total distance, accelerations, sprint distance
- **Rest days** are filled with zero load and contribute to EWMA decay
- **`daily.parquet`** columns: `player_id`, `date`, `total_distance`, `accelerations`, `sprint_distance`, `n_periods`, `n_exercise_types`, `height`, `weight`, `age`, `position`, `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`

---

## Project Status

- [x] Data exploration and cleaning complete (see `data_decisions.md`)
- [x] Daily aggregation and full-calendar grid built (28 players)
- [x] Feature engineering pipeline complete (`daily.parquet`)
- [x] EWMA-ACWR computed for all players × all load metrics, with warmup masking
- [x] Load prediction models trained and validated (`total_distance`, `accelerations`, `sprint_distance`)
- [x] 15-day recursive ACWR forecast (`build_forecast` in `src/app/forecasting.py`)
- [x] Interactive Streamlit application — Dashboard + Planning & Forecast
- [x] Professional UI with Real Madrid branding (official colours, club logo, team logo)
- [x] English / Spanish bilingual support
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
