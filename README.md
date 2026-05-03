# Prediction of Acute vs Chronic Workload Ratio for Players

**Client:** Departamento de Data del Club (Club Data Department)  
**Team:** Group A  
**Date:** March 2026  
**License:** Apache License 2.0

---

## Overview

This repository contains the end-to-end solution developed by Group A for the Real Madrid internship project. The goal is to design, build, and deliver a **prediction and visualization tool for the Acute vs Chronic Workload Ratio (ACWR)** for football players — a key indicator for assessing injury risk and optimizing athletic performance.

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

Input data is provided as a single CSV containing one row per training period (drill/block within a session) over the 2024–25 season, with the following key fields:

| Field | Description |
|---|---|
| `player_id`, `position_name_en`, `height`, `weight`, `date_of_birth` | Player identity and anthropometrics |
| `period_start_time`, `period_name`, `activity_id` | Session timing and drill identification |
| `is_official_match` | Flag distinguishing matches from training |
| `total_distance` | External load: aerobic volume (metres) |
| `acc_band7plus_total_effort_count` | External load: high-intensity accelerations (count) |
| `velocity_band6plus7_total_distance` | External load: high-speed running (metres) |

Training categories (extracted from `period_name` prefix): **G** (game-based/SSG), **TAC** (tactical), **BP** (set pieces), **TEC** (technical), **MATCH** (official).

### 2. Data Pipeline (`notebooks/data_pipeline.ipynb`)

| Step | Description |
|---|---|
| Row-level cleaning | Type coercion, outlier treatment (player 94884 `total_distance` replaced with player median), trialist/placeholder-metadata player exclusion |
| Daily aggregation | One row per player-day; loads summed → `total_distance`, `acc_total`, `vel_total` |
| Feature engineering | Session flags (`has_G`, `has_TAC`, …), position one-hots, calendar features, activity history |
| Persist | `data/processed/model_data.parquet` — the input consumed by model training and the app |

### 3. Load Prediction Models

Three independent XGBoost models, one per load metric, trained via `train_models.py`:

| Model | Loss | Test MAE | Test R² |
|---|---|---|---|
| `acc_total` | Tweedie | ≈ 3.63 efforts | ≈ 0.38 |
| `total_distance` | log-MSE | ≈ 800 m | ≈ 0.43 |
| `vel_total` | Raw MSE | ≈ 18.3 m | ≈ 0.16 |

Models are saved as XGBoost native JSON to avoid sklearn/scipy version conflicts across environments.

### 4. Interactive Visualization App (`app.py`)

A Streamlit application with three pages:

- **Dashboard** — current ACWR status for all 28 squad players across all three load metrics, with risk zone flags
- **Session Planner** — coaches select session types (G / TAC / BP / TEC / MATCH / REST) for each of 15 upcoming days
- **Forecast Results** — predicted 15-day ACWR trajectory per player, interactive Plotly chart, day-15 summary table with risk zones

```bash
streamlit run app.py
```

---

## Running the Project

```bash
# Step 1 — Data pipeline (requires raw CSV)
jupyter nbconvert --to notebook --execute notebooks/data_pipeline.ipynb

# Step 2 — Train models
python train_models.py

# Step 3 — Launch app
streamlit run app.py
# or, if using a specific conda env:
conda run -n <env> streamlit run app.py
```

---

## Repository Structure

```
.
├── app.py                             # Streamlit application (3 pages)
├── train_models.py                    # Model training script
├── requirements.txt                   # Python dependencies
├── data/
│   ├── data_acute_vs_chronic.csv      # Raw input (gitignored)
│   └── processed/
│       └── model_data.parquet         # Feature-engineered pipeline output
├── models/
│   ├── README.md
│   ├── {target}_model.json            # XGBoost Booster (native format)
│   ├── {target}_feature_cols.pkl      # Ordered feature name list
│   └── {target}_transform.pkl         # Inverse-transform metadata
├── notebooks/
│   ├── README.md
│   ├── data_pipeline.ipynb            # Data engineering → model_data.parquet
│   ├── acc_total.ipynb                # Acceleration count model
│   ├── total_distance.ipynb           # Running distance model
│   └── vel_total.ipynb                # High-speed running model (SHAP selection)
├── utils/
│   └── acwr.py                        # EWMA-ACWR computation utilities
├── static/
│   └── img/
│       └── Real-Madrid-CF-v2002.svg   # Club logo (Streamlit sidebar)
└── data_decisions.md                  # Cleaning & methodology decision log
```

---

## Current Status

- [x] Data exploration and cleaning complete (see `data_decisions.md`)
- [x] Daily aggregation and full-calendar grid built (2,103 active rows; 28 players)
- [x] Feature engineering pipeline complete (`model_data.parquet`)
- [x] EWMA-ACWR computed for all players × all load metrics, with warmup masking
- [x] Load prediction models trained and validated (`acc_total`, `total_distance`, `vel_total`)
- [x] SHAP interpretability analysis complete for all three models
- [x] 15-day roll-forward ACWR simulation (`build_forecast` in `app.py`)
- [x] Interactive Streamlit application (`app.py`) — Dashboard, Planner, Forecast Results
- [ ] Final documentation and presentation

---

## Project Timeline (8 Weeks)

| Week | Focus |
|---|---|
| 1–2 | Data exploration, approach definition, work plan submission |
| 3–4 | Data preparation pipeline and initial model development |
| 5–6 | Model validation, iteration, and visualization design |
| 7 | App finalisation, documentation, and pre-delivery review |
| 8 | Final presentation and live demo delivery |

---

## Mentoring Sessions

| Session | Timing | Duration | Focus |
|---|---|---|---|
| 1 | End of Week 2 | 45 min | Validate approach and data understanding |
| 2 | End of Week 4 | 60 min | Review pipeline and model validation |
| 3 | End of Week 6 | 60 min | Feedback on visualization and what-if design |
| 4 | End of Week 7 | 45 min | Pre-delivery review and presentation rehearsal |

---

## License

This project is licensed under the **Apache License, Version 2.0**.
