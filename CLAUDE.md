# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

End-to-end ACWR (Acute:Chronic Workload Ratio) prediction tool for Real Madrid's fitness coaching staff. Coaches enter a planned 15-day session composition and see predicted ACWR trajectories for each squad player across three load metrics. The interactive app is a Streamlit single-file application (`app.py`).

The raw dataset (`data_acute_vs_chronic.csv`) is gitignored and must be present locally to run the data pipeline.

## Repository layout

```
app.py                              ← Streamlit application (run: streamlit run app.py)
train_models.py                     ← Train XGBoost models; saves to models/

notebooks/data_pipeline.ipynb       ← Run first; produces model_data.parquet
notebooks/acc_total.ipynb           ← EDA + model exploration (acc_total)
notebooks/total_distance.ipynb      ← EDA + model exploration (total_distance)
notebooks/vel_total.ipynb           ← EDA + model exploration (vel_total, two-round SHAP)

data/processed/model_data.parquet   ← Bridge between pipeline and app/models
models/                             ← XGBoost JSON artifacts + pkl feature/transform metadata
utils/acwr.py                       ← EWMA-ACWR computation utilities
static/img/                         ← Real Madrid SVG logo (used by Streamlit sidebar)

data_decisions.md                   ← Every cleaning / methodology decision
requirements.txt                    ← Python dependencies
```

## Running the app

```bash
# 1. Run data pipeline once (needs raw CSV)
jupyter nbconvert --to notebook --execute notebooks/data_pipeline.ipynb

# 2. Train models (uses model_data.parquet)
python train_models.py

# 3. Start app
streamlit run app.py
```

Models are saved as XGBoost native JSON (`{target}_model.json`) plus pkl metadata — no sklearn dependency at load time, avoiding scipy binary conflicts across environments.

## Model artifacts (`models/`)

Each of the three targets has three files:

| File | Content |
|---|---|
| `{target}_model.json` | XGBoost `Booster` in native JSON format |
| `{target}_feature_cols.pkl` | Ordered list of 45 feature names |
| `{target}_transform.pkl` | Dict with `type` (`log1p` or `none`) for inverse-transform |

The app loads models with `xgb.Booster().load_model()` — intentionally avoiding sklearn Pipeline pickle to prevent scipy/sklearn binary incompatibilities.

## App architecture (`app.py`)

Single-file Streamlit app. Three pages routed via `st.session_state.nav`:

| Page | Function | Description |
|---|---|---|
| Dashboard | `page_dashboard()` | Current ACWR status for all 28 players; stat cards + player grid |
| Plan Sessions | `page_planner()` | `st.data_editor` with checkbox columns for 15-day session plan |
| Forecast Results | `page_results()` | Plotly ACWR chart (historical + forecast) + day-15 summary table |

Key functions:
- `load_models()` — `@st.cache_resource`; loads all three XGBoost Boosters
- `load_player_data()` — `@st.cache_resource`; returns `player_data`, `all_pids`, `current_acwr`
- `build_forecast(plan_days)` — roll-forward inference; calls `compute_acwr_with_forecast` per player
- `build_acwr_chart(mdata, meta)` — returns `go.Figure` with zone bands and dual traces

## Pipeline architecture (`notebooks/data_pipeline.ipynb`)

| Section | Key output |
|---|---|
| 0. Setup & Data Loading | raw `df` |
| 1. Data Cleaning | cleaned `df` (3,802 rows, 28 players) |
| 2. Outlier Treatment | player 94884 `total_distance` replaced with median; `weight=200` players dropped |
| 3. Daily Aggregation | `daily` (2,103 rows), one row per `(player_id, date)` |
| 4. Modeling Input | `model_data` with session flags, position one-hots, activity history |
| Persist | `data/processed/model_data.parquet` |

## Model notebooks

| Notebook | Target | Loss | Test MAE | Test R² |
|---|---|---|---|---|
| `acc_total.ipynb` | `acc_total` (effort count) | Tweedie | ≈ 3.63 | ≈ 0.38 |
| `total_distance.ipynb` | `total_distance` (m) | log-MSE | ≈ 800 m | ≈ 0.43 |
| `vel_total.ipynb` | `vel_total` (m) | Raw MSE | ≈ 18.3 | ≈ 0.16 |

`vel_total.ipynb` runs a two-round pipeline: Round 1 trains on all 52 features, SHAP selects features covering 90% cumulative importance, Round 2 re-tunes on the reduced set.

## Key data facts

- **Granularity shift:** raw data is one row per *period* (drill); pipeline aggregates to one row per *player-day*.
- **Load metrics** (three, computed independently): `total_distance` (m), `acc_total` (high-intensity acceleration count), `vel_total` (high-speed running, m).
- **Metric correlations (EDA):** largely decoupled — Pearson r ≈ 0.48 (td↔vel), 0.22 (vel↔acc), 0.19 (td↔acc). Do not collapse them.
- **EWMA parameters:** α_acute = 2/(7+1) = 0.25; α_chronic = 2/(28+1) ≈ 0.069. Initialized from 0.
- **Modelable players:** 18 of 28 have ≥50 active days (natural gap: no players between 46–71 active days). All 28 appear in `pid_*` one-hot columns.
- **Safe ACWR range:** 0.8–1.3; elevated-risk threshold: ≥1.5.
- **Feature count:** 45 = 17 base features + 28 `pid_*` player one-hots.
- **`model_data.parquet`** columns: `player_id`, `height`, `weight`, `age`, `total_distance`, `acc_total`, `vel_total`, `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`, `n_session_types`, `pos_*` (5), `days_since_start`, `days_since_last_activity`, `days_since_last_match`.

## Decision log

`data_decisions.md` documents every cleaning and methodology choice. Check it before changing pipeline logic. Open items are marked `[OPEN]`.
