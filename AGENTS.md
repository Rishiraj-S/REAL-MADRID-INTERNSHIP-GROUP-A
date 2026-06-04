# AGENTS.md

This file provides guidance to Codex and other coding agents when working with
this repository.

## Project

End-to-end ACWR (Acute:Chronic Workload Ratio) prediction tool for Real Madrid's
fitness coaching staff. Coaches schedule a 15-day session plan, see predicted
ACWR trajectories for each squad player across three load metrics, customise
plans per player, and export the full squad plan as a PDF.
The interactive app is a Streamlit application (`main.py`).

The repository keeps `data/data_acute_vs_chronic.zip` as the bootstrap archive.
The extracted raw CSV lives at `data/raw/data_acute_vs_chronic.csv` and is
ignored by git.

## Repository layout

```text
main.py                                          <- Streamlit entry point (4 pages)
train_models.py                                  <- CLI wrapper for model training
pyproject.toml                                   <- Package metadata, dependencies, tool config
Makefile                                         <- Common local commands

src/app/
  charts.py                                      <- Plotly ACWR chart builder
  constants.py                                   <- Domain constants (imports TRANSLATIONS)
  translations.py                                <- Full ENG/ESP translation table (162 keys)
  forecasting.py                                 <- Direct load prediction + ACWR stitching
  i18n.py                                        <- t(), t_pos(), date-format helpers
  loaders.py                                     <- @st.cache_resource model + player data loaders
  pages.py                                       <- All 4 page renderers + sidebar
  planning.py                                    <- Calendar event model + plan helpers
  styles.py                                      <- CSS injection

src/real_madrid_acwr/
  acwr.py                                        <- EWMA-ACWR computation utilities
  config.py                                      <- Shared project Path constants
  modeling/
    artifacts.py                                 <- Bundle loader + contract validation
    datapipeline.py                              <- Shared preprocessing (app + training)
    train.py                                     <- Compatibility shim → training/train.py
    training/
      train.py                                   <- Orchestrator: builds daily.parquet, trains all 3 models
      acceleration_model_train.py                <- XGBoost training for accelerations
      sprint_distance_model_train.py             <- XGBoost training for sprint_distance
      total_distance_model_train.py              <- XGBoost training for total_distance

notebooks/
  datapipeline.py                                <- Shared preprocessing module (EDA only)
  accelerations.ipynb                            <- EDA + XGBoost for accelerations
  sprint_distance.ipynb                          <- EDA + XGBoost for sprint_distance
  total_distance.ipynb                           <- EDA + XGBoost for total_distance

data/
  data_acute_vs_chronic.zip                      <- Bootstrap archive (committed)
  raw/data_acute_vs_chronic.csv                  <- Extracted raw data (gitignored)
  processed/daily.parquet                        <- Player-day grid (produced by train_models.py)

models/xgboost/
  accelerations/bundle.joblib
  sprint_distance/bundle.joblib
  total_distance/bundle.joblib

static/img/                                      <- App image assets
tests/                                           <- Pytest suite (28 tests)
references/                                      <- Reference documents and briefs
data_decisions.md                                <- Cleaning/methodology decision log
```

## Environment

`pyproject.toml` is the single dependency source. Do not reintroduce
`requirements.txt` unless the project intentionally moves back to that workflow.

```bash
python -m pip install -e ".[dev]"
```

Notebook-only dependencies are separate:

```bash
python -m pip install -e ".[notebooks]"
```

## Common commands

```bash
make quality      # ruff + mypy + pytest
make run          # streamlit run main.py
make train        # python train_models.py
```

Equivalent direct commands:

```bash
ruff check main.py train_models.py src tests
mypy
pytest
streamlit run main.py
python train_models.py
```

## Application pages

| Page | Key | Description |
|---|---|---|
| Dashboard | `PAGES[0]` | ACWR methodology, formula, risk zones, current squad status |
| Planning & Forecast | `PAGES[1]` | FullCalendar session planner, 15-day ACWR forecast for all 28 players |
| Player Customization | `PAGES[2]` | Modify squad plan for individual at-risk players, compare custom vs original ACWR |
| Export Plan | `PAGES[3]` | Full squad training schedule table with A3 PDF export |

Language toggle (ENG/ESP) is in the sidebar. All UI text is routed through
`t(key)` from `app.i18n`, backed by `app.translations.TRANSLATIONS`.

## Model artifacts

Targets: `accelerations`, `sprint_distance`, `total_distance`.

Each target produces one artifact:

| File | Content |
|---|---|
| `models/xgboost/{target}/bundle.joblib` | `joblib`-serialised dict: `model` (XGBRegressor), `scaler` (MinMaxScaler), `feature_cols` (list[str]), `ewma_spans` (dict) |

The app loads bundles via `src/real_madrid_acwr/modeling/artifacts.py`.
Prediction uses `np.clip(np.expm1(model.predict(scaler.transform(X))), 0, None)`.

Feature set (17–18 per model, cross-sectional):
- Session flags: `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`
- Activity: `n_periods`, `n_exercise_types`
- Anthropometrics: `height`, `weight`, `age`
- Calendar: `dow_0 … dow_6` (day-of-week OHE)
- Cross-metric: `total_distance` only for `accelerations` and `sprint_distance` models

A processed daily history file is also required at inference time:

| File | Content |
|---|---|
| `data/processed/daily.parquet` | Daily frame per player with all load columns and session flags |

Both files are produced by `python train_models.py`.

## Important conventions

- All production Python lives under `src/`.
- All UI strings go through `t(key)` — never hardcode English in pages.py.
  Add missing keys to `src/app/translations.py` in both ENG and ESP.
- Keep private raw data out of git.
- Check `data_decisions.md` before changing pipeline or modeling methodology.
- Prefer small, test-backed changes over notebook-only edits for production behaviour.
- `notebooks/` is for EDA only; production training uses `train_models.py`.
