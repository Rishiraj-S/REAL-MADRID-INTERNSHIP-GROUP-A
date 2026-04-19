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
- Which athletes are most likely to have a dangerously high ACWR based on their training and injury history?

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
| `total_distance` | External load: aerobic volume (meters) |
| `acc_band7plus_total_effort_count` | External load: high-intensity accelerations (count) |
| `velocity_band6plus7_total_distance` | External load: high-speed running (meters) |

Training categories (extracted from `period_name` prefix): **G** (game-based/SSG), **TAC** (tactical), **BP** (set pieces), **TEC** (technical), **MATCH** (official).

**Note:** Session duration, sRPE, and wellness data are not available in the current dataset. All load computations are therefore based on external (GPS/IMU) metrics only. See `docs/data_decisions.md` for full cleaning details and limitations.

### 2. Data Pipeline

| Step | Description |
|---|---|
| Row-level cleaning | Type coercion, outlier review, trialist/placeholder-metadata player exclusion |
| Daily aggregation | One row per player-day; loads summed, exercise-type composition preserved via pivot |
| Full calendar grid | Zero-filled rest days, common end date across players |
| ACWR computation | EWMA (Williams et al., 2017) with λ_acute = 2/8, λ_chronic = 2/29; computed independently for all three load metrics |
| Predictive model | ML model(s) predicting daily load from planned session composition and player history; ACWR derived from predicted loads |
| Model validation | Time-series cross-validation; evaluated using RMSE, MAE, and R² per metric |

Multiple algorithms are explored and compared. Notebooks are developed in **Databricks**.

### 3. Interactive Visualization App

A deployable, user-friendly application that allows the technical team to:

- Select the **proposed session composition** (counts of G / TAC / BP / TEC periods, or predefined session templates based on empirical patterns) for upcoming days
- View the **predicted ACWR for each athlete over the next 15 days** based on that plan, for all three load metrics independently
- Identify at-risk athletes at a glance through clear, actionable visualizations

---

## Deliverables

| # | Deliverable | Description |
|---|---|---|
| 1 | Pipeline & Model Code | Databricks notebooks for data preparation and ML model development |
| 2 | Interactive Application | Deployable app for coaching staff to interact with the predictive model |
| 3 | Technical Documentation | Architecture, data model, methodology, and user guide |
| 4 | Final Presentation | 15–20 min executive PowerPoint with live demo |

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

## Current Status

- [x] Data exploration and cleaning complete (see `docs/data_decisions.md`)
- [x] Daily aggregation and full-calendar grid built
- [x] EWMA-ACWR computed for all players × all load metrics, with unit tests
- [x] Pipeline validated via player-level trajectory plots
- [ ] Load prediction model
- [ ] 15-day simulation (roll-forward prediction of future ACWR)
- [ ] Interactive visualization app
- [ ] Final documentation and presentation

---

## Mentoring Sessions

Up to **4 hours of remote tutoring** with a Club Data Department expert:

| Session | Timing | Duration | Focus |
|---|---|---|---|
| 1 | End of Week 2 | 45 min | Validate approach and data understanding |
| 2 | End of Week 4 | 60 min | Review pipeline and model validation |
| 3 | End of Week 6 | 60 min | Feedback on visualization and what-if design |
| 4 | End of Week 7 | 45 min | Pre-delivery review and presentation rehearsal |

---

## Week 2 Work Plan Submission

By end of Week 2, the team must submit a document containing:

1. **Team Identity** — Team name and description of the approach
2. **Roles and Responsibilities** — Assigned roles (Project Lead, Data Engineer, Visualization Specialist, ML/Analytics Lead) with clear responsibilities per member
3. **8-Week Activity Plan** — Gantt diagram or equivalent breakdown of tasks by role and week
4. **Mentoring Schedule** — Proposed dates, durations, and agenda items for each session

---

## Repository Structure

```
.
├── data/                  # Raw and processed datasets (CSV/JSON)
├── notebooks/             # Databricks notebooks for pipeline and modelling
├── app/                   # Interactive visualization application
├── docs/                  # Technical documentation
│   └── data_decisions.md  # Data cleaning & processing decision log
└── README.md
```

---

## License

This project is licensed under the **Apache License, Version 2.0**. All code and deliverables are submitted under this open-source license.