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

### 1. Data (`3.1`)

Input data includes:

| Data Type | Details |
|---|---|
| Training & match history | Intensity, duration, exercise type per session |
| Workload & fatigue metrics | Session RPE, distance, accelerations, etc. |

Data is provided in **CSV or JSON** format at project start.

### 2. Data Pipeline (`3.2`)

| Step | Description |
|---|---|
| Data Preparation | Clean, transform, and engineer features for modelling |
| Predictive Model Development | ML model(s) predicting ACWR from training characteristics and athlete metrics |
| Model Validation | Evaluated using RMSE, MAE, and R² |

Multiple algorithms are explored and compared. Notebooks are developed in **Databricks**.

### 3. Interactive Visualization App (`3.3`)

A deployable, user-friendly application that allows the technical team to:

- Select the **proposed training type** (e.g., strength, cardiovascular, mixed) for upcoming days
- View the **predicted ACWR for each athlete over the next 15 days** based on that selection
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
├── docs/                  # Technical documentation and architecture notes
└── README.md
```

---

## License

This project is licensed under the **Apache License, Version 2.0**. All code and deliverables are submitted under this open-source license.
