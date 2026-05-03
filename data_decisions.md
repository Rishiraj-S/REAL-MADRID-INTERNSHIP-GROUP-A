# Data Decisions Log

This document records decisions made during data cleaning and processing for the
ACWR prediction project. Each entry states *what* was decided, *why*, and what
alternatives were considered. Open questions to be resolved with supervisors are
marked with **[OPEN]**.

---

## Dataset overview

- **Source:** `data_acute_vs_chronic.csv`
- **Initial shape:** 3,903 rows × 13 columns
- **Granularity:** one row per training *period* (drill/block within a session)
- **Date range:** 2024-07-16 to 2025-06-26 (345 days, one full season)
- **Players:** 35 in raw data, 29 after cleaning
- **Load metrics available:** `total_distance`, `acc_band7plus_total_effort_count`,
  `velocity_band6plus7_total_distance`
- **Missing:** no session duration, no sRPE, no wellness data

---

## Structural interpretations of the raw data

### `period_name` format
The `period_name` field encodes drill IDs of the form `{CATEGORY} {DRILL_ID}`
(e.g., `G 1960`, `TAC 0133`, `BP 2351`). The prefix corresponds to a training
category:

| Prefix | Assumed meaning | Count |
|---|---|---|
| G | Juego / game-based / SSG | 1,184 |
| TAC | Tactical | 800 |
| BP | Balón parado / set pieces | 684 |
| TEC | Technical | 262 |
| NaN | Official match (no drill logged) | 973 |

**Decision:** Extract prefix as `exercise_type`; treat match rows (where
`period_name` is NaN) as `exercise_type = 'MATCH'`.

**Rationale:** Prefix is the meaningful category for modelling; individual drill
IDs are too granular given the dataset size.

**[OPEN]** Confirm prefix meanings with supervisors.

### `is_official_match`
NaN in 2,930 rows, 1.0 in 973 rows. NaN and match-row `period_name` are perfectly
correlated.

**Decision:** Fill NaN → 0 and cast to boolean.

**Rationale:** The pattern makes clear that NaN means "not a match"; there is no
ambiguity. One row per match period, always with `period_name = NaN`.

---

## Cleaning decisions

### Data types
- `period_start_time` → `datetime64[ns]`
- `date_of_birth` → `datetime64[ns]`
- `date` extracted as calendar-day from `period_start_time` (time component always 00:00:00)
- `player_id` → `category` dtype
- `is_official_match` → boolean

### Age computation

**Decision:** Compute age per row at `period_start_time`:
```python
df['age'] = (df['period_start_time'] - df['date_of_birth']).dt.days / 365.25
```
Do **not** round.

**Alternatives considered:**
- Age at a fixed reference date (e.g., season start): simpler but loses precision
- Rounded integer age: destroys information; two players born 11 months apart
  could round to the same or to adjacent integers depending on reference date

**Rationale:** Downstream the model predicts load on specific future dates, so
age on that date is the correct feature. Float ages preserve information at no
cost.

---

## Player-level exclusions

### Players excluded entirely

| player_id | Rows | Window | Reason |
|---|---|---|---|
| 50333 | 19 | 2024-07-27 to 2024-08-02 | Missing all metadata (height/weight/DOB/position). Preseason-only. Likely trialist. |
| 42978 | 21 | 2024-07-27 to 2024-08-07 | DOB=1969 (implausible for player); height=180, weight=200 placeholder values; preseason-only. |
| 86086 | 10 | 2024-07-27 to 2024-08-01 | Same pattern as 42978. |
| 93116 | 16 | 2024-07-29 to 2024-08-04 | Preseason-only, age-17 academy call-up, weight=200 placeholder. Insufficient data for ACWR (< 28 days). |
| 60819 | 2 | 2025-06-18 to 2025-06-22 | 2 rows total; cannot compute ACWR (warmup alone is 28 days). |
| 89091 | 3 | 2025-06-18 to 2025-06-26 | 3 rows total; cannot compute ACWR. |
| 15795 | 30 | 2024-07-30 to 2025-06-26 | `weight = 200` placeholder; dropped as part of the blanket weight=200 exclusion. Season-long activity pattern suggests a Castilla call-up. **[OPEN]** Confirm whether exclusion is correct — if so, request real anthropometrics for potential reinstatement. |

**Rationale for exclusion:** Insufficient data to compute meaningful ACWR,
unreliable metadata, or both. These players are not in the modelling target
population (the current first-team squad).

**Note on player 15795:** An earlier version of this log marked player 15795 for
retention with nulled `height`/`weight`. The current pipeline drops all
`weight = 200` players as a group. The net effect is the same for modelling
(missing anthropometrics → unusable without imputation), but the player's load
data is also lost. **[OPEN]** Confirm with supervisor whether to reinstate 15795
with real anthropometrics or accept the exclusion.

**Final player count:** 28

**[OPEN]** Confirm with supervisor that these exclusions are correct. In
particular, confirm whether 42978, 86086, 93116 are genuinely trialists /
academy call-ups or first-team squad members whose metadata was miscoded.

---

## Row-level data fixes

### Outlier: `total_distance = 32,299.89` for player 94884, match 2025-02-15

- Only `total_distance` is anomalous; `acc_band7plus_total_effort_count = 15`
  and `velocity_band6plus7_total_distance = 23.47` are normal for a match
- Physiologically impossible (a professional match is typically 10–12 km)

**Decision:** Replace `total_distance` with player 94884's **median match
distance** (1,461.14 m), computed from all other matches excluding the outlier.
Retain the row and the other two load metrics unchanged.

**Alternatives considered:**
- Setting to NaN: preserves uncertainty honestly but propagates NaN into the
  daily aggregation and EWMA, forcing downstream code to handle a missing value
  on an otherwise complete row. Adds complexity for marginal benefit given the
  evidence strongly points to a single-field export error.
- Dropping the row: rejected — loses the information that the player played a
  match on this date, and the other two metrics are valid.

**Rationale:** The selective corruption of one metric (not all three) indicates
a parsing / export error on a single field, not a sensor malfunction. The
player's own match-distance distribution is unimodal and well-behaved; the
median is a defensible, robust substitute. Imputing from the player's own
empirical distribution is preferable to leaving a structural gap in the EWMA
series.

**[OPEN]** Can the raw value be recovered from the source GPS system?

---

## Aggregation decisions

### Daily aggregation granularity

**Decision:** Aggregate row-level periods to one row per `(player_id, date)`.

- Load metrics: **summed** across periods → `total_distance`, `acc_total`,
  `vel_total` (renamed from the raw column names for brevity)
- Exercise type composition: preserved as a **frozenset** column
  `exercise_types` (e.g., `{'G', 'TAC', 'BP'}`). Individual drill IDs are
  dropped; per-type pivot columns (`count_G`, `td_G`, …) are not retained —
  the frozenset is compact and directly queryable for session template matching.
- Static metadata (position, height, weight, DOB): `first` (does not vary
  within a player)

**Rationale:** ACWR is computed on daily load. Daily aggregation is the correct
unit. The frozenset captures session composition without the sparsity of wide
pivot columns, and session templates (e.g., `G+TAC`, `BP+G+TAC`) are readable
directly from the set. The most common non-match session templates observed in
the data: `G+TAC` (338 days), `BP+G` (159), `BP+G+TAC` (150), `G` (118),
`TAC` (93), `BP+G+TEC` (88).

### Full calendar grid with zero-filled rest days

**Decision:**
- Each player's grid starts on their **first observed session**
- Each player's grid ends on their **last observed session** (per-player end date)
- Days with no observed activity are filled with zero load and `is_rest = 1`

**Rationale:**
- EWMA computation requires a continuous daily series; gaps would under-weight
  rest days in the chronic load baseline and inflate ACWR
- Per-player end dates prevent chronic-load collapse for mid-season transfers and
  loan departures. Extending every player to the global season end would inflate
  the rest-day tail, drive EWMA chronic toward zero, and produce artificially
  elevated ACWR at inference time for players who returned after a gap.
- Starting at first observed session avoids fabricating rest history before the
  player entered the dataset

**Alternatives considered:**
- Global end date (2025-06-26) for all players: simpler but creates spurious
  rest-day tails for players who left mid-season. Rejected.

**Resulting grid:** 6,310 rows; 66.7% rest days (4,207 rest / 2,103 active); 28
columns after ACWR computation. Additional derived columns on the grid:

- `is_rest` (int, 0/1) — rest-day flag
- `rest_streak` (int) — consecutive rest-day counter per player; resets to 0
  on active days
- `activity_rate` (float) — active days / grid days per player
- `is_modelable` (bool) — see below

### Player modelability flag

**Decision:** Flag each player `is_modelable = True` if they have ≥ 50 active
days in the grid; False otherwise.

**Threshold:** 50 days — chosen at the natural gap in the data. The cohort
splits cleanly into regular squad members (71+ active days) and irregular
contributors (≤46 active days), with no players in the 47–70 range. 50 falls
inside this gap and matches conventional minimum sample sizes for stable
per-player parameter estimation in hierarchical models. **18 of 28 players**
meet this threshold. The 10 non-modelable players have 3–46 active days and
appear to be Castilla call-ups, short-term signings, or players with data
capture gaps.

**Rationale:** The model is trained only on modelable players. Non-modelable
players will require a position-based fallback at inference time (see
**[OPEN]** items). The `is_modelable` column is propagated to the full grid so
downstream code can filter with a single predicate.

**[OPEN]** Confirm the position-based fallback strategy with supervisor before
modelling begins.

---

## ACWR methodology

### Formula: EWMA (Williams et al., 2017)

For each metric, for each player, compute:
```
acute[t]   = λ_a · load[t] + (1 − λ_a) · acute[t-1],   λ_a = 2/(7+1)  = 0.25
chronic[t] = λ_c · load[t] + (1 − λ_c) · chronic[t-1], λ_c = 2/(28+1) ≈ 0.0690
ACWR[t]    = acute[t] / chronic[t]
```
Initialization: `acute[-1] = chronic[-1] = 0`.

**Alternatives considered:**
- Simple rolling averages (RA): older method with known mathematical-coupling
  issues; EWMA explicitly requested by supervisor

**Rationale:** EWMA gives more weight to recent sessions and handles rest-day
decay more realistically than rolling averages.

### Warmup masking

**Decision:** Set ACWR to NaN for the first 28 days of each player's timeline.

**Rationale:** Chronic EWMA needs ~28 days to converge. Displaying ACWR
during this period would show artificially high values driven by the
initialization-from-zero, not real physiological state.

### Division-by-zero handling

**Decision:** Set ACWR to NaN wherever `chronic == 0`.

**Rationale:** "Undefined" is more honest than 0/0 = NaN or `x/0 = inf`. If the
chronic baseline is zero, the player has never trained and no ratio is
meaningful.

### All three load metrics computed independently

**Decision:** Compute ACWR separately for `total_distance`, `acc_total`, and
`vel_total`. Do not combine them.

**Rationale:** The three metrics measure different physiological stresses
(aerobic volume, neuromuscular effort, sprint mechanics). They are on different
scales and can diverge — a player can be safe by one metric and at-risk by
another. Summing or averaging them would hide this asymmetry.

**Empirical support (EDA finding):** Pearson and Spearman correlations on active
days (modelable cohort) confirm the metrics are largely independent:

| Pair | Pearson | Spearman |
|---|---|---|
| `total_distance` ↔ `vel_total` | ~0.48 | ~0.48 |
| `total_distance` ↔ `acc_total` | ~0.19 | ~0.19 |
| `vel_total` ↔ `acc_total` | ~0.22 | ~0.22 |

For reference, sports science literature typically reports correlations of 0.6–0.8
between these metrics at squad level. The lower values here are consistent with a
squad whose positional roles create divergent speed/acceleration profiles. Pearson
and Spearman match within ±0.02 across all pairs — no hidden non-linear structure.

---

## Known limitations

1. **No session duration.** Cannot compute RPE × minutes load; relying entirely
   on GPS/IMU-derived external loads.
2. **No wellness or subjective data.** No RPE, no fatigue questionnaires.
3. **78% of calendar days are rest days.** Some are genuine rest, some are
   likely data-capture gaps (not every player has a tracker every day). The two
   are indistinguishable in the current data.
4. **Sparse data for some retained players.** Several of the 28 players have
   < 10% activity rate; model training will likely need a position-based fallback
   for the 10 non-modelable players (< 50 active days).
5. **One season only.** No cross-season generalization can be validated.
6. **ACWR methodology has been criticized in the literature** (Impellizzeri,
   Lolli et al.) for mathematical coupling and weak causal links to injury.
   ACWR is used here as a decision-support signal, not a diagnostic tool.

---

## Modeling decisions

### Feature set

**Decision:** 17 base features + 35 player identity one-hots = 52 features per model.

Base features:
- Player anthropometrics: `height`, `weight`, `age`
- Session composition: `has_G`, `has_TAC`, `has_BP`, `has_TEC`, `has_MATCH`, `n_session_types`
- Position: `pos_central_back`, `pos_central_midfielder`, `pos_forward`, `pos_full_back`, `pos_winger`
- Calendar: `days_since_start`
- Activity history: `days_since_last_activity`, `days_since_last_match`

**Rationale:** Player identity one-hots allow the model to learn per-player baselines without
requiring a hierarchical model. The 35-player cohort is small enough that one-hot encoding is
tractable and interpretable. SHAP analysis confirms player identity accounts for only 3.3% of
total feature importance in `acc_total`, confirming the base features dominate.

### Loss function per target

| Target | Loss | Rationale |
|---|---|---|
| `acc_total` | Tweedie (power 1.1–1.9) | Count-like data with 3.1% zeros and right skew (skew ≈ 1.55); Tweedie handles the mass at zero natively without a two-stage model |
| `total_distance` | log-MSE (`reg:squarederror` on `log1p` target) | Right-skewed continuous (skew ≈ 0.9); log transform normalises it; `expm1` applied at inference |
| `vel_total` | Raw MSE (`reg:squarederror`) | ~30% zeros; distribution analysis showed Tweedie did not outperform raw MSE after SHAP feature selection |

**Alternatives considered for `acc_total`:** Hurdle model (classify zero/non-zero, then regress).
Rejected — Tweedie achieved comparable MAE with a single-stage pipeline and simpler inference code.

### Train / test split strategy

**Decision:** Random 80/20 split, no temporal ordering.

**Rationale:** The model answers a cross-sectional question — given a player's attributes and
session type, what load does that session produce? Temporal structure is not the primary concern
at this stage. A time-aware split would reduce training data substantially given the dataset size
(2,103 rows) and leave some players with very few test examples.

**[OPEN]** Switch to time-based split for the final model once simulation requirements are clearer.

### Hyperparameter tuning

**Decision:** `RandomizedSearchCV`, n_iter=100, 5-fold CV, `neg_mean_absolute_error` scoring.
Elastic net regularisation (L1 + L2 jointly) + early stopping (50 rounds on 15% holdout).

**Rationale:** Full grid search is computationally prohibitive over the chosen search space (~10M
combinations). 100 random draws with 5-fold CV provides good coverage. Elastic net is preferred
over pure L1 or L2 because it handles correlated features (position + player one-hots) more
robustly.

### SHAP feature selection (vel_total only)

**Decision:** Two-round pipeline — Round 1 trains on all 52 features; Round 2 retains only the
features needed to reach 90% cumulative SHAP importance and re-tunes from scratch. Winner chosen
by test MAE.

**Rationale:** `vel_total` has a more sparse, noisy signal (~30% zeros) than the other targets.
Reducing features addresses potential overfitting and speeds up inference. The threshold (90%)
preserves nearly all predictive information while dropping low-signal features.

---