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

**Rationale for exclusion:** Insufficient data to compute meaningful ACWR,
unreliable metadata, or both. These players are not in the modelling target
population (the current first-team squad).

**Final player count:** 29

**[OPEN]** Confirm with supervisor that these exclusions are correct. In
particular, confirm whether 42978, 86086, 93116 are genuinely trialists /
academy call-ups or first-team squad members whose metadata was miscoded.

### Player retained with modified metadata: 15795

- 30 rows across the full season (Jul 2024 – Jun 2025)
- Age 19, plausible DOB (2005-06-02)
- `height = 180` and `weight = 200` match the placeholder pattern seen in
  excluded players

**Decision:** Retain the player; set `height` and `weight` to NaN.

**Rationale:** Player has a real season-long appearance pattern — likely a
Castilla (reserve team) player on occasional first-team training — so their
training data is valid. The anthropometric values are clearly placeholder
defaults and cannot be trusted. Nulling is more honest than keeping false values
or imputing a guess.

**[OPEN]** Request real height/weight values for player 15795 from supervisor /
club HR system.

---

## Row-level data fixes

### Outlier: `total_distance = 32,299.89` for player 94884, match 2025-02-15

- Only `total_distance` is anomalous; `acc_band7plus_total_effort_count = 15`
  and `velocity_band6plus7_total_distance = 23.47` are normal for a match
- Physiologically impossible (a professional match is typically 10–12 km)

**Decision:** Set `total_distance` to NaN for this row. Retain the row and the
other two load metrics.

**Alternatives considered:**
- Imputing the player's median match distance: rejected — invents a specific
  value the model and coaches would treat as real, hiding uncertainty in a
  decision-support tool
- Dropping the row: rejected — loses the information that the player played a
  match on this date, and the other two metrics are valid

**Rationale:** The selective corruption of one metric (not all three) indicates
a parsing / export error on a single value, not a sensor malfunction. Setting
that one value to NaN preserves all valid information and propagates "unknown"
honestly downstream.

**[OPEN]** Can the raw value be recovered from the source GPS system?

---

## Aggregation decisions

### Daily aggregation granularity

**Decision:** Aggregate row-level periods to one row per `(player_id, date)`.

- Load metrics: **summed** across periods on the same day
- Exercise types: preserved via **pivot** — columns `count_G`, `count_TAC`, etc.
- Per-type loads also preserved: `td_G`, `td_TAC`, etc.
- Static metadata (position, height, weight, DOB): `first` (does not vary
  within a player)

**Rationale:** ACWR is computed on daily load. Daily aggregation is the correct
unit. Pivoting exercise types rather than dropping them preserves session
composition information for the model, which will be asked to predict load
given a *planned* session composition.

### Full calendar grid with zero-filled rest days

**Decision:**
- Each player's grid starts on their **first observed session**
- All players' grids end on the **global maximum date** (2025-06-26)
- Days with no observed activity are filled with zero load and `is_rest = 1`

**Rationale:**
- EWMA computation requires a continuous daily series; gaps would under-weight
  rest days in the chronic load baseline and inflate ACWR
- Aligning end dates across players allows direct comparison (e.g., team-wide
  ACWR on any given date)
- Starting at first observed session avoids fabricating rest history before the
  player entered the dataset

**Resulting grid:** 8,872 rows; 76.1% rest days (2,119 active days).

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
`hsr_total`. Do not combine them.

**Rationale:** The three metrics measure different physiological stresses
(aerobic volume, neuromuscular effort, sprint mechanics). They are on different
scales and can diverge — a player can be safe by one metric and at-risk by
another. Summing or averaging them would hide this asymmetry.

---

## Known limitations

1. **No session duration.** Cannot compute RPE × minutes load; relying entirely
   on GPS/IMU-derived external loads.
2. **No wellness or subjective data.** No RPE, no fatigue questionnaires.
3. **78% of calendar days are rest days.** Some are genuine rest, some are
   likely data-capture gaps (not every player has a tracker every day). The two
   are indistinguishable in the current data.
4. **Sparse data for some retained players.** 7 of 29 players have < 10%
   activity rate; model training will likely need a position-based fallback for
   these.
5. **One season only.** No cross-season generalization can be validated.
6. **ACWR methodology has been criticized in the literature** (Impellizzeri,
   Lolli et al.) for mathematical coupling and weak causal links to injury.
   ACWR is used here as a decision-support signal, not a diagnostic tool.

---