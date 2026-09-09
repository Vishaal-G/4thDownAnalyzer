# 4th Down Decision Model

## Goal

A tool that takes a 4th-down game state (field position, distance to go, score
differential, time remaining) and recommends go-for-it / field goal / punt, backed
by a model trained on real NFL play-by-play data (via `nfl_data_py`).

Equally important: this is a **guided learning project**. The user knows Python well
but has never used nfl_data_py, pandas, PyTorch, Flask, or React. The point is for
them to learn each library hands-on, with Claude acting as a teacher/guide.

## Working style — read this before writing any code

**Do not implement stages for the user.** For each stage:
- Explain the concept and point at the exact function/class/doc needed.
- A tiny illustrative snippet (a few lines) to show a pattern is fine — never a full solution.
- The user writes the actual code themselves.
- Review what they share (code, errors, output) and help debug — don't rewrite it
  for them unless they're stuck and explicitly ask you to just do it.
- Checkpoint at the end of each stage before moving to the next.

## Confirmed design decisions

1. The model computes an **independent expected-value estimate for each option**
   (go / FG / punt) from historical outcomes and recommends the max — it does not
   imitate historical coach behavior (which is systematically biased, e.g. over-punting).
2. Reuse nflfastR's precomputed `epa`/`ep`/`wp` columns (already in nfl_data_py's
   play-by-play data) as the value signal, rather than deriving an Expected Points
   model from scratch. Keeps scope to the four target libraries.
3. Three separate small PyTorch regression models (one per decision: GO / FIELD_GOAL /
   PUNT), each trained only on historical rows where that decision occurred, predicting
   `epa` from `[yardline_100, ydstogo, score_differential, game_seconds_remaining]`.
   A single shared multi-head model is a stretch goal for later, not v1.
4. Win-probability-based decisions (`wp`/`wpa` instead of `epa`) are a Phase 2, after
   the EPA version works end-to-end.

## Architecture

```
nfl_data_py (pbp data)
      │  pandas: filter to 4th downs, clean, label decision
      ▼
data/clean_4th_downs.parquet  [yardline_100, ydstogo, score_differential,
                                game_seconds_remaining, decision, epa, season]
      │  split by decision into 3 subsets
      ▼
3 small PyTorch MLPs (GO / FIELD_GOAL / PUNT) → predicted EPA
      │
      ▼
backend/decision_engine.py: run all 3 models, argmax → recommendation
      │
      ▼
Flask API: POST /api/recommend
      │
      ▼
React dashboard (frontend/): form → fetch → show 3 EVs + recommended call
```

## Environment (Stage 0 — done)

- Python venv at `~/.venvs/4thDownAnalyzer` (native fs, not `/mnt/c`) with
  torch (CPU, 2.14.0), pandas (3.0.5), numpy, nfl_data_py (0.3.2), flask (3.1.3),
  jupyter installed. Activate with `source ~/.venvs/4thDownAnalyzer/bin/activate`.
- `frontend/` — Vite/React scaffold, dependencies installed, untouched placeholder content.
- `data/` and `*.parquet` are gitignored — data artifacts stay local.

## Stages & progress

- [ ] **Stage 1 — Data familiarization** (nfl_data_py + pandas): pull 2014–2023 pbp
      data, filter to 4th downs, explore `play_type` value_counts and key columns,
      save `data/raw_4th_downs.parquet`. *(almost done — see "Where we left off" below
      for the one remaining step)*
- [ ] **Stage 2 — Feature engineering & labeling** (pandas): map plays to
      GO/FIELD_GOAL/PUNT, build `data/clean_4th_downs.parquet`.
- [ ] **Stage 3 — Naive baseline** (pandas): mean `epa` by decision × distance bucket,
      via `groupby` — sanity check for Stage 4.
- [ ] **Stage 4 — PyTorch models**: 3 MLPs predicting `epa`, split by season,
      `backend/train.py`, weights in `backend/models/{go,fg,punt}.pt`.
- [ ] **Stage 5 — Decision engine**: `backend/decision_engine.py`,
      `recommend(yardline_100, ydstogo, score_diff, seconds_remaining) -> dict`.
- [ ] **Stage 6 — Flask API**: `backend/app.py`, `POST /api/recommend`.
- [ ] **Stage 7 — React dashboard**: replace `frontend/src/App.jsx` placeholder with
      the input form + results panel.
- [ ] **Stage 8 — Backtest & iterate**: compare model recommendations against real
      historical coach decisions, tabulate agreement rate and biggest EV gaps.

Full stage-by-stage detail (what to learn, what to build, deliverables) is in the
original plan; the checklist above is the source of truth for where we are — check a
box off and add a one-line note here as each stage's deliverable is confirmed working.

## Where we left off (2026-09-08)

Working in `notebooks/01_explore.ipynb` (VS Code + Jupyter, kernel = the
`4thDownAnalyzer` venv — VS Code must be connected to the WSL remote for that kernel
to be visible/selectable; opening the plain Windows VS Code window will not see it).

Done so far in that notebook:
- Pulled `pbp` for 2014–2023 via `nfl_data_py.import_pbp_data(years=range(2014,2024),
  downcast=True)`.
- Confirmed shape/dtypes (396 columns: 206 float32, 183 object, 6 int32, 1 int64).
- Filtered to `fourth_downs = pbp[pbp['down'] == 4]` — 41,526 rows.
- Ran `fourth_downs['play_type'].value_counts()`: punt 23393, field_goal 9603,
  pass 4012, run 2270, no_play 2133, qb_kneel 32 (sums to 41,443; the remaining 83
  rows have `play_type` == NaN, confirmed via `.isna().sum()`). These three groups
  (`no_play`, `qb_kneel`, NaN) are noise, not real decisions — will be dropped in
  Stage 2. Real decisions collapse to: `pass`+`run` → GO, `field_goal` → FIELD_GOAL,
  `punt` → PUNT.
- Ran `.describe()` on `yardline_100`, `ydstogo`, `score_differential`,
  `game_seconds_remaining` — ranges all sane. Investigated the `ydstogo` max (46):
  it's a real punt where a "running into the kicker" penalty was *declined* — the
  46-yards-to-go itself came from earlier penalties/plays in that same drive, not
  the punt play itself. Good example of why raw play-by-play needs cleaning.

**One step still not done — resume here:** the notebook's last cell calls
`fourth_downs.to_parquet("fourth_downs.parquet")` (no folder, wrong filename per our
convention) and, as of this note, has not actually been executed — `data/` is still
empty on disk. Next session: fix that line to
`fourth_downs.to_parquet("../data/raw_4th_downs.parquet")`, run it, confirm the file
appears in `data/`, then Stage 1 is complete and Stage 2 (labeling) can start.

Reminder of the working style for whoever picks this up: per the rules above, walk
the user through fixing/running that line rather than doing it for them.
