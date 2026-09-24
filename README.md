# 4th Down Analyzer

**Should you go for it, kick, or punt?** A decision-support tool that takes
a 4th-down game situation — field position, distance to go, score, time left
— and recommends a call, backed by models trained on ten seasons of real NFL
play-by-play data (2014–2023, via [`nfl_data_py`](https://github.com/nflverse/nfl_data_py)).

Rather than imitating what real coaches historically did (which is a known,
documented bias — coaches punt far more often than the numbers support), the
model estimates the value of **each option independently** from historical
outcomes and recommends whichever comes out on top.

## What it does

- Takes a 4th-down situation as input (yard line, yards to go, score
  differential, time remaining).
- Runs it through **six small PyTorch models** — one per play call (go for
  it / field goal / punt), trained twice over: once to predict **EPA**
  (expected points added) and once to predict **win probability added**,
  the metric that actually accounts for score and clock context.
- Flags any recommendation that falls outside the range of situations its
  model actually saw in training, so it doesn't confidently extrapolate into
  the unknown.
- Shows both the win-probability call and the points-only call side by side,
  so you can see when the two disagree and why.

## Why it's more than a lookup table

- **Two independent value signals, not one.** A field goal can look great in
  raw points-added terms while doing almost nothing for your actual win
  probability (e.g. kicking to cut a 6-point deficit to 3 with 30 seconds
  left doesn't give you a realistic path to winning). Building the
  win-probability layer surfaced this exact gap, then an engineered feature
  (the margin left over after a made field goal) was added specifically to
  fix it — see `CLAUDE.md` for the before/after.
- **Backtested against real history, not just spot-checked.** Comparing the
  model's recommendation to what coaches actually did on every held-out test
  play (2022–2023) shows only ~45–48% overall agreement — and it's uneven:
  ~86% agreement on field goal calls, but only ~27–28% on punts. On
  short-yardage 4th downs in competitive games specifically, the model would
  have gone for it instead of punting **98%+ of the time** — a large-scale,
  quantified confirmation of the "coaches over-punt" finding well known in
  NFL analytics circles (e.g. Romer's original research, the nflfastR
  4th-down-bot community).
- **Honest about its own blind spots.** Every model here only ever saw
  historical rows where that exact decision was made, so it's extrapolating
  whenever it's asked to judge a road not taken. An out-of-range guardrail
  catches the most obvious cases (e.g. a "field goal" from 80 yards out);
  the rest is documented as a known limitation, not hidden.

## Tech stack

| Layer | Tools |
|---|---|
| Data | `nfl_data_py` (nflfastR play-by-play), `pandas` |
| Modeling | `PyTorch` — 6 small MLPs (`Linear → ReLU → Linear`), one per (decision × value signal) |
| Backend | `Flask` REST API |
| Frontend | `React` + `Vite`, hand-styled (no CSS framework) |

## Architecture

```
nfl_data_py (play-by-play data)
      │  pandas: filter to 4th downs, clean, label the decision made
      ▼
data/clean_4th_downs*.parquet
      │  split by decision (GO / FIELD_GOAL / PUNT)
      ▼
6 small PyTorch MLPs → predicted EPA and predicted win probability added
      │
      ▼
backend/decision_engine.py: run all models, apply an out-of-range guardrail,
                             recommend the best in-range option
      │
      ▼
Flask API — POST /api/recommend
      │
      ▼
React dashboard — situation form → both recommendations, compared
```

## Running it locally

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install torch pandas nfl_data_py flask
cd backend
python app.py            # http://127.0.0.1:5000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev               # http://localhost:5173
```

The Vite dev server proxies `/api/*` to the Flask server, so no CORS setup
is needed. Pretrained model weights are included under `backend/models/`, so
the app runs immediately without retraining. The underlying play-by-play
data itself isn't checked in (it's pulled fresh via `nfl_data_py`); to
reproduce the pipeline from scratch, work through `notebooks/01`–`04` and
`backend/train_wp.py` in order.

## Project structure

```
notebooks/        Data exploration, cleaning, baseline, and EPA model training
backend/
  decision_engine.py   Core recommendation logic + the out-of-range guardrail
  train_wp.py           Trains the win-probability models
  backtest.py            Backtests recommendations against real coach decisions
  app.py                  Flask API
  models/                  Trained weights + per-feature scaling stats
frontend/          React dashboard
CLAUDE.md          Full build log — every design decision, finding, and
                   dead end, in the order they happened
```

## Known limitations

- The out-of-range guardrail checks each input feature independently, so it
  can miss situations where every feature is individually plausible but the
  *combination* was rarely or never seen historically (e.g. going for it on
  4th & long from deep in your own territory).
- All values are model estimates from historical outcomes, not guarantees.
