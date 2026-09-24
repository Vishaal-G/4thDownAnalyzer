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

- [x] **Stage 1 — Data familiarization** (nfl_data_py + pandas): pull 2014–2023 pbp
      data, filter to 4th downs, explore `play_type` value_counts and key columns,
      save `data/raw_4th_downs.parquet`. Done 2026-09-09 — `data/raw_4th_downs.parquet`
      confirmed on disk (41,526 rows).
- [x] **Stage 2 — Feature engineering & labeling** (pandas): map plays to
      GO/FIELD_GOAL/PUNT, build `data/clean_4th_downs.parquet`. Done 2026-09-09 —
      dropped no_play/qb_kneel/NaN play_type rows, mapped pass+run→GO,
      field_goal→FIELD_GOAL, punt→PUNT (39,278 rows: 6,282/9,603/23,393), no
      missing epa, trimmed to `[yardline_100, ydstogo, score_differential,
      game_seconds_remaining, decision, epa, season]`.
- [x] **Stage 3 — Naive baseline** (pandas): mean `epa` by decision × distance bucket,
      via `groupby` — sanity check for Stage 4. Done 2026-09-12 —
      `notebooks/03_baseline.ipynb`: bucketed `ydstogo` via `pd.cut` into
      short/medium/long (0–2/3–6/7+), `groupby(['decision','ydstogo_bucket'])['epa'].mean()`.
      Results sane: GO short 0.282 vs long -0.758, FIELD_GOAL flat ~0.04–0.06 across
      buckets (expected — FG attempts are gated by field position, not `ydstogo`),
      PUNT worst on short (-0.375), roughly neutral on long (-0.009).
- [x] **Stage 4 — PyTorch models**: 3 MLPs predicting `epa`, split by season,
      weights in `backend/models/`. Done 2026-09-18 — built in
      `notebooks/04_train.ipynb` (not yet ported to `backend/train.py`, see note
      below). Train/test split at season <= 2021 vs > 2021. Each model: `Linear(4,8)
      → ReLU → Linear(8,1)`, `nn.MSELoss`-equivalent (manual `torch.mean((pred-target)**2)`),
      `Adam(lr=0.001)`, 1000 epochs (GO plateaus by ~epoch 2000 in a longer run,
      final loss ~8.44 vs naive mean-baseline 8.82 — GO's outcome is inherently
      high-variance, see note below). **Key fix mid-stage:** input features
      (`yardline_100`, `ydstogo`, `score_differential`, `game_seconds_remaining`)
      had to be standardized (zero mean/unit std, stats computed on train only,
      applied to both train/test) before training — unscaled, `game_seconds_remaining`'s
      much larger magnitude dominated training and the GO model couldn't beat
      the naive baseline. Validated by bucketing each model's *predictions* by
      `ydstogo` (short/medium/long, same cut points as Stage 3) on both train and
      test sets and comparing to the Stage 3 baseline — all three models tracked
      the real pattern closely on both splits (e.g. GO test: 0.304/0.048/-0.620 vs
      baseline 0.282/0.055/-0.758). Weights saved to `backend/models/model_GO.pt`,
      `model_FieldGoal.pt`, `model_Punt.pt` (named per-decision rather than the
      originally-planned `go.pt`/`fg.pt`/`punt.pt` — use these actual filenames in
      Stage 5).
- [x] **Stage 5 — Decision engine**: `backend/decision_engine.py`,
      `recommend(yardline_100, ydstogo, score_diff, seconds_remaining) -> dict`.
      Done 2026-09-23 — `scaling_stats.json` stores per-decision `mean`/`std` (used
      to scale raw inputs before feeding each model) plus `min`/`max` (raw,
      unscaled — used for the guardrail below), keys `GO`/`PUNT`/`FIELD_GOAL`.
      `recommend()`: pairs each label with its model via `labelToModel`; for each
      decision, scales the 4 raw inputs with **that decision's own** mean/std,
      orders them per `feature_names`, builds a `(1, 4)` float32 tensor, and runs
      a forward pass under `torch.no_grad()`, extracting the predicted EPA via
      `.item()`. Returns `{"GO": {"epa": ..., "out_of_range": [...]}, "PUNT": {...},
      "FIELD_GOAL": {...}, "BEST": <label>}`. **Guardrail:** each raw input is also
      checked against that decision's own `min`/`max`; anything outside is added to
      that decision's `out_of_range` list, and `'BEST'` is computed via `max()` over
      only the decisions with an empty `out_of_range` — a decision extrapolating
      outside its training range can no longer win the recommendation just because
      its (unreliable) predicted EPA is highest. **Known gap:** no handling yet for
      all three decisions being out-of-range at once (the filtered `max()` would
      raise `ValueError` on an empty sequence) — not hit by testing, left as-is.
      Sanity tests passed: 4th & 1 at opp. 40 (tied, lots of time) → GO (0.79 vs
      FIELD_GOAL 0.35 vs PUNT -0.67), all in-range. 4th & 15 at own 20 →
      FIELD_GOAL's `yardline_100=80` correctly flagged out-of-range (its training
      max was `49.0`, confirmed by hand in `04_train.ipynb`), so `'BEST'` falls
      through to GO once FIELD_GOAL is excluded — see the "Where we left off" note
      for why GO winning there is itself a known, separate limitation (not caught
      by this guardrail) flagged for Stage 8.
- [x] **Stage 6 — Flask API**: `backend/app.py`, `POST /api/recommend`. Done
      2026-09-23 — `app.py` creates the Flask app, registers `/api/recommend`
      (`methods=['POST']`), pulls the JSON body via `request.get_json(silent=True)`,
      and calls `recommend(**inputData)` (dict-unpacked directly into `recommend()`'s
      4 named parameters, since the JSON keys match `feature_names` exactly). Flask
      3.x auto-converts the returned dict to a JSON response. **Error handling
      written directly by Claude, at the user's explicit request**, after the happy
      path was already working and tested: non-JSON/empty body → 400
      `{"error": "Request body must be valid JSON"}`; a missing required field →
      400 naming which field(s), checked against `decision_engine.feature_names`
      (one source of truth for the field list, not duplicated); a wrong-typed or
      unexpected-extra field → `recommend(**inputData)` wrapped in
      `try`/`except TypeError`, 400 with the underlying error message. Verified
      end-to-end via curl for both Stage 5 sanity cases (opp. 40/4th & 1 →
      `BEST: GO`, all in-range; own 20/4th & 15 → `FIELD_GOAL.out_of_range: [80]`,
      `BEST: GO`) — matches the bare-`recommend()` results exactly, confirming the
      Flask layer doesn't change behavior. Run via `python app.py`
      (`app.run(debug=True)`), default `http://127.0.0.1:5000`.
- [x] **Stage 7 — React dashboard**: replace `frontend/src/App.jsx` placeholder with
      the input form + results panel. Done 2026-09-23 — **built entirely by Claude
      at the user's explicit request** ("build it yourself, I don't need to be a
      part of it"), unlike every prior stage. Used `EnterPlanMode`/`ExitPlanMode`
      first; user picked a visual direction ("Stadium lights": dark charcoal/navy,
      turf-green/end-zone-gold/steel-blue accents, glow on the winning call) via
      `AskUserQuestion`. Plan saved at
      `~/.claude/plans/happy-bouncing-kazoo.md`. Structure: `src/lib/api.js`
      (fetch wrapper for `POST /api/recommend`), `src/components/DecisionForm.jsx`
      (yard line / yards to go / score differential / MM:SS time, converted to
      `game_seconds_remaining`), `src/components/ResultModal.jsx` +
      `EpaBar.jsx` (popup showing the `BEST` call plus all 3 EPAs as a
      zero-anchored comparison — bars support negative EPA, shared scale across
      all three, not self-scaled), `src/index.css`/`App.css` re-themed. No new
      npm dependencies. **No backend changes** — CORS avoided via a Vite dev
      proxy (`vite.config.js`: `/api` → `http://127.0.0.1:5000`) rather than
      touching `backend/app.py`. Removed the unused Vite-starter assets
      (`hero.png`, `react.svg`, `vite.svg`, `public/icons.svg`); replaced the
      favicon. **Colors went through the `dataviz` skill**, not picked by eye —
      the categorical GO/FIELD_GOAL/PUNT trio was validated with
      `node scripts/validate_palette.js "#3987e5,#d95926,#199e70" --mode dark
      --surface "#151a24" --pairs all` (ALL CHECKS PASS) before use; status
      colors (out-of-range warning badge) kept separate from the categorical
      series colors per the skill's rules. **Verification:** `npm run lint`
      (0 warnings/errors) and `npm run build` both clean; backend confirmed
      responding correctly via curl (same results as Stage 5/6). Could **not**
      get a self-driven browser screenshot in this sandbox — `chromium-cli` isn't
      installed and no local Playwright/Chromium fallback was available (see
      queued feedback draft); also hit a WSL/Windows interop snag where the
      Windows-native npm's Vite server wasn't reachable from the WSL shell via
      `localhost`, only via the WSL vEthernet address (`172.28.0.1`) after
      adding `--host 0.0.0.0` — irrelevant for the user's own browser, which
      reaches it at the normal `http://localhost:5173/`. **User confirmed the
      look afterward** ("the frontend looks good") and asked for two follow-up
      passes, both done: (1) stripped em/en dashes from all visible copy
      (hero subtext, yard-line helper, out-of-range tooltip, modal note) and
      reworded to read less AI-generated; (2) leaned further into the football
      theme — a live "4TH & X" broadcast-style down-and-distance badge on the
      form, faint yard-line stripes in the hero background, and small
      football/goalpost/punt-arc icons (`DecisionIcon.jsx`) replacing the
      plain color dots in the EPA rows and modal headline.
- [x] **Stage 8 — Backtest & iterate**: compare model recommendations against real
      historical coach decisions, tabulate agreement rate and biggest EV gaps.
      Done 2026-09-23 — **built entirely by Claude at the user's explicit
      request** ("can you do the backtest for me then and explain ur thought
      process"). `backend/backtest.py`: for every test-split play (season >
      2021, same held-out split used throughout, never trained on), calls
      `recommend()` and `recommend_wp()` on the play's real situation and
      compares `BEST` to the real `decision` column. Reports overall
      agreement, agreement by actual decision, a full confusion matrix, the
      value gap on disagreements (model's predicted value for its own pick
      minus for what the coach did), and the 10 biggest gaps. Saves
      `data/backtest_epa.csv`/`data/backtest_wp.csv` (full per-play results,
      gitignored like other `data/` artifacts).
      **Headline result:** overall agreement is low — EPA 45.4%, WP 48.1% —
      but it's wildly uneven by decision: FIELD_GOAL agreement is high
      (EPA 86.0%, WP 68.2% — coaches' kick/no-kick calls are mostly
      validated), GO agreement is much better under WP than EPA (46.0% →
      79.4%, a big win for Phase 2), and **PUNT agreement is very low both
      ways (EPA 27.2%, WP 28.4%)** — most real punts, the model would have
      gone for it instead (confusion matrix: of 4646 real punts, EPA
      recommends GO on 2957 of them, WP on 2848). This is exactly the
      "coaches over-punt" bias the project was built around (Confirmed
      Design Decision #1) — now quantified, not just assumed, and it matches
      real NFL analytics research (Romer's original work, nflfastR/4th-down-
      bot community findings) on this same question.
      **Checked this wasn't blowout-garbage-time noise before trusting it:**
      restricting to competitive one-score games (|score_differential| <= 8)
      barely moves the punt-agreement numbers (24.7%/30.1%), and blowouts
      are only 4.6% of the test set with agreement rates close to the
      overall figure either way — so the low PUNT-agreement finding holds up
      outside extreme score situations, it isn't an artifact of them.
      Short-yardage real punts (`ydstogo` <= 2) in competitive games are the
      most extreme slice: the model would have gone for it **98%+ of the
      time** — matching Stage 3's original naive baseline almost exactly
      (GO short +0.282 vs PUNT short -0.375), just now shown against real
      decisions at scale instead of an aggregate bucket mean.
      **Caveat applied throughout, not just mentioned once:** every model
      here was trained only on rows where that exact decision was actually
      made, so evaluating a model on a play where a *different* decision
      happened is extrapolation for that model — the same selection-bias
      risk flagged since Stage 5. The out-of-range guardrail catches the
      worst of it (0% of disagreements here had the *actual* decision itself
      flagged, for what that's worth), but it's univariate and known to miss
      joint/interaction sparsity, so "not flagged" isn't the same as
      "verified reliable." **Spot-checked the individual "biggest gap" list
      critically rather than taking it at face value** — a couple of the
      single largest WP gaps turned out to be a team already leading by 14
      going for it late (arguably a defensible aggressive call the model
      just disagrees with, not a clear coach error) and a team up 42 (a
      42-point blowout) punting in the final seconds, where win probability
      is already ~100% either way and the model's WPA prediction there is
      likely just noise, not real signal — flagging this so the "biggest
      gaps" table isn't read as a clean list of coaching mistakes.
- [x] **Stage 9 — Phase 2: win-probability decision engine** (done out of the
      original order — Stage 8's backtest hasn't happened yet — triggered by a
      concrete case found while testing Stage 7, see below). Done 2026-09-23
      — **built entirely by Claude at the user's explicit request** ("since I
      did the epa model, can you do everything for the wpa model"). Confirmed
      `wpa` exists in `data/raw_4th_downs.parquet` (dropped during Stage 2's
      trim) and that `clean_4th_downs.parquet`'s index is a row-identity
      subset of `raw`'s, so `wpa` was attached by index join rather than
      re-deriving Stage 2's filter/label logic — saved as
      `data/clean_4th_downs_wp.parquet` (39,278 rows, same rows as the EPA
      clean set plus a `wpa` column). Confirmed by hand that the existing
      `backend/models/scaling_stats.json` mean/std/min/max are numerically
      identical whether computed against the EPA or WP clean set (same input
      features, same per-decision train-row selection, season <= 2021 —
      only the *target* column differs) — so it's reused as-is, no new
      scaling file. New `backend/train_wp.py` mirrors Stage 4's architecture/
      methodology exactly (`Linear(4,8) → ReLU → Linear(8,1)`, `Adam(lr=0.001)`,
      manual MSE, same season train/test split) but targets `wpa`.
      **Key tuning finding:** `wpa` is a much smaller-magnitude, noisier
      target than `epa` (std ≈ 0.058 vs epa's much larger spread — most single
      plays barely move win probability). At Stage 4's 1000 epochs, GO beat
      its naive mean-baseline but FIELD_GOAL and PUNT did not; diagnosed by
      sweeping epochs (1000/2000/4000/8000) and learning rate — both catch up
      to and pass their baselines by ~2000 epochs, diminishing returns past
      ~4000, and GO doesn't overfit at 4000 either — so **all three trained at
      4000 epochs**, all three beat their naive baseline (GO test MSE 0.00727
      vs baseline 0.00744; FIELD_GOAL 0.00386 vs 0.00398; PUNT 0.00247 vs
      0.00255), and predicted-`wpa`-by-`ydstogo`-bucket tracked the same
      directional pattern as the raw groupby baseline on both train and test.
      Weights: `backend/models/model_GO_wp.pt`, `model_FieldGoal_wp.pt`,
      `model_Punt_wp.pt`. `backend/decision_engine.py` gained a parallel,
      clearly-separated block (not a replacement — existing `recommend()`/
      `EPAToResult` code is untouched): `model_GO_wp`/`model_FieldGoal_wp`/
      `model_Punt_wp`, `labelToModelWp`, and `recommend_wp(...)` — same
      structure/style as `recommend()` (scale per-decision → tensor → forward
      pass → out-of-range guardrail → argmax over in-range decisions only),
      returning `{"GO": {"wpa": ..., "out_of_range": [...]}, ..., "BEST": ...}`.
      **Validation — the actual point of this work:** re-ran the 4th & 2 at
      the opp. 2, down 6, 0:30-left case that surfaced the EPA-vs-WP gap
      (see the Stage 7 entry above): EPA recommends FIELD_GOAL (+0.058 EPA);
      `recommend_wp()` correctly flips this to **GO** — `FIELD_GOAL`'s `wpa`
      is actually negative (-0.015, worse than GO's -0.008), because a field
      goal down 6 doesn't give a realistic path to winning. The two original
      Stage 5 sanity cases (opp. 40/4th & 1 and own 20/4th & 15) still both
      return `BEST: GO` under WP too, so no regression there.
      **Wired into the API and dashboard** (done 2026-09-23, same day, at the
      user's follow-up request — "wire the other function to front end too
      and show difference as well"): `backend/app.py` now calls both
      `recommend()` and `recommend_wp()` per request and nests the WP result
      under a new `"WP"` key in the JSON response — purely additive, the
      existing top-level `GO`/`PUNT`/`FIELD_GOAL`/`BEST` (EPA) shape is
      unchanged, so nothing that already worked broke. `EpaBar.jsx` was
      generalized from a hardcoded `epa` prop to `value`/`formatValue`/
      `metricName`, so the same component renders either metric.
      `ResultModal.jsx` now shows **both** recommendations: the win-
      probability call is the primary headline (it's the more complete
      signal — the whole point of Phase 2), with a visible note when the two
      disagree ("Points alone would say &lt;X&gt; instead") or agree, followed by
      two stacked comparison panels — win probability added (formatted as a
      signed percentage, e.g. `+3.1%`, since raw `wpa` decimals like `0.031`
      aren't intuitive) and expected points added (existing formatting,
      visually secondary/dimmer) — each with its own independent bar scale,
      since the two metrics are on very different orders of magnitude and
      sharing one scale would make the WP bars invisible next to EPA's.
      Verified via curl that `/api/recommend` returns both nested correctly
      (including on the down-6/opp.-2 case: top-level `BEST: FIELD_GOAL`,
      `WP.BEST: GO`); `npm run lint`/`npm run build` clean.

      **Follow-up fixes from the first visual check (same day):** user
      screenshotted the down-6/opp.-2 case and flagged the bars as "weird."
      Diagnosis: the original bar length encoded raw distance from zero, which
      is technically correct but backwards for a recommendation UI when every
      option is negative (as here) — the *winner* (closest to zero) got the
      visually *shortest* bar. Redesigned `EpaBar.jsx`/`ResultModal.jsx`
      (`relativeBarPcts()`) so bar length now means "how close to the best
      option," not "distance from zero": the recommended row always renders
      at 100% width, others scale down by how far behind they are (floored at
      4% so a bad-but-present option doesn't disappear); the signed numeric
      value is still shown as text either way, so no information was lost,
      only what the bar length communicates changed. Simplified `.epa-track`
      from a zero-anchored diverging bar (pos/neg classes, center zero-line)
      to a single left-anchored bar. Also, at the user's request, reworded
      score differential from a signed number (confusing: typing "-6" to
      mean "down by 6") to an "Ahead"/"Behind" toggle plus a plain positive
      number in `DecisionForm.jsx` — converted to the signed
      `score_differential` the model expects right before the API call, same
      pattern already used for the MM:SS → `game_seconds_remaining`
      conversion; the API/model contract itself is untouched. `npm run lint`/
      `npm run build` clean; both dev servers confirmed still serving.

      **v2 of the WP model — engineered feature (same day):** user found
      another case WP didn't catch: down 5, 4th & 1 at the opp. 20, 0:40 left
      — logically similar to the down-6 case (a FG only gets you to down 2,
      no realistic time to get the ball back), but WP had GO/FIELD_GOAL
      within 0.1 percentage points of each other, essentially a coin flip,
      not a confident call. Neither option was out-of-range, so this wasn't
      the guardrail/extrapolation problem — a different, subtler gap: the
      model had no direct signal for "does a field goal even help here,"
      only score_differential to infer it from indirectly. Added a 5th,
      engineered input to the WP models only: `score_after_fg =
      score_differential + 3` (the margin right after an otherwise-successful
      FG — >=0 means a FG alone ties/wins). `backend/train_wp.py` v2:
      computes it inline (no new data file needed), can no longer reuse
      `scaling_stats.json` (different feature count than the EPA models) so
      it now saves its own `backend/models/scaling_stats_wp.json`, and
      `build_model()` is `Linear(5,8)` instead of `Linear(4,8)`. All three
      still beat their naive baseline at the same 4000 epochs (GO test MSE
      0.00729, FIELD_GOAL 0.00387, PUNT 0.00245 — all ~same as v1).
      `decision_engine.py`'s WP block updated to match: loads
      `scaling_stats_wp.json`, `feature_names_wp` (base 4 + the new one),
      5-input architectures, and `recommend_wp()` computes `score_after_fg`
      from the raw `score_differential` argument before scaling/the
      out-of-range check (which now also covers this feature, same loop, no
      special-casing needed). **Result: fixed the target case** —
      `recommend_wp()` now gives GO a clear, decisive margin (+1.2% vs
      FIELD_GOAL's -0.5%), not a coin flip; the original down-6 case still
      says GO too, and even more decisively than v1 (-0.6% vs -2.1%, wider
      than v1's -0.8%/-1.5%). **Side effect, not obviously a bug:** the
      Stage 5 "own 20, 4th & 15" sanity case, where EPA and v1-WP both said
      GO, now has v2-WP saying PUNT instead. Punting from your own 20 on
      4th & 15 is what a real coach would almost always do — recall this
      exact scenario (GO on long distance from own territory) was already
      flagged back in Stage 5 as a known, documented limitation of the GO
      model's selection bias. So this flip plausibly makes the tool *more*
      realistic, not less, though it can't be attributed with certainty to
      the new feature specifically vs. ordinary retraining variance (a new
      random weight initialization). Worth watching in Stage 8's backtest.
      Old 4-feature WP weights were overwritten (no v1 preserved separately)
      -- if ever needed for comparison, this note plus `train_wp.py`'s git
      history has everything needed to reproduce v1. Verified via curl
      through the full `/api/recommend` endpoint, not just
      `decision_engine.py` directly.

Full stage-by-stage detail (what to learn, what to build, deliverables) is in the
original plan; the checklist above is the source of truth for where we are — check a
box off and add a one-line note here as each stage's deliverable is confirmed working.

## Where we left off (2026-09-23)

**All 9 stages are done.** The tool works end-to-end: form → Flask →
`decision_engine.py` (both EPA and WP models) → dashboard, backtested against
real historical coach decisions. Summary of where everything lives:

- `data/`: `raw_4th_downs.parquet` (Stage 1), `clean_4th_downs.parquet`
  (Stage 2, EPA features), `clean_4th_downs_wp.parquet` (Stage 9, same rows
  + `wpa`), `backtest_epa.csv`/`backtest_wp.csv` (Stage 8, full per-play
  backtest results). All gitignored, local only.
- `notebooks/01-04`: Stages 1-4 (exploration, labeling, naive baseline,
  training the 3 EPA models) — the only guided-learning stages; everything
  from Stage 5 onward that's marked "built by Claude" below was done at the
  user's explicit request, not guided.
- `backend/models/`: `model_GO.pt`/`model_FieldGoal.pt`/`model_Punt.pt` (EPA,
  4 inputs) + `scaling_stats.json`; `model_GO_wp.pt`/etc. (WP v2, 5 inputs —
  see Stage 9 note for the `score_after_fg` engineered feature) +
  `scaling_stats_wp.json` (its own file, different feature count than EPA's).
- `backend/decision_engine.py`: `recommend()` (EPA, Stage 5) and
  `recommend_wp()` (WP, Stage 9), each with its own out-of-range guardrail.
- `backend/train_wp.py`: trains the WP models (EPA training is still only in
  `notebooks/04_train.ipynb` — porting it to a `backend/train.py` remains
  the one still-optional, never-done item, see below).
  `backend/backtest.py`: Stage 8.
- `backend/app.py`: one endpoint, `POST /api/recommend`, returns both EPA
  (top-level) and WP (nested under `"WP"`) — see Stage 6/9 notes.
- `frontend/`: the "Stadium lights" dashboard (Stage 7), showing both
  recommendations with the win-probability call as primary, EPA as a visible
  secondary comparison. User has seen and confirmed this in-browser, with
  three follow-up rounds of polish since (dash/copy cleanup, deeper football
  theming, the relative-scale bar redesign + Ahead/Behind score toggle).

**Still-open, non-blocking items**, all previously flagged and never picked
back up because nothing required it:
1. `recommend()`/`recommend_wp()` don't handle all-three-decisions-out-of-
   range (would raise `ValueError` on the filtered `max()`) — never hit in
   testing or the backtest.
2. `04_train.ipynb` (EPA training) was never ported to a `backend/train.py`
   — always marked optional, `train_wp.py` and `backtest.py` exist as plain
   scripts instead since Claude authored those directly.
3. The univariate out-of-range guardrail is known to miss joint/interaction
   sparsity (flagged since Stage 5, restated in Stage 8's backtest note) —
   a real limitation, not something either stage tried to fully solve.

If picking this project back up for further work, the natural next things
(none asked for yet) would be: acting on Stage 8's over-punting finding
somehow (e.g. surfacing "how does this compare to real coaching?" in the
dashboard itself), or the interaction-sparsity guardrail gap.

Notebooks so far (VS Code + Jupyter, kernel = the `4thDownAnalyzer` venv — VS Code
must be connected to the WSL remote for that kernel to be visible/selectable):
- `notebooks/01_explore.ipynb` — Stage 1 exploration.
- `notebooks/02_label.ipynb` — Stage 2 labeling.
- `notebooks/03_baseline.ipynb` — Stage 3 naive baseline.
- `notebooks/04_train.ipynb` — Stage 4 model training/validation, plus (added
  2026-09-23) the `min`/`max` per-decision aggregation and the updated
  `json.dump` cell.

**Finding from sanity-testing the guardrail — carry into Stage 8:** at 4th & 15
from your own 20, once FIELD_GOAL is correctly excluded (its `yardline_100=80` is
flagged out-of-range — confirmed by hand that FIELD_GOAL's training data topped
out at `yardline_100=49.0`, a ~66-yard kick), `recommend()` still lands on GO
(0.097 EPA) over PUNT (0.055) — but a real coach would essentially always punt
there. The guardrail is **univariate** (checks each feature against its own
min/max independently), so it misses unrealistic *combinations* of otherwise
in-range features: neither `ydstogo=15` nor `yardline_100=80` alone is outside
GO's marginal training range, but GO's training data is dominated by
short-yardage/midfield situations, so this specific (long distance) × (deep in
own territory) corner of the input space is likely sparse even though no single
feature trips the flag. This is exactly the "GO on long distance from own
territory" selection-bias risk this doc already flagged for Stage 8 — now
confirmed with a concrete example rather than just a theoretical concern.
Decided **not** to try to patch this into `recommend()` now — a proper fix is
real out-of-distribution/density detection, bigger scope than a min/max
guardrail — leave it for Stage 8's backtest to quantify how often it happens
against real historical coach decisions.

**Open decision, asked but not yet answered as of this note:** whether to do
Stage 7 (React dashboard) next, or skip/defer straight to Stage 8 (backtest vs.
real coach decisions). The user has said they care mostly about the ML + backend
and less about frontend/UI work, but has NOT yet explicitly said to skip Stage 7
— don't mark it skipped until they say so; check with them first if picking this
back up. Either way, Stage 8 is where the extrapolation/selection-bias examples
already found (very long FGs, GO on long distance from own territory) should get
surfaced and quantified against real historical coach decisions; also watch for
punts near midfield.

Optional: port `04_train.ipynb` to `backend/train.py` (notebook is the only copy
of training code; a shared architecture helper there could also remove the
triple-defined architecture in the engine).

Reminder of the working style for whoever picks this up: per the rules above, explain
the concept and point at the function/doc, let the user write the actual code, and
debug from what they share rather than writing it for them.
