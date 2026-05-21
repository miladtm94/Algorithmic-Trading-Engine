# ROADMAP

## Project Goal
Turn the current prototype into a research-valid and operationally realistic crypto setup-selection system, starting with ETH/USDT 1h.

## Current Phase
Selective-deployment research refactor.

The project is no longer optimizing for generic candle classification. The active goal is to improve sparse candidate quality, expectancy-aware labeling, validation realism, and eventual live alignment.

## Phase Summary

### Phase 1: Core Engine Foundation
Status: done

Delivered:
- market snapshot validation
- indicator computation
- regime classification
- candidate generation
- confluence scoring
- risk filters
- execution planning
- optional LLM validation

### Phase 2: Sparse Research Foundations
Status: done

Delivered:
- sparse dataset built from engine-generated setups
- richer realized-path labels including `net_r`, excursion stats, and timing stats
- validation split with purge support
- calibration support
- weekly-cap threshold evaluation

### Phase 3: Explicit Setup Families
Status: done

Delivered:
- `TREND_PULLBACK_CONTINUATION`
- `BREAKOUT_RETEST_CONTINUATION`
- `RANGE_REJECTION_MEAN_REVERSION`
- `FAILED_BREAKOUT_REVERSAL`
- family-aware dataset metadata
- family-aware feature plumbing
- family-level diagnostics in dataset/train outputs
- standalone sparse-dataset diagnostics for family expectancy, exit damage, and oracle limits
- sparse labels now honor family `max_hold_bars`
- selector training now reports overfit gaps and calibrated probability-range collapse

### Phase 4: Candidate Quality Tuning
Status: active

Current focus:
- redesign setup families and family-specific exits until the candidate pool itself is positive under honest labels
- reduce over-dominance of `BREAKOUT_RETEST_CONTINUATION`
- improve no-trade filters and family balance
- expand structure/state features only where they improve setup discrimination

Success criteria:
- healthier family mix
- smaller but better candidate pool
- improved post-cost expectancy by family
- selector begins surfacing a non-degenerate probability spread

### Phase 5: Validation Hardening
Status: planned

Planned work:
- rolling walk-forward evaluation
- purge and embargo across folds
- family-level stability reporting
- rule baseline, selector baseline, and oracle upper-bound reporting on each fold

### Phase 6: Model Objective Upgrade
Status: planned

Planned work:
- move beyond pure `WIN/LOSS` classification
- add expectancy-aware ranking or `net_r` regression
- keep models simple unless added complexity is clearly justified

### Phase 7: Live Wiring Alignment
Status: planned

Planned work:
- align live engine candidate universe with sparse selector research
- load the selector artifact in the live decision path
- isolate or retire legacy dense-model usage for this workflow

## Current Working Milestone
Candidate family coverage tuning for ETH/USDT 1h.

## Next Concrete Tasks
1. Replace the breakout-retest 20-bar `prior_high/low` reference with a 60- or 80-bar structural swing so the family fires only on real breakouts, and widen the stop to ~`0.7 * atr` below the breakout level.
2. After breakout-retest is no longer net-negative on test, redesign trend-pullback and failed-breakout family-specific exits until stop-loss-heavy subsets no longer average worse than `-1.2` net R.
3. Run `make diagnose-signal-dataset asset=ETH/USDT timeframe=1h weekly_cap=1` and `weekly_cap=10` after each candidate-rule change.
4. Confirm whether the next rule change improves `weekly_cap=1` validation oracle before trusting any selector threshold.
5. Add the next tranche of structure/state features only if they help explain family differences.
6. Rebuild the sparse dataset and retrain after each meaningful family-rule change.
7. Reassess whether ranking/regression is justified only after the candidate-pool oracle turns positive across all folds, not just `weekly_cap=1`.
8. Use BTC/USDC 15m as a control dataset for selector-objective and calibration checks, because its oracle is already strong enough to test whether the model stack can exploit a healthier candidate universe.

## Paper-Trading Path (rule-based, evidence-aligned)
A conservative `strict` preset is now wired through CLI, Makefile, and the backtester:
- families: `RANGE_REJECTION_MEAN_REVERSION` only
- weekly cap: `1` per ISO week
- confluence floor: `55` (matches research)
- per-trade risk: `0.5%`
- minimum risk-reward: `1.0` (matches the family's native TP1)
- kill-switch: `5` consecutive losses (was `3` — `3` locks out the engine for months at this trade tempo)

Backtest commands:
- `make backtest-strict history_csv=data/historical/ETH_USDT_1h.csv start=2024-10-01` — expect ~`15` trades, `60%` win, `+3.5%` total return, `~1.5%` max DD, Sharpe near `1.1`.
- `make backtest-strict history_csv=data/historical/ETH_USDT_1h.csv` (full 5-year) — expect ~`21` trades, `~24%` win, `-5.4%` total return; the same policy is regime-dependent and not yet a stable edge.

Live paper command: `make run-paper-strict`. See `docs/research_upgrade_status.md` "Paper-Trading Runbook" for the full pre-flight, sanity check, and abort criteria.

## Current Measured State
- 5-year ETH/USDT 1h history is available locally: `43,820` candles.
- After tightening range-rejection to require the structural range extreme (2026-04-25), the sparse dataset is now `330` rows, down from `670`, with a much higher-quality range-rejection subset.
- Overall realized quality (post-change): `34.2%` win, `-0.19%` avg PnL, `-0.28` avg net R, `2.26`/week.
- Test slice (post-change): `66` rows, `37.9%` win, `-0.13%` avg PnL, `-0.19` avg net R, `1.83`/week.
- Test family expectancy (post-change):
  - `RANGE_REJECTION_MEAN_REVERSION`: `17` rows, `64.7%` win, `+0.33%` avg PnL, `+0.35` avg net R (previously `49.4%` win / `-0.04` net R on `77` rows).
  - `FAILED_BREAKOUT_REVERSAL`: `26` rows, `38.5%` win, `-0.22` net R.
  - `BREAKOUT_RETEST_CONTINUATION`: `12` rows, `8.3%` win, `-0.67` net R. Longs are `0.0%` win on `5` rows at `-1.20` net R.
  - `TREND_PULLBACK_CONTINUATION`: `11` rows, `27.3%` win, `-0.46` net R.
- Ex-post `weekly_cap=1` oracle on test improves to `36` trades, `+0.08%` avg PnL, `+0.12` net R overall; on just range-rejection it is `8` rows, `87.5%` win, `+1.05` net R.
- Validation slice is now thin (`26` rows, `-0.65` net R); it covers a choppy 2024-summer window, so a single slice should not yet be read as a regression.
- Default selector training is not re-run yet after this rule change because the overall pool is still negative; selector improvements without candidate-pool improvement remain confounded.
- Exploratory BTC/USDC 15m research: `70,128` candles and `2,242` sparse candidate rows; full test pool is still negative at about `-0.12%` avg PnL / `-0.27` net R, but the ex-post oracle is strongly positive at both `weekly_cap=10` and `weekly_cap=1`.
- On that BTC/USDC 15m control set, default calibrated selector training still abstains because validation calibrated `p(WIN)` only spans about `35.7%` to `43.6%`, and an exploratory no-calibration run selected trades but stayed negative on test.

## Operational Rule
After any change to candidate logic, labels, core features, or validation:
1. rebuild sparse dataset
2. run sparse dataset diagnosis at `weekly_cap=1` and `weekly_cap=10`
3. retrain sparse selector and note overfit/probability-range warnings
4. update `docs/research_upgrade_status.md`
5. update this roadmap if the active phase or next tasks changed
