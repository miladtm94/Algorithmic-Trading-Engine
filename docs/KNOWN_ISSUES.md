# KNOWN ISSUES

## Active Issues

### 1. Breakout-retest and trend-pullback families are the current worst offenders
Severity: high
Status: open

Why it matters:
- After tightening range-rejection (2026-04-25), the dominant remaining damage on the ETH/USDT 1h test slice is `BREAKOUT_RETEST_CONTINUATION` and `TREND_PULLBACK_CONTINUATION`.
- Breakout-retest longs are `0.0%` win on `5` test rows at `-1.20` net R; the family as a whole is `8.3%` win on `12` rows at `-0.67` net R.
- Trend-pullback is `27.3%` win on `11` rows at `-0.46` net R, and its stop-loss rows average `-1.25` net R.

Current impact:
- These families pull the overall test-slice expectancy back down from a positive range-rejection contribution to an overall `-0.19` net R.
- The next concrete rule-level change belongs in breakout-retest entry gating (trend-stack alignment inside the rule itself, not just the regime gate) and in family-specific exits for trend-pullback.

### 2. Candidate pool is still negative under honest exits
Severity: high
Status: open

Why it matters:
- Even after the 2026-04-25 range-rejection tightening, the full ETH/USDT 1h sparse pool is still negative at `-0.28` net R overall and `-0.19` net R on the test slice.

Current impact:
- The bottleneck remains candidate design and family exit design, not just selector modeling.
- Selector retraining is intentionally deferred until the candidate pool stops being overall-negative on honest labels.

Latest diagnosis:
- `weekly_cap=1` test oracle is `+0.12` net R overall (`+1.05` net R restricted to range rejection, `+0.18` net R on failed breakout, `-0.51` net R on breakout retest, `-0.49` net R on trend pullback).
- `weekly_cap=10` no longer binds on the current pool (`1.83` candidates/week on test), so `weekly_cap=1` remains the primary strict diagnostic.

### 3. Sparse selector still fails to produce deployable thresholds
Severity: high
Status: open

Why it matters:
- The current selector does not yet convert either weak or moderately promising candidate pools into a deployable trading policy.

Current impact:
- There is still no deployable selective-trading policy from the current sparse model, and the classifier/calibration stack may be a bottleneck in its own right.

Latest note:
- with 5-year history the strict workflow is trainable again, but it still does not produce a deployable positive-expectancy threshold
- the latest default selector run overfits (`100%` train accuracy) and calibrated validation probabilities only span about `19%` to `46%`, so the default threshold grid forces abstention
- a BTC/USDC 15m control run is more encouraging at the candidate level but still fails at the selector level:
  - the full test pool is still negative at about `-0.12%` avg PnL / `-0.27` net R
  - the ex-post oracle is strongly positive at `weekly_cap=10` (`74.2%` win rate, `+0.26%` avg PnL, `+0.71` avg net R) and `weekly_cap=1` (`95.2%` win rate, `+0.70%` avg PnL, `+1.94` avg net R)
  - the default calibrated selector still collapses to about `35.7%` to `43.6%` validation probabilities and abstains
  - an exploratory no-calibration run selected `134` test trades at the validation-chosen threshold but still lost money (`34.3%` precision, `-0.13%` avg PnL)

### 4. Labels are richer but exits are still mostly generic
Severity: medium
Status: open

Why it matters:
- Family-specific exits are still only partially implemented.
- That means setup-family edge can still be distorted by a generic exit template.

Current impact:
- sparse labels now honor family `max_hold_bars`, which is more realistic
- but the stop/TP path is still mostly generic, so `net_r` is better than raw `WIN/LOSS` but still not the final form of the research target

### 5. Walk-forward validation is not fully implemented
Severity: medium
Status: open

Why it matters:
- Single-holdout temporal validation is better than random split, but still not enough for robust deployment claims.

Current impact:
- Stability across time is not proven yet.

### 6. Live engine and sparse selector are not fully wired together
Severity: medium
Status: improved (partially)

Why it matters:
- Research can improve while the live path still uses older logic or artifacts.

Current impact:
- Research and live engine now share the **rule-based** decision policy: backtester now uses `dataset.build_research_snapshot`, the engine enforces a family whitelist + ISO-weekly cap, the LLM-rule no longer hard-codes a confluence floor that diverges from the configured threshold, and the backtester respects family-aware `max_hold_bars` horizon exits. So a `strict` preset run in research, backtest, and paper modes uses the same policy.
- The sparse selector itself is still not loaded into the live decision path — that is a separate Phase 7 task and remains open.

### 7. Some context features are still simplified historical proxies
Severity: medium
Status: open

Why it matters:
- Placeholder or simplified market-context features can pollute research if treated as real signal.

Current impact:
- Research realism is still constrained by data quality on some non-price inputs.

### 8. Default `weekly_cap=10` is not meaningfully selective on the current ETH/USDT 1h reference pool
Severity: medium
Status: open

Why it matters:
- The sparse ETH/USDT 1h test pool currently averages only about `2.5` candidates per week.
- That means `weekly_cap=10` mostly replays the whole pool instead of testing true selectivity.

Current impact:
- The more informative strict diagnostic right now is `weekly_cap=1`, which still shows only a small unstable edge.

## Recently Improved

### 1. Threshold selection leakage to test set
Status: improved

Resolution:
- Threshold choice now comes from the validation slice, not the final test slice.

### 2. Candidate engine genericity
Status: improved

Resolution:
- Candidate generation is now explicit by setup family instead of pretending all regimes share one generic template.

### 3. Missing low-token project handoff docs
Status: improved

Resolution:
- `AGENTS.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`, and this file now provide session-start context.

### 4. Missing repeatable family-generation diagnostics
Status: improved

Resolution:
- `scripts/diagnose_signal_families.py` and `make diagnose-signal-families` now provide a repeatable family-level inspection path.

### 5. Missing sparse-dataset expectancy diagnostics
Status: improved

Resolution:
- `scripts/diagnose_signal_dataset.py` and `make diagnose-signal-dataset` now audit family expectancy, side asymmetry, exit-reason damage, and realized oracle limits from the exact sparse CSV used for selector training.

### 6. Range-rejection trigger admitted mid-range BB-only setups
Status: improved

Resolution:
- `_build_range_rejection_long/short` now require the structural range extreme to be touched; the BB-only alternative trigger was removed and the BB tag now only raises setup quality when both conditions align.
- On the rebuilt ETH/USDT 1h sparse dataset, range-rejection test-slice expectancy moved from `-0.04` net R on `77` rows to `+0.35` net R on `17` rows, and the `weekly_cap=1` test oracle on range rejection is `+1.05` net R on `8` rows.

### 7. Engine, backtester, and rule-based LLM gate were silently misaligned
Status: improved

Resolution:
- Backtester now uses `dataset.build_research_snapshot` instead of its own ad-hoc snapshot builder, so candidate generation in research and backtest is bit-for-bit identical.
- Backtester staleness allowance bumped to 10 years of minutes (the prior `999_999` clamp rejected anything older than ~1.9 years and silently zeroed out 5-year backtests).
- Backtester now enforces family-aware `max_hold_bars` horizon exits, matching `dataset.label_trade_path`.
- `_rule_based_validation` no longer hard-codes a `confluence < 75` reject — the engine's configured `confluence_threshold` is the single source of truth.
- Engine now supports `allowed_setup_families` and `max_trades_per_iso_week` natively, and a `strict` CLI preset wires this end-to-end.

### 8. Strict-preset paper trading is honest but not yet a deployable edge
Status: documented

Resolution:
- A `strict` preset (RANGE_REJECTION only, weekly cap 1, confluence floor 55, risk 0.5%/trade, kill-switch 5) is now available via `make run-paper-strict` and `make backtest-strict`.
- On the recent ~18-month test slice the preset is `+3.55%` total return, `60%` win rate, Sharpe `1.10`. Across the full 5 years it is `-5.4%` (with kill-switch) or `-1.97%` (kill-switch disabled). This is regime-dependent edge, not a stable strategy.
- Paper-trade only as observational deployment. See the runbook in `docs/research_upgrade_status.md`.

## When Updating This File
Update this file when:
- an issue is resolved
- a limitation remains after implementation
- a new blocker is discovered
- a research caveat meaningfully affects interpretation of results
