# DECISIONS

## Decision Log

### 1. Optimize for selective deployment, not candle prediction
Status: active

Reasoning:
- The intended use case is a sparse set of high-quality trades per week.
- Random-split candle prediction metrics do not answer the deployment question.
- The system should prefer `no trade` over forcing weak setups.

Implication:
- Candidate quality, calibration, expectancy, and weekly-cap performance matter more than raw accuracy.

### 2. Keep the sparse selector separate from the dense baseline
Status: active

Reasoning:
- The dense pipeline is still useful as a baseline and research comparison.
- The sparse selector serves a different decision problem: which engine-generated setups are worth taking.

Implication:
- Do not collapse the sparse selector back into a generic candle model.

### 3. Replace generic candidate logic with explicit setup families
Status: active

Reasoning:
- The earlier candidate engine reused overly generic logic across regimes.
- Distinct setup families are easier to reason about, debug, diagnose, and eventually score differently.

Current families:
- `TREND_PULLBACK_CONTINUATION`
- `BREAKOUT_RETEST_CONTINUATION`
- `RANGE_REJECTION_MEAN_REVERSION`
- `FAILED_BREAKOUT_REVERSAL`

Implication:
- Dataset rows, features, diagnostics, and evaluation should remain family-aware.

### 4. Richer trade-path labels are preferred over bare `WIN/LOSS`
Status: active

Reasoning:
- Binary labels hide expectancy, path quality, and time behavior.
- `net_r`, excursion stats, and timing stats better describe tradability.

Implication:
- `WIN/LOSS` remains a derived diagnostic, not the ultimate objective.

### 5. Validation decisions must be made on validation, not test
Status: active

Reasoning:
- Threshold selection on the test set leaks policy selection into the final evaluation.
- The pipeline must keep thresholding and calibration off the final holdout.

Implication:
- Use `train / validation / test` for the current workflow and move to rolling walk-forward next.

### 6. Prefer simple models until feature/state information is exhausted
Status: active

Reasoning:
- Data volume is limited and setup families are still evolving.
- Better labels, validation, and structure features will likely add more value than deep sequence models right now.

Implication:
- Prefer tree-based baselines, ranking/regression upgrades, and clearer diagnostics before considering more complex architectures.

### 7. Honest abstention is better than false confidence
Status: active

Reasoning:
- A selector that refuses to take trades is more useful than one that overstates weak edge.
- Recent calibrated runs show the current selector is still not decision-ready.

Implication:
- Do not lower standards just to produce trades.

### 8. Docs are part of the implementation
Status: active

Reasoning:
- This repo is evolving quickly and research phases matter.
- Future sessions need a low-token, accurate summary of the current architecture and bottlenecks.

Implication:
- Update `AGENTS.md`, `docs/ROADMAP.md`, `docs/KNOWN_ISSUES.md`, and `docs/research_upgrade_status.md` whenever the phase or findings change materially.

### 9. Sparse labels should honor family max-hold
Status: active

Reasoning:
- The candidate families already declare different intended holding periods.
- Labeling every sparse setup with one generic lookahead distorts family edge and can count late failures against trades that should already be flat.

Implication:
- Sparse labeling and point-in-time sparse audits should cap the realized path by `candidate.max_hold_bars`, bounded by the requested research lookahead.

### 10. Use `weekly_cap=1` as the strict selector diagnostic until the pool grows
Status: active

Reasoning:
- On the current 5-year ETH/USDT 1h sample, the sparse pool only produces about `2.5` candidates per week on the test slice.
- That means the default `weekly_cap=10` is not binding and does not tell us much about selective deployment.

Implication:
- Keep the official deployment target as `<= 10/week`, but use `weekly_cap=1` alongside `weekly_cap=10` when diagnosing whether the candidate pool has any narrow deployable edge at all.

### 11. Use BTC/USDC 15m as a selector control dataset
Status: active

Reasoning:
- ETH/USDT 1h is still the main deployment target, but its current candidate pool is weak enough that selector failure can be confounded with candidate-quality failure.
- BTC/USDC 15m now provides a larger sparse pool with a strongly positive ex-post oracle at both `weekly_cap=10` and `weekly_cap=1`.
- The current selector still fails there, which makes BTC/USDC 15m a useful control for testing whether model, calibration, or objective changes can exploit a healthier candidate universe.

Implication:
- Keep ETH/USDT 1h as the primary milestone, but sanity-check future selector and objective changes on BTC/USDC 15m before concluding that a model-stack change is or is not useful.
