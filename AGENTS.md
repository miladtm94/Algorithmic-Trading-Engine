# AGENTS.md

## Project Summary
This repository contains `AI Trading Engine`, a modular hybrid trading engine and research pipeline for crypto setup generation, filtering, and selective ML-based trade selection.
Primary goal: build a deployable ETH/USDT 1h research and execution stack that prefers `no trade` over weak trade and selects only a small number of high-quality setups.
Current stage: prototype moving through research refactor toward production hardening.

## What matters most
The agent must quickly understand:
1. what the project does
2. how the codebase is structured
3. what is already working
4. what is currently being built
5. how to verify changes safely
6. what documents must be updated after implementation

## Repository Map
- `src/ai_trading_engine/...` — core engine, research pipeline, ML utilities, and shared models
- `scripts/...` — operational entrypoints for history fetch, dataset builds, training, evaluation, visualization, and demos
- `tests/...` — unit and integration tests for engine behavior and research-pipeline behavior
- `data/historical/...` — downloaded OHLCV history used for research
- `data/features/...` — built dense and sparse research datasets
- `data/models/...` — saved model artifacts, metadata, and prediction audits
- `data/reports/...` — generated HTML model comparison reports
- `docs/...` — project documentation and working-session reference material
- `docs/ROADMAP.md` — current implementation plan and next phase
- `docs/DECISIONS.md` — architectural and product decisions
- `docs/KNOWN_ISSUES.md` — bugs, edge cases, and known limitations
- `docs/research_upgrade_status.md` — detailed live log of the current selective-deployment research upgrade
- `README.md` — setup and human-facing overview

## Tech Stack
- Language(s): Python 3.11+
- Framework(s): standard-library-first package with optional `scikit-learn`, `joblib`, `openai`, `ccxt`
- Database: none required for core workflow; local CSV artifacts are the main research store
- Infra / deployment: local CLI, optional Docker, optional Telegram, optional exchange connectivity
- Package manager: `pip` with editable install via `pyproject.toml`

## Run / Build / Test
Use the following commands exactly unless there is a good reason not to:
- Install: `make setup && make install-ml`
- Dev server: `make run`
- Build: `make build-signal-dataset asset=ETH/USDT timeframe=1h`
- Lint: `make lint`
- Typecheck: `make typecheck`
- Test: `make test`

Useful research commands:
- Sparse selector train: `make train-signal asset=ETH/USDT timeframe=1h`
- Sparse selector audit: `make evaluate-signal-at asset=ETH/USDT timeframe=1h at='2026-04-18 08:00'`
- Sparse dataset diagnosis: `make diagnose-signal-dataset asset=ETH/USDT timeframe=1h weekly_cap=1`
- Family diagnostics: `make diagnose-signal-families asset=ETH/USDT timeframe=1h`
- Dense baseline train: `make train asset=ETH/USDT timeframe=1h`

## Working Rules
- Do not introduce new dependencies unless necessary.
- Prefer minimal, focused changes over broad rewrites.
- Preserve existing architecture unless the task explicitly requires refactoring.
- Keep functions/modules small and readable.
- Add or update tests when behavior changes.
- Do not break existing public APIs without documenting it.
- If assumptions are required, state them in code comments or docs.
- For this repo, prefer improving candidate quality, labels, validation, and deployment realism over chasing generic ML metrics.

## Coding Standards
- Follow existing style in the touched files.
- Reuse existing helpers/components before creating new ones.
- Keep naming consistent with the repository.
- Prefer explicit error handling over silent failure.
- Prefer typed interfaces/contracts where applicable.
- Keep research output honest: if the model should abstain, let it abstain.

## Definition of Done
A task is complete only if:
- the requested feature/fix is implemented
- relevant tests pass
- lint/typecheck pass when applicable to the touched code
- affected docs are updated
- `docs/ROADMAP.md` is updated if plan/status changed
- `docs/DECISIONS.md` is updated for non-trivial design choices
- `docs/KNOWN_ISSUES.md` is updated if limitations remain
- `docs/research_upgrade_status.md` is updated for meaningful research-pipeline changes

## Current Status
### Working
- Core rule-based engine pipeline: validation, regime classification, candidate generation, confluence, risk, execution, and optional LLM validation
- Richer sparse research labels with `net_r`, path stats, and validation-based threshold selection
- Explicit setup-family candidate engine with family metadata and diagnostics
- Sparse dataset build and selector training pipeline for ETH/USDT 1h
- Standalone sparse-dataset audit for family expectancy, exit damage, and oracle limits
- Sparse labels now honor family `max_hold_bars`, and selector training now reports overfit/probability-collapse warnings
- 5-year ETH/USDT 1h history is now available locally and the strict sparse-selector workflow is trainable again
- Cross-asset/timeframe control runs now work on BTC/USDC 15m, including sparse dataset build, oracle diagnostics, and selector training
- Engine supports `allowed_setup_families` whitelist and `max_trades_per_iso_week` cap natively, applied uniformly in research, backtest, and paper modes
- Backtester now uses `dataset.build_research_snapshot` and family-aware horizon exits, so backtest behavior matches the sparse research pipeline
- A `strict` preset is wired through `make run-paper-strict`, `make backtest-strict`, and `--preset strict` on the CLI; a paper-trading runbook is in `docs/research_upgrade_status.md`

### In Progress
- Selective-deployment research refactor for ETH/USDT 1h
- Candidate family coverage tuning and state/structure feature upgrades
- Cross-asset control checks to separate candidate-quality limits from selector-modeling limits

### Next Priorities
1. Tighten `BREAKOUT_RETEST_CONTINUATION` entry rules to stop the `-1.20` net R long-side failure (trend-stack alignment inside the rule; tighter `HIGH_VOLATILITY` regime branch).
2. After breakout-retest is healthier, redesign trend-pullback and failed-breakout family-specific exits until stop-loss-heavy subsets no longer average worse than `-1.2` net R.
3. Use the sparse dataset diagnosis after each candidate-rule change to verify family expectancy, side asymmetry, stop-loss damage, and oracle limits.
4. Treat `weekly_cap=1` as the strict selective-deployment diagnostic until the pool is rich enough for `weekly_cap=10` to bind.
5. Only revisit model-objective changes after the candidate pool is no longer overall-negative under honest labels.
6. Use BTC/USDC 15m as a control dataset for selector and calibration experiments, because its oracle is strong enough to expose model-stack failures that ETH/USDT 1h can hide behind weak candidate quality.

## Known Constraints
- The sparse selector currently abstains after calibration; there is no deployable threshold yet.
- The selector currently overfits the training slice and calibrated validation probabilities collapse below the default threshold grid.
- Walk-forward validation with multi-fold purge/embargo is not fully implemented yet.
- Labels now honor family max-hold, but exits are still not fully family-specific beyond stop/TP/horizon templates.
- Some market-context fields are still placeholders or simplified historical proxies.
- Live engine wiring is not yet fully aligned with the sparse selector deployment path.
- After the 2026-04-25 range-rejection tightening, the ETH/USDT 1h sparse pool is `330` rows and the candidate rate is `1.83`/week on test, so `weekly_cap=1` remains the useful strict diagnostic there.
- The full ETH/USDT 1h pool is still overall-negative (`-0.28` net R overall, `-0.19` net R on test), even though range-rejection alone is now positive on the test slice and on the `weekly_cap=1` oracle.
- On BTC/USDC 15m, the ex-post oracle is strongly positive, but the current calibrated selector still collapses below threshold and the uncalibrated exploratory run remains negative on test.

## Safe Change Policy
Before large refactors:
1. understand existing flow
2. identify impacted modules
3. preserve backward compatibility where possible
4. update tests and docs together

## Documentation Update Policy
When a feature is completed or behavior changes, update:
- `docs/ROADMAP.md`
- `docs/DECISIONS.md` if architectural reasoning changed
- `README.md` if setup/usage changed
- `docs/KNOWN_ISSUES.md` if issues were added/resolved
- `docs/research_upgrade_status.md` if research behavior, artifacts, or current findings changed

## Handoff Notes for Future Sessions
When starting a new session, first read:
1. `AGENTS.md`
2. `docs/ROADMAP.md`
3. `docs/DECISIONS.md`
4. `docs/KNOWN_ISSUES.md`
5. `docs/research_upgrade_status.md`
6. relevant module-level `AGENTS.md` if present

Then summarize:
- project purpose
- current phase
- active task
- risks/blockers
- recommended next step
