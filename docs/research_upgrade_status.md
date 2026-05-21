# Research Upgrade Status

Last updated: 2026-04-25

This is the live implementation tracker for the ETH/USDT 1h research-pipeline upgrade. It also records cross-asset control experiments when they materially change how we interpret the selector stack. Update it whenever we change labels, validation, candidate logic, feature design, or live deployment wiring.

## Objective

Move the pipeline away from generic classification and toward selective deployment:

- prefer `no trade` over weak trade
- enforce a practical cap of `<= 10` trades per week
- rank setups by deployable expectancy, not candle prediction
- keep research, saved artifacts, and live decision rules aligned

## Implemented

### Step 1: richer trade-path labels

Status: implemented in code

Files:

- [src/ai_trading_engine/dataset.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/dataset.py)
- [src/ai_trading_engine/signal_learning.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/signal_learning.py)
- [scripts/build_dataset.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/build_dataset.py)
- [scripts/build_signal_dataset.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/build_signal_dataset.py)

What changed:

- labels now keep `risk_pct`, `net_return_pct`, `net_r`
- labels now keep `max_favorable_r`, `max_adverse_r`
- labels now keep `bars_to_target`, `bars_to_stop`
- labels now emit `r_bucket` and `meta_label`
- existing `outcome`, `pnl_pct`, `MFE/MAE pct`, and exit fields are preserved for backward compatibility

What works:

- both dense and sparse dataset builders can now emit richer realized-path information
- downstream code can still read `outcome` and `pnl_pct` exactly as before
- same-candle TP/SL ambiguity still resolves pessimistically with stop-first logic

What still needs work:

- the label is still tied to the current candidate stop/target template
- family max-hold is now honored in the sparse path, but the stop/TP exit template is still not fully family-specific
- the live engine is not yet ranking on `expected_net_r`

### Step 2: selector validation cleanup

Status: implemented in code

Files:

- [src/ai_trading_engine/validation.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/validation.py)
- [scripts/train_signal_model.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/train_signal_model.py)
- [scripts/evaluate_signal_model.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/evaluate_signal_model.py)

What changed:

- added a shared validation utility module
- training now uses `train / validation / test` instead of choosing thresholds on the test set
- temporal training supports `purge_rows` to reduce overlap leakage
- probability calibration support added with `none`, `platt`, and `isotonic`
- saved test predictions now mark `selected_recommended` using the actual weekly-cap policy instead of raw threshold hits
- metadata now records Brier score and calibration-bin summaries

What works:

- threshold selection source is now the validation slice, not the final test slice
- the saved selector artifact and the audit script now agree on calibrated probability handling
- selection CSVs are now consistent with the weekly-cap evaluation logic

What still needs work:

- this is still a single holdout workflow, not full rolling walk-forward
- there is no embargo-aware multi-fold evaluator yet
- calibration is only as good as the current candidate pool and current target

### Step 3: regression tests for the upgrade foundation

Status: implemented in code

Files:

- [tests/test_research_pipeline.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/tests/test_research_pipeline.py)

What changed:

- added tests for `net_r` and timing labels
- added tests for stop-first ambiguity handling
- added tests for weekly-cap selection consistency
- added tests for purged temporal splits

### Step 4: explicit setup-family candidate engine

Status: implemented in code

Files:

- [src/ai_trading_engine/models.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/models.py)
- [src/ai_trading_engine/signal_generation.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/signal_generation.py)
- [src/ai_trading_engine/signal_learning.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/signal_learning.py)
- [src/ai_trading_engine/feature_extractor.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/feature_extractor.py)
- [src/ai_trading_engine/demo_data.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/demo_data.py)
- [scripts/build_signal_dataset.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/build_signal_dataset.py)
- [tests/test_research_pipeline.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/tests/test_research_pipeline.py)

What changed:

- replaced the generic candidate constructor with explicit setup families
- implemented `trend pullback continuation`
- implemented `breakout retest continuation`
- implemented `range rejection mean reversion`
- implemented `failed breakout reversal`
- candidate rows now carry `setup_family`, `setup_quality`, `max_hold_bars`, and `reference_level`
- sparse-dataset features now include setup-family flags and family-state features like range width, range position, pullback depth, breakout distance, rejection wick bias, and compression state
- dataset build and selector training now print and save family-level diagnostics
- added repeatable family-generation diagnostics via `scripts/diagnose_signal_families.py`
- relaxed trend-pullback and breakout-retest gates, and compressed research support/resistance levels into wider structural zones
- regression tests now cover family generation and setup metadata flow into features
- demo data was updated so the stricter setup logic still produces a valid engine trade in tests

What works:

- the candidate engine is now family-explicit instead of pretending every regime is one generic template
- the sparse dataset rebuild succeeded with the new family metadata
- family-level stats are now visible in both the dataset build output and the saved selector metadata
- trend pullback now appears in the rebuilt sparse dataset instead of remaining completely absent
- research and engine tests pass with the stricter candidate rules
- the candidate universe became much closer to the selective-deployment objective: `693` rows instead of `9552`

What still needs work:

- family coverage is still imbalanced and now too sparse overall for the default strict temporal split
- trend pullback now exists, but still only contributes a small number of rows
- the selector still does not find deployable calibrated probabilities from this narrower pool
- family-specific exit logic is still missing, so labels are richer but exits are still generic
- setup-quality scoring is still heuristic and not yet validated against post-cost expectancy

### Step 5: standalone sparse-dataset diagnostics

Status: implemented in code

Files:

- [src/ai_trading_engine/signal_diagnostics.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/signal_diagnostics.py)
- [scripts/diagnose_signal_dataset.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/diagnose_signal_dataset.py)
- [tests/test_research_pipeline.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/tests/test_research_pipeline.py)
- [Makefile](/Users/mtmamaghani/GitHub/AI-Trading-Engine/Makefile)

What changed:

- added reusable sparse-dataset summary helpers for expectancy, `net_r`, MFE/MAE, exit mix, and trades/week
- added ex-post weekly oracle selection by realized `net_r`
- added a CLI that audits all/train/validation/test splits, family quality, family+side quality, family+exit damage, and oracle family mix
- added `make diagnose-signal-dataset`
- added regression tests for summary, grouping, and weekly oracle behavior

What works:

- the dataset audit reads the exact sparse CSV used by selector training
- the audit can be run with `weekly_cap=10` to match the selector cap or stricter caps like `weekly_cap=1` to inspect selectivity limits
- the output now makes it obvious when the oracle is merely replaying the whole sparse test pool because candidate frequency is already below the cap

What still needs work:

- the audit is diagnostic only; it does not fix weak family logic by itself
- the next setup-rule pass should use these diagnostics to reduce stop-loss clusters and side asymmetry before model-objective changes

### Step 6: family-aware sparse label horizon and selector warnings

Status: implemented in code

Files:

- [src/ai_trading_engine/signal_learning.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/signal_learning.py)
- [scripts/evaluate_signal_model.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/evaluate_signal_model.py)
- [scripts/train_signal_model.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/scripts/train_signal_model.py)
- [tests/test_research_pipeline.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/tests/test_research_pipeline.py)

What changed:

- sparse labels now cap the realized path by `candidate.max_hold_bars`, bounded by the requested research lookahead
- point-in-time sparse audits now use the same family-aware label horizon and report it when it differs from the requested lookahead
- selector training now prints probability ranges and explicit warnings when train/validation overfit or calibration collapse would force abstention

What works:

- range-rejection and failed-breakout rows are no longer judged by an unnecessarily long generic sparse horizon
- selector output now makes probability collapse obvious instead of silently looking like an ordinary abstention
- regression coverage now includes the family-aware label-horizon helper

What still needs work:

- honoring family max-hold made the labels more honest, but it did not fix candidate expectancy on its own
- the next rule pass still belongs in candidate design and family exits, not in relaxing selector standards

## Not Implemented Yet

These are still the highest-priority open items.

### Candidate family coverage tuning

Status: in progress

Needed:

- expand the trend-following families to a trainable but still selective candidate count
- check whether `breakout retest continuation` is still too broad relative to the weekly deployment goal
- tighten family-specific rules based on diagnostics now that family-level reporting is available

Why it matters:

- the candidate engine is now explicit, but the pool is still too concentrated in one family
- selector quality will stay capped if one family dominates the pool with weak setups

### Feature upgrade for structure/state memory

Status: not started in code

Needed:

- swing-state and range-state features
- compression and failed-break features
- regime-memory features
- session/time-of-day features

Why it matters:

- current feature set is still too snapshot-heavy

### Live engine wiring

Status: not started in code

Needed:

- connect the sparse selector path to live engine decisions
- ensure live inference uses the same candidate universe and thresholding policy as research
- retire or isolate the old dense `rf` path for this use case

## Known Remaining Research Risks

- candidate pool is still too broad relative to the weekly deployment objective
- several research context fields are still synthetic placeholders, not real historical microstructure
- walk-forward validation is still missing, so robustness is not yet proven
- the model stack is still classification-first; ranking/regression on `net_r` is the next major upgrade

## Operational Notes

- this step has been rebuilt through the sparse dataset and selector artifacts
- after changing candidate logic, features, or labels again, rerun:
  - `PYTHONPATH=src python scripts/build_signal_dataset.py --asset ETH/USDT --timeframe 1h`
  - `PYTHONPATH=src python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 1`
  - `PYTHONPATH=src python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 10`
  - `PYTHONPATH=src python scripts/train_signal_model.py --asset ETH/USDT --timeframe 1h`
- if we later change setup families or features, both the dataset and the selector must be rebuilt again before judging results

## Latest Observed Result

Run date: 2026-04-21

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/build_signal_dataset.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_families.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset ETH/USDT --timeframe 1h --validation-pct 0.12 --test-pct 0.18 --purge-rows 12`

Observed output summary:

- dataset rows: `693`
- candidate families: `508 breakout retest`, `141 range rejection`, `44 failed breakout`, `0 trend pullback`
- engine-grade rows at default threshold `75`: `17`
- split: `train 65% / validation 15% / test 20%`
- purge rows: `24`
- calibration: `platt`
- validation accuracy: `65.0%`
- test accuracy: `55.4%`
- baseline accuracy: `52.5%`
- validation Brier: `0.2409`
- test Brier: `0.2461`
- recommended threshold: `None`
- selected test trades at recommended threshold: `0`
- calibrated test probability range: about `0.386` to `0.522`
- family diagnostics:
  - all rows: `breakout retest 508 rows, +0.03 avg net R`; `range rejection 141 rows, -0.12 avg net R`; `failed breakout 44 rows, -0.24 avg net R`
  - test rows: `breakout retest 111 rows, +0.09 avg net R`; `range rejection 25 rows, +0.26 avg net R`; `failed breakout 3 rows, ~0.00 avg net R`
- trend-pullback generation diagnostics:
  - `0` raw trend-pullback candidates across `13,241` trending bars
  - this is not a head-to-head selection problem
  - top bullish blockers: `trend_stack_or_reclaim 4307`, `pullback_depth_band 1372`, `confirmation_candle 452`, `resistance_too_close 404`
  - top bearish blockers: `trend_stack_or_reclaim 4286`, `pullback_depth_band 1266`, `confirmation_candle 474`, `support_too_close 398`

Interpretation:

- the explicit-family refactor successfully narrowed the candidate pool into something much closer to the actual deployment objective
- that said, the pool is still not good enough to trade: calibration still refuses to surface a deployable threshold
- the family distribution is telling us where the next bottleneck is: candidate coverage and family-specific structure are still not balanced enough
- the new family diagnostics show that `breakout retest` still dominates the pool while `trend pullback` is missing entirely
- the new trend-pullback inspection shows the immediate bottleneck is rule gating, not model ranking
- `range rejection` currently looks better on the holdout slices, but the sample is still too small to trust without more targeted generation and walk-forward checks
- this is still a useful result, because the research pipeline is now rejecting weak candidate universes instead of masking them with generic classification metrics

## Latest Candidate-Generation Tuning Result

Run date: 2026-04-22

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_families.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/build_signal_dataset.py --asset ETH/USDT --timeframe 1h`

Observed output summary:

- dataset rows after tuning: `217`
- engine-grade rows at default threshold `75`: `4`
- family mix:
  - `140 range rejection`
  - `43 failed breakout`
  - `19 breakout retest`
  - `15 trend pullback`
- trend-pullback diagnostics:
  - raw trend-pullback candidates: `4`
  - raw breakout-retest candidates in trend regimes: `2`
  - trend pullback is no longer fully absent, but trend-following families are still too sparse

Interpretation:

- the rule changes succeeded in one important way: `trend pullback` is no longer stuck at zero
- the structural-zone compression also reduced unrealistic nearby support/resistance blocking
- however, the overall candidate pool is now too small for the default strict `15% validation / 20% test / purge 24` training workflow
- this means the next tuning pass still belongs in candidate generation, not model complexity

## Latest 5-Year Data Expansion Result

Run date: 2026-04-22

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/fetch_history.py --asset ETH/USDT --exchange binance --timeframe 1h --years 5`
- `PYTHONPATH=src .venv/bin/python scripts/build_signal_dataset.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset ETH/USDT --timeframe 1h`

Observed output summary:

- history expanded to `43,820` candles from `2021-04-22` to `2026-04-22`
- rebuilt sparse dataset on the larger sample became trainable again under the strict temporal workflow
- best strict classifier result on the 5-year sample before the stricter mean-reversion first-target change:
  - dataset rows `700`
  - test accuracy `65.0%`
  - baseline `57.1%`
  - test Brier `0.2245`
  - validation-selected threshold `65%`
  - test result at the selected threshold: `26 trades`, `1.3/week`, `80.8% precision`, `-0.02% avg pnl`

Interpretation:

- more history solved the sample-size problem and brought the strict workflow back into play
- however, more data did not solve the deployment problem: the selected policy was still slightly negative expectancy

## Latest Honest-Exit Result

Run date: 2026-04-22

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/build_signal_dataset.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset ETH/USDT --timeframe 1h`

Observed output summary:

- dataset rows after stricter mean-reversion first-target logic: `670`
- family mix:
  - `426 range rejection`
  - `128 failed breakout`
  - `63 trend pullback`
  - `53 breakout retest`
- strict classifier result:
  - test accuracy `59.0%`
  - baseline `59.0%`
  - test Brier `0.2650`
  - no threshold met the minimum trade count
  - selected test trades: `0`

Interpretation:

- the stricter family-exit logic made the labels more honest by removing some inflated high-win/low-payoff setups
- but it also exposed the deeper issue: the candidate pool itself is still too weak under honest exits

## Oracle Check

Run date: 2026-04-22

Observed result:

- on the current stricter-label 5-year test pool, even the oracle top-k weekly selection is negative:
  - oracle trades: `134`
  - trades per week: `2.48`
  - avg pnl: `-0.16%`
  - avg net R: `-0.21`

Interpretation:

- this is the key current finding
- under the current honest family exits, the bottleneck is not just the selector
- the candidate pool itself is not yet strong enough to support a valid working model
- the next high-ROI work is setup-family redesign and family-specific exit design, not more threshold tuning

## Latest Sparse-Dataset Diagnosis

Run date: 2026-04-23

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 10`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 1 --limit 6`

Observed output summary:

- dataset rows: `670`
- split with `purge_rows=24`:
  - train `411` rows from `2021-05-04` to `2023-12-02`
  - validation `77` rows from `2024-02-02` to `2024-08-21`
  - test `134` rows from `2024-10-15` to `2026-04-19`
- overall realized quality:
  - all rows: `37.9%` win rate, `-0.19%` avg PnL, `-0.26` avg net R
  - validation: `35.1%` win rate, `-0.27%` avg PnL, `-0.40` avg net R
  - test: `38.8%` win rate, `-0.19%` avg PnL, `-0.23` avg net R
- test family quality:
  - `RANGE_REJECTION_MEAN_REVERSION`: `77` rows, `49.4%` win rate, `-0.06%` avg PnL, `-0.04` avg net R
  - `FAILED_BREAKOUT_REVERSAL`: `32` rows, `31.2%` win rate, `-0.35%` avg PnL, `-0.40` avg net R
  - `BREAKOUT_RETEST_CONTINUATION`: `14` rows, `7.1%` win rate, `-0.40%` avg PnL, `-0.75` avg net R
  - `TREND_PULLBACK_CONTINUATION`: `11` rows, `27.3%` win rate, `-0.36%` avg PnL, `-0.46` avg net R
- test side asymmetry:
  - range-rejection longs: `45` rows, `-0.10%` avg PnL, `-0.01` avg net R
  - range-rejection shorts: `32` rows, `-0.01%` avg PnL, `-0.09` avg net R
  - breakout-retest longs are especially poor in the test slice: `6` rows, `-0.64%` avg PnL, `-1.20` avg net R
- exit-reason damage:
  - failed-breakout stop-loss rows average about `-1.13` net R
  - breakout-retest stop-loss rows average about `-1.17` net R
  - trend-pullback stop-loss rows average about `-1.25` net R
  - range-rejection stop-loss rows average about `-1.13` net R, and the shorter family horizon now exposes a larger horizon bucket instead of letting all mean-reversion trades run to the generic lookahead
- oracle observations:
  - with `weekly_cap=10`, the test oracle selects all `134` test rows because the pool is only `2.48` candidates/week
  - with `weekly_cap=1`, the ex-post validation oracle improves to `24` trades, about `+0.00%` avg PnL, and `+0.07` avg net R
  - with `weekly_cap=1`, the ex-post test oracle improves to `54` trades, `+0.09%` avg PnL, and `+0.18` avg net R

Interpretation:

- honoring family `max_hold_bars` made the sparse labels more honest
- the full candidate pool is still negative, but the stricter `1/week` oracle is now at least mildly positive on both validation and test
- that is not enough to declare success, but it does support keeping the focus on extremely selective candidate quality rather than widening the trade rate
- next tuning should still focus on reducing stop-loss clusters in failed breakout, breakout retest, and trend pullback

## Latest Selector Overfit Result

Run date: 2026-04-23

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset ETH/USDT --timeframe 1h`
- exploratory checks also used `--weekly-cap 1` and lower threshold grids for diagnosis only

Observed output summary:

- default selector run after the family max-hold label change:
  - train accuracy `100.0%`
  - validation accuracy `64.9%`
  - test accuracy `61.2%`
  - baseline accuracy `61.2%`
  - validation Brier `0.2111`
  - test Brier `0.2506`
- calibrated probability range collapsed below the default threshold grid:
  - validation calibrated `p(WIN)` range about `19.3%` to `46.3%`
  - test calibrated `p(WIN)` range about `19.2%` to `46.8%`
- result:
  - no default threshold (`55%` to `75%`) selected any trades
  - lowering thresholds for diagnostics could create validation-selected trades, but those thresholds failed on the test slice
- simple alternative classifier and regressor checks remained negative on the test slice, so the problem still looks structural rather than model-choice-only

Interpretation:

- the selector is now honestly abstaining for a visible reason: overfit plus probability-range collapse
- changing the model family alone did not recover a deployable edge
- the next work still belongs in candidate design and family-specific exit logic, not just in retuning the classifier

## Latest Candidate-Priority Check

Run date: 2026-04-23

Command used:

- a local exploratory script that rebuilt all same-bar family candidates and compared the current chosen candidate with the ex-post best `net_r` candidate on bars with multiple valid candidates

Observed output summary:

- candidate bars: `670`
- multi-candidate bars: `83`
- current choice was already the ex-post best family on `604` of `670` candidate bars
- only `66` bars had a different same-bar family with better ex-post `net_r`
- the largest overlap issue was `FAILED_BREAKOUT_REVERSAL -> RANGE_REJECTION_MEAN_REVERSION` on `66` bars, but both sides of that overlap were still negative on average

Interpretation:

- candidate-family priority is not the main bottleneck
- simply reordering family preference will not rescue the sparse pool
- the next high-ROI changes remain family-rule and exit redesign, especially around stop-loss-heavy continuation and failed-break setups

## Latest Exploratory Training Result

Run date: 2026-04-22

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset ETH/USDT --timeframe 1h --validation-pct 0.12 --test-pct 0.18 --purge-rows 12`

Observed output summary:

- dataset rows: `217`
- split: `train 70% / validation 12% / test 18%`
- purge rows: `12`
- validation accuracy: `64.3%`
- test accuracy: `70.0%`
- baseline accuracy: `70.0%`
- validation Brier: `0.2131`
- test Brier: `0.2042`
- recommended threshold from validation: `55%`
- recommended-threshold test result: `40 trades`, `2.1/week`, `70.0% precision`, `-0.05% avg pnl`
- test result at `65%`: `17 trades`, `1.5/week`, `82.3% precision`, `+0.01% avg pnl`

Interpretation:

- this exploratory run is useful for direction, but it is not apples-to-apples with the stricter reference workflow
- the selector is still not delivering clear positive expectancy at the validation-selected threshold
- the positive `+0.01%` avg pnl at `65%` is too small and too exploratory to count as a real improvement
- the most important takeaway is still structural: family balance is improving, but the candidate pool is not yet in the right zone

## Latest Cross-Asset Control Result

Run date: 2026-04-24

Command used:

- `PYTHONPATH=src .venv/bin/python scripts/build_signal_dataset.py --asset BTC/USDC --timeframe 15m`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset BTC/USDC --timeframe 15m --weekly-cap 10 --limit 12`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset BTC/USDC --timeframe 15m --weekly-cap 1 --limit 12`
- `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset BTC/USDC --timeframe 15m`
- exploratory diagnosis only: `PYTHONPATH=src .venv/bin/python scripts/train_signal_model.py --asset BTC/USDC --timeframe 15m --calibration none --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65`

Observed output summary:

- history and sparse dataset:
  - `70,128` candles from `2024-04-23 05:45:00+00:00` to `2026-04-23 17:30:00+00:00`
  - `2,242` sparse candidate rows
  - overall pool: `37.0%` win rate, `-0.11%` avg PnL, `-0.30` avg net R, about `21.35` candidates/week
  - test pool: `37.4%` win rate, `-0.12%` avg PnL, `-0.27` avg net R, about `21.38` candidates/week
- ex-post oracle quality is materially better than the raw pool:
  - `weekly_cap=10` validation oracle: `160` trades, `72.5%` win rate, `+0.22%` avg PnL, `+0.61` avg net R
  - `weekly_cap=10` test oracle: `198` trades, `74.2%` win rate, `+0.26%` avg PnL, `+0.71` avg net R
  - `weekly_cap=1` validation oracle: `16` trades, `100.0%` win rate, `+0.81%` avg PnL, `+2.84` avg net R
  - `weekly_cap=1` test oracle: `21` trades, `95.2%` win rate, `+0.70%` avg PnL, `+1.94` avg net R
- default calibrated selector run still fails to capture that oracle gap:
  - train accuracy `98.9%`
  - validation accuracy `60.9%`
  - test accuracy `62.6%`
  - baseline accuracy `62.6%`
  - validation Brier `0.2375`
  - test Brier `0.2350`
  - validation calibrated probability range only `35.7%` to `43.6%`
  - no default threshold (`55%` to `75%`) selected any trades
- exploratory no-calibration run is useful only as diagnosis:
  - validation-chosen threshold became `50%`
  - test selection: `134` trades, about `6.7/week`, `34.3%` precision, `-0.13%` avg PnL

Interpretation:

- BTC/USDC 15m is a useful control dataset because candidate frequency is high enough and the ex-post oracle is genuinely strong
- that means selector failure on this slice cannot be explained only by ETH/USDT 1h candidate scarcity or weak oracle limits
- the current classifier-plus-calibration stack still does not convert a healthier candidate universe into deployable selected trades
- ETH/USDT 1h remains the main milestone, but future selector, calibration, or objective changes should be sanity-checked on BTC/USDC 15m to separate model-stack limits from candidate-quality limits

## Latest Breakout-Retest Tightening + Engine Alignment Result

Run date: 2026-04-25

What changed in code:

- `_build_breakout_retest_long/short` in [src/ai_trading_engine/signal_generation.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/signal_generation.py) now require trend-stack alignment inside the rule itself: long demands `ema20 > ema50` and `entry > ema50`, short the symmetric inversion. This is in addition to the regime classifier's prior gate.
- `EngineConfig` gained `allowed_setup_families: frozenset[str] | None` and `max_trades_per_iso_week: int | None`. Both are `None` by default and preserve research engine behavior.
- `PortfolioState` gained `recent_trade_timestamps: list[str]` so engine-level selectivity gates can enforce a real ISO-weekly cap end-to-end.
- `HybridTradingEngine.evaluate` now applies (a) the family whitelist immediately after candidate generation and (b) the ISO-weekly cap from the snapshot's last-bar timestamp + portfolio history.
- `Backtester` now uses `dataset.build_research_snapshot` to construct snapshots, so engine candidate generation in the backtester is bit-for-bit identical to the sparse research pipeline.
- `Backtester._check_close` now enforces a per-family `max_hold_bars` horizon exit and records `exit_reason` (TAKE_PROFIT / STOP_LOSS / HORIZON), matching `dataset.label_trade_path` so research labels and backtest exits agree.
- `Backtester` extends the historical-staleness allowance to 10 years of minutes (the prior `999_999` clamp rejected anything older than ~1.9 years).
- `_rule_based_validation` no longer hard-codes a `confluence < 75` reject; the engine's configured `confluence_threshold` is the single source of truth.
- `scripts/backtest.py` gained `--history-csv`, `--start`, `--end`, `--preset {default,strict}`, and `--trades-csv` so a full local backtest is one command.
- `scripts/__main__.py` gained `--preset {default,strict}`. The strict preset is `confluence_threshold=55`, `allowed_setup_families={RANGE_REJECTION_MEAN_REVERSION}`, `max_trades_per_iso_week=1`, `risk.risk_per_trade_pct=0.005`, `risk.min_rr=1.0` (mean-reversion TP1 is 1.1R), `risk.kill_switch_after_losses=5` (the default 3 locks out the engine for months at this trade tempo).
- `Makefile` gained `make run-paper-strict`, `make backtest-history`, `make backtest-strict`.
- `tests/test_engine.py` covers the new family whitelist and weekly-cap gates.

Commands used:

- `PYTHONPATH=src .venv/bin/python scripts/build_signal_dataset.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 10`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 1`
- `PYTHONPATH=src .venv/bin/python scripts/backtest.py --history-csv data/historical/ETH_USDT_1h.csv --asset ETH/USDT --timeframe 1h --preset strict --trades-csv data/reports/backtest_strict_full.csv`
- `PYTHONPATH=src .venv/bin/python scripts/backtest.py --history-csv data/historical/ETH_USDT_1h.csv --asset ETH/USDT --timeframe 1h --preset strict --start 2024-10-01 --trades-csv data/reports/backtest_strict_test_slice.csv`
- `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` → `44 passed`

Observed sparse-dataset summary (ETH/USDT 1h after both range-rejection and breakout-retest changes):

- dataset rows: `322` (after the trend-stack gate; was `330` after only the range-rejection change)
- family mix:
  - `124` failed breakout
  - `90` range rejection
  - `63` trend pullback
  - `45` breakout retest (was `53`)
- split with `purge_rows=24`: train `185`, validation `24`, test `65`
- test slice (post-change): `65` rows, `38.5%` win, `-0.13%` avg PnL, `-0.19` avg net R
- test family expectancy:
  - `RANGE_REJECTION_MEAN_REVERSION`: `19` rows, `57.9%` win, `+0.20` avg net R (longs `+0.44`, shorts `-0.14`)
  - `FAILED_BREAKOUT_REVERSAL`: `27` rows, `37.0%` win, `-0.25` net R
  - `BREAKOUT_RETEST_CONTINUATION`: `12` rows, `8.3%` win, `-0.56` net R (longs still `0%` win on `4` rows at `-1.20`)
  - `TREND_PULLBACK_CONTINUATION`: `11` rows, `27.3%` win, `-0.46` net R
- ex-post `weekly_cap=1` test oracle:
  - overall: `34` trades, `50.0%` win, `+0.04` avg PnL, `+0.08` net R
  - on range rejection only: `9` rows, `77.8%` win, `+0.81` net R

End-to-end backtest of the strict preset on ETH/USDT 1h:

- full history (2021-04-22 → 2026-04-22, ~5 years):
  - `21` trades, `23.8%` win, `-5.42%` total return, `5.57%` max drawdown, Sharpe `-1.14`, profit factor `0.27`
  - Note: kill-switch fires at 5 losses and locks the engine out for months at the 1/week tempo — so this is partly a structural artifact, not just edge failure.
- recent test slice (2024-10-01 → 2026-04-22, ~18 months):
  - `15` trades, `60.0%` win, `+3.55%` total return, `1.49%` max drawdown, Sharpe `1.10`, profit factor `2.18`
  - This matches the sparse research finding (range rejection at `64.7%` win on test, `+0.35` net R).
- diagnostic with kill-switch disabled across the full history: `88` trades, `44.3%` win, `-1.97%` total return, `7.44%` max drawdown — roughly break-even with slight negative drift, again matching the sparse pool's overall `-0.22` net R for range rejection.

Interpretation:

- The breakout-retest trend-stack gate filtered `8` candidates from the family but did not cure the `0%` long-side win rate on the `4` test rows. The next blocker for breakout-retest is structural (probably the `prior_high` 20-bar lookback and the tight `0.35*atr` stop), not entry-side conviction.
- The range-rejection edge from the prior change held: test slice `+0.20` net R with `+0.81` net R on the strict `weekly_cap=1` oracle. The end-to-end strict-preset backtest reproduced this on the recent test slice (`+3.55%` over 18 months at `0.5%` risk).
- Across the full 5-year history, the same strict-preset policy is **negative**, with or without the kill-switch. So the recent positive expectancy is at least partly regime-dependent and is **not yet a deployable, robust edge**.
- The engine, backtester, and live runner now use the same snapshot construction, the same horizon-exit semantics, and the same family/weekly-cap gates. That closes the largest research↔deployment alignment gap that previously made the diagnostic numbers untrustworthy in the live path.

## Latest Range-Rejection Tightening Result

Run date: 2026-04-25

What changed in code:

- `_build_range_rejection_long` and `_build_range_rejection_short` in [src/ai_trading_engine/signal_generation.py](/Users/mtmamaghani/GitHub/AI-Trading-Engine/src/ai_trading_engine/signal_generation.py) now require the structural range extreme to be touched.
- The prior Bollinger-band alternative trigger (`entry <= bb_lower * 1.01` for longs, `entry >= bb_upper * 0.99` for shorts) is no longer enough by itself to qualify the setup.
- When both the structural touch and the BB tag are present, the BB touch now raises setup quality by `+3` instead of being a trigger leg on its own.

Motivation:

- Mid-range rejection off the BB envelope alone has a materially weaker mean-reversion thesis than structural range-extreme rejection.
- On the 2026-04-23 diagnosis, range rejection was the largest family (`426/670`) and nearly break-even on the test slice, which made it the most leverage-able candidate family for a single focused rule change.

Commands used:

- `PYTHONPATH=src .venv/bin/python scripts/build_signal_dataset.py --asset ETH/USDT --timeframe 1h`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 10`
- `PYTHONPATH=src .venv/bin/python scripts/diagnose_signal_dataset.py --asset ETH/USDT --timeframe 1h --weekly-cap 1 --limit 6`
- `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`

Observed output summary (after rebuild):

- dataset rows: `330` (down from `670`)
- family mix:
  - `124` failed breakout
  - `90` range rejection (down from `426`)
  - `63` trend pullback
  - `53` breakout retest
- split with `purge_rows=24`:
  - train `190` rows from `2021-05-04` to `2023-12-02`
  - validation `26` rows from `2024-04-27` to `2024-08-26`
  - test `66` rows from `2025-01-01` to `2026-04-19`
- overall realized quality:
  - all rows: `34.2%` win, `-0.19%` avg PnL, `-0.28` avg net R, `2.26`/week
  - test: `37.9%` win, `-0.13%` avg PnL, `-0.19` avg net R, `1.83`/week
- test family quality:
  - `RANGE_REJECTION_MEAN_REVERSION`: `17` rows, `64.7%` win, `+0.33%` avg PnL, `+0.35` avg net R (was `49.4%` win and `-0.04` net R on `77` rows)
    - LONG: `9` rows, `77.8%` win, `+0.79` net R
    - SHORT: `8` rows, `50.0%` win, `-0.14` net R
  - `FAILED_BREAKOUT_REVERSAL`: `26` rows, `38.5%` win, `-0.22` net R
  - `BREAKOUT_RETEST_CONTINUATION`: `12` rows, `8.3%` win, `-0.67` net R
    - LONG: `5` rows, `0.0%` win, `-1.20` net R
    - SHORT: `7` rows, `14.3%` win, `-0.30` net R
  - `TREND_PULLBACK_CONTINUATION`: `11` rows, `27.3%` win, `-0.46` net R
- ex-post oracle at `weekly_cap=1`:
  - validation: `10` trades, `+0.12%` avg PnL, `+0.04` net R
  - test: `36` trades, `+0.08%` avg PnL, `+0.12` net R
  - test oracle family mix:
    - `RANGE_REJECTION_MEAN_REVERSION`: `8` rows, `87.5%` win, `+0.68%` avg PnL, `+1.05` avg net R
    - `FAILED_BREAKOUT_REVERSAL`: `15` rows, `53.3%` win, `+0.18` net R
    - `BREAKOUT_RETEST_CONTINUATION`: `9` rows, `11.1%` win, `-0.51` net R
    - `TREND_PULLBACK_CONTINUATION`: `4` rows, `25.0%` win, `-0.49` net R
- regression tests: `40 passed`

Interpretation:

- The targeted rule change did exactly what it was designed to do: it preserved the structural range-extreme rejection thesis and removed mid-range BB-only setups. Range-rejection expectancy on the test slice went from `-0.04` net R to `+0.35` net R, and the `weekly_cap=1` oracle on test range-rejection is now strongly positive at `+1.05` net R.
- However, the overall candidate pool is still negative. The cut reduced volume from `670` to `330` rows, so remaining families now dominate the test slice proportionally. Overall test net R only improved marginally from `-0.23` to `-0.19`.
- The validation slice (`26` rows, `-0.65` net R) is small and covers a choppy 2024-summer window; it is a caution flag, not decisive evidence that the change hurts out-of-sample.
- The dominant remaining blocker is now clearly `BREAKOUT_RETEST_CONTINUATION`. Its longs are `0.0%` win across `5` test rows at `-1.20` net R, and the whole family averages `-0.67` net R on test. This is the next high-ROI family to tighten.
- `TREND_PULLBACK_CONTINUATION` expectancy is unchanged but still weak, and stop-loss rows average `-1.25` net R; that family still likely needs family-specific exits, not just entry rules.
- We did not retrain the selector yet, because the candidate pool is still overall negative, so a selector improvement would still be confounded by candidate-quality failure. Selector retraining is only worth doing after the next family-rule pass.

## Next Recommended Step

The breakout-retest trend-stack gate did not cure the long-side failure (`0%` win on `4` test rows at `-1.20` net R). The next levers, in order of likely return:

1. Replace `structure.prior_high/low` (20-bar lookback) with a longer structural reference (e.g. last `60` or `80`-bar swing) so a "breakout" must clear a meaningful prior high, not just a 20-bar oscillation high. Combined with the existing `is_compressed` / `bb_width_pct <= 0.04` gate this should remove a lot of pseudo-breakouts.
2. Widen the breakout-retest stop. The current `breakout_level - 0.35 * atr` puts stops on top of the same level the breakout just cleared, which is exactly where rejection wicks finish. A stop at `breakout_level - 0.7 * atr` (or below the most recent swing low before the breakout) should reduce stop-out density without inflating risk much.
3. After breakout-retest is no longer net-negative, revisit `TREND_PULLBACK_CONTINUATION` exits — its stop-loss bucket averages `-1.25` net R, so a partial-profit / breakeven exit rather than a single-target template is probably the highest leverage there.
4. Only after the candidate pool stops being overall-negative on honest labels should we retrain the sparse selector. Until then a selector improvement is confounded by candidate-quality failure.
5. Keep BTC/USDC 15m as the selector control dataset; its oracle is strong enough to expose model-stack failure independent of candidate-quality failure.
6. Preserve `weekly_cap=1` as the primary selectivity diagnostic on ETH/USDT 1h, because the pool is still below `10/week` on every slice.

## Paper-Trading Runbook (current, evidence-aligned)

This is the recipe a user can run today to paper-trade what the research currently supports. **It is not a guarantee of profit.** The same policy is positive on the recent test slice and negative across the full 5-year history. Treat the run as observational deployment, not a deployable strategy.

Pre-flight:

- `make setup && make install-ml`
- `cp .env.example .env` and fill in `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET` (read-only is enough for paper mode), `INITIAL_EQUITY=10000`, `EXCHANGE=binance`. Telegram is optional.
- `make test` → expect `44 passed`.

Sanity check on local history before going live:

- `make backtest-strict history_csv=data/historical/ETH_USDT_1h.csv asset=ETH/USDT timeframe=1h start=2024-10-01`
- This should produce roughly: `~15 trades`, win rate near `60%`, `+3.5%` total return, `~1.5%` max DD, Sharpe near `1.1`. If the numbers are materially different, do not paper-trade — investigate first.

Run paper trading:

- `make run-paper-strict` (foreground)
- The strict preset only takes `RANGE_REJECTION_MEAN_REVERSION` setups, at most one per ISO week, with `0.5%` risk per trade and a kill-switch that activates after `5` consecutive losses.

What to expect:

- Most cycles will print `NO TRADE` with a clear no-trade reason. That is the correct behavior — selectivity is the strategy.
- Trades, when they fire, will sit in `data/trades.db` (SQLite) and in the runtime log. The first take-profit is `~1.1R`, the stop is below the structural range low, and a `12`-bar horizon exit is enforced.
- A typical month at this preset on ETH/USDT 1h is `0` to `4` trades.

When to abort the run:

- If the kill-switch fires (you'll see `Kill-switch active after 5 consecutive losses`) — restart only after re-checking `make backtest-strict` on the most recent 90 days, and only if that recent slice still shows a positive profit factor.
- If candidate generation rate drops to zero for more than 4 weeks, the regime has likely changed and the strict preset is no longer applicable.
