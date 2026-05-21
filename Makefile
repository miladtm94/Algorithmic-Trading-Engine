.PHONY: help setup install install-dev install-prod install-ml clean \
        run run-once run-paper run-paper-strict run-live \
        backtest backtest-strict backtest-history test test-cov lint format typecheck \
        fetch-history build-dataset build-signal-dataset diagnose-signal-dataset diagnose-signal-families train train-signal train-deep optimize visualize-models \
        evaluate evaluate-as-of evaluate-at evaluate-signal-at \
        demo docker-build docker-run docker-stop env logs \
        git-status git-remote git-commit-staged git-commit-all git-push git-push-force git-publish

PYTHON  ?= python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
PYTEST  := $(VENV)/bin/pytest
REMOTE  ?= origin
BRANCH  ?= main
REMOTE_URL ?= git@github.com:miladtm94/Algorithmic-Trading-Engine.git
COMMIT_MSG ?= chore: update

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@printf "\n\033[1mAI Trading Engine\033[0m — Available Commands\n"
	@printf "══════════════════════════════════════════════════\n\n"
	@printf "\033[1mSetup\033[0m\n"
	@printf "  make setup            Create venv and install all dependencies\n"
	@printf "  make install          Install core package (editable)\n"
	@printf "  make install-dev      Install dev/lint/test tools\n"
	@printf "  make install-prod     Install production extras (ccxt, dotenv)\n"
	@printf "\n\033[1mRunning\033[0m\n"
	@printf "  make run              Demo mode — no real data or money needed\n"
	@printf "  make run-once         Single evaluation cycle (demo) and exit\n"
	@printf "  make run-paper        Paper trading with real exchange data (default preset)\n"
	@printf "  make run-paper-strict Paper trading with the strict RANGE_REJECTION preset\n"
	@printf "  make run-live         Live trading — REAL MONEY (read .env.example first)\n"
	@printf "\n\033[1mBacktesting\033[0m\n"
	@printf "  make backtest         Run backtest on demo candle data\n"
	@printf "  make backtest-live    Fetch historical data from exchange and backtest\n"
	@printf "  make backtest-history Replay a local OHLCV CSV (history_csv=... preset=strict start=... end=...)\n"
	@printf "  make backtest-strict  Shortcut for 'backtest-history preset=strict'\n"
	@printf "\n\033[1mML Pipeline\033[0m\n"
	@printf "  make install-ml       Install scikit-learn + joblib\n"
	@printf "  make fetch-history    Download multi-year OHLCV (asset=ETH/USDT years=2)\n"
	@printf "  make build-dataset    Build dense LONG/SHORT feature+label CSV\n"
	@printf "  make build-signal-dataset  Build sparse engine-candidate dataset\n"
	@printf "  make diagnose-signal-dataset Audit sparse dataset family expectancy and oracle limits\n"
	@printf "  make diagnose-signal-families Inspect why setup families are or are not firing\n"
	@printf "  make train            Train RandomForest model (time-series split)\n"
	@printf "  make train-signal     Train sparse selector model (temporal split + weekly cap)\n"
	@printf "  make train-deep       Train sequence deep NN model (default 70/30 random split)\n"
	@printf "  make optimize         Search best label settings and train final model\n"
	@printf "  make visualize-models Build HTML dashboard comparing saved models\n"
	@printf "  make evaluate         Score live signal with trained model (--now)\n"
	@printf "  make evaluate-as-of   Evaluate accuracy at a past date (as_of=2024-01-01)\n"
	@printf "  make evaluate-at      Audit prediction vs outcome (at='2026-03-15 09:00')\n"
	@printf "  make evaluate-signal-at  Audit sparse selector at one timestamp\n"
	@printf "\n\033[1mTesting & Quality\033[0m\n"
	@printf "  make test             Run unit tests\n"
	@printf "  make test-cov         Tests with HTML coverage report\n"
	@printf "  make lint             Lint with ruff\n"
	@printf "  make format           Auto-format with ruff\n"
	@printf "  make typecheck        Static type check with mypy\n"
	@printf "\n\033[1mDocker\033[0m\n"
	@printf "  make docker-build     Build container image\n"
	@printf "  make docker-run       Start in Docker (paper mode by default)\n"
	@printf "  make docker-stop      Tear down Docker containers\n"
	@printf "  make logs             Tail Docker container logs\n"
	@printf "\n\033[1mMisc\033[0m\n"
	@printf "  make demo             Send demo signal to Telegram\n"
	@printf "  make env              Show active environment variables\n"
	@printf "  make clean            Remove build/cache artefacts\n"
	@printf "\n\033[1mGit Publishing\033[0m\n"
	@printf "  make git-status       Show branch, remote, and changed files\n"
	@printf "  make git-remote       Set origin URL (REMOTE_URL=... optional)\n"
	@printf "  make git-commit-staged Commit only already staged files (COMMIT_MSG='...' optional)\n"
	@printf "  make git-commit-all   Stage all visible changes and commit (COMMIT_MSG='...' optional)\n"
	@printf "  make git-push         Push current branch as main to origin\n"
	@printf "  make git-push-force   Push with --force-with-lease for recreated remotes\n"
	@printf "  make git-publish      Set remote, rename branch to main, and push\n\n"

# ── Virtualenv & dependencies ─────────────────────────────────────────────────

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: $(VENV)/bin/activate
	$(PIP) install -e .

install-dev: $(VENV)/bin/activate
	$(PIP) install ruff mypy pytest pytest-cov

install-prod: $(VENV)/bin/activate
	$(PIP) install "ccxt>=4.0.0" "python-dotenv>=1.0.0"

install-ml: $(VENV)/bin/activate
	$(PIP) install "scikit-learn>=1.4.0" "joblib>=1.3.0"

setup: $(VENV)/bin/activate install install-dev install-prod
	@cp -n .env.example .env 2>/dev/null && printf "\033[33m.env created from .env.example — fill in your credentials\033[0m\n" || true
	@printf "\033[32mSetup complete. Run 'make run' to try the demo.\033[0m\n"

# ── Running ───────────────────────────────────────────────────────────────────

run: install
	$(PY) -m ai_trading_engine --mode demo

run-once: install
	$(PY) -m ai_trading_engine --mode demo --once

run-paper: install install-prod
	@test -f .env || (printf "\033[31mCreate .env first (copy .env.example)\033[0m\n" && exit 1)
	$(PY) -m ai_trading_engine --mode paper

run-paper-strict: install install-prod
	@test -f .env || (printf "\033[31mCreate .env first (copy .env.example)\033[0m\n" && exit 1)
	@printf "\033[33mStrict preset: RANGE_REJECTION only, <=1 trade per ISO week, confluence>=55, risk 0.5%%/trade.\033[0m\n"
	$(PY) -m ai_trading_engine --mode paper --preset strict

run-live: install install-prod
	@test -f .env || (printf "\033[31mCreate .env first (copy .env.example)\033[0m\n" && exit 1)
	@printf "\033[31m⚠  WARNING: LIVE MODE USES REAL MONEY.\033[0m\n"
	@printf "Press Ctrl+C within 5 seconds to abort...\n"
	@sleep 5
	$(PY) -m ai_trading_engine --mode live

# ── Backtesting ───────────────────────────────────────────────────────────────

backtest: install
	PYTHONPATH=src $(PY) scripts/backtest.py

backtest-live: install install-prod
	@test -f .env || (printf "\033[31mCreate .env first\033[0m\n" && exit 1)
	PYTHONPATH=src $(PY) scripts/backtest.py --live

backtest-history: install install-ml
	PYTHONPATH=src $(PY) scripts/backtest.py \
	  --history-csv $(or $(history_csv),data/historical/ETH_USDT_1h.csv) \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --preset $(or $(preset),default) \
	  --confluence $(or $(confluence),75) \
	  $(if $(start),--start $(start),) \
	  $(if $(end),--end $(end),) \
	  $(if $(trades_csv),--trades-csv $(trades_csv),) \
	  --equity $(or $(equity),10000) \
	  --risk $(or $(risk),0.01)

backtest-strict: install install-ml
	PYTHONPATH=src $(PY) scripts/backtest.py \
	  --history-csv $(or $(history_csv),data/historical/ETH_USDT_1h.csv) \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --preset strict \
	  $(if $(start),--start $(start),) \
	  $(if $(end),--end $(end),) \
	  $(if $(trades_csv),--trades-csv $(trades_csv),data/reports/backtest_strict_trades.csv)

# ── Tests & Quality ───────────────────────────────────────────────────────────

test: install
	PYTHONPATH=src $(PYTEST) tests/ -v

test-cov: install install-dev
	PYTHONPATH=src $(PYTEST) tests/ -v \
	  --cov=ai_trading_engine \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov
	@printf "\n\033[32mCoverage report: htmlcov/index.html\033[0m\n"

lint: install-dev
	$(RUFF) check src/ tests/ scripts/

format: install-dev
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

typecheck: install-dev
	$(MYPY) src/ai_trading_engine/ --ignore-missing-imports --strict-optional

# ── ML Pipeline ───────────────────────────────────────────────────────────────

# Example ML workflow:
#   make install-ml
#   make fetch-history asset=ETH/USDT exchange=binance timeframe=1h years=2
#   make build-dataset asset=ETH/USDT timeframe=1h lookahead=24 stop_atr=1.5 reward_risk=2.0
#   make train asset=ETH/USDT timeframe=1h
#   make build-signal-dataset asset=ETH/USDT timeframe=1h lookahead=24 min_confluence=55
#   make diagnose-signal-dataset asset=ETH/USDT timeframe=1h weekly_cap=10
#   make train-signal asset=ETH/USDT timeframe=1h weekly_cap=10
#   make train-deep asset=ETH/USDT timeframe=1h sequence_length=24 split_mode=random
#   make optimize asset=ETH/USDT timeframe=1h lookaheads=12,24,48 stop_atrs=1.2,1.5,2.0 reward_risks=1.5,2.0
#   make visualize-models asset=ETH/USDT timeframe=1h
#
# Example model checks:
#   make evaluate-as-of asset=ETH/USDT timeframe=1h as_of=2025-11-26
#   make evaluate-at asset=ETH/USDT timeframe=1h at='2026-03-15 09:00' side=ALL walk_forward=1
#   make evaluate-signal-at asset=ETH/USDT timeframe=1h at='2026-04-18 08:00' walk_forward=1
#   make evaluate-at asset=ETH/USDT timeframe=1h at='2026-03-15 09:00' side=LONG
#
# Optional sparse engine-signal dataset:
#   make build-dataset asset=ETH/USDT timeframe=1h mode=engine confluence=65

fetch-history: install install-prod install-ml
	PYTHONPATH=src $(PY) scripts/fetch_history.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --exchange $(or $(exchange),binance) \
	  --timeframe $(or $(timeframe),1h) \
	  --years $(or $(years),2)

build-dataset: install install-ml
	PYTHONPATH=src $(PY) scripts/build_dataset.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --mode $(or $(mode),opportunity) \
	  --confluence $(or $(confluence),75.0) \
	  --lookahead $(or $(lookahead),24) \
	  --stop-atr $(or $(stop_atr),1.5) \
	  --reward-risk $(or $(reward_risk),2.0)

build-signal-dataset: install install-ml
	PYTHONPATH=src $(PY) scripts/build_signal_dataset.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --window-size $(or $(window_size),220) \
	  --lookahead $(or $(lookahead),24) \
	  --fee-bps $(or $(fee_bps),10.0) \
	  --min-profit-pct $(or $(min_profit_pct),0.0) \
	  --min-confluence $(or $(min_confluence),55.0)

diagnose-signal-dataset: install install-ml
	PYTHONPATH=src $(PY) scripts/diagnose_signal_dataset.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --validation-pct $(or $(validation_pct),0.15) \
	  --test-pct $(or $(test_pct),0.20) \
	  --purge-rows $(or $(purge_rows),24) \
	  --weekly-cap $(or $(weekly_cap),10) \
	  --min-group-size $(or $(min_group_size),3)

diagnose-signal-families: install install-ml
	PYTHONPATH=src $(PY) scripts/diagnose_signal_families.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --window-size $(or $(window_size),220) \
	  --lookahead $(or $(lookahead),24) \
	  --min-confluence $(or $(min_confluence),55.0) \
	  --family $(or $(family),trend_pullback)

train: install install-ml
	PYTHONPATH=src $(PY) scripts/train_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h)

train-signal: install install-ml
	PYTHONPATH=src $(PY) scripts/train_signal_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --test-pct $(or $(test_pct),0.20) \
	  --validation-pct $(or $(validation_pct),0.15) \
	  --split-mode $(or $(split_mode),temporal) \
	  --weekly-cap $(or $(weekly_cap),10) \
	  --thresholds $(or $(thresholds),0.55,0.60,0.65,0.70,0.75) \
	  --min-threshold-count $(or $(min_threshold_count),12)

train-deep: install install-ml
	PYTHONPATH=src $(PY) scripts/train_deep_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --sequence-length $(or $(sequence_length),24) \
	  --split-mode $(or $(split_mode),random) \
	  --test-pct $(or $(test_pct),0.30) \
	  --hidden-layers $(or $(hidden_layers),256,128,64) \
	  --selection-threshold $(or $(selection_threshold),0.65)

optimize: install install-ml
	PYTHONPATH=src $(PY) scripts/optimize_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --lookaheads $(or $(lookaheads),12,24,48) \
	  --stop-atrs $(or $(stop_atrs),1.2,1.5,2.0) \
	  --reward-risks $(or $(reward_risks),1.5,2.0) \
	  --selection-threshold $(or $(selection_threshold),0.65) \
	  --min-threshold-count $(or $(min_threshold_count),30)

visualize-models: install
	PYTHONPATH=src $(PY) scripts/visualize_model_performance.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h)

evaluate: install install-prod install-ml
	@test -f .env || (printf "\033[31mCreate .env first\033[0m\n" && exit 1)
	PYTHONPATH=src $(PY) scripts/evaluate_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --now

evaluate-as-of: install install-ml
	@test -n "$(as_of)" || (printf "\033[31mUsage: make evaluate-as-of as_of=2024-01-01\033[0m\n" && exit 1)
	PYTHONPATH=src $(PY) scripts/evaluate_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --as-of $(as_of)

evaluate-at: install install-ml
	@test -n "$(at)" || (printf "\033[31mUsage: make evaluate-at at='2026-03-15 09:00' side=ALL\033[0m\n" && exit 1)
	PYTHONPATH=src $(PY) scripts/evaluate_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --at "$(at)" \
	  --side $(or $(side),ALL) \
	  $(if $(walk_forward),--walk-forward,)

evaluate-signal-at: install install-ml
	@test -n "$(at)" || (printf "\033[31mUsage: make evaluate-signal-at at='2026-04-18 08:00'\033[0m\n" && exit 1)
	PYTHONPATH=src $(PY) scripts/evaluate_signal_model.py \
	  --asset $(or $(asset),ETH/USDT) \
	  --timeframe $(or $(timeframe),1h) \
	  --at "$(at)" \
	  --lookahead $(or $(lookahead),24) \
	  --window-size $(or $(window_size),220) \
	  --fee-bps $(or $(fee_bps),10.0) \
	  --min-profit-pct $(or $(min_profit_pct),0.0) \
	  --min-confluence $(or $(min_confluence),55.0) \
	  --threshold $(or $(threshold),auto) \
	  $(if $(walk_forward),--walk-forward,)

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:
	docker build -t ai-trading-engine:latest .

docker-run:
	@test -f .env || (printf "\033[31mCreate .env first\033[0m\n" && exit 1)
	docker compose up -d
	@printf "\033[32mContainer started. Run 'make logs' to watch.\033[0m\n"

docker-stop:
	docker compose down

logs:
	docker compose logs -f trading-engine

# ── Misc ──────────────────────────────────────────────────────────────────────

demo: install
	PYTHONPATH=src $(PY) scripts/send_demo_signal.py

env:
	@printf "\n\033[1mActive Environment\033[0m\n"
	@printf "  TRADING_MODE        = $${TRADING_MODE:-demo}\n"
	@printf "  DEFAULT_ASSET       = $${DEFAULT_ASSET:-ETH/USDT}\n"
	@printf "  DEFAULT_TIMEFRAME   = $${DEFAULT_TIMEFRAME:-1h}\n"
	@printf "  EXCHANGE            = $${EXCHANGE:-binance}\n"
	@printf "  INITIAL_EQUITY      = $${INITIAL_EQUITY:-10000}\n"
	@printf "  EXCHANGE_API_KEY    = $${EXCHANGE_API_KEY:+(set)}\n"
	@printf "  EXCHANGE_API_SECRET = $${EXCHANGE_API_SECRET:+(set)}\n"
	@printf "  TELEGRAM_BOT_TOKEN  = $${TELEGRAM_BOT_TOKEN:+(set)}\n"
	@printf "  TELEGRAM_CHAT_ID    = $${TELEGRAM_CHAT_ID:+(set)}\n"
	@printf "  OPENAI_API_KEY      = $${OPENAI_API_KEY:+(set)}\n\n"

# ── Git publishing ────────────────────────────────────────────────────────────

git-status:
	@printf "\n\033[1mBranch\033[0m\n"
	git branch --show-current
	@printf "\n\033[1mRemotes\033[0m\n"
	git remote -v
	@printf "\n\033[1mWorking tree\033[0m\n"
	git status --short
	@printf "\n\033[1mStaged changes\033[0m\n"
	git diff --cached --name-status

git-remote:
	git remote set-url $(REMOTE) $(REMOTE_URL) 2>/dev/null || git remote add $(REMOTE) $(REMOTE_URL)
	git remote -v

git-commit-staged:
	git diff --cached --check
	git diff --cached --name-status
	git commit -m "$(COMMIT_MSG)"

git-commit-all:
	git add -A
	git diff --cached --check
	git diff --cached --name-status
	git commit -m "$(COMMIT_MSG)"

git-push:
	git branch -M $(BRANCH)
	git push -u $(REMOTE) $(BRANCH)

git-push-force:
	git branch -M $(BRANCH)
	git push -u $(REMOTE) $(BRANCH) --force-with-lease

git-publish: git-remote git-push

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -o -name "*.pyo" | xargs rm -f 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	rm -rf dist/ build/ *.egg-info/ src/*.egg-info/
	@printf "\033[32mClean done.\033[0m\n"
