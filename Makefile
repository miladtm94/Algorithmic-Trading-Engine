.PHONY: help setup install install-dev install-prod clean \
        run run-once run-paper run-live \
        backtest test test-cov lint format typecheck \
        demo docker-build docker-run docker-stop env logs

PYTHON  ?= python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
PYTEST  := $(VENV)/bin/pytest

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
	@printf "  make run-paper        Paper trading with real exchange data\n"
	@printf "  make run-live         Live trading — REAL MONEY (read .env.example first)\n"
	@printf "\n\033[1mBacktesting\033[0m\n"
	@printf "  make backtest         Run backtest on demo candle data\n"
	@printf "  make backtest-live    Fetch historical data from exchange and backtest\n"
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
	@printf "  make git-push         Stage all, commit, and push (msg='...' optional)\n\n"

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

git-push:
	git add -A && git commit -m "$(or $(msg),chore: update)" && git push

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -o -name "*.pyo" | xargs rm -f 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .coverage htmlcov/
	rm -rf dist/ build/ *.egg-info/ src/*.egg-info/
	@printf "\033[32mClean done.\033[0m\n"
