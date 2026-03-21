# StockAI — Claude Code Project Briefing

> Read this entire file before doing anything. This is the source of truth
> for all architecture decisions, conventions, and current project state.

---

## What This Project Is

An AI-assisted trading intelligence system built in Python. It fetches market
data, computes technical indicators, generates trade signals with confidence
scores, enriches them with LLM analysis, enforces risk rules, and executes
paper/live trades via Alpaca. A Streamlit dashboard provides visibility.

**Owner:** Sribalaji Prabhakar  
**Current phase:** Phase 1 — Signal Intelligence (starting now)  
**Trading mode:** PAPER only until Phase 2 gate passes  

---

## Project Structure

```
stock-ai-system/
├── CLAUDE.md                  ← you are here
├── config.yaml                ← single source of truth for ALL settings
├── requirements.txt
├── .env                       ← API keys (never commit this)
├── .env.example               ← template (safe to commit)
├── .gitignore
│
├── src/
│   ├── config.py              ← typed Pydantic config loader
│   ├── secrets.py             ← API key manager (.env locally, AWS prod)
│   │
│   ├── ingestion/             ← PHASE 0 — COMPLETE
│   │   ├── storage.py         ← Parquet read/write (OHLCV, signals, audit)
│   │   ├── polygon_client.py  ← historical OHLCV from Polygon.io
│   │   ├── alpaca_client.py   ← real-time data + order execution
│   │   ├── market_data_service.py ← orchestrates all data fetching
│   │   └── news_service.py    ← Finnhub news + basic sentiment
│   │
│   ├── features/              ← PHASE 0 — COMPLETE
│   │   └── feature_engine.py  ← 30+ technical indicators (RSI, MACD, BB, ATR...)
│   │
│   ├── signals/               ← PHASE 1 — BUILD THIS NEXT
│   │   ├── signal_detector.py     ← pattern matching on features per strategy
│   │   ├── confidence_scorer.py   ← statistical win-rate scoring (NOT LLM)
│   │   └── strategies/            ← one file per strategy
│   │       ├── momentum.py
│   │       ├── trend_following.py
│   │       └── volatility_breakout.py
│   │
│   ├── llm/                   ← PHASE 1 — BUILD THIS NEXT
│   │   └── llm_analysis_service.py ← Claude API for news enrichment only
│   │
│   ├── risk/                  ← PHASE 1 — BUILD THIS NEXT
│   │   └── risk_manager.py    ← position sizing, exposure, circuit breaker
│   │
│   ├── execution/             ← PHASE 1 — BUILD THIS NEXT
│   │   └── decision_engine.py ← combines signal + confidence + risk
│   │
│   ├── monitoring/            ← PHASE 2
│   │   └── model_monitor.py   ← drift detection, win-rate tracking
│   │
│   └── scheduler/             ← PHASE 2
│       └── scheduler.py       ← APScheduler jobs
│
├── lean/                      ← PHASE 1.5
│   └── strategies/            ← LEAN algorithm files
│
├── dashboard/                 ← PHASE 2
│   └── app.py                 ← Streamlit dashboard
│
├── tests/
│   ├── test_phase0.py         ← 22 tests — ALL PASSING
│   ├── test_phase1.py         ← BUILD THIS with Phase 1 code
│   └── test_backtest.py       ← BUILD THIS with Phase 1.5
│
├── scripts/
│   └── setup.py               ← one-time data fetch + health check
│
└── data/                      ← gitignored — local Parquet storage
    ├── raw/                   ← TICKER.parquet (OHLCV, 4 years)
    ├── signals/               ← YYYY-MM-DD.parquet
    ├── news/                  ← TICKER_YYYY-MM.parquet
    └── audit/                 ← YYYY-MM.parquet
```

---

## Technology Stack

| Component | Tool | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Historical data | Polygon.io | Always use for backtesting |
| Real-time data | Alpaca (free IEX feed) | Use for live signal compute |
| Order execution | Alpaca | Paper and live, same API |
| News | Finnhub | Free tier sufficient |
| LLM enrichment | Anthropic Claude API | Sonnet model, enrichment only |
| Indicators | pandas-ta | 130+ indicators available |
| Storage | Parquet (pyarrow) | No database — Parquet is faster |
| Backtesting | LEAN engine | Free, open source |
| Scheduling | APScheduler | Not Step Functions — overkill |
| Dashboard | Streamlit | Not React — overkill for personal tool |
| Hosting | AWS Lightsail | Single Python process |
| Secrets (prod) | AWS Secrets Manager | Never .env in production |

---

## Architecture Rules — NEVER Violate These

### LLM Boundaries (critical)
- LLM **DOES**: summarize news, explain signals, generate reasoning text for dashboard
- LLM **DOES NOT**: calculate indicators, generate confidence scores, make trade decisions
- Confidence scores are computed **statistically** (historical win rate × regime adjustment)
- If you find yourself asking the LLM to score confidence — STOP and use ConfidenceScorer

### Risk Rules — Hardcoded, Never Configurable at Runtime
- Max position size: 5% of portfolio per trade
- Max crypto exposure: 10% of portfolio total
- Max open positions: 5 simultaneously
- Daily loss circuit breaker: -2% triggers pause
- Max drawdown kill switch: -15% closes all positions
- Minimum confidence to trade: 0.60
- Stop loss on every trade: 7% below entry price

### Data Rules
- Historical backtest data: **always** from Polygon.io (adjusted=True)
- Real-time signal data: Alpaca recent bars (last 60 days)
- Never use Yahoo Finance — unreliable adjusted prices
- Storage: Parquet only, no SQLite, no RDS, no DynamoDB

### Code Rules
- Every module imports config via `from src.config import get_config`
- Every module gets secrets via `from src.secrets import Secrets`
- All data goes through `ParquetStore` — no direct file I/O elsewhere
- Use `loguru` for all logging — never `print()` in production code
- Use `tenacity` for all API retry logic
- Pydantic models for all data validation

---

## Asset Universe

**Stocks (tradeable):** NVDA, QQQ, GOOGL, CRWD, ORCL, MSFT, AAPL, META, AMD  
**Crypto (tradeable):** BTC, SOL  
**Watchlist (signals only, no trades):** AVGO, AMZN, TSLA  

---

## Signal Pipeline Flow

```
MarketDataService
    → loads OHLCV from ParquetStore
    → FeatureEngine.compute_all()        # 30+ indicators
    → SignalDetector.detect()            # BUY / SELL / HOLD per strategy
    → ConfidenceScorer.score()           # 0.0–1.0 statistical score
    → NewsService.get_sentiment_summary() # fast keyword sentiment
    → LLMAnalysisService.enrich()        # reasoning text only
    → RiskManager.validate()             # position size, exposure, limits
    → DecisionEngine.decide()            # APPROVE / BLOCK + reason
    → ParquetStore.save_signals()        # persist for dashboard + audit
```

---

## Confidence Scoring — How It Works

The confidence score is **NOT** from an LLM. It is computed as:

```
base_confidence    = historical_win_rate(pattern, strategy, lookback=60_trades)
regime_multiplier  = 1.0 if bull_regime else 0.85
volume_multiplier  = 1.05 if high_volume else 0.95
final_confidence   = base_confidence × regime_multiplier × volume_multiplier
                     clamped to [0.0, 1.0]
```

Signals below 0.60 are **blocked** — never reach DecisionEngine.

---

## Quality Gate — Backtest Must Pass All 7

| Metric | Threshold |
|---|---|
| CAGR | ≥ 15% |
| Sharpe Ratio | ≥ 1.0 |
| Max Drawdown | ≤ 25% |
| Win Rate | ≥ 45% |
| Profit Factor | ≥ 1.5 |
| Calmar Ratio | ≥ 0.5 |
| Min Trades | ≥ 30 |

---

## Phase Status

| Phase | Status | What Was Built |
|---|---|---|
| 0 — Data Pipeline | ✅ COMPLETE | Polygon, Alpaca, Parquet store, FeatureEngine, NewsService, 22 tests |
| 1 — Signal Intelligence | 🔲 IN PROGRESS | SignalDetector, ConfidenceScorer, LLMAnalysisService, RiskManager, DecisionEngine |
| 1.5 — Backtesting | 🔲 PENDING | LEAN strategies, LEANBridge, QualityGate |
| 2 — Paper Trading | 🔲 PENDING | Scheduler, ModelMonitor, live Streamlit dashboard |
| 3 — Live Trading | 🔲 PENDING | Flip config to live, S3 sync, CloudWatch alerts |

---

## Dynamic Universe Discovery (Phase 2.5)

Files: src/discovery/trend_scanner.py, stock_screener.py, universe_manager.py
Dashboard: dashboard/pages/discovery.py
Tests: tests/test_discovery.py

Purpose: Automatically surface trending stocks NOT in the predefined universe.
Flow: TrendScanner → StockScreener → UniverseManager → human approval → signal pipeline

Key rules:
- auto_approve is ALWAYS False — human clicks Approve in dashboard before any ticker trades
- Approved tickers enter signal pipeline but still need confidence >= 0.60 to generate orders
- All existing RiskManager rules apply to dynamically approved tickers
- Reddit scanning degrades gracefully if PRAW not installed (logs warning, returns [])
- Never modify config.yaml automatically — approved tickers stored in data/discovery/watchlist.parquet
- Ticker blocklist for Reddit parsing (common English words that look like tickers):
  IT, AT, BE, DO, GO, IF, IN, IS, ME, MY, NO, OF, ON, OR, SO, TO, UP, US, WE,
  ARE, FOR, THE, CAN, BIG, ALL, NEW, NOW, WHO, HOW, WHY, GET, SET, GOOD, REAL,
  COST, MOVE, HIGH, FULL, OPEN, JUST, LAST, NEXT, ALSO, THEN, WELL, BACK, INTO

---

## Current Task — Build Phase 1

Files to create (in order):
1. `src/signals/strategies/momentum.py`
2. `src/signals/strategies/trend_following.py`
3. `src/signals/strategies/volatility_breakout.py`
4. `src/signals/signal_detector.py`
5. `src/signals/confidence_scorer.py`
6. `src/llm/llm_analysis_service.py`
7. `src/risk/risk_manager.py`
8. `src/execution/decision_engine.py`
9. `tests/test_phase1.py`

After writing each module, run: `pytest tests/test_phase1.py -v`
Fix all failures before moving to the next module.

---

## API Keys Setup

Keys live in `.env` (local dev). Copy `.env.example` → `.env` and fill in:
- `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` — from paper trading dashboard
- `POLYGON_API_KEY` — from polygon.io
- `FINNHUB_API_KEY` — from finnhub.io
- `ANTHROPIC_API_KEY` — from console.anthropic.com

---

## How to Run Things

```powershell
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run only Phase 0 tests
pytest tests/test_phase0.py -v

# First-time data fetch (needs API keys in .env)
python scripts/setup.py

# Start Streamlit dashboard (Phase 2+)
streamlit run dashboard/app.py
```

---

## Coding Conventions

- Type hints on all function signatures
- Docstrings on all public methods (what it does, args, returns)
- Loguru for logging: `from loguru import logger`
- Retry with tenacity on all external API calls
- Return empty DataFrame (not None) when no data available
- Never raise exceptions silently — log then re-raise or return safe default
- Test files mirror src structure: `src/signals/x.py` → `tests/test_phase1.py`
