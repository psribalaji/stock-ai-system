# StockAI — System Architecture

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  Polygon.io (historical)    Alpaca (real-time)    Finnhub (news)│
└──────────┬──────────────────────┬──────────────────┬────────────┘
           │                      │                  │
           ▼                      ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                              │
│  PolygonClient          AlpacaClient          NewsService       │
│       └──────────────────────┴──────────────────┘              │
│                        MarketDataService                        │
│                              │                                  │
│                        ParquetStore                             │
│                    data/raw/TICKER.parquet                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL PIPELINE                              │
│                                                                 │
│  FeatureEngine ──► SignalDetector ──► ConfidenceScorer          │
│  (30+ indicators)  (3 strategies)    (win rate × regime)        │
│                                             │                   │
│                                    LLMAnalysisService           │
│                                    (Claude — reasoning only)    │
│                                             │                   │
│                                      RiskManager               │
│                                   (position size + rules)       │
│                                             │                   │
│                                     DecisionEngine             │
│                                  BUY / SELL / BLOCKED           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                              │
│                                                                 │
│               OrderExecutor                                     │
│         (paper mode → Alpaca API)                               │
│                    │                                            │
│           data/audit/YYYY-MM.parquet                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING & UI                              │
│  ModelMonitor            TradingScheduler     Streamlit         │
│  (drift detection)       (APScheduler)        Dashboard         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Signal Pipeline (per ticker, every 5 min)

```
OHLCV DataFrame (501 bars)
         │
         ▼
  FeatureEngine.compute_all()
  ┌──────────────────────────┐
  │ RSI, MACD, Bollinger     │
  │ SMA50/200, ADX, ATR      │
  │ vol_ratio, bb_pct        │
  │ bull_regime, golden_cross│
  └──────────┬───────────────┘
             │  features dict
             ▼
  ┌──────────────────────────────────────┐
  │           SignalDetector             │
  │                                      │
  │  MomentumStrategy.evaluate()         │
  │  TrendFollowingStrategy.evaluate()   │
  │  VolatilityBreakoutStrategy.evaluate()│
  └──────────┬───────────────────────────┘
             │  RawSignal (BUY/SELL/HOLD)
             │  (HOLDs filtered out)
             ▼
  ConfidenceScorer.score()
  ┌──────────────────────────┐
  │ base  = historical       │
  │        win_rate          │
  │ × regime_multiplier      │
  │   (1.0 bull / 0.85 bear) │
  │ × volume_multiplier      │
  │   (1.05 high / 0.95 low) │
  │ = confidence 0.0–1.0     │
  └──────────┬───────────────┘
             │
      confidence < 0.60?
         YES │                NO
             ▼                ▼
         BLOCKED     LLMAnalysisService
                     (Claude Sonnet 4.6)
                     - 2-3 sentence reasoning
                     - 2 risk factors
                             │
                             ▼
                     RiskManager.validate()
                  ┌──────────────────────┐
                  │ ✓ confidence ≥ 0.60  │
                  │ ✓ positions < 5      │
                  │ ✓ daily PnL > -2%    │
                  │ ✓ size ≤ 5% portfolio│
                  │ ✓ crypto ≤ 10%       │
                  └──────────┬───────────┘
                             │
                    APPROVED / BLOCKED
                             │
                             ▼
                     OrderExecutor
                  paper order → Alpaca
```

---

## 3. Scheduler Jobs

```
TradingScheduler (APScheduler)
│
├── [Every 5 min, market hours only]
│    job_signal_pipeline()
│    └── DecisionEngine.decide_all()
│         └── OrderExecutor.execute_all()  ← paper orders
│
├── [5:00am ET daily, Mon–Fri]
│    job_data_sync()
│    └── Polygon.fetch_daily_bars() → ParquetStore
│
├── [Sunday 8:00pm ET weekly]
│    job_drift_check()
│    └── ModelMonitor.check_drift()
│         ├── WIN_RATE_DROP  → WARNING
│         ├── SHARPE_FLOOR   → WARNING
│         ├── DRAWDOWN_PAUSE → CRITICAL (pauses strategy)
│         └── BENCHMARK_LAG  → WARNING
│
└── [1st of Jan/Apr/Jul/Oct]
     job_recalibrate()
     └── LEANBridge.run_all_strategies()
          └── ModelMonitor.recalibrate()
               └── updates confidence baselines
```

---

## 4. Confidence Score Breakdown

```
                    Historical trades
                    (last 60 matching)
                           │
                    win_rate = wins/total
                    (fallback: signal_strength
                     if < 10 trades)
                           │
              ┌────────────┴────────────┐
              │                         │
        bull_regime?             high_volume?
         YES      NO              YES      NO
          ×1.0   ×0.85           ×1.05   ×0.95
              │                         │
              └────────────┬────────────┘
                           │
                    clamp [0.0, 1.0]
                           │
                    final_confidence
                           │
                      ≥ 0.60?
                    YES       NO
                     │         │
                  TRADE     BLOCKED
```

---

## 5. Risk Rules (Hardcoded — Never Configurable at Runtime)

| Rule | Threshold |
|---|---|
| Min confidence to trade | 0.60 |
| Max position size | 5% of portfolio |
| Max crypto exposure | 10% of portfolio |
| Max open positions | 5 simultaneous |
| Stop loss | 7% below entry |
| Daily loss circuit breaker | −2% triggers pause |
| Max drawdown kill switch | −15% closes all |

---

## 6. Quality Gate (Phase 1.5 Backtest Must Pass All 7)

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

## 7. Storage Layout

```
data/
├── raw/
│   ├── NVDA.parquet       ← 4 years OHLCV
│   ├── AAPL.parquet
│   └── ...                ← one file per ticker
├── signals/
│   └── YYYY-MM-DD.parquet ← daily approved decisions
├── news/
│   └── TICKER_YYYY-MM.parquet
└── audit/
    └── YYYY-MM.parquet    ← all order results
```

---

## 8. Phase Status

| Phase | Status | What Was Built |
|---|---|---|
| 0 — Data Pipeline | ✅ Complete | Polygon, Alpaca, Parquet store, FeatureEngine, NewsService |
| 1 — Signal Intelligence | ✅ Complete | 3 strategies, SignalDetector, ConfidenceScorer, LLM enrichment, RiskManager, DecisionEngine |
| 1.5 — Backtesting | ✅ Complete | LEANBridge (Python backtest), QualityGate, 7 threshold checks |
| 2 — Paper Trading | ✅ Complete | TradingScheduler, ModelMonitor, Streamlit dashboard, OrderExecutor wired |
| 3 — Live Trading | ✅ Complete | LiveTrader pre-flight, S3 sync, CloudWatch alerts (gate: flip config to live) |
