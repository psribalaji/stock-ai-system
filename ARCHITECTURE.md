# StockAI — System Architecture

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  Polygon.io (historical)    Alpaca (real-time)    Finnhub (news)│
│                          ApeWisdom (Reddit trends)              │
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
│  (30+ indicators)  (5 strategies)    (win rate × regime ×      │
│                                       volume × sentiment)       │
│                                             │                   │
│                                    LLMAnalysisService           │
│                                    (Claude — sentiment +        │
│                                     reasoning enrichment)       │
│                                             │                   │
│                                      RiskManager               │
│                                   (position size + correlation  │
│                                    + trailing stop + TP)        │
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
│         + Trailing stop tracker                                 │
│         + Take-profit tracker                                   │
│                    │                                            │
│           data/audit/YYYY-MM.parquet                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING & UI                              │
│  ModelMonitor            TradingScheduler     Streamlit         │
│  (drift detection)       (two-loop arch)      Dashboard         │
│                                               (11 pages)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Signal Pipeline (per ticker, every 30 min)

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
  │ stoch_k, stoch_d         │
  └──────────┬───────────────┘
             │  features dict
             ▼
  ┌───────────────────────────────────────────┐
  │              SignalDetector               │
  │                                           │
  │  MomentumStrategy.evaluate()              │
  │  TrendFollowingStrategy.evaluate()        │
  │  VolatilityBreakoutStrategy.evaluate()    │
  │  MeanReversionStrategy.evaluate()         │
  │  MLEnsembleStrategy.evaluate() (if enabled)│
  └──────────┬────────────────────────────────┘
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
  │ × sentiment_multiplier   │
  │   (0.70–1.10 from LLM)   │
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
                  │ ✓ correlation < 0.70 │
                  │ ✓ size ≤ 5% portfolio│
                  │ ✓ crypto ≤ 10%       │
                  └──────────┬───────────┘
                             │
                    APPROVED / BLOCKED
                             │
                             ▼
                     OrderExecutor
                  paper order → Alpaca
                  + register trailing stop
                  + register take-profit
```

---

## 3. Scheduler Jobs (Two-Loop Architecture)

```
TradingScheduler (APScheduler)
│
├── [Every 5 min, market hours only] — FAST LOOP
│    job_position_check()
│    └── Fetch live prices (Alpaca)
│         ├── check_take_profits(price_map)
│         │    └── price ≥ target → SELL (lock profit)
│         └── update_trailing_stops(price_map)
│              ├── price > high_water → trail stop up
│              └── price ≤ stop → SELL (protect capital)
│
├── [Every 30 min, market hours only] — SLOW LOOP
│    job_signal_pipeline()
│    └── DecisionEngine.decide_all()
│         └── OrderExecutor.execute_all()
│              └── register stops + TPs for each BUY
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
├── [1st of Jan/Apr/Jul/Oct]
│    job_recalibrate()
│    └── LEANBridge.run_all_strategies()
│         └── ModelMonitor.recalibrate()
│
└── [Every 30 min + 8am pre-market]
     job_discovery_scan()
     └── ApeWisdom + Finnhub news velocity
          └── StockScreener → UniverseManager
```

---

## 4. Confidence Score Breakdown

```
                    Historical trades
                    (last 60 matching)
                           │
                    win_rate = wins/total
                    (fallback: seed defaults
                     if < 10 trades)
                           │
              ┌────────────┼────────────┐
              │            │            │
        bull_regime?  high_volume?  sentiment?
         YES    NO    YES    NO    +1.0  -1.0
         ×1.0  ×0.85 ×1.05 ×0.95 ×1.10 ×0.90
              │            │            │
              └────────────┼────────────┘
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

## 5. Risk Rules

| Rule | Threshold | Notes |
|---|---|---|
| Min confidence to trade | 0.60 | After all multipliers applied |
| Max position size | 5% of portfolio | Capped at available cash |
| Max crypto exposure | 10% of portfolio | |
| Max open positions | 5 simultaneous | |
| Max portfolio correlation | 0.70 avg | Blocks if new position too correlated with held |
| Stop loss (hard) | 7% below entry | Floor — never worse than this |
| Trailing stop | 2× ATR below high-water | Trails up, never down |
| Take-profit target | 2:1 reward-to-risk | entry + 2×(entry - stop) |
| Daily loss circuit breaker | −2% triggers pause | Resets next trading day |
| Max drawdown kill switch | −15% closes all | Requires manual reset |

---

## 6. Position Exit Logic

```
After BUY order submitted:
  │
  ├── register_trailing_stop(entry, ATR)
  │     stop = entry - (2 × ATR)
  │
  └── register_take_profit(target)
        target = entry + 2 × (entry - stop)

Every 5 minutes (fast loop):
  │
  ├── Price ≥ take-profit target?
  │     YES → SELL (profit locked)
  │
  ├── Price > high-water mark?
  │     YES → trail stop up: new_stop = price - (2 × ATR)
  │
  └── Price ≤ trailing stop?
        YES → SELL (loss capped)

Every 30 minutes (slow loop):
  │
  └── SELL signal from strategy?
        YES + confidence ≥ 0.60 → SELL
```

---

## 7. Quality Gate (Backtest Must Pass All 7)

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

## 8. Strategies

| Strategy | Type | BUY Patterns | SELL Patterns |
|---|---|---|---|
| Momentum | Rule-based | RSI oversold + MACD cross, MACD + volume, strong ROC | RSI overbought + MACD, MACD below EMA, negative ROC |
| Trend Following | Rule-based | Golden cross, MA alignment + ADX, near 52w high | Death cross, below SMA50 downtrend, trend weakening |
| Volatility Breakout | Rule-based | BB squeeze breakout up, BB upper + volume, vol expansion bull | BB squeeze breakdown, BB lower + volume, vol expansion bear |
| Mean Reversion | Rule-based | BB lower bounce + stoch, deep oversold + volume, BB mean revert up | BB upper rejection + stoch, deep overbought + volume, BB mean revert down |
| ML Ensemble | LightGBM | Predict probability > 0.55 | Predict probability < 0.45 |

ML Ensemble is **disabled by default** — enable after accumulating 300+ labeled trades.

---

## 9. Dashboard Pages

| Page | Purpose |
|---|---|
| Signals Today | Today's approved trade decisions |
| Discovery | Trending tickers, signal explanation, approve/ignore |
| Portfolio | Open positions and P&L summary |
| Monitor | Strategy health and drift alerts |
| Risk | Correlation heatmap, risk limits, circuit breaker status |
| Strategy Performance | Per-strategy comparison, signals over time |
| Stops & TPs | Active trailing stops, take-profit targets |
| Sentiment | Per-ticker news headlines, LLM sentiment scores |
| ML Ensemble | Training data progress, model status, feature importance |
| Audit Log | Historical trade records |
| Live Trading | Pre-flight checklist, go-live gate, kill switch |

Reusable widget: `render_price_chart(ticker)` — interactive candlestick/line chart with volume + MAs.

---

## 10. Storage Layout

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
├── audit/
│   └── YYYY-MM.parquet    ← all order results
└── ml_ensemble_model.pkl  ← trained model (when enabled)
```

---

## 11. Configuration Modes

| Mode | Config | Behavior |
|---|---|---|
| Paper trading | `trading.mode: paper` | Orders go to Alpaca paper account |
| Live trading | `trading.mode: live` | Real money — requires pre-flight pass |
| Simulation (24/7) | `schedule.force_market_hours: true` | Bypasses market hours check for testing |
| ML Ensemble | `ml_ensemble.enabled: true` | Adds 5th strategy after 300+ trades |

---

## 12. Phase Status

| Phase | Status | What Was Built |
|---|---|---|
| 0 — Data Pipeline | ✅ Complete | Polygon, Alpaca, Parquet store, FeatureEngine, NewsService |
| 1 — Signal Intelligence | ✅ Complete | 5 strategies, SignalDetector, ConfidenceScorer, LLM sentiment + enrichment, RiskManager, DecisionEngine |
| 1.5 — Backtesting | ✅ Complete | LEANBridge (Python backtest), QualityGate, 7 threshold checks |
| 2 — Paper Trading | ✅ Complete | TradingScheduler (two-loop), ModelMonitor, Streamlit dashboard (11 pages), OrderExecutor with trailing stops + take-profits |
| 2.5 — Discovery | ✅ Complete | ApeWisdom trending, Finnhub news velocity, StockScreener, UniverseManager, signal explanation |
| 3 — Live Trading | ✅ Complete | LiveTrader pre-flight, S3 sync, CloudWatch alerts, correlation checks |
