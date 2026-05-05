"""
backtesting/backtester.py — Walk-forward backtester.

Simulates the full signal pipeline on historical OHLCV data with no
lookahead bias. Fills at next-day open; stops and take-profits checked
intraday using high/low of the fill day.

Usage:
    bt = Backtester()
    result = bt.run(tickers=["NVDA", "AAPL"], start="2025-01-01", end="2026-01-01")
    print(result.summary)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.config import get_config
from src.features.feature_engine import FeatureEngine
from src.signals.strategies.momentum import MomentumStrategy, RawSignal
from src.signals.strategies.trend_following import TrendFollowingStrategy
from src.signals.strategies.volatility_breakout import VolatilityBreakoutStrategy
from src.signals.strategies.mean_reversion import MeanReversionStrategy
from src.signals.confidence_scorer import ConfidenceScorer
from src.backtesting.metrics import compute_metrics, check_quality_gate, format_report


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BacktestPosition:
    ticker: str
    entry_date: date
    entry_price: float
    quantity: float
    stop_price: float
    take_profit: float
    strategy: str
    confidence: float


@dataclass
class BacktestTrade:
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    direction: str = "BUY"
    exit_reason: str = ""
    strategy: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticker":       self.ticker,
            "entry_date":   str(self.entry_date),
            "exit_date":    str(self.exit_date),
            "entry_price":  round(self.entry_price, 4),
            "exit_price":   round(self.exit_price, 4),
            "quantity":     round(self.quantity, 4),
            "pnl":          round(self.pnl, 2),
            "direction":    self.direction,
            "exit_reason":  self.exit_reason,
            "strategy":     self.strategy,
            "confidence":   round(self.confidence, 3),
        }


@dataclass
class BacktestResult:
    metrics: dict
    quality_gate: dict
    trades: list[BacktestTrade]
    equity_curve: list[tuple]
    period_start: str
    period_end: str
    initial_capital: float
    tickers: list[str]
    summary: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.summary = format_report(
            self.metrics, self.quality_gate, self.period_start, self.period_end
        )

    def save(self, path: str = "./data/backtest") -> None:
        """Save results to JSON + CSV files."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        tag = f"{self.period_start}_{self.period_end}".replace("-", "")

        # Equity curve CSV
        eq_df = pd.DataFrame(self.equity_curve, columns=["date", "value"])
        eq_df.to_csv(out / f"equity_{tag}.csv", index=False)

        # Trades CSV
        if self.trades:
            tr_df = pd.DataFrame([t.to_dict() for t in self.trades])
            tr_df.to_csv(out / f"trades_{tag}.csv", index=False)

        # Metrics JSON — convert numpy types to plain Python for serialisation
        def _to_json_safe(obj):
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_safe(i) for i in obj]
            return obj

        summary_data = _to_json_safe({
            "period_start":    self.period_start,
            "period_end":      self.period_end,
            "initial_capital": self.initial_capital,
            "tickers":         self.tickers,
            "metrics":         self.metrics,
            "quality_gate":    self.quality_gate,
        })
        with open(out / f"summary_{tag}.json", "w") as fh:
            json.dump(summary_data, fh, indent=2)

        logger.info(f"[Backtester] Results saved to {out}/")


# ── Backtester ───────────────────────────────────────────────────────────────

class Backtester:
    """
    Walk-forward simulator for the stock-ai signal pipeline.

    Key design decisions:
    - Features precomputed once per ticker (rolling windows are correct; no lookahead)
    - Fills at next trading day's open price
    - Stop losses: triggered if day.low <= stop_price; filled at max(open, stop_price)
    - Take profits: triggered if day.high >= tp_price; filled at tp_price
    - No short selling — SELL signals only close existing long positions
    - Max concurrent positions from config.risk.max_open_positions
    - Position size: config.risk.max_position_pct × portfolio value
    """

    def __init__(self, data_path: str = "./data") -> None:
        self.config    = get_config()
        self.engine    = FeatureEngine()
        self.scorer    = ConfidenceScorer(store=None)
        self.strategies = [
            MomentumStrategy(),
            TrendFollowingStrategy(),
            VolatilityBreakoutStrategy(),
            MeanReversionStrategy(),
        ]
        self._data_path = Path(data_path)

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        tickers: Optional[list[str]] = None,
        start: str = "",
        end: str = "",
        initial_capital: float = 100_000.0,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            tickers:         List of ticker symbols. Defaults to config tradeable stocks.
            start:           ISO date string (inclusive). Defaults to earliest data.
            end:             ISO date string (inclusive). Defaults to latest data.
            initial_capital: Starting portfolio cash.

        Returns:
            BacktestResult with metrics, trades, equity curve, and quality gate.
        """
        cfg = self.config

        if tickers is None:
            tickers = list(cfg.assets.stocks)

        logger.info(f"[Backtester] Loading data for {len(tickers)} tickers…")
        price_data, feature_data = self._load_and_compute(tickers)

        # Build sorted trading calendar from all available dates
        all_dates = sorted({
            d
            for ticker in price_data
            for d in price_data[ticker].index
        })

        if not all_dates:
            raise ValueError("No trading dates found in data.")

        start_dt = pd.Timestamp(start).date() if start else all_dates[0]
        end_dt   = pd.Timestamp(end).date()   if end   else all_dates[-1]

        trading_days = [d for d in all_dates if start_dt <= d <= end_dt]

        if len(trading_days) < 10:
            raise ValueError(f"Only {len(trading_days)} trading days in range — too few.")

        logger.info(
            f"[Backtester] Simulating {len(trading_days)} days "
            f"({trading_days[0]} → {trading_days[-1]}) | capital=${initial_capital:,.0f}"
        )

        # Run simulation
        equity_curve, trades = self._simulate(
            tickers, price_data, feature_data, trading_days, initial_capital
        )

        metrics = compute_metrics(equity_curve, [t.to_dict() for t in trades], initial_capital)
        gate    = check_quality_gate(metrics)

        return BacktestResult(
            metrics=metrics,
            quality_gate=gate,
            trades=trades,
            equity_curve=equity_curve,
            period_start=str(trading_days[0]),
            period_end=str(trading_days[-1]),
            initial_capital=initial_capital,
            tickers=tickers,
        )

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_and_compute(
        self,
        tickers: list[str],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
        """
        Load OHLCV and precompute features for each ticker.

        Returns:
            price_data:   {ticker: DataFrame indexed by date, cols: open/high/low/close/volume}
            feature_data: {ticker: {date: feature_dict}}
        """
        price_data: dict[str, pd.DataFrame] = {}
        feature_data: dict[str, dict] = {}

        for ticker in tickers:
            raw = self._load_ohlcv(ticker)
            if raw.empty or len(raw) < 60:
                logger.warning(f"[Backtester] {ticker}: insufficient data ({len(raw)} rows), skipping")
                continue

            # Compute features on full history (rolling windows use only past rows — no lookahead)
            feat_df = self.engine.compute_all(raw.copy(), ticker)

            # Store OHLCV indexed by date for simulation price lookups
            price_data[ticker] = self._to_date_indexed(raw)
            # Store feature snapshot per date for fast walk-forward lookup
            feature_data[ticker] = self._extract_feature_timeline(feat_df)

        logger.info(f"[Backtester] Loaded {len(price_data)} tickers with features.")
        return price_data, feature_data

    def _load_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Load OHLCV parquet in original format (with timestamp column) for FeatureEngine."""
        path = self._data_path / "raw" / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(f"[Backtester] No parquet for {ticker}")
            return pd.DataFrame()

        df = pd.read_parquet(path)
        if df.empty:
            return df

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    def _to_date_indexed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp-indexed raw/feature DataFrame to date-indexed."""
        out = df.copy()
        out["date"] = pd.to_datetime(out["timestamp"]).dt.date
        out = out.set_index("date").sort_index()
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
        return out[keep].astype(float)

    def _extract_feature_timeline(self, feat_df: pd.DataFrame) -> dict[date, dict]:
        """Convert feature DataFrame (has timestamp col, range index) to {date: feature_dict}."""
        timeline: dict[date, dict] = {}
        feat_df = feat_df.copy()
        feat_df["_date"] = pd.to_datetime(feat_df["timestamp"]).dt.date
        for _, row in feat_df.iterrows():
            d = row["_date"]
            timeline[d] = row.drop("_date").to_dict()
        return timeline

    # ── Simulation loop ───────────────────────────────────────────────────────

    def _simulate(
        self,
        tickers: list[str],
        price_data: dict[str, pd.DataFrame],
        feature_data: dict[str, dict],
        trading_days: list[date],
        initial_capital: float,
    ) -> tuple[list[tuple], list[BacktestTrade]]:
        """Walk-forward simulation. Returns (equity_curve, closed_trades)."""
        cfg            = self.config
        min_conf       = cfg.risk.min_confidence
        max_positions  = cfg.risk.max_open_positions
        pos_size_pct   = cfg.risk.max_position_pct
        stop_pct       = cfg.risk.stop_loss_pct
        rr_ratio       = cfg.risk.reward_risk_ratio

        cash: float                             = initial_capital
        positions: dict[str, BacktestPosition] = {}
        pending_buys: dict[str, dict]           = {}   # ticker → order info
        pending_sells: set[str]                 = set() # tickers to sell at next open
        closed_trades: list[BacktestTrade]      = []
        equity_curve: list[tuple]               = []
        cooldowns: dict[str, date]              = {}   # ticker → can_reenter_after date

        for i, today in enumerate(trading_days):
            day_price: dict[str, dict] = {}
            for ticker in tickers:
                if ticker not in price_data or today not in price_data[ticker].index:
                    continue
                row = price_data[ticker].loc[today]
                day_price[ticker] = {
                    "open":  float(row["open"]),
                    "high":  float(row["high"]),
                    "low":   float(row["low"]),
                    "close": float(row["close"]),
                }

            # ── 1. Execute pending sells (signal sell from yesterday) ─────────
            for ticker in list(pending_sells):
                if ticker in positions and ticker in day_price:
                    pos   = positions.pop(ticker)
                    fill  = day_price[ticker]["open"]
                    pnl   = (fill - pos.entry_price) * pos.quantity
                    cash += fill * pos.quantity
                    closed_trades.append(BacktestTrade(
                        ticker=ticker,
                        entry_date=pos.entry_date,
                        exit_date=today,
                        entry_price=pos.entry_price,
                        exit_price=fill,
                        quantity=pos.quantity,
                        pnl=pnl,
                        exit_reason="signal_sell",
                        strategy=pos.strategy,
                        confidence=pos.confidence,
                    ))
                    cooldowns[ticker] = today + timedelta(days=2)
            pending_sells.clear()

            # ── 2. Execute pending buys (queued from yesterday's signals) ─────
            for ticker, order in list(pending_buys.items()):
                if ticker not in day_price:
                    continue
                if ticker in positions:
                    continue  # already holding
                if len(positions) >= max_positions:
                    continue  # position limit hit

                fill        = day_price[ticker]["open"]
                portfolio_v = cash + sum(
                    positions[t].quantity * price_data[t].loc[today, "close"]
                    if t in price_data and today in price_data[t].index else 0
                    for t in positions
                )
                # Recalculate quantity at fill price
                alloc    = portfolio_v * pos_size_pct
                quantity = alloc / fill if fill > 0 else 0

                if quantity <= 0 or cash < alloc:
                    continue

                stop  = fill * (1 - stop_pct)
                tp    = fill + rr_ratio * (fill - stop)

                cash -= fill * quantity
                positions[ticker] = BacktestPosition(
                    ticker=ticker,
                    entry_date=today,
                    entry_price=fill,
                    quantity=quantity,
                    stop_price=stop,
                    take_profit=tp,
                    strategy=order["strategy"],
                    confidence=order["confidence"],
                )
            pending_buys.clear()

            # ── 3. Check stops and take-profits for held positions ────────────
            for ticker in list(positions.keys()):
                if ticker not in day_price:
                    continue

                pos  = positions[ticker]
                ohlc = day_price[ticker]

                # Stop loss: triggered if low breaches stop
                if ohlc["low"] <= pos.stop_price:
                    fill  = max(ohlc["open"], pos.stop_price)  # gap-down protection
                    pnl   = (fill - pos.entry_price) * pos.quantity
                    cash += fill * pos.quantity
                    closed_trades.append(BacktestTrade(
                        ticker=ticker,
                        entry_date=pos.entry_date,
                        exit_date=today,
                        entry_price=pos.entry_price,
                        exit_price=fill,
                        quantity=pos.quantity,
                        pnl=pnl,
                        exit_reason="stop_loss",
                        strategy=pos.strategy,
                        confidence=pos.confidence,
                    ))
                    del positions[ticker]
                    cooldowns[ticker] = today + timedelta(days=2)

                # Take profit: triggered if high reaches target
                elif ohlc["high"] >= pos.take_profit:
                    fill  = pos.take_profit
                    pnl   = (fill - pos.entry_price) * pos.quantity
                    cash += fill * pos.quantity
                    closed_trades.append(BacktestTrade(
                        ticker=ticker,
                        entry_date=pos.entry_date,
                        exit_date=today,
                        entry_price=pos.entry_price,
                        exit_price=fill,
                        quantity=pos.quantity,
                        pnl=pnl,
                        exit_reason="take_profit",
                        strategy=pos.strategy,
                        confidence=pos.confidence,
                    ))
                    del positions[ticker]

            # ── 4. Generate new signals ────────────────────────────────────────
            if len(positions) < max_positions:
                for ticker in tickers:
                    if ticker not in feature_data:
                        continue
                    if today not in feature_data[ticker]:
                        continue
                    if ticker in positions or ticker in pending_buys:
                        continue
                    if cooldowns.get(ticker, date.min) > today:
                        continue

                    feats = feature_data[ticker][today]

                    for strategy in self.strategies:
                        try:
                            raw_signal = strategy.evaluate(feats, ticker)
                        except Exception as exc:
                            logger.debug(f"[Backtester] {strategy.NAME}/{ticker} error: {exc}")
                            continue

                        if raw_signal.direction == "HOLD":
                            continue

                        scored = self.scorer.score(raw_signal, ticker, features=feats)
                        if scored.blocked:
                            continue

                        if raw_signal.direction == "BUY":
                            if ticker not in pending_buys and len(positions) + len(pending_buys) < max_positions:
                                pending_buys[ticker] = {
                                    "strategy":   strategy.NAME,
                                    "confidence": scored.confidence,
                                }
                                break  # one signal per ticker per day

                        elif raw_signal.direction == "SELL":
                            if ticker in positions:
                                pending_sells.add(ticker)
                                break

            # ── 5. Mark-to-market equity ──────────────────────────────────────
            held_value = 0.0
            for ticker, pos in positions.items():
                if ticker in day_price:
                    held_value += pos.quantity * day_price[ticker]["close"]
                else:
                    # Use entry price if no data for today
                    held_value += pos.quantity * pos.entry_price

            portfolio_value = cash + held_value
            equity_curve.append((today, portfolio_value))

            if (i + 1) % 50 == 0:
                logger.info(
                    f"[Backtester] Day {i+1}/{len(trading_days)} {today} | "
                    f"positions={len(positions)} | equity=${portfolio_value:,.0f}"
                )

        # ── 6. Close all remaining positions at last close ────────────────────
        last_day = trading_days[-1]
        for ticker, pos in list(positions.items()):
            if ticker in price_data and last_day in price_data[ticker].index:
                fill = float(price_data[ticker].loc[last_day, "close"])
            else:
                fill = pos.entry_price

            pnl   = (fill - pos.entry_price) * pos.quantity
            cash += fill * pos.quantity
            closed_trades.append(BacktestTrade(
                ticker=ticker,
                entry_date=pos.entry_date,
                exit_date=last_day,
                entry_price=pos.entry_price,
                exit_price=fill,
                quantity=pos.quantity,
                pnl=pnl,
                exit_reason="end_of_period",
                strategy=pos.strategy,
                confidence=pos.confidence,
            ))

        logger.info(
            f"[Backtester] Done. Trades={len(closed_trades)} | "
            f"Final equity=${cash:,.0f}"
        )
        return equity_curve, closed_trades
