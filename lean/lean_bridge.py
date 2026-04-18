"""
lean/lean_bridge.py — Bridge between StockAI strategies and backtesting.

Provides two execution modes:
  1. Python backtest (always available): pure-Python event simulation using
     FeatureEngine + strategy classes. Fast, no external deps.
  2. LEAN CLI backtest (optional, Phase 1.5+): if LEAN is installed, delegates
     to the LEAN engine for higher-fidelity simulation. Falls back to Python
     mode automatically if LEAN is not available.

Usage:
    bridge = LEANBridge()

    # Single strategy
    result = bridge.run_python_backtest("momentum", df, "NVDA")

    # All 3 strategies
    results = bridge.run_all_strategies(df, "NVDA")

    # Auto-mode (LEAN if available, else Python)
    result = bridge.run_backtest("momentum", df, "NVDA")
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config
from lean.quality_gate import BacktestResult


class LEANBridge:
    """
    Runs strategy backtests and returns standardised BacktestResult objects.

    The Python backtest mode is self-contained and requires no external tools.
    It uses FeatureEngine to compute indicators and the strategy classes to
    generate signals, then simulates a simple long-only portfolio.
    """

    STRATEGIES = ["momentum", "trend_following", "volatility_breakout", "mean_reversion"]
    MIN_ROWS = 60          # minimum bars needed for feature computation

    def __init__(self) -> None:
        self.config = get_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        strategy: str,
        df: pd.DataFrame,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run a backtest, preferring LEAN if available, else Python.

        Args:
            strategy:   One of "momentum", "trend_following", "volatility_breakout"
            df:         OHLCV DataFrame (must have open/high/low/close/volume columns)
            ticker:     Ticker symbol
            start_date: Optional ISO date string to filter data (inclusive)
            end_date:   Optional ISO date string to filter data (inclusive)

        Returns:
            BacktestResult with full performance metrics
        """
        # Currently always uses Python backtest.
        # LEAN CLI integration is a future extension point.
        return self.run_python_backtest(strategy, df, ticker,
                                        start_date=start_date, end_date=end_date)

    def run_python_backtest(
        self,
        strategy: str,
        df: pd.DataFrame,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Pure-Python event-driven backtest simulation.

        Simulation rules:
          - Long-only: enter on BUY signal, exit on SELL signal
          - Stop loss: 7% below entry (from risk config)
          - Slippage: applied to entry and exit prices (from backtest config)
          - One position at a time per ticker
          - Open position at end-of-history is closed at last price

        Args:
            strategy:   Strategy name (raises ValueError if unknown)
            df:         OHLCV DataFrame
            ticker:     Ticker symbol
            start_date: Optional filter start (ISO date string)
            end_date:   Optional filter end (ISO date string)

        Returns:
            BacktestResult. Returns zero-filled result on empty/insufficient data.

        Raises:
            ValueError: If strategy name is not recognised.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. Must be one of {self.STRATEGIES}"
            )

        cfg_bt   = self.config.backtest
        cfg_risk = self.config.risk

        # ── Default date range from config ───────────────────────────────────
        actual_start = start_date or cfg_bt.train_start
        actual_end   = end_date   or cfg_bt.train_end

        # ── Guard: empty input ────────────────────────────────────────────────
        if df is None or df.empty:
            logger.debug(f"[LEANBridge] {strategy}/{ticker}: empty DataFrame")
            return self._empty_result(strategy, ticker, actual_start, actual_end)

        # ── Apply date filter ─────────────────────────────────────────────────
        work_df = df.copy()
        if "timestamp" in work_df.columns:
            work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
            if start_date:
                sd = pd.Timestamp(start_date, tz="UTC")
                work_df = work_df[work_df["timestamp"] >= sd]
            if end_date:
                ed = pd.Timestamp(end_date, tz="UTC")
                work_df = work_df[work_df["timestamp"] <= ed]

            if not work_df.empty:
                actual_start = str(work_df["timestamp"].iloc[0].date())
                actual_end   = str(work_df["timestamp"].iloc[-1].date())

        # ── Guard: insufficient rows ──────────────────────────────────────────
        if len(work_df) < self.MIN_ROWS:
            logger.debug(
                f"[LEANBridge] {strategy}/{ticker}: only {len(work_df)} rows "
                f"(need {self.MIN_ROWS}) — returning empty result"
            )
            return self._empty_result(strategy, ticker, actual_start, actual_end)

        # ── Compute features ──────────────────────────────────────────────────
        try:
            from src.features.feature_engine import FeatureEngine
            fe = FeatureEngine()
            features_df = fe.compute_all(work_df.reset_index(drop=True), ticker=ticker)
        except Exception as exc:
            logger.warning(
                f"[LEANBridge] {strategy}/{ticker}: FeatureEngine failed — {exc}"
            )
            return self._empty_result(strategy, ticker, actual_start, actual_end)

        if features_df is None or features_df.empty or len(features_df) < self.MIN_ROWS:
            return self._empty_result(strategy, ticker, actual_start, actual_end)

        # ── Load strategy class ───────────────────────────────────────────────
        strategy_obj = self._get_strategy(strategy)

        # ── Simulate trades ───────────────────────────────────────────────────
        trades = self._simulate(
            strategy_obj=strategy_obj,
            features_df=features_df,
            slippage=cfg_bt.slippage_pct,
            stop_loss_pct=cfg_risk.stop_loss_pct,
        )

        if not trades:
            logger.debug(f"[LEANBridge] {strategy}/{ticker}: no trades generated")
            return self._empty_result(strategy, ticker, actual_start, actual_end)

        # ── Compute and return metrics ────────────────────────────────────────
        return self._compute_metrics(
            trades=trades,
            strategy=strategy,
            ticker=ticker,
            start_date=actual_start,
            end_date=actual_end,
        )

    def run_all_strategies(
        self,
        df: pd.DataFrame,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[BacktestResult]:
        """
        Run all 3 strategies and return results.

        Args:
            df:         OHLCV DataFrame
            ticker:     Ticker symbol
            start_date: Optional filter start
            end_date:   Optional filter end

        Returns:
            List of 3 BacktestResult objects (one per strategy)
        """
        results = []
        for strategy in self.STRATEGIES:
            result = self.run_python_backtest(
                strategy, df, ticker,
                start_date=start_date, end_date=end_date,
            )
            results.append(result)
            logger.info(
                f"[LEANBridge] {strategy}/{ticker}: "
                f"trades={result.total_trades}, "
                f"CAGR={result.cagr:.1%}, "
                f"Sharpe={result.sharpe_ratio:.2f}"
            )
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_strategy(self, name: str):
        """Lazily import and return a strategy instance."""
        if name == "momentum":
            from src.signals.strategies.momentum import MomentumStrategy
            return MomentumStrategy()
        elif name == "trend_following":
            from src.signals.strategies.trend_following import TrendFollowingStrategy
            return TrendFollowingStrategy()
        elif name == "volatility_breakout":
            from src.signals.strategies.volatility_breakout import VolatilityBreakoutStrategy
            return VolatilityBreakoutStrategy()
        elif name == "mean_reversion":
            from src.signals.strategies.mean_reversion import MeanReversionStrategy
            return MeanReversionStrategy()
        else:
            raise ValueError(f"Unknown strategy: '{name}'")

    def _simulate(
        self,
        strategy_obj,
        features_df: pd.DataFrame,
        slippage: float,
        stop_loss_pct: float,
    ) -> list:
        """
        Row-by-row long-only simulation.

        Returns:
            List of trade dicts, each with keys: pnl (float), won (bool)
        """
        trades: list = []

        if "close" not in features_df.columns:
            logger.warning("[LEANBridge] _simulate: 'close' column not found")
            return trades

        in_position = False
        entry_price = 0.0

        for i in range(len(features_df)):
            row = features_df.iloc[i]
            price = row.get("close")

            if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0:
                continue

            # Convert row to feature dict, replacing NaN with None for strategy evals
            features = {
                k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                for k, v in row.to_dict().items()
            }

            try:
                signal = strategy_obj.evaluate(features)
            except Exception as exc:
                logger.debug(f"[LEANBridge] strategy eval error at row {i}: {exc}")
                continue

            if not in_position:
                if signal.direction == "BUY":
                    entry_price = price * (1 + slippage)
                    in_position = True
            else:
                stop_price = entry_price * (1 - stop_loss_pct)

                if price <= stop_price:
                    # Stop loss triggered
                    exit_price = stop_price * (1 - slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({"pnl": pnl, "won": pnl > 0})
                    in_position = False

                elif signal.direction == "SELL":
                    exit_price = price * (1 - slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({"pnl": pnl, "won": pnl > 0})
                    in_position = False

        # Close any open position at end of data
        if in_position:
            last_close = features_df["close"].dropna()
            if not last_close.empty:
                exit_price = float(last_close.iloc[-1]) * (1 - slippage)
                pnl = (exit_price - entry_price) / entry_price
                trades.append({"pnl": pnl, "won": pnl > 0})

        return trades

    def _compute_metrics(
        self,
        trades: list,
        strategy: str,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Compute all 7 quality-gate metrics from a list of trade results."""
        pnls = [t["pnl"] for t in trades]
        wins = [t["won"] for t in trades]

        # Equity curve (compounded)
        equity = np.cumprod([1.0 + p for p in pnls])
        total_return = float(equity[-1] - 1.0)

        # CAGR
        try:
            start_ts = pd.Timestamp(start_date)
            end_ts   = pd.Timestamp(end_date)
            years    = max((end_ts - start_ts).days / 365.25, 0.1)
        except Exception:
            years = max(len(trades) / 252, 0.1)

        cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)

        # Sharpe (annualised, clamped >= 0)
        arr = np.array(pnls, dtype=float)
        if len(arr) >= 2:
            std = float(arr.std(ddof=1))
            sharpe = float((arr.mean() / std) * math.sqrt(252)) if std > 1e-12 else 0.0
        else:
            sharpe = 0.0
        sharpe = max(0.0, sharpe)

        # Max drawdown (positive fraction)
        peak       = np.maximum.accumulate(equity)
        drawdowns  = (peak - equity) / np.where(peak > 0, peak, 1.0)
        max_dd     = float(np.clip(drawdowns.max(), 0.0, 1.0))

        # Win rate
        win_rate = float(sum(wins) / len(wins)) if wins else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss   = abs(sum(p for p in pnls if p < 0))
        if gross_loss > 1e-12:
            profit_factor = float(gross_profit / gross_loss)
        elif gross_profit > 0:
            profit_factor = float(gross_profit * 100)   # no losses — large PF
        else:
            profit_factor = 0.0

        # Calmar ratio
        calmar = float(cagr / max_dd) if max_dd > 1e-9 else (float(cagr) if cagr > 0 else 0.0)

        return BacktestResult(
            strategy=strategy,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100_000.0,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            total_trades=len(trades),
            metadata={"source": "python_backtest"},
        )

    def _empty_result(
        self,
        strategy: str,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Return a zero-filled BacktestResult for insufficient/missing data."""
        return BacktestResult(
            strategy=strategy,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100_000.0,
            total_return=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            metadata={"source": "python_backtest"},
        )
