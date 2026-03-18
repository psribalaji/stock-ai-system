"""
monitoring/model_monitor.py — Live strategy drift detection and win-rate tracking.

Compares live trading performance against backtest baselines and flags
degraded strategies before they cause significant losses.

Drift checks (from config.drift):
  - Sharpe floor:       live Sharpe < 0.5  → WARNING
  - Win-rate drop:      live win-rate dropped > 15pp vs baseline → WARNING
  - Drawdown pause:     live drawdown > 15%  → CRITICAL (pause strategy)
  - Benchmark lag:      live return lags SPY by > 10% over 30 days → WARNING
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import get_config


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class StrategyMetrics:
    """Live performance snapshot for a single strategy."""
    strategy: str
    live_win_rate: float          # fraction of winning trades (0–1)
    live_sharpe: float            # annualised Sharpe ratio
    live_drawdown: float          # max drawdown as positive fraction
    live_return: float            # total return fraction
    baseline_win_rate: float      # from backtest (used for drift delta)
    trade_count: int
    last_updated: datetime


@dataclass
class DriftAlert:
    """A single drift event for a strategy."""
    strategy: str
    alert_type: str               # "WIN_RATE_DROP" | "SHARPE_FLOOR" | "DRAWDOWN_PAUSE" | "BENCHMARK_LAG"
    severity: str                 # "WARNING" | "CRITICAL"
    message: str
    triggered_at: datetime


@dataclass
class DriftReport:
    """Full drift check result across all strategies."""
    checked_at: datetime
    alerts: List[DriftAlert]
    strategy_metrics: List[StrategyMetrics]
    any_paused: bool
    paused_strategies: List[str]

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0

    def summary(self) -> str:
        if not self.alerts:
            return "All strategies nominal — no drift detected."
        lines = [f"Drift report — {len(self.alerts)} alert(s):"]
        for a in self.alerts:
            lines.append(f"  [{a.severity}] {a.strategy}: {a.alert_type} — {a.message}")
        if self.paused_strategies:
            lines.append(f"  PAUSED strategies: {', '.join(self.paused_strategies)}")
        return "\n".join(lines)


# ── Monitor ───────────────────────────────────────────────────────────────────

class ModelMonitor:
    """
    Tracks live strategy performance and detects drift vs backtest baselines.

    Usage:
        monitor = ModelMonitor()

        # Record completed trades (call after each trade closes)
        monitor.record_trade("momentum", ticker="NVDA", pnl_pct=0.04, won=True)

        # Check for drift (call weekly or on-demand)
        report = monitor.check_drift()
        print(report.summary())

        # Get current metrics for dashboard
        metrics = monitor.get_metrics("momentum")
    """

    # Baseline win-rates from Phase 1.5 backtest results (per strategy)
    # These are updated by recalibrate() from actual backtest data
    DEFAULT_BASELINES: Dict[str, float] = {
        "momentum":            0.55,
        "trend_following":     0.52,
        "volatility_breakout": 0.50,
    }

    def __init__(self, baselines: Optional[Dict[str, float]] = None) -> None:
        self.config = get_config()
        self._baselines: Dict[str, float] = baselines or dict(self.DEFAULT_BASELINES)

        # Trade history per strategy: list of {"pnl_pct": float, "won": bool, "ts": datetime}
        self._trades: Dict[str, List[dict]] = {s: [] for s in self._baselines}

        # Paused strategies (CRITICAL drift)
        self._paused: set[str] = set()

        logger.info("[ModelMonitor] Initialised — tracking strategies: "
                    f"{list(self._baselines)}")

    # ── Trade recording ───────────────────────────────────────────────────────

    def record_trade(
        self,
        strategy: str,
        ticker: str,
        pnl_pct: float,
        won: bool,
    ) -> None:
        """
        Record a completed trade result.

        Args:
            strategy: Strategy that generated the signal
            ticker:   Ticker symbol
            pnl_pct:  Realised P&L as a fraction (e.g. 0.04 = +4%)
            won:      Whether the trade was profitable
        """
        if strategy not in self._trades:
            self._trades[strategy] = []

        self._trades[strategy].append({
            "ticker":  ticker,
            "pnl_pct": pnl_pct,
            "won":     won,
            "ts":      datetime.now(timezone.utc),
        })
        logger.debug(f"[ModelMonitor] Recorded {strategy}/{ticker}: "
                     f"{'WIN' if won else 'LOSS'} {pnl_pct:+.2%}")

    def record_trades_from_df(self, df: pd.DataFrame) -> int:
        """
        Bulk-record trades from a DataFrame (e.g. loaded from audit log).

        Expected columns: strategy, ticker, pnl_pct, won
        Returns number of trades recorded.
        """
        required = {"strategy", "ticker", "pnl_pct", "won"}
        if df.empty or not required.issubset(df.columns):
            return 0

        count = 0
        for _, row in df.iterrows():
            self.record_trade(
                strategy=str(row["strategy"]),
                ticker=str(row["ticker"]),
                pnl_pct=float(row["pnl_pct"]),
                won=bool(row["won"]),
            )
            count += 1
        return count

    # ── Metrics computation ───────────────────────────────────────────────────

    def get_metrics(self, strategy: str) -> Optional[StrategyMetrics]:
        """
        Compute live performance metrics for a strategy.

        Returns None if fewer than 5 trades recorded (not enough data).
        """
        trades = self._trades.get(strategy, [])
        if len(trades) < 5:
            return None

        pnl_series = [t["pnl_pct"] for t in trades]
        wins       = [t["won"]     for t in trades]

        win_rate   = sum(wins) / len(wins)
        total_ret  = sum(pnl_series)
        sharpe     = self._compute_sharpe(pnl_series)
        drawdown   = self._compute_max_drawdown(pnl_series)
        baseline   = self._baselines.get(strategy, 0.50)

        return StrategyMetrics(
            strategy=strategy,
            live_win_rate=win_rate,
            live_sharpe=sharpe,
            live_drawdown=drawdown,
            live_return=total_ret,
            baseline_win_rate=baseline,
            trade_count=len(trades),
            last_updated=datetime.now(timezone.utc),
        )

    def get_all_metrics(self) -> List[StrategyMetrics]:
        """Return metrics for all strategies that have enough data."""
        results = []
        for strategy in self._trades:
            m = self.get_metrics(strategy)
            if m is not None:
                results.append(m)
        return results

    # ── Drift detection ───────────────────────────────────────────────────────

    def check_drift(
        self,
        benchmark_return: Optional[float] = None,
    ) -> DriftReport:
        """
        Run all drift checks across all strategies.

        Args:
            benchmark_return: SPY return over same period (fraction).
                              Used for benchmark-lag check. Skip if None.

        Returns:
            DriftReport with all alerts and current paused strategies.
        """
        cfg        = self.config.drift
        alerts: List[DriftAlert] = []
        paused: List[str] = []
        metrics_list = self.get_all_metrics()

        for m in metrics_list:
            strategy_alerts = self._check_strategy(m, cfg, benchmark_return)
            alerts.extend(strategy_alerts)

            # Pause on CRITICAL
            is_critical = any(a.severity == "CRITICAL" for a in strategy_alerts)
            if is_critical:
                self._paused.add(m.strategy)
                paused.append(m.strategy)
                logger.warning(f"[ModelMonitor] {m.strategy} PAUSED due to critical drift")
            elif m.strategy in self._paused and not strategy_alerts:
                # Auto-resume if no alerts
                self._paused.discard(m.strategy)
                logger.info(f"[ModelMonitor] {m.strategy} auto-resumed — drift cleared")

        report = DriftReport(
            checked_at=datetime.now(timezone.utc),
            alerts=alerts,
            strategy_metrics=metrics_list,
            any_paused=len(self._paused) > 0,
            paused_strategies=list(self._paused),
        )

        if alerts:
            logger.warning(f"[ModelMonitor] Drift check: {len(alerts)} alert(s)\n"
                           + report.summary())
        else:
            logger.info("[ModelMonitor] Drift check: all strategies nominal")

        return report

    def is_paused(self, strategy: str) -> bool:
        """Return True if a strategy is currently paused due to critical drift."""
        return strategy in self._paused

    def resume(self, strategy: str) -> None:
        """Manually resume a paused strategy."""
        self._paused.discard(strategy)
        logger.info(f"[ModelMonitor] {strategy} manually resumed")

    def recalibrate(self, new_baselines: Dict[str, float]) -> None:
        """
        Update baseline win-rates from fresh backtest results.
        Called quarterly by the scheduler.

        Args:
            new_baselines: Dict mapping strategy name → new baseline win-rate
        """
        for strategy, baseline in new_baselines.items():
            if 0.0 <= baseline <= 1.0:
                self._baselines[strategy] = baseline
                logger.info(f"[ModelMonitor] Recalibrated {strategy} baseline → {baseline:.1%}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_strategy(
        self,
        m: StrategyMetrics,
        cfg,
        benchmark_return: Optional[float],
    ) -> List[DriftAlert]:
        """Run all drift checks for a single strategy. Returns list of alerts."""
        alerts: List[DriftAlert] = []
        now = datetime.now(timezone.utc)

        # 1. Sharpe floor
        if math.isfinite(m.live_sharpe) and m.live_sharpe < cfg.sharpe_floor:
            alerts.append(DriftAlert(
                strategy=m.strategy,
                alert_type="SHARPE_FLOOR",
                severity="WARNING",
                message=(f"Sharpe {m.live_sharpe:.2f} is below floor {cfg.sharpe_floor:.2f} "
                         f"over {m.trade_count} trades"),
                triggered_at=now,
            ))

        # 2. Win-rate drop
        win_rate_delta = m.baseline_win_rate - m.live_win_rate
        if win_rate_delta >= cfg.win_rate_drop:
            alerts.append(DriftAlert(
                strategy=m.strategy,
                alert_type="WIN_RATE_DROP",
                severity="WARNING",
                message=(f"Win rate dropped {win_rate_delta:.1%} "
                         f"(baseline {m.baseline_win_rate:.1%} → live {m.live_win_rate:.1%})"),
                triggered_at=now,
            ))

        # 3. Drawdown pause (CRITICAL)
        if m.live_drawdown >= cfg.drawdown_pause:
            alerts.append(DriftAlert(
                strategy=m.strategy,
                alert_type="DRAWDOWN_PAUSE",
                severity="CRITICAL",
                message=(f"Live drawdown {m.live_drawdown:.1%} exceeds pause threshold "
                         f"{cfg.drawdown_pause:.1%}"),
                triggered_at=now,
            ))

        # 4. Benchmark lag
        if benchmark_return is not None:
            lag = benchmark_return - m.live_return
            if lag >= cfg.benchmark_lag:
                alerts.append(DriftAlert(
                    strategy=m.strategy,
                    alert_type="BENCHMARK_LAG",
                    severity="WARNING",
                    message=(f"Lagging benchmark by {lag:.1%} "
                             f"(strategy {m.live_return:+.1%}, benchmark {benchmark_return:+.1%})"),
                    triggered_at=now,
                ))

        return alerts

    @staticmethod
    def _compute_sharpe(pnl_series: List[float], risk_free: float = 0.0) -> float:
        """Annualised Sharpe ratio from a list of per-trade P&L fractions."""
        if len(pnl_series) < 2:
            return 0.0
        arr  = np.array(pnl_series, dtype=float)
        mean = arr.mean() - risk_free
        std  = arr.std(ddof=1)
        if std < 1e-12:
            return 0.0
        # Approximate annualisation: assume ~252 trades/year
        return float((mean / std) * math.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(pnl_series: List[float]) -> float:
        """Max drawdown as a positive fraction from cumulative P&L."""
        if not pnl_series:
            return 0.0
        cumulative = np.cumprod([1 + r for r in pnl_series])
        peak       = np.maximum.accumulate(cumulative)
        drawdowns  = (peak - cumulative) / peak
        return float(drawdowns.max())
