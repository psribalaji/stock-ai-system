"""
lean/quality_gate.py — Quality gate for backtest results.

Validates a BacktestResult against the 7 required thresholds from config.yaml.
All thresholds come from config.quality_gate — never hardcoded here.

Usage:
    gate = QualityGate()
    result = gate.validate(backtest_result)
    if result.passed:
        print("Ready for paper trading")
    else:
        print(result.summary)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

from loguru import logger

from src.config import get_config


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """
    Output of a single strategy backtest run.

    All rate/fraction fields are in decimal form (0.18 = 18%).
    """
    strategy: str
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    total_return: float          # fractional total return (0.5 = +50%)
    cagr: float                  # compound annual growth rate
    sharpe_ratio: float          # annualised Sharpe
    max_drawdown: float          # max peak-to-trough drawdown (positive fraction)
    win_rate: float              # fraction of winning trades
    profit_factor: float         # gross profit / gross loss
    calmar_ratio: float          # CAGR / max_drawdown
    total_trades: int
    metadata: dict = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """
    Result of validating a BacktestResult against the quality gate.

    Attributes:
        strategy:     Strategy name (or "combined")
        passed:       True only if ALL 7 checks pass
        passed_count: Number of checks that passed
        total_checks: Always 7
        checks:       List of per-check dicts with keys: metric, value, threshold, passed
        summary:      Human-readable summary string
    """
    strategy: str
    passed: bool
    passed_count: int
    total_checks: int
    checks: List[dict]
    summary: str


# ── Quality Gate ──────────────────────────────────────────────────────────────

class QualityGate:
    """
    Validates backtest results against the 7 quality thresholds.

    Thresholds are loaded from config.quality_gate — never hardcoded.
    """

    def __init__(self) -> None:
        self.config = get_config()

    def get_thresholds(self) -> dict:
        """
        Return the 7 quality gate thresholds from config.

        Returns:
            Dict with keys: min_cagr, min_sharpe, max_drawdown, min_win_rate,
            min_profit_factor, min_calmar, min_trades
        """
        cfg = self.config.quality_gate
        return {
            "min_cagr":           cfg.min_cagr,
            "min_sharpe":         cfg.min_sharpe,
            "max_drawdown":       cfg.max_drawdown,
            "min_win_rate":       cfg.min_win_rate,
            "min_profit_factor":  cfg.min_profit_factor,
            "min_calmar":         cfg.min_calmar,
            "min_trades":         cfg.min_trades,
        }

    def validate(self, result: BacktestResult) -> QualityGateResult:
        """
        Validate a single BacktestResult against all 7 thresholds.

        Args:
            result: BacktestResult from LEANBridge or Python backtest

        Returns:
            QualityGateResult with pass/fail details for each metric
        """
        cfg = self.config.quality_gate

        checks = [
            {
                "metric":    "cagr",
                "value":     result.cagr,
                "threshold": cfg.min_cagr,
                "passed":    result.cagr >= cfg.min_cagr,
            },
            {
                "metric":    "sharpe_ratio",
                "value":     result.sharpe_ratio,
                "threshold": cfg.min_sharpe,
                "passed":    result.sharpe_ratio >= cfg.min_sharpe,
            },
            {
                "metric":    "max_drawdown",
                "value":     result.max_drawdown,
                "threshold": cfg.max_drawdown,
                "passed":    result.max_drawdown <= cfg.max_drawdown,  # lower is better
            },
            {
                "metric":    "win_rate",
                "value":     result.win_rate,
                "threshold": cfg.min_win_rate,
                "passed":    result.win_rate >= cfg.min_win_rate,
            },
            {
                "metric":    "profit_factor",
                "value":     result.profit_factor,
                "threshold": cfg.min_profit_factor,
                "passed":    result.profit_factor >= cfg.min_profit_factor,
            },
            {
                "metric":    "calmar_ratio",
                "value":     result.calmar_ratio,
                "threshold": cfg.min_calmar,
                "passed":    result.calmar_ratio >= cfg.min_calmar,
            },
            {
                "metric":    "total_trades",
                "value":     result.total_trades,
                "threshold": cfg.min_trades,
                "passed":    result.total_trades >= cfg.min_trades,
            },
        ]

        passed_count = sum(1 for c in checks if c["passed"])
        passed = passed_count == 7

        if passed:
            summary = (
                f"{result.strategy}/{result.ticker}: PASSED all 7 quality checks "
                f"(CAGR={result.cagr:.1%}, Sharpe={result.sharpe_ratio:.2f}, "
                f"WinRate={result.win_rate:.1%}, Trades={result.total_trades})"
            )
        else:
            failed_names = [c["metric"] for c in checks if not c["passed"]]
            summary = (
                f"{result.strategy}/{result.ticker}: FAILED {len(failed_names)}/7 checks "
                f"— {', '.join(failed_names)}"
            )

        level = logger.info if passed else logger.warning
        level(f"[QualityGate] {summary}")

        return QualityGateResult(
            strategy=result.strategy,
            passed=passed,
            passed_count=passed_count,
            total_checks=7,
            checks=checks,
            summary=summary,
        )

    def validate_combined(self, results: List[BacktestResult]) -> QualityGateResult:
        """
        Validate a list of BacktestResults as a combined portfolio.

        Aggregation rules:
          - cagr, sharpe_ratio, win_rate, profit_factor, calmar_ratio: mean
          - max_drawdown: worst-case (max)
          - total_trades: sum

        Args:
            results: List of BacktestResult from multiple strategies/tickers

        Returns:
            QualityGateResult for the combined result (strategy="combined")
        """
        if not results:
            logger.warning("[QualityGate] validate_combined: empty results list")
            return QualityGateResult(
                strategy="combined",
                passed=False,
                passed_count=0,
                total_checks=7,
                checks=[],
                summary="No backtest results to validate",
            )

        combined = BacktestResult(
            strategy="combined",
            ticker=",".join(r.ticker for r in results),
            start_date=results[0].start_date,
            end_date=results[0].end_date,
            initial_capital=results[0].initial_capital,
            total_return=float(np.mean([r.total_return for r in results])),
            cagr=float(np.mean([r.cagr for r in results])),
            sharpe_ratio=float(np.mean([r.sharpe_ratio for r in results])),
            max_drawdown=float(max(r.max_drawdown for r in results)),   # worst case
            win_rate=float(np.mean([r.win_rate for r in results])),
            profit_factor=float(np.mean([r.profit_factor for r in results])),
            calmar_ratio=float(np.mean([r.calmar_ratio for r in results])),
            total_trades=sum(r.total_trades for r in results),
        )

        return self.validate(combined)
