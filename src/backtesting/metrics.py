"""
backtesting/metrics.py — Performance metrics for backtest results.

All metrics are computed from an equity curve and a trade log.
No external dependencies beyond numpy/pandas.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config


def compute_metrics(
    equity_curve: list[tuple],   # [(date, portfolio_value), ...]
    trades: list[dict],          # list of closed trade dicts
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """
    Compute full performance metrics from an equity curve and trade list.

    Args:
        equity_curve: Ordered list of (date, portfolio_value) tuples.
        trades:       List of closed trade dicts with keys:
                      ticker, entry_date, exit_date, entry_price,
                      exit_price, quantity, pnl, direction, exit_reason
        initial_capital: Starting capital

    Returns:
        Dict with CAGR, Sharpe, Sortino, max_drawdown, win_rate,
        profit_factor, calmar, total_trades, avg_hold_days, etc.
    """
    if not equity_curve:
        return _empty_metrics()

    eq_df = pd.DataFrame(equity_curve, columns=["date", "value"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.sort_values("date").set_index("date")

    final_value   = float(eq_df["value"].iloc[-1])
    start_value   = float(eq_df["value"].iloc[0])
    n_days        = (eq_df.index[-1] - eq_df.index[0]).days
    n_years       = max(n_days / 365.25, 1 / 365.25)

    # CAGR
    cagr = (final_value / initial_capital) ** (1 / n_years) - 1

    # Daily returns
    daily_returns = eq_df["value"].pct_change().dropna()

    # Sharpe (annualised, 0% risk-free)
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino (downside deviation only)
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * math.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    rolling_max = eq_df["value"].cummax()
    drawdown     = (eq_df["value"] - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())   # negative number

    # Calmar ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Total return
    total_return = (final_value - initial_capital) / initial_capital

    if not trades:
        return {
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "total_return": total_return,
            "final_value": final_value,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_hold_days": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }

    # Trade-level metrics
    pnls        = [t["pnl"] for t in trades]
    winners     = [p for p in pnls if p > 0]
    losers      = [p for p in pnls if p <= 0]

    win_rate        = len(winners) / len(pnls) if pnls else 0.0
    gross_profit    = sum(winners)
    gross_loss      = abs(sum(losers))
    profit_factor   = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_pnl         = sum(pnls) / len(pnls) if pnls else 0.0

    hold_days = []
    for t in trades:
        try:
            d = (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
            hold_days.append(d)
        except Exception:
            pass
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0.0

    return {
        "cagr":              round(cagr, 4),
        "sharpe":            round(sharpe, 3),
        "sortino":           round(sortino, 3),
        "max_drawdown":      round(max_drawdown, 4),
        "calmar":            round(calmar, 3),
        "total_return":      round(total_return, 4),
        "final_value":       round(final_value, 2),
        "total_trades":      len(trades),
        "win_rate":          round(win_rate, 4),
        "profit_factor":     round(profit_factor, 3),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "avg_hold_days":     round(avg_hold, 1),
        "largest_win":       round(max(pnls), 2) if pnls else 0.0,
        "largest_loss":      round(min(pnls), 2) if pnls else 0.0,
    }


def check_quality_gate(metrics: dict[str, Any]) -> dict[str, bool]:
    """
    Check each metric against the quality gate thresholds from config.yaml.

    Args:
        metrics: Output of compute_metrics()

    Returns:
        Dict mapping metric name → True (pass) / False (fail).
        Also includes "overall_pass" key.
    """
    cfg = get_config().quality_gate

    results = {
        "cagr":           metrics.get("cagr", 0)          >= cfg.min_cagr,
        "sharpe":         metrics.get("sharpe", 0)        >= cfg.min_sharpe,
        "max_drawdown":   abs(metrics.get("max_drawdown", 1)) <= cfg.max_drawdown,
        "win_rate":       metrics.get("win_rate", 0)      >= cfg.min_win_rate,
        "profit_factor":  metrics.get("profit_factor", 0) >= cfg.min_profit_factor,
        "calmar":         metrics.get("calmar", 0)        >= cfg.min_calmar,
        "total_trades":   metrics.get("total_trades", 0)  >= cfg.min_trades,
    }
    results["overall_pass"] = all(results.values())
    return results


def format_report(
    metrics: dict[str, Any],
    gate: dict[str, bool],
    period_start: str,
    period_end: str,
) -> str:
    """
    Render a text report suitable for terminal output.
    """
    check = lambda k: "✅" if gate.get(k, True) else "❌"

    lines = [
        "",
        "═" * 55,
        "  BACKTEST RESULTS",
        f"  Period : {period_start} → {period_end}",
        "═" * 55,
        f"  CAGR             {metrics['cagr']*100:>8.2f}%  {check('cagr')}",
        f"  Total Return     {metrics['total_return']*100:>8.2f}%",
        f"  Final Value      ${metrics['final_value']:>10,.2f}",
        "─" * 55,
        f"  Sharpe Ratio     {metrics['sharpe']:>8.3f}   {check('sharpe')}",
        f"  Sortino Ratio    {metrics['sortino']:>8.3f}",
        f"  Calmar Ratio     {metrics['calmar']:>8.3f}   {check('calmar')}",
        f"  Max Drawdown     {metrics['max_drawdown']*100:>8.2f}%  {check('max_drawdown')}",
        "─" * 55,
        f"  Total Trades     {metrics['total_trades']:>8d}   {check('total_trades')}",
        f"  Win Rate         {metrics['win_rate']*100:>8.2f}%  {check('win_rate')}",
        f"  Profit Factor    {metrics['profit_factor']:>8.3f}   {check('profit_factor')}",
        f"  Avg PnL / Trade  ${metrics['avg_pnl_per_trade']:>9.2f}",
        f"  Avg Hold Days    {metrics['avg_hold_days']:>8.1f}",
        f"  Largest Win      ${metrics['largest_win']:>9.2f}",
        f"  Largest Loss     ${metrics['largest_loss']:>9.2f}",
        "─" * 55,
        f"  Quality Gate     {'✅ PASS' if gate['overall_pass'] else '❌ FAIL'}",
        "═" * 55,
        "",
    ]
    return "\n".join(lines)


def consecutive_losses(pnl_series: list[float]) -> dict[str, Any]:
    """
    Compute current and worst-ever consecutive losing streak.

    Args:
        pnl_series: P&L fractions in chronological order, e.g. [0.02, -0.01, 0.03]

    Returns:
        current_streak: losses in a row right now
        max_streak:     worst streak ever in the series
        last_10:        last 10 pnl values for sparkline display
        alert:          True when current_streak >= 3 (pause and review)
    """
    if not pnl_series:
        return {"current_streak": 0, "max_streak": 0, "last_10": [], "alert": False}

    max_streak = 0
    run = 0
    for pnl in pnl_series:
        if pnl < 0:
            run += 1
            max_streak = max(max_streak, run)
        else:
            run = 0

    current_streak = 0
    for pnl in reversed(pnl_series):
        if pnl < 0:
            current_streak += 1
        else:
            break

    return {
        "current_streak": current_streak,
        "max_streak": max_streak,
        "last_10": pnl_series[-10:],
        "alert": current_streak >= 3,
    }


def _empty_metrics() -> dict[str, Any]:
    return {k: 0.0 for k in [
        "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
        "total_return", "final_value", "total_trades", "win_rate",
        "profit_factor", "avg_pnl_per_trade", "avg_hold_days",
        "largest_win", "largest_loss",
    ]}
