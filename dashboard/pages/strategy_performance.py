"""
dashboard/pages/strategy_performance.py — Strategy Performance comparison page.
Shows win rate, profit factor, trade count per strategy.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_strategy_performance_page() -> None:
    st.header("Strategy Performance")
    from src.ingestion.storage import ParquetStore
    from src.config import get_config

    cfg = get_config()
    store = ParquetStore(cfg.data.storage_path)

    days = st.slider("Look-back (days)", 7, 90, 30)
    end = date.today()
    start = end - timedelta(days=days)

    audit = store.load_audit(start=start, end=end)
    signals = store.load_signals(start=start, end=end)

    if signals.empty:
        st.info("No signal data yet. Run the pipeline to generate signals.")
        return

    # ── Per-strategy summary
    st.subheader("Strategy Comparison")
    strategies = ["momentum", "trend_following", "volatility_breakout", "mean_reversion"]
    if cfg.ml_ensemble.enabled:
        strategies.append("ml_ensemble")

    rows = []
    for strat in strategies:
        mask = signals["strategy"] == strat if "strategy" in signals.columns else pd.Series(False, index=signals.index)
        subset = signals[mask]
        total = len(subset)
        if total == 0:
            rows.append({"Strategy": strat, "Signals": 0, "Approved": 0, "Blocked": 0, "Approval Rate": "—"})
            continue

        approved = len(subset[subset["approved"] == True]) if "approved" in subset.columns else total
        blocked = total - approved
        rows.append({
            "Strategy": strat,
            "Signals": total,
            "Approved": approved,
            "Blocked": blocked,
            "Approval Rate": f"{approved / total:.0%}" if total > 0 else "—",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Signals over time chart
    if "timestamp" in signals.columns and "strategy" in signals.columns:
        st.subheader("Signals Over Time")
        signals["date"] = pd.to_datetime(signals["timestamp"]).dt.date
        chart_data = signals.groupby(["date", "strategy"]).size().unstack(fill_value=0)
        st.bar_chart(chart_data)

    # ── Confidence distribution
    if "confidence" in signals.columns and "strategy" in signals.columns:
        st.subheader("Confidence Distribution by Strategy")
        for strat in strategies:
            subset = signals[signals["strategy"] == strat]
            if not subset.empty:
                st.write(f"**{strat}** — avg: {subset['confidence'].mean():.3f}, "
                         f"min: {subset['confidence'].min():.3f}, "
                         f"max: {subset['confidence'].max():.3f}")

    # ── Audit log stats (if available)
    if not audit.empty and "strategy" in audit.columns:
        st.subheader("Execution Results")
        exec_stats = audit.groupby("strategy").agg(
            Orders=("status", "count"),
            Submitted=("status", lambda x: (x == "submitted").sum()),
            Skipped=("status", lambda x: (x == "skipped").sum()),
            Failed=("status", lambda x: (x == "failed").sum()),
        ).reset_index()
        st.dataframe(exec_stats, use_container_width=True, hide_index=True)
