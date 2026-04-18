"""
dashboard/pages/stops_tp.py — Trailing Stops & Take-Profits monitor page.
Shows active trailing stops, take-profit targets, and how positions are tracking.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_stops_tp_page() -> None:
    st.header("Trailing Stops & Take-Profits")
    from src.config import get_config

    cfg = get_config()

    st.subheader("Configuration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hard Stop Loss", f"{cfg.risk.stop_loss_pct:.0%}")
    c2.metric("Trailing Stop", f"{cfg.risk.trailing_stop_atr_mult:.1f}× ATR")
    c3.metric("Take-Profit Ratio", f"{cfg.risk.reward_risk_ratio:.1f}:1")
    c4.metric("Position Check", f"Every {cfg.schedule.position_check_interval_min} min")

    # ── Try to get live executor state
    st.subheader("Active Positions")
    try:
        from src.execution.order_executor import OrderExecutor
        executor = OrderExecutor(dry_run=True)

        trailing = getattr(executor, "_trailing_stops", {})
        take_profits = getattr(executor, "_take_profits", {})

        if not trailing and not take_profits:
            st.info("No active trailing stops or take-profit targets. "
                    "Positions are tracked when the scheduler is running.")

            # Show example of how it works
            st.subheader("How It Works")
            st.markdown("""
            **Trailing Stop** — follows price up, never moves down:
            ```
            Buy @ $120, ATR = $4.50 → Stop starts at $111.00
            Price → $135 → Stop trails to $126.00
            Price → $148 → Stop trails to $139.00
            Price → $139 → STOP TRIGGERED, profit locked
            ```

            **Take-Profit** — automatic exit at reward target:
            ```
            Buy @ $120, Stop = $111 → Risk = $9
            Take-profit = $120 + (2 × $9) = $138
            Price → $138 → TAKE-PROFIT HIT, sell for +$18
            ```
            """)
        else:
            rows = []
            all_tickers = set(list(trailing.keys()) + list(take_profits.keys()))
            for ticker in sorted(all_tickers):
                ts = trailing.get(ticker, {})
                tp = take_profits.get(ticker)
                rows.append({
                    "Ticker": ticker,
                    "High Water": f"${ts.get('high_water', 0):.2f}" if ts else "—",
                    "Trailing Stop": f"${ts.get('stop_price', 0):.2f}" if ts else "—",
                    "ATR": f"${ts.get('atr', 0):.2f}" if ts else "—",
                    "Take-Profit": f"${tp:.2f}" if tp else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load executor state: {e}")

    # ── Recent audit showing stop/TP exits
    st.subheader("Recent Exits")
    try:
        from src.ingestion.storage import ParquetStore
        store = ParquetStore(cfg.data.storage_path)
        audit = store.load_audit()
        if not audit.empty and "direction" in audit.columns:
            sells = audit[audit["direction"] == "SELL"].tail(20)
            if not sells.empty:
                display_cols = [c for c in ["timestamp_submitted", "ticker", "strategy", "qty", "entry_price", "stop_loss_price", "status"] if c in sells.columns]
                st.dataframe(sells[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No SELL orders recorded yet")
        else:
            st.info("No audit data yet")
    except Exception:
        st.info("No audit data available")
