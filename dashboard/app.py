"""
dashboard/app.py — Streamlit dashboard for StockAI.

Pages:
  1. Signals Today   — today's approved trade decisions
  2. Portfolio       — open positions and P&L summary
  3. Monitor         — strategy health and drift alerts
  4. Audit Log       — historical trade records
  5. Live Trading    — pre-flight checklist, go-live gate, kill switch

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="StockAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (avoid crashing dashboard if src not on path) ────────────────
import sys
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_config, reload_config
from src.live.live_trader import LiveTrader
from src.ingestion.storage import ParquetStore
from src.monitoring.model_monitor import ModelMonitor, DriftReport


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_store() -> ParquetStore:
    cfg = get_config()
    return ParquetStore(cfg.data.storage_path)


@st.cache_resource
def _get_monitor() -> ModelMonitor:
    return ModelMonitor()


@st.cache_data(ttl=60)
def _load_signals(days_back: int = 7) -> pd.DataFrame:
    store = _get_store()
    end   = date.today()
    start = end - timedelta(days=days_back)
    return store.load_signals(start=start, end=end)


@st.cache_data(ttl=60)
def _load_audit(days_back: int = 30) -> pd.DataFrame:
    store = _get_store()
    end   = date.today()
    start = end - timedelta(days=days_back)
    return store.load_audit(start=start, end=end)


def _fmt_pct(v: float) -> str:
    return f"{v:+.1%}" if math.isfinite(v) else "—"


def _fmt_num(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if math.isfinite(v) else "—"


def _direction_color(direction: str) -> str:
    return "🟢" if direction == "BUY" else "🔴"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> str:
    cfg = reload_config()
    is_live = cfg.is_live

    st.sidebar.title("StockAI")

    # Dynamic live/paper badge
    if is_live:
        st.sidebar.error("🔴 LIVE TRADING — real money at risk")
    else:
        st.sidebar.success("🟡 PAPER TRADING — simulation mode")

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["Signals Today", "Portfolio", "Monitor", "Audit Log", "Live Trading"],
        index=0,
    )

    st.sidebar.divider()
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    # Account snapshot (always shown)
    try:
        from src.ingestion.alpaca_client import AlpacaClient
        _acct = AlpacaClient()._trading_client.get_account()
        _total = float(_acct.portfolio_value)
        _pnl   = _total - 100_000
        st.sidebar.metric("Portfolio",  f"${_total:,.0f}", delta=f"${_pnl:+,.0f}")
        st.sidebar.metric("Cash",       f"${float(_acct.cash):,.0f}")
    except Exception:
        st.sidebar.caption(
            f"Mode: **{cfg.trading.mode.upper()}**  \n"
            f"Universe: {', '.join(cfg.assets.all_tradeable)}"
        )

    return page


# ── Page: Signals Today ───────────────────────────────────────────────────────

def page_signals() -> None:
    st.header("Signals Today")

    days = st.slider("Look-back (days)", min_value=1, max_value=30, value=1)
    df   = _load_signals(days_back=days)

    if df.empty:
        st.info("No signals found for the selected period.")
        return

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        tickers = ["All"] + sorted(df["ticker"].unique().tolist()) if "ticker" in df.columns else ["All"]
        ticker_filter = st.selectbox("Ticker", tickers)
    with col2:
        strategies = ["All"] + sorted(df["strategy"].unique().tolist()) if "strategy" in df.columns else ["All"]
        strategy_filter = st.selectbox("Strategy", strategies)
    with col3:
        direction_filter = st.selectbox("Direction", ["All", "BUY", "SELL"])

    filtered = df.copy()
    if ticker_filter    != "All" and "ticker"    in filtered.columns:
        filtered = filtered[filtered["ticker"]    == ticker_filter]
    if strategy_filter  != "All" and "strategy"  in filtered.columns:
        filtered = filtered[filtered["strategy"]  == strategy_filter]
    if direction_filter != "All" and "direction" in filtered.columns:
        filtered = filtered[filtered["direction"] == direction_filter]

    st.caption(f"{len(filtered)} signal(s) shown")

    # Metrics row
    if not filtered.empty and "confidence" in filtered.columns:
        approved = filtered[filtered.get("approved", pd.Series(True, index=filtered.index)) == True] if "approved" in filtered.columns else filtered
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total signals",  len(filtered))
        m2.metric("Approved",       len(approved))
        m3.metric("Avg confidence", f"{filtered['confidence'].mean():.1%}" if not filtered.empty else "—")
        m4.metric("Tickers",        filtered["ticker"].nunique() if "ticker" in filtered.columns else "—")

    # Table
    display_cols = [c for c in [
        "timestamp", "ticker", "direction", "strategy", "pattern",
        "confidence", "position_size_pct", "entry_price", "stop_loss_price",
        "approved", "block_reason", "llm_summary",
    ] if c in filtered.columns]

    if "direction" in filtered.columns:
        filtered["direction"] = filtered["direction"].apply(
            lambda d: f"{_direction_color(d)} {d}"
        )

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )


# ── Page: Portfolio ───────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_alpaca_portfolio():
    """Fetch live account + positions from Alpaca. Returns (account_dict, positions_df)."""
    try:
        from src.ingestion.alpaca_client import AlpacaClient
        client  = AlpacaClient()
        account = client._trading_client.get_account()
        positions = client.get_positions()
        return {
            "portfolio_value": float(account.portfolio_value),
            "cash":            float(account.cash),
            "equity":          float(account.equity),
            "buying_power":    float(account.buying_power),
        }, positions
    except Exception as exc:
        return None, pd.DataFrame()


def page_portfolio() -> None:
    st.header("Portfolio")

    STARTING_BALANCE = 100_000.0

    acct, positions = _load_alpaca_portfolio()

    # ── Account summary ───────────────────────────────────────────
    st.subheader("Account Summary")

    if acct:
        total_value   = acct["portfolio_value"]
        cash          = acct["cash"]
        invested      = total_value - cash
        total_pnl     = total_value - STARTING_BALANCE
        total_pnl_pct = total_pnl / STARTING_BALANCE

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Value",      f"${total_value:,.2f}")
        m2.metric("Cash Available",   f"${cash:,.2f}")
        m3.metric("Invested",         f"${invested:,.2f}")
        m4.metric(
            "Total P&L",
            f"${total_pnl:+,.2f}",
            delta=f"{total_pnl_pct:+.2%}",
            delta_color="normal",
        )
        m5.metric("Buying Power",     f"${acct['buying_power']:,.2f}")
    else:
        st.warning("Could not connect to Alpaca — showing cached audit data only.")

    st.divider()

    # ── Open positions ─────────────────────────────────────────────
    st.subheader("Open Positions")

    if positions is not None and not positions.empty:
        pos = positions.copy()

        # Add unrealized P&L %
        if "unrealized_pl" in pos.columns and "market_value" in pos.columns:
            pos["cost_basis"] = pos["market_value"] - pos["unrealized_pl"]
            pos["pnl_pct"]    = pos["unrealized_pl"] / pos["cost_basis"].replace(0, float("nan"))

        # Format for display
        display_pos = pd.DataFrame({
            "Ticker":       pos["ticker"],
            "Shares":       pos["qty"],
            "Avg Entry $":  pos.get("avg_entry_price", pd.Series(["—"] * len(pos))),
            "Current $":    pos.get("current_price",   pd.Series(["—"] * len(pos))),
            "Market Value": pos["market_value"].apply(lambda v: f"${v:,.2f}"),
            "Unrealized P&L": pos["unrealized_pl"].apply(
                lambda v: f"${v:+,.2f}" if pd.notna(v) else "—"
            ),
            "P&L %": pos["pnl_pct"].apply(
                lambda v: f"{v:+.2%}" if pd.notna(v) else "—"
            ) if "pnl_pct" in pos.columns else "—",
            "Side": pos.get("side", pd.Series(["—"] * len(pos))),
        })

        st.dataframe(display_pos.reset_index(drop=True), use_container_width=True)

        # Mini bar chart — P&L per ticker
        if "unrealized_pl" in pos.columns:
            import plotly.graph_objects as go
            colors = ["green" if v >= 0 else "red" for v in pos["unrealized_pl"]]
            fig = go.Figure(go.Bar(
                x=pos["ticker"],
                y=pos["unrealized_pl"],
                marker_color=colors,
                text=[f"${v:+,.0f}" for v in pos["unrealized_pl"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Unrealized P&L per Position",
                yaxis_title="P&L ($)",
                height=320,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No open positions.")

    st.divider()

    # ── Closed trades from audit ───────────────────────────────────
    st.subheader("Closed Trades")
    audit = _load_audit(days_back=90)

    submitted = audit[audit["status"] == "submitted"] if not audit.empty and "status" in audit.columns else pd.DataFrame()

    if submitted.empty:
        st.info("No closed trades yet.")
    else:
        # Summary metrics
        total_trades = len(submitted)
        m1, m2, m3 = st.columns(3)
        m1.metric("Orders submitted", total_trades)

        if "pnl_pct" in submitted.columns and not submitted["pnl_pct"].isna().all():
            winners  = (submitted["pnl_pct"] > 0).sum()
            win_rate = winners / total_trades
            m2.metric("Win rate", f"{win_rate:.1%}")
            m3.metric("Winners / Losers", f"{winners} / {total_trades - winners}")
        else:
            m2.metric("Strategies", submitted["strategy"].nunique() if "strategy" in submitted.columns else "—")
            m3.metric("Tickers traded", submitted["ticker"].nunique() if "ticker" in submitted.columns else "—")

        # Table
        display_cols = [c for c in [
            "timestamp_submitted", "ticker", "direction", "strategy",
            "qty", "entry_price", "position_size_usd", "confidence",
            "stop_loss_price", "order_id",
        ] if c in submitted.columns]
        st.dataframe(submitted[display_cols].reset_index(drop=True), use_container_width=True, height=300)


# ── Page: Monitor ─────────────────────────────────────────────────────────────

def page_monitor() -> None:
    st.header("Strategy Monitor")

    monitor = _get_monitor()
    audit   = _load_audit(days_back=90)

    # Load historical trades into monitor
    if not audit.empty:
        required = {"strategy", "ticker", "pnl_pct", "won"}
        if required.issubset(audit.columns):
            monitor.record_trades_from_df(audit)

    # Run drift check
    report: DriftReport = monitor.check_drift()

    # Status banner
    if report.any_paused:
        st.error(f"PAUSED strategies: {', '.join(report.paused_strategies)}")
    elif report.has_alerts:
        st.warning(f"{len(report.alerts)} drift alert(s) detected")
    else:
        st.success("All strategies nominal — no drift detected")

    st.caption(f"Last checked: {report.checked_at.strftime('%Y-%m-%d %H:%M UTC')}")
    st.divider()

    # Alerts table
    if report.alerts:
        st.subheader("Drift Alerts")
        alert_rows = [
            {
                "Severity": a.severity,
                "Strategy": a.strategy,
                "Type":     a.alert_type,
                "Message":  a.message,
                "Time":     a.triggered_at.strftime("%Y-%m-%d %H:%M"),
            }
            for a in report.alerts
        ]
        alert_df = pd.DataFrame(alert_rows)
        st.dataframe(alert_df, use_container_width=True)
        st.divider()

    # Strategy metrics
    st.subheader("Strategy Health")
    metrics = report.strategy_metrics

    if not metrics:
        st.info("Not enough trade data yet (need ≥ 5 trades per strategy).")
        cfg = get_config()
        st.caption("Strategies being tracked: momentum, trend_following, volatility_breakout")
        return

    for m in metrics:
        with st.expander(f"{m.strategy}  |  {m.trade_count} trades", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Win Rate",    f"{m.live_win_rate:.1%}",
                      delta=f"{m.live_win_rate - m.baseline_win_rate:+.1%} vs baseline")
            c2.metric("Sharpe",      _fmt_num(m.live_sharpe))
            c3.metric("Max Drawdown", _fmt_pct(m.live_drawdown))
            c4.metric("Total Return", _fmt_pct(m.live_return))
            c5.metric("Paused",      "YES" if monitor.is_paused(m.strategy) else "NO")


# ── Page: Audit Log ───────────────────────────────────────────────────────────

def page_audit() -> None:
    st.header("Audit Log")

    days = st.slider("Look-back (days)", min_value=7, max_value=365, value=30)
    df   = _load_audit(days_back=days)

    if df.empty:
        st.info("No audit records found.")
        return

    st.caption(f"{len(df)} trade record(s)")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        tickers = ["All"] + sorted(df["ticker"].unique().tolist()) if "ticker" in df.columns else ["All"]
        ticker_filter = st.selectbox("Ticker", tickers, key="audit_ticker")
    with col2:
        strategies = ["All"] + sorted(df["strategy"].unique().tolist()) if "strategy" in df.columns else ["All"]
        strategy_filter = st.selectbox("Strategy", strategies, key="audit_strat")

    filtered = df.copy()
    if ticker_filter   != "All" and "ticker"   in filtered.columns:
        filtered = filtered[filtered["ticker"]   == ticker_filter]
    if strategy_filter != "All" and "strategy" in filtered.columns:
        filtered = filtered[filtered["strategy"] == strategy_filter]

    display_cols = [c for c in [
        "timestamp_submitted", "ticker", "direction", "strategy",
        "confidence", "entry_price", "position_size_usd",
        "stop_loss_price", "pnl_pct", "won", "approved",
    ] if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

    # Download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"audit_{date.today().isoformat()}.csv",
        mime="text/csv",
    )


# ── Page: Live Trading ────────────────────────────────────────────────────────

def page_live_trading() -> None:
    st.header("Live Trading")

    cfg = reload_config()
    is_live = cfg.is_live

    # Mode banner
    if is_live:
        st.error("🔴 System is currently in **LIVE** mode — real money is being traded.")
    else:
        st.warning("🟡 System is in **PAPER** mode. Complete the pre-flight checklist to go live.")

    st.divider()

    # ── Pre-flight checklist ──────────────────────────────────────
    st.subheader("Pre-flight Checklist")
    st.caption("All checks must pass before going live.")

    try:
        from src.ingestion.alpaca_client import AlpacaClient
        trader = LiveTrader(alpaca_client=AlpacaClient())
    except Exception:
        trader = LiveTrader()

    with st.spinner("Running pre-flight checks..."):
        report = trader.run_preflight()

    for check in report.checks:
        col_icon, col_name, col_msg = st.columns([0.5, 2, 5])
        with col_icon:
            st.write("✅" if check.passed else "❌")
        with col_name:
            st.write(f"**{check.name}**")
        with col_msg:
            st.write(check.message)

    st.divider()

    passed = sum(1 for c in report.checks if c.passed)
    total  = len(report.checks)

    if report.all_passed:
        st.success(f"All {total}/{total} checks passed — system is ready to go live.")
    else:
        st.error(f"{total - passed} of {total} checks failed — resolve issues before going live.")

    st.divider()

    # ── Go-Live section ───────────────────────────────────────────
    st.subheader("Go Live")

    if is_live:
        st.info("Already in live mode.")
    else:
        st.markdown(
            "> **Warning:** Going live submits real orders with real money.  \n"
            "> Set `trading.mode: live` in `config.yaml` first, then click below."
        )

        col_btn, col_help = st.columns([1, 3])
        with col_btn:
            go_live_clicked = st.button(
                "Go Live",
                type="primary",
                disabled=not report.all_passed,
            )
        with col_help:
            if not report.all_passed:
                st.caption("Button disabled until all pre-flight checks pass.")

        if go_live_clicked:
            try:
                trader.go_live()
                st.success("Live trading activated. Monitor positions closely.")
                st.balloons()
            except RuntimeError as e:
                st.error(str(e))

    st.divider()

    # ── Kill Switch ───────────────────────────────────────────────
    st.subheader("Emergency Kill Switch")
    st.markdown(
        "> Cancels **all open orders** and **closes all positions** at market price.  \n"
        "> Use only in emergencies."
    )

    if "kill_switch_confirm" not in st.session_state:
        st.session_state["kill_switch_confirm"] = False

    if not st.session_state["kill_switch_confirm"]:
        if st.button("Activate Kill Switch", type="secondary"):
            st.session_state["kill_switch_confirm"] = True
            st.rerun()
    else:
        st.error("Are you sure? This will close ALL positions at market price.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("YES — Close Everything", type="primary"):
                try:
                    from src.ingestion.alpaca_client import AlpacaClient
                    AlpacaClient().close_all_positions()
                    st.success("Kill switch activated — all positions closed.")
                    st.session_state["kill_switch_confirm"] = False
                except Exception as e:
                    st.error(f"Kill switch failed: {e}")
        with col_no:
            if st.button("Cancel"):
                st.session_state["kill_switch_confirm"] = False
                st.rerun()

    st.divider()

    # ── Config snapshot ───────────────────────────────────────────
    st.subheader("Current Config")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mode",            cfg.trading.mode.upper())
    col2.metric("Max position",    f"{cfg.risk.max_position_pct:.0%}")
    col3.metric("Min confidence",  f"{cfg.risk.min_confidence:.0%}")
    col4.metric("Stop loss",       f"{cfg.risk.stop_loss_pct:.0%}")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Max positions",   cfg.risk.max_open_positions)
    col6.metric("Circuit breaker", f"{cfg.risk.daily_loss_limit:.0%} daily loss")
    col7.metric("Kill switch",     f"{cfg.risk.max_drawdown_pct:.0%} drawdown")
    col8.metric("S3 sync",         "ON" if cfg.data.sync_to_s3 else "OFF")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    page = _sidebar()

    if page == "Signals Today":
        page_signals()
    elif page == "Portfolio":
        page_portfolio()
    elif page == "Monitor":
        page_monitor()
    elif page == "Audit Log":
        page_audit()
    elif page == "Live Trading":
        page_live_trading()


if __name__ == "__main__":
    main()
