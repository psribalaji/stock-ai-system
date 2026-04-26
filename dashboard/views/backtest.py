"""
dashboard/views/backtest.py — Run backtests from the UI.
Pick strategy + ticker + date range, see results with equity curve.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_backtest_page() -> None:
    st.header("📈 Backtesting")
    st.caption("Test strategies against historical data before risking real money.")

    from src.config import get_config
    from src.ingestion.storage import ParquetStore

    cfg = get_config()
    store = ParquetStore(cfg.data.storage_path)

    # ── Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        strategies = ["momentum", "trend_following", "volatility_breakout", "mean_reversion"]
        strategy = st.selectbox("Strategy", strategies,
                                help="Which strategy to backtest")
    with col2:
        all_tickers = cfg.assets.all_tradeable
        ticker = st.selectbox("Ticker", all_tickers,
                              help="Stock to test against")
    with col3:
        run_all = st.checkbox("Run all strategies", value=False,
                              help="Compare all strategies on this ticker")

    col4, col5 = st.columns(2)
    with col4:
        start = st.date_input("Start date", value=date(2023, 1, 1))
    with col5:
        end = st.date_input("End date", value=date.today())

    if st.button("🚀 Run Backtest", type="primary"):
        df = store.load_ohlcv(ticker)
        if df.empty or len(df) < 60:
            st.error(f"Not enough data for {ticker} ({len(df)} bars). Need 60+.")
            return

        try:
            from lean.lean_bridge import LEANBridge
            from lean.quality_gate import QualityGate

            bridge = LEANBridge()
            gate = QualityGate()

            if run_all:
                with st.spinner("Running all strategies..."):
                    results = bridge.run_all_strategies(df, ticker,
                                                        start_date=str(start), end_date=str(end))
            else:
                with st.spinner(f"Running {strategy}..."):
                    result = bridge.run_python_backtest(strategy, df, ticker,
                                                        start_date=str(start), end_date=str(end))
                    results = [result]

            # ── Results
            for result in results:
                st.divider()
                st.subheader(f"{result.strategy} — {result.ticker}")

                # Quality gate
                gate_result = gate.validate(result)
                if gate_result.passed:
                    st.success(f"✅ PASSED quality gate ({gate_result.passed_count}/7)")
                else:
                    st.error(f"❌ FAILED quality gate ({gate_result.passed_count}/7)")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("CAGR", f"{result.cagr:.1%}",
                          help="Compound Annual Growth Rate. Quality gate: ≥15%")
                m2.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}",
                          help="Risk-adjusted return. >1.0 is good. Quality gate: ≥1.0")
                m3.metric("Max Drawdown", f"{result.max_drawdown:.1%}",
                          help="Worst peak-to-trough drop. Quality gate: ≤25%")
                m4.metric("Win Rate", f"{result.win_rate:.1%}",
                          help="% of trades that were profitable. Quality gate: ≥45%")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Profit Factor", f"{result.profit_factor:.2f}",
                          help="Gross profit / gross loss. >1.5 is good")
                m6.metric("Calmar Ratio", f"{result.calmar_ratio:.2f}",
                          help="CAGR / max drawdown. Quality gate: ≥0.5")
                m7.metric("Total Trades", result.total_trades,
                          help="Number of trades executed. Quality gate: ≥30")
                m8.metric("Total Return", f"{result.total_return:.1%}")

                # Quality gate detail
                with st.expander("Quality Gate Details"):
                    for check in gate_result.checks:
                        icon = "✅" if check["passed"] else "❌"
                        st.write(f"{icon} **{check['metric']}**: {check['value']:.4f} "
                                 f"(threshold: {check['threshold']})")

            # ── Combined results comparison
            if run_all and len(results) > 1:
                st.divider()
                st.subheader("Strategy Comparison")
                comp_rows = []
                for r in results:
                    comp_rows.append({
                        "Strategy": r.strategy,
                        "CAGR": f"{r.cagr:.1%}",
                        "Sharpe": f"{r.sharpe_ratio:.2f}",
                        "Max DD": f"{r.max_drawdown:.1%}",
                        "Win Rate": f"{r.win_rate:.1%}",
                        "Profit Factor": f"{r.profit_factor:.2f}",
                        "Trades": r.total_trades,
                        "Gate": "✅" if gate.validate(r).passed else "❌",
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

                # Combined quality gate
                combined = gate.validate_combined(results)
                if combined.passed:
                    st.success(f"✅ Combined portfolio PASSES quality gate")
                else:
                    st.warning(f"⚠️ Combined portfolio FAILS quality gate ({combined.passed_count}/7)")

        except Exception as e:
            st.error(f"Backtest failed: {e}")
