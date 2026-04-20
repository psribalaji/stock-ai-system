"""
dashboard/pages/discovery.py — Dynamic Universe Discovery page.
Shows trending tickers, allows approval/ignore, displays approved universe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_discovery_page() -> None:
    """
    Render the Dynamic Universe Discovery dashboard page.
    Displays trending candidates, approval controls, approved universe,
    and scan trigger controls.
    Does NOT call st.set_page_config().
    """
    st.header("Dynamic Universe Discovery")

    from src.discovery.universe_manager import UniverseManager, STATUS_CANDIDATE, STATUS_APPROVED, STATUS_EXPIRED

    manager = UniverseManager()
    stats   = manager.get_stats()

    # ── 1. HEADER METRICS ROW ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates (All Time)", stats["total"])
    m2.metric("Currently Watching",          stats["candidate"])
    m3.metric("Approved Tickers",            stats["approved"])
    m4.metric("Expired",                     stats["expired"])

    st.divider()

    # ── 2. TRENDING NOW ───────────────────────────────────────────────────────
    st.subheader("Trending Now")

    candidates_df = manager.get_watchlist(status_filter=STATUS_CANDIDATE)

    if candidates_df.empty:
        st.info("No candidate tickers found. Run a scan to discover trending stocks.")
    else:
        # Sort by mention_spike descending
        if "mention_spike" in candidates_df.columns:
            candidates_df = candidates_df.sort_values("mention_spike", ascending=False)

        # ── Compute current signal for each candidate ─────────────────────────
        @st.cache_data(ttl=300, show_spinner=False)
        def _get_signal(ticker: str) -> str:
            """Run signal detector on OHLCV (local Parquet → Alpaca fallback). Returns BUY/SELL/HOLD/NO DATA."""
            try:
                from src.ingestion.storage import ParquetStore
                from src.ingestion.alpaca_client import AlpacaClient
                from src.signals.signal_detector import SignalDetector
                from src.config import get_config
                store = ParquetStore(get_config().data.storage_path)
                df    = store.load_ohlcv(ticker)
                if df.empty or len(df) < 50:
                    # Discovery candidates won't be in local Parquet — fetch live
                    df = AlpacaClient().get_recent_bars(ticker, days=90)
                if df.empty or len(df) < 20:
                    return "NO DATA"
                detector = SignalDetector()
                signals  = detector.detect_actionable(ticker, df)
                if not signals:
                    return "HOLD"
                directions = [s.direction for s in signals if s.direction != "HOLD"]
                if not directions:
                    return "HOLD"
                return "BUY" if "BUY" in directions else directions[0]
            except Exception:
                return "—"

        signal_map = {row["ticker"]: _get_signal(row["ticker"])
                      for _, row in candidates_df.iterrows()}

        _SIGNAL_ICON = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD", "NO DATA": "⬛ NO DATA", "—": "—"}

        # Display table with action buttons
        display_cols = [
            "ticker", "company_name", "sector", "mention_spike",
            "avg_sentiment", "sources", "latest_price", "market_cap", "added_at",
        ]
        show_cols = [c for c in display_cols if c in candidates_df.columns]

        rename_map = {
            "ticker":        "Ticker",
            "company_name":  "Company",
            "sector":        "Sector",
            "mention_spike": "Mention Spike",
            "avg_sentiment": "Sentiment",
            "sources":       "Sources",
            "latest_price":  "Price",
            "market_cap":    "Market Cap",
            "added_at":      "Added",
        }
        display_df = candidates_df[show_cols].rename(columns=rename_map)

        # Inject signal column right after Ticker
        display_df.insert(1, "Signal", display_df["Ticker"].map(
            lambda t: _SIGNAL_ICON.get(signal_map.get(t, "—"), "—")
        ))

        # Format numeric columns
        if "Mention Spike" in display_df.columns:
            display_df["Mention Spike"] = display_df["Mention Spike"].apply(
                lambda v: f"{v:.1f}x" if pd.notna(v) else "—"
            )
        if "Sentiment" in display_df.columns:
            display_df["Sentiment"] = display_df["Sentiment"].apply(
                lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
            )
        if "Price" in display_df.columns:
            display_df["Price"] = display_df["Price"].apply(
                lambda v: f"${v:,.2f}" if pd.notna(v) and v > 0 else "—"
            )
        if "Market Cap" in display_df.columns:
            display_df["Market Cap"] = display_df["Market Cap"].apply(
                lambda v: f"${v / 1e9:.1f}B" if pd.notna(v) and v >= 1e9
                else (f"${v / 1e6:.0f}M" if pd.notna(v) and v > 0 else "—")
            )
        if "Added" in display_df.columns:
            display_df["Added"] = pd.to_datetime(display_df["Added"]).dt.strftime("%Y-%m-%d %H:%M")

        st.caption("🟢 BUY / 🔴 SELL = signal detected on local data  |  ⚪ HOLD = no pattern  |  ⬛ NO DATA = fetch OHLCV first")
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

        # Approve / Ignore buttons per ticker
        st.caption("Approve or ignore each candidate:")
        for _, row in candidates_df.iterrows():
            ticker = row["ticker"]
            sig    = signal_map.get(ticker, "—")
            col_ticker, col_approve, col_ignore = st.columns([2, 1, 1])
            with col_ticker:
                spike    = row.get("mention_spike", 0)
                sig_icon = _SIGNAL_ICON.get(sig, "—")
                st.write(f"**{ticker}** {sig_icon} — {row.get('company_name', '')} ({spike:.1f}x spike)")
            with col_approve:
                if st.button("Approve", key=f"approve_{ticker}"):
                    manager.approve(ticker)
                    st.toast(f"{ticker} approved for trading universe!", icon="✅")
                    st.rerun()
            with col_ignore:
                if st.button("Ignore", key=f"ignore_{ticker}"):
                    manager.ignore(ticker)
                    st.toast(f"{ticker} ignored.", icon="✗")
                    st.rerun()

    st.divider()

    # ── 3. APPROVED UNIVERSE ──────────────────────────────────────────────────
    st.subheader("Approved Universe")

    approved_df = manager.get_watchlist(status_filter=STATUS_APPROVED)

    if approved_df.empty:
        st.info("No tickers approved yet.")
    else:
        disp_cols = ["ticker", "company_name", "sector", "approved_at", "mention_spike", "market_cap"]
        show_cols = [c for c in disp_cols if c in approved_df.columns]

        approved_display = approved_df[show_cols].rename(columns={
            "ticker":        "Ticker",
            "company_name":  "Company",
            "sector":        "Sector",
            "approved_at":   "Approved At",
            "mention_spike": "Mention Spike",
            "market_cap":    "Market Cap",
        })

        if "Approved At" in approved_display.columns:
            approved_display["Approved At"] = pd.to_datetime(
                approved_display["Approved At"]
            ).dt.strftime("%Y-%m-%d %H:%M")
        if "Mention Spike" in approved_display.columns:
            approved_display["Mention Spike"] = approved_display["Mention Spike"].apply(
                lambda v: f"{v:.1f}x" if pd.notna(v) else "—"
            )
        if "Market Cap" in approved_display.columns:
            approved_display["Market Cap"] = approved_display["Market Cap"].apply(
                lambda v: f"${v / 1e9:.1f}B" if pd.notna(v) and v >= 1e9
                else (f"${v / 1e6:.0f}M" if pd.notna(v) and v > 0 else "—")
            )

        st.dataframe(approved_display.reset_index(drop=True), use_container_width=True)

        # Remove (set to IGNORED) button per approved ticker
        st.caption("Remove ticker from approved universe:")
        for _, row in approved_df.iterrows():
            ticker = row["ticker"]
            col_t, col_remove = st.columns([3, 1])
            with col_t:
                st.write(f"**{ticker}** — {row.get('company_name', '')}")
            with col_remove:
                if st.button("Remove", key=f"remove_{ticker}"):
                    manager.ignore(ticker)
                    st.toast(f"{ticker} removed from approved universe.")
                    st.rerun()

    st.divider()

    # ── 4. RECENTLY EXPIRED ───────────────────────────────────────────────────
    with st.expander("Recently Expired Candidates"):
        expired_df = manager.get_watchlist(status_filter=STATUS_EXPIRED)

        if expired_df.empty:
            st.caption("No expired candidates.")
        else:
            # Show last 10
            exp_show = expired_df.sort_values("added_at", ascending=False).head(10)
            disp_cols = ["ticker", "company_name", "added_at", "mention_spike"]
            show_cols = [c for c in disp_cols if c in exp_show.columns]
            exp_display = exp_show[show_cols].rename(columns={
                "ticker":        "Ticker",
                "company_name":  "Company",
                "added_at":      "Added At",
                "mention_spike": "Mention Spike",
            })
            if "Added At" in exp_display.columns:
                exp_display["Added At"] = pd.to_datetime(
                    exp_display["Added At"]
                ).dt.strftime("%Y-%m-%d")
            if "Mention Spike" in exp_display.columns:
                exp_display["Mention Spike"] = exp_display["Mention Spike"].apply(
                    lambda v: f"{v:.1f}x" if pd.notna(v) else "—"
                )
            st.dataframe(exp_display.reset_index(drop=True), use_container_width=True)

    # ── 5. SCAN CONTROLS ──────────────────────────────────────────────────────
    st.subheader("Scan Controls")
    st.caption("Manually trigger a discovery scan to find new trending tickers.")

    if st.button("Run Scan Now", type="primary"):
        from src.discovery.trend_scanner import TrendScanner
        from src.discovery.stock_screener import StockScreener

        scanner  = TrendScanner()
        screener = StockScreener()
        manager2 = UniverseManager()

        try:
            with st.spinner("Scanning news..."):
                news_results   = scanner._scan_news_velocity()

            with st.spinner("Scanning Reddit (ApeWisdom)..."):
                reddit_results = scanner._scan_apewisdom()

            # Deduplicate and merge (same logic as scanner.scan())
            with st.spinner("Screening candidates..."):
                candidates = scanner.scan()
                screened   = screener.screen(candidates)
                passed     = [s for s in screened if s.passed]
                added      = manager2.add_candidates(passed)

            # ── Per-source breakdown ───────────────────────────────
            st.markdown("**Scan Results:**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("News mentions",    len(news_results),
                      help="Tickers with a spike in Finnhub news mentions (48h vs 30d avg)")
            c2.metric("Reddit mentions",  len(reddit_results),
                      help="Tickers trending on WSB / investing / stocks subreddits")
            c3.metric("Passed screening", len(passed),
                      help="Met all 4 criteria: market cap, volume, price, Alpaca-tradeable")
            c4.metric("New candidates",   added,
                      help="Not already in watchlist — added for your review")

            if len(news_results) == 0:
                st.warning("News scan: 0 results — Finnhub returned no articles with ticker spikes "
                           "above the 3x threshold, or API returned empty.")
            if len(reddit_results) == 0:
                st.warning("ApeWisdom scan: 0 results — API may be unreachable or no tickers met the spike threshold.")

            if candidates and not passed:
                # Show why each candidate failed screening
                st.markdown("**Why candidates failed screening:**")
                failed = [s for s in screened if not s.passed]
                for s in failed[:10]:
                    st.caption(f"❌ **{s.ticker}**: {', '.join(s.fail_reasons)}")

            if added > 0:
                st.success(f"{added} new candidate(s) added — see Trending Now above.")
            elif len(passed) > 0:
                st.info("Candidates found but already in watchlist — no duplicates added.")
            else:
                st.info("No new candidates this scan.")

        except Exception as e:
            st.error(f"Scan failed: {e}")

        st.rerun()
