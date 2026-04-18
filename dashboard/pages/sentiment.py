"""
dashboard/pages/sentiment.py — Sentiment Monitor page.
Shows LLM sentiment scores, news headlines, and sentiment impact on confidence.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def render_sentiment_page() -> None:
    st.header("Sentiment Monitor")
    from src.config import get_config
    from src.ingestion.storage import ParquetStore

    cfg = get_config()
    store = ParquetStore(cfg.data.storage_path)

    st.subheader("Sentiment Impact on Confidence")
    st.markdown("""
    Sentiment from news headlines is scored by Claude (-1.0 to +1.0) and mapped to a multiplier:
    | Sentiment Score | Multiplier | Effect |
    |---|---|---|
    | +1.0 (very bullish) | 1.10× | Boosts confidence 10% |
    | 0.0 (neutral) | 1.00× | No effect |
    | -1.0 (very bearish) | 0.90× | Reduces confidence 10% |
    """)

    # ── Per-ticker sentiment
    st.subheader("Ticker Sentiment")
    tickers = cfg.assets.all_tradeable
    selected = st.multiselect("Select tickers", tickers, default=tickers[:5])

    for ticker in selected:
        news_df = store.load_news(ticker, days_back=7)
        if news_df.empty:
            continue

        with st.expander(f"📰 {ticker} — {len(news_df)} articles", expanded=False):
            if "sentiment" in news_df.columns:
                avg = news_df["sentiment"].mean()
                pos = (news_df["sentiment"] > 0.2).sum()
                neg = (news_df["sentiment"] < -0.2).sum()

                c1, c2, c3 = st.columns(3)
                color = "🟢" if avg > 0.1 else "🔴" if avg < -0.1 else "⚪"
                c1.metric("Avg Sentiment", f"{color} {avg:.3f}")
                c2.metric("Positive", pos)
                c3.metric("Negative", neg)

            if "headline" in news_df.columns:
                display_cols = [c for c in ["datetime", "headline", "sentiment", "source"] if c in news_df.columns]
                st.dataframe(news_df[display_cols].head(10), use_container_width=True, hide_index=True)

    if not selected:
        st.info("Select tickers above to view sentiment data")

    # ── Sentiment distribution
    st.subheader("Overall Sentiment Distribution")
    all_news = []
    for ticker in tickers[:10]:
        df = store.load_news(ticker, days_back=7)
        if not df.empty and "sentiment" in df.columns:
            df["ticker"] = ticker
            all_news.append(df)

    if all_news:
        combined = pd.concat(all_news, ignore_index=True)
        st.bar_chart(combined.groupby("ticker")["sentiment"].mean())
    else:
        st.info("No news data available. Run the pipeline to fetch news.")
