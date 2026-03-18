"""
news_service.py — Finnhub news ingestion + basic sentiment scoring.
Note: Sentiment scoring here is rule-based keyword matching.
      LLM-based deeper analysis happens in LLMAnalysisService — NOT here.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.secrets import Secrets
from src.ingestion.storage import ParquetStore
from src.config import get_config


# Keyword lists for basic sentiment classification
_POSITIVE_KEYWORDS = [
    "beat", "beats", "exceeds", "record", "upgrade", "outperform",
    "raises guidance", "strong", "growth", "partnership", "contract",
    "buyback", "dividend", "breakout", "expansion", "wins",
]
_NEGATIVE_KEYWORDS = [
    "miss", "misses", "disappoints", "downgrade", "underperform",
    "cuts guidance", "layoffs", "recall", "lawsuit", "breach",
    "investigation", "decline", "loss", "debt", "warning", "concern",
]


class NewsService:
    """
    Fetches and stores news headlines from Finnhub.
    Provides basic rule-based sentiment for quick signal enrichment.
    Full LLM analysis is done downstream in LLMAnalysisService.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        store: Optional[ParquetStore] = None,
    ):
        self.api_key = api_key or Secrets.finnhub_api_key()
        self.config  = get_config()
        self.store   = store or ParquetStore(self.config.data.storage_path)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import finnhub
            self._client = finnhub.Client(api_key=self.api_key)
            logger.info("Finnhub client initialized")
        except ImportError:
            raise ImportError("finnhub-python not installed. Run: pip install finnhub-python")

    # ── FETCH ─────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
    def fetch_company_news(
        self,
        ticker: str,
        days_back: int = 7,
    ) -> pd.DataFrame:
        """
        Fetch recent news headlines for a ticker from Finnhub.
        Results are cached in Parquet to avoid re-fetching.
        """
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days_back)

        raw = self._client.company_news(
            ticker,
            _from=start_dt.strftime("%Y-%m-%d"),
            to=end_dt.strftime("%Y-%m-%d"),
        )

        if not raw:
            logger.debug(f"No news found for {ticker}")
            return pd.DataFrame()

        records = []
        for item in raw:
            records.append({
                "id":        str(item.get("id", "")),
                "ticker":    ticker,
                "headline":  item.get("headline", ""),
                "summary":   item.get("summary", ""),
                "source":    item.get("source", ""),
                "url":       item.get("url", ""),
                "datetime":  pd.Timestamp(item.get("datetime", 0), unit="s", tz="UTC"),
                "sentiment": self._score_sentiment(item.get("headline", "")),
            })

        df = pd.DataFrame(records).sort_values("datetime", ascending=False)
        self.store.save_news(ticker, df)
        logger.debug(f"Fetched {len(df)} articles for {ticker}")
        return df

    def fetch_market_news(self, category: str = "general") -> pd.DataFrame:
        """Fetch broad market news (macro events, Fed, etc.)."""
        raw = self._client.general_news(category, min_id=0)
        if not raw:
            return pd.DataFrame()

        records = [
            {
                "id":        str(item.get("id", "")),
                "ticker":    "MARKET",
                "headline":  item.get("headline", ""),
                "summary":   item.get("summary", ""),
                "source":    item.get("source", ""),
                "url":       item.get("url", ""),
                "datetime":  pd.Timestamp(item.get("datetime", 0), unit="s", tz="UTC"),
                "sentiment": self._score_sentiment(item.get("headline", "")),
            }
            for item in raw
        ]
        return pd.DataFrame(records)

    def fetch_all_tickers(
        self,
        tickers: Optional[list[str]] = None,
        days_back: int = 3,
    ) -> dict[str, pd.DataFrame]:
        """Fetch news for all tickers in the asset universe."""
        tickers = tickers or self.config.assets.all_tradeable
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_company_news(ticker, days_back)
            except Exception as e:
                logger.warning(f"News fetch failed for {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results

    # ── SENTIMENT ────────────────────────────────────────────────

    @staticmethod
    def _score_sentiment(text: str) -> float:
        """
        Basic rule-based sentiment score.
        Returns: +1.0 (very positive) to -1.0 (very negative), 0.0 = neutral.
        This is fast and cheap — used for quick filtering.
        Full LLM analysis happens in LLMAnalysisService.
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text_lower)
        neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text_lower)

        if pos_hits == 0 and neg_hits == 0:
            return 0.0

        total = pos_hits + neg_hits
        score = (pos_hits - neg_hits) / total
        return round(score, 2)

    def get_sentiment_summary(
        self,
        ticker: str,
        days_back: int = 7,
    ) -> dict:
        """
        Get aggregated sentiment for a ticker over the past N days.
        Used by SignalDetector to add news context to signals.
        """
        df = self.store.load_news(ticker, days_back)

        if df.empty:
            return {
                "ticker":         ticker,
                "article_count":  0,
                "avg_sentiment":  0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count":  0,
                "top_headlines":  [],
            }

        return {
            "ticker":         ticker,
            "article_count":  len(df),
            "avg_sentiment":  round(df["sentiment"].mean(), 3),
            "positive_count": int((df["sentiment"] > 0.2).sum()),
            "negative_count": int((df["sentiment"] < -0.2).sum()),
            "neutral_count":  int((df["sentiment"].between(-0.2, 0.2)).sum()),
            "top_headlines":  df["headline"].head(5).tolist(),
        }

    def validate_connection(self) -> bool:
        """Test Finnhub connection."""
        try:
            self._client.company_news(
                "AAPL",
                _from=(datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                to=datetime.now().strftime("%Y-%m-%d"),
            )
            logger.info("Finnhub connection: OK")
            return True
        except Exception as e:
            logger.error(f"Finnhub connection failed: {e}")
            return False
