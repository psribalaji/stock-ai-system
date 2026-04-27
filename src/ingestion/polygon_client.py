"""
polygon_client.py — Polygon.io REST API wrapper.
Used for: historical OHLCV data, split/dividend adjustments.
DO NOT use Yahoo Finance — Polygon is the source of truth for backtesting.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
import time
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_not_exception_message

from src.secrets import Secrets


class PolygonClient:
    """
    Wrapper around polygon-api-client.
    All data returned is split/dividend adjusted (adjusted=True).
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Secrets.polygon_api_key()
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from polygon import RESTClient
            self._client = RESTClient(api_key=self.api_key)
            logger.info("Polygon.io client initialized")
        except ImportError:
            raise ImportError(
                "polygon-api-client not installed.\n"
                "Run: pip install polygon-api-client"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=15, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def fetch_daily_bars(
        self,
        ticker: str,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a ticker.

        Args:
            ticker:   Stock symbol e.g. "NVDA"
            start:    Start date (inclusive)
            end:      End date (inclusive)
            adjusted: Use split/dividend adjusted prices (always True for backtesting)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap, ticker
        """
        logger.info(f"Fetching {ticker} daily bars {start} → {end}")

        bars = self._client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start.isoformat(),
            to=end.isoformat(),
            adjusted=adjusted,
            sort="asc",
            limit=50000,
        )

        if not bars:
            logger.warning(f"No data returned for {ticker} from Polygon")
            return pd.DataFrame()

        records = []
        for bar in bars:
            records.append({
                "timestamp": pd.Timestamp(bar.timestamp, unit="ms", tz="UTC"),
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
                "vwap":   getattr(bar, "vwap", None),
                "ticker": ticker,
                "source": "polygon",
            })

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} bars for {ticker}")
        return df

    def fetch_multiple_tickers(
        self,
        tickers: list[str],
        start: date,
        end: date,
        delay_seconds: float = 0.25,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch daily bars for multiple tickers.
        Adds delay between requests to respect rate limits.

        Returns:
            Dict mapping ticker → DataFrame
        """
        results = {}
        for i, ticker in enumerate(tickers):
            try:
                df = self.fetch_daily_bars(ticker, start, end)
                results[ticker] = df
                if i < len(tickers) - 1:
                    time.sleep(delay_seconds)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results

    def fetch_latest_bar(self, ticker: str) -> pd.DataFrame:
        """Fetch the most recent daily bar for a ticker."""
        today = date.today()
        start = today - timedelta(days=5)  # Go back 5 days to handle weekends
        df = self.fetch_daily_bars(ticker, start, today)
        if df.empty:
            return df
        return df.tail(1).reset_index(drop=True)

    def validate_connection(self) -> bool:
        """Ping Polygon to check API key is valid."""
        try:
            self.fetch_daily_bars("AAPL", date.today() - timedelta(days=5), date.today())
            logger.info("Polygon.io connection: OK")
            return True
        except Exception as e:
            logger.error(f"Polygon.io connection failed: {e}")
            return False
