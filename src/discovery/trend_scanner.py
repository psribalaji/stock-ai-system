"""
discovery/trend_scanner.py — Scans news and Reddit for trending tickers
not already in the configured asset universe.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional

from loguru import logger

from src.config import get_config


# ── Blocklist for Reddit — common English words that look like tickers ──────
_REDDIT_BLOCKLIST: set[str] = {
    "IT", "AT", "BE", "DO", "GO", "IF", "IN", "IS", "ME", "MY", "NO", "OF",
    "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "ARE", "FOR", "THE", "CAN", "BIG", "ALL", "NEW", "NOW", "WHO", "HOW",
    "WHY", "GET", "SET", "GOOD", "REAL", "COST", "MOVE", "HIGH", "FULL",
    "OPEN", "JUST", "LAST", "NEXT", "ALSO", "THEN", "WELL", "BACK", "INTO",
    "USD", "ETF", "CEO", "CFO", "IPO", "FDA", "SEC", "GDP", "IMO", "LOL",
    "EPS", "YOY", "QOQ", "ATH", "FUD", "YOLO", "FOMO", "HODL", "BTFD",
    "PUTS", "CALL", "CALLS", "BULL", "BEAR", "LONG", "SHORT",
    "BUY", "SELL", "HOLD", "PLAY", "MOON", "DUMP", "PUMP", "LOSS", "GAIN",
    "CASH", "DEBT", "RISK", "SAFE",
}

_TICKER_PATTERN     = re.compile(r'\$([A-Z]{1,5})\b')
_REDDIT_TICKER_PAT  = re.compile(r'\b([A-Z]{2,5})\b')


@dataclass
class TrendingTicker:
    """A ticker surfaced by the trend scanner."""
    ticker:        str
    company_name:  str
    sector:        str
    mention_count: int
    mention_spike: float         # ratio vs 30d average (e.g. 5.2 = 5.2x normal)
    avg_sentiment: float         # -1.0 to +1.0
    sources:       list[str]
    first_seen:    datetime
    price:         float         # latest price from Polygon
    market_cap:    float         # from Polygon


class TrendScanner:
    """
    Scans news (Finnhub) and Reddit for trending tickers that are NOT
    already in the configured asset universe.
    """

    def __init__(self) -> None:
        self.config = get_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self) -> list[TrendingTicker]:
        """
        Calls all sub-scanners, deduplicates by ticker, ranks by mention_spike,
        filters out tickers already in the universe, returns top N candidates.

        Returns:
            List of TrendingTicker sorted by mention_spike descending.
        """
        existing = set(
            self.config.assets.all_tradeable + self.config.assets.watchlist
        )

        news_results   = self._scan_news_velocity()
        reddit_results = self._scan_reddit()

        # Merge: deduplicate by ticker, aggregate across sources
        merged: dict[str, TrendingTicker] = {}

        for tt in news_results + reddit_results:
            if tt.ticker in existing:
                continue
            if tt.ticker in merged:
                existing_tt = merged[tt.ticker]
                # Combine sources, sum mentions, average sentiment, take max spike
                combined_sources = list(set(existing_tt.sources + tt.sources))
                merged[tt.ticker] = TrendingTicker(
                    ticker        = tt.ticker,
                    company_name  = existing_tt.company_name if existing_tt.company_name != "Unknown" else tt.company_name,
                    sector        = existing_tt.sector if existing_tt.sector != "Unknown" else tt.sector,
                    mention_count = existing_tt.mention_count + tt.mention_count,
                    mention_spike = max(existing_tt.mention_spike, tt.mention_spike),
                    avg_sentiment = (existing_tt.avg_sentiment + tt.avg_sentiment) / 2,
                    sources       = combined_sources,
                    first_seen    = min(existing_tt.first_seen, tt.first_seen),
                    price         = tt.price if tt.price > 0 else existing_tt.price,
                    market_cap    = tt.market_cap if tt.market_cap > 0 else existing_tt.market_cap,
                )
            else:
                merged[tt.ticker] = tt

        # Sort by mention_spike descending, return top N
        max_candidates = getattr(getattr(self.config, "discovery", None), "max_candidates", 20)
        ranked = sorted(merged.values(), key=lambda x: x.mention_spike, reverse=True)
        return ranked[:max_candidates]

    # ── Sub-scanners ──────────────────────────────────────────────────────────

    def _scan_news_velocity(self) -> list[TrendingTicker]:
        """
        Fetch market news from Finnhub for the last 48h.
        Parse tickers mentioned in each article.
        Compute spike: 48h count vs estimated 30d average.
        Return tickers with spike > config.discovery.mention_spike_factor.

        Returns:
            List of TrendingTicker from news sources.
        """
        try:
            from src.secrets import Secrets
            import finnhub

            api_key = Secrets.finnhub_api_key()
            client  = finnhub.Client(api_key=api_key)

            lookback_hours = getattr(
                getattr(self.config, "discovery", None), "news_lookback_hours", 48
            )
            spike_factor = getattr(
                getattr(self.config, "discovery", None), "mention_spike_factor", 3.0
            )

            now       = datetime.now(timezone.utc)
            cutoff_ts = int((now - timedelta(hours=lookback_hours)).timestamp())

            # Fetch general and forex categories
            all_articles: list[dict] = []
            for category in ("general", "forex"):
                try:
                    articles = client.general_news(category, min_id=0)
                    if articles:
                        all_articles.extend(articles)
                except Exception as e:
                    logger.warning(f"[TrendScanner] Failed to fetch '{category}' news: {e}")

            # Filter to lookback window
            recent = [
                a for a in all_articles
                if a.get("datetime", 0) >= cutoff_ts
            ]

            if not recent:
                logger.debug("[TrendScanner] No recent news articles found")
                return []

            # Extract ticker mentions
            ticker_mentions: dict[str, int] = defaultdict(int)
            ticker_sentiments: dict[str, list[float]] = defaultdict(list)

            existing = set(self.config.assets.all_tradeable + self.config.assets.watchlist)

            for article in recent:
                headline = article.get("headline", "")
                summary  = article.get("summary", "")
                text     = f"{headline} {summary}"

                tickers_found = _TICKER_PATTERN.findall(text)
                for ticker in tickers_found:
                    if ticker in existing:
                        continue
                    if len(ticker) < 1 or len(ticker) > 5:
                        continue
                    ticker_mentions[ticker] += 1
                    ticker_sentiments[ticker].append(0.0)  # neutral default

            if not ticker_mentions:
                return []

            # Estimate 30d average from the current window (scale up proportionally)
            # 48h window → scale to 30 days: 30*24/48 = 15x
            scale_factor = (30 * 24) / lookback_hours

            results: list[TrendingTicker] = []
            for ticker, count in ticker_mentions.items():
                estimated_30d_avg = count / scale_factor  # what 48h implies over 30d
                # spike = current 48h rate vs the rolling average
                # If estimated_30d_avg is very small, spike can be large
                spike = count / max(estimated_30d_avg / 30, 1.0)

                if spike < spike_factor:
                    continue

                avg_sentiment = (
                    sum(ticker_sentiments[ticker]) / len(ticker_sentiments[ticker])
                    if ticker_sentiments[ticker] else 0.0
                )

                company_name, sector = self._get_sector_and_name(ticker)
                price, market_cap    = self._get_price_and_market_cap(ticker)

                results.append(TrendingTicker(
                    ticker        = ticker,
                    company_name  = company_name,
                    sector        = sector,
                    mention_count = count,
                    mention_spike = round(spike, 2),
                    avg_sentiment = round(avg_sentiment, 3),
                    sources       = ["news"],
                    first_seen    = now,
                    price         = price,
                    market_cap    = market_cap,
                ))

            logger.info(f"[TrendScanner] News scan: {len(results)} trending tickers found")
            return results

        except Exception as e:
            logger.warning(f"[TrendScanner] _scan_news_velocity failed: {e}")
            return []

    def _scan_reddit(self) -> list[TrendingTicker]:
        """
        Scan configured subreddits for trending tickers.
        Uses PRAW library — returns [] gracefully if not installed or credentials missing.

        Returns:
            List of TrendingTicker from Reddit sources.
        """
        try:
            import praw  # type: ignore
        except ImportError:
            logger.warning("[TrendScanner] PRAW not installed — Reddit scan skipped. "
                           "Install with: pip install praw>=7.7.0")
            return []

        try:
            import os

            client_id     = os.environ.get("REDDIT_CLIENT_ID")
            client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
            user_agent    = os.environ.get("REDDIT_USER_AGENT", "StockAI/1.0")

            if not client_id or not client_secret:
                logger.warning("[TrendScanner] REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET "
                               "not set — Reddit scan skipped")
                return []

            reddit = praw.Reddit(
                client_id     = client_id,
                client_secret = client_secret,
                user_agent    = user_agent,
            )

            spike_factor  = getattr(
                getattr(self.config, "discovery", None), "mention_spike_factor", 3.0
            )
            subreddits_cfg = getattr(
                getattr(self.config, "discovery", None),
                "reddit_subreddits",
                ["wallstreetbets", "investing", "stocks", "stockmarket"],
            )

            existing = set(self.config.assets.all_tradeable + self.config.assets.watchlist)

            ticker_mentions:  dict[str, int]          = defaultdict(int)
            ticker_sentiments: dict[str, list[float]] = defaultdict(list)

            for subreddit_name in subreddits_cfg:
                try:
                    sub = reddit.subreddit(subreddit_name)
                    posts = list(sub.hot(limit=100)) + list(sub.new(limit=50))

                    for post in posts:
                        text = f"{post.title} {post.selftext if hasattr(post, 'selftext') else ''}"
                        found = _REDDIT_TICKER_PAT.findall(text)

                        for ticker in found:
                            if ticker in _REDDIT_BLOCKLIST:
                                continue
                            if ticker in existing:
                                continue
                            if len(ticker) < 2 or len(ticker) > 5:
                                continue

                            ticker_mentions[ticker] += 1

                            # Sentiment from upvote ratio: maps [0.5, 1.0] -> [0.0, 1.0]
                            upvote_ratio = getattr(post, "upvote_ratio", 0.5)
                            sentiment    = (upvote_ratio - 0.5) * 2
                            ticker_sentiments[ticker].append(sentiment)

                except Exception as e:
                    logger.warning(f"[TrendScanner] Failed to scan r/{subreddit_name}: {e}")

            now = datetime.now(timezone.utc)
            results: list[TrendingTicker] = []

            for ticker, count in ticker_mentions.items():
                if count < 10:
                    continue

                # Estimate spike: assume normal baseline is ~2 mentions per scan
                baseline = 2.0
                spike    = count / baseline

                if spike < spike_factor:
                    continue

                avg_sentiment = (
                    sum(ticker_sentiments[ticker]) / len(ticker_sentiments[ticker])
                    if ticker_sentiments[ticker] else 0.0
                )

                company_name, sector = self._get_sector_and_name(ticker)
                price, market_cap    = self._get_price_and_market_cap(ticker)

                results.append(TrendingTicker(
                    ticker        = ticker,
                    company_name  = company_name,
                    sector        = sector,
                    mention_count = count,
                    mention_spike = round(spike, 2),
                    avg_sentiment = round(avg_sentiment, 3),
                    sources       = ["reddit"],
                    first_seen    = now,
                    price         = price,
                    market_cap    = market_cap,
                ))

            logger.info(f"[TrendScanner] Reddit scan: {len(results)} trending tickers found")
            return results

        except Exception as e:
            logger.warning(f"[TrendScanner] _scan_reddit failed: {e}")
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @lru_cache(maxsize=256)
    def _get_sector_and_name(self, ticker: str) -> tuple[str, str]:
        """
        Fetch sector and company name from Polygon ticker details.
        Cached per ticker. Returns ("Unknown", "Unknown") on failure.

        Args:
            ticker: Stock symbol.

        Returns:
            Tuple of (sector, company_name).
        """
        try:
            from src.ingestion.polygon_client import PolygonClient
            client  = PolygonClient()
            details = client._client.get_ticker_details(ticker)
            sector  = getattr(details, "sic_description", None) or "Unknown"
            name    = getattr(details, "name", None) or "Unknown"
            return (sector, name)
        except Exception as e:
            logger.debug(f"[TrendScanner] Could not fetch details for {ticker}: {e}")
            return ("Unknown", "Unknown")

    def _get_price_and_market_cap(self, ticker: str) -> tuple[float, float]:
        """
        Fetch latest price and market cap from Polygon.

        Args:
            ticker: Stock symbol.

        Returns:
            Tuple of (price, market_cap). Returns (0.0, 0.0) on failure.
        """
        try:
            from src.ingestion.polygon_client import PolygonClient
            client  = PolygonClient()
            details = client._client.get_ticker_details(ticker)
            market_cap = float(getattr(details, "market_cap", 0) or 0)

            # Try to get latest price from daily bars
            from datetime import date
            df = client.fetch_latest_bar(ticker)
            price = float(df["close"].iloc[-1]) if not df.empty else 0.0

            return (price, market_cap)
        except Exception as e:
            logger.debug(f"[TrendScanner] Could not fetch price/market_cap for {ticker}: {e}")
            return (0.0, 0.0)
