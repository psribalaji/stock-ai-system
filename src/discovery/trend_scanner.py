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
    # Common English words
    "IT", "AT", "BE", "DO", "GO", "IF", "IN", "IS", "ME", "MY", "NO", "OF",
    "ON", "OR", "SO", "TO", "UP", "US", "WE", "AN", "AS",
    "ARE", "FOR", "THE", "CAN", "BIG", "ALL", "NEW", "NOW", "WHO", "HOW",
    "WHY", "GET", "SET", "GOOD", "REAL", "COST", "MOVE", "HIGH", "FULL",
    "OPEN", "JUST", "LAST", "NEXT", "ALSO", "THEN", "WELL", "BACK", "INTO",
    "OVER", "FROM", "WITH", "THIS", "THAT", "THEY", "WHEN", "WILL", "MORE",
    "BEEN", "HAVE", "SAID", "SAYS", "AFTER", "ABOUT", "THEIR", "WHICH",
    "WOULD", "COULD", "SHOULD", "THERE", "THESE", "THOSE",
    # Finance / trading terms
    "USD", "ETF", "CEO", "CFO", "CTO", "COO", "IPO", "FDA", "SEC", "GDP",
    "IMO", "LOL", "EPS", "YOY", "QOQ", "ATH", "FUD", "YOLO", "FOMO",
    "HODL", "BTFD", "PUTS", "CALL", "CALLS", "BULL", "BEAR", "LONG", "SHORT",
    "BUY", "SELL", "HOLD", "PLAY", "MOON", "DUMP", "PUMP", "LOSS", "GAIN",
    "CASH", "DEBT", "RISK", "SAFE", "RATE", "BOND", "FUND", "BANK", "LOAN",
    "RATE", "FOMC", "FED", "BOJ", "ECB", "IMF", "REPO", "LIBOR", "SOFR",
    # Geopolitical / news organisations
    "NATO", "IRAN", "IRAQ", "ISIS", "ISIL", "OPEC", "SWIFT", "BRICS",
    "EU", "UK", "UN", "US", "UAE", "WHO", "WTO", "IMF", "IAEA",
    "AI", "ML", "IT", "HR", "PR", "IR",
    "INC", "LLC", "LTD", "CORP", "PLC", "ETF", "REIT",
    # Common news words that appear in caps
    "BREAKING", "UPDATE", "REPORT", "ALERT", "NEWS", "LIVE", "WATCH",
    "SAYS", "TOLD", "WARN", "PLAN", "DEAL", "TALK", "MEET", "CALL",
    # Press release / wire boilerplate
    "FORM", "GLOBE", "WIRE", "RELEASE", "PRESS", "MEDIA", "GROUP", "PLC",
    "CORP", "HOLD", "GLOBAL", "TRUST", "FUND", "REIT", "NYSE", "NASDAQ",
    "CNBC", "MSNBC", "WSJ", "FT", "EUR", "GBP", "JPY", "CNY",
    "NYC", "CEO", "DOJ", "FTC", "CFTC", "FINRA", "PCAOB",
    "LNG", "EPT", "RI", "TV", "HRX", "ET", "HDFC",
    # File formats / tech abbreviations common on Reddit
    "CSV", "API", "SQL", "JSON", "XML", "PDF", "SaaS", "AWS", "GCP", "SAAS",
    "IPO", "EPS", "ROI", "ROE", "DCF", "TTM", "YTD", "QTD", "MOM",
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
        reddit_results = self._scan_apewisdom()

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

            # Fetch general, merger (M&A deals = ticker-rich) and forex categories
            all_articles: list[dict] = []
            for category in ("general", "merger", "forex"):
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
                related  = article.get("related", "") or ""
                text     = f"{headline} {summary}"

                # Strategy 1: explicit $TICKER mentions (rare in news but high confidence)
                dollar_tickers = _TICKER_PATTERN.findall(text)

                # Strategy 2: tickers in the 'related' field (Finnhub populates this for
                # company-specific articles, e.g. "AAPL,MSFT")
                related_tickers = [
                    t.strip().upper() for t in related.split(",")
                    if t.strip() and 1 < len(t.strip()) <= 5
                ]

                # Strategy 3: bare uppercase words in headline (e.g. "NVDA surges")
                # Apply same blocklist as Reddit to filter noise
                bare_tickers = [
                    t for t in _REDDIT_TICKER_PAT.findall(headline)
                    if t not in _REDDIT_BLOCKLIST and 2 <= len(t) <= 5
                ]

                all_found = set(dollar_tickers + related_tickers + bare_tickers)
                for ticker in all_found:
                    if ticker in existing:
                        continue
                    ticker_mentions[ticker] += 1
                    ticker_sentiments[ticker].append(0.0)  # neutral default

            if not ticker_mentions:
                return []

            # Spike = mention count in the window vs the minimum threshold.
            # Without a true 30d history, we use total article count as a normaliser:
            # a ticker mentioned in >1% of articles is notable; spike = count / baseline
            # where baseline = 1 (i.e. being mentioned at all is the baseline).
            # Spike threshold of 3.0 means mentioned 3+ times across all articles.
            total_articles = max(len(recent), 1)

            results: list[TrendingTicker] = []
            for ticker, count in ticker_mentions.items():
                # Normalise by article count so a spike of 3.0 means the ticker
                # appeared in ≥3% of articles (3 mentions per 100 articles baseline)
                baseline = max(total_articles / 100, 1.0)
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

    def _scan_apewisdom(self) -> list[TrendingTicker]:
        """
        Fetch trending stock tickers from ApeWisdom (https://apewisdom.io).
        No API key required. Aggregates mentions from WSB, investing, stocks,
        and stockmarket subreddits. Returns [] gracefully on any failure.

        Spike ratio = mentions_now / max(1, mentions_24h_ago) — a real measured
        ratio rather than an estimate, since ApeWisdom provides both values.

        Returns:
            List of TrendingTicker from Reddit/ApeWisdom sources.
        """
        try:
            import httpx

            spike_factor = getattr(
                getattr(self.config, "discovery", None), "mention_spike_factor", 3.0
            )
            existing = set(self.config.assets.all_tradeable + self.config.assets.watchlist)
            now      = datetime.now(timezone.utc)

            # ApeWisdom paginates at 25 results per page; fetch pages 1–4 (top 100)
            rows: list[dict] = []
            with httpx.Client(timeout=10) as client:
                for page in range(1, 5):
                    try:
                        resp = client.get(
                            f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        rows.extend(data.get("results", []))
                        if not data.get("next_page"):
                            break
                    except Exception as e:
                        logger.warning(f"[TrendScanner] ApeWisdom page {page} failed: {e}")
                        break

            if not rows:
                logger.debug("[TrendScanner] ApeWisdom returned no results")
                return []

            results: list[TrendingTicker] = []

            for row in rows:
                ticker = (row.get("ticker") or "").strip().upper()
                if not ticker or len(ticker) > 5:
                    continue
                if ticker in existing:
                    continue
                if ticker in _REDDIT_BLOCKLIST:
                    continue

                # Use `or 0` to handle explicit null values from the API
                mentions_now  = int(row.get("mentions") or 0)
                mentions_prev = int(row.get("mentions_24h_ago") or 0)
                upvotes       = int(row.get("upvotes") or 0)

                # Real spike: current 24h mentions vs prior 24h mentions
                spike = mentions_now / max(1, mentions_prev)
                if spike < spike_factor:
                    continue

                # Sentiment: normalise upvote count relative to mentions
                # (upvotes / max(mentions,1) capped at 1.0, then scaled to [-1, 1])
                raw_sentiment = min(upvotes / max(mentions_now, 1), 1.0)
                avg_sentiment = round((raw_sentiment * 2) - 1.0, 3)  # [0,1] → [-1,1]

                # ApeWisdom provides the company name directly — use it to avoid
                # a Polygon call per ticker here (screener fetches details anyway)
                company_name = row.get("name") or "Unknown"
                sector       = "Unknown"
                price        = 0.0
                market_cap   = 0.0

                results.append(TrendingTicker(
                    ticker        = ticker,
                    company_name  = company_name,
                    sector        = sector,
                    mention_count = mentions_now,
                    mention_spike = round(spike, 2),
                    avg_sentiment = avg_sentiment,
                    sources       = ["reddit"],
                    first_seen    = now,
                    price         = price,
                    market_cap    = market_cap,
                ))

            logger.info(f"[TrendScanner] ApeWisdom scan: {len(results)} trending tickers found")
            return results

        except Exception as e:
            logger.warning(f"[TrendScanner] _scan_apewisdom failed: {e}")
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
