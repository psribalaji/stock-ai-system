"""
discovery/stock_screener.py — Validates trending tickers against
fundamental and tradability criteria.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from src.config import get_config
from src.discovery.trend_scanner import TrendingTicker
from src.ingestion.polygon_client import PolygonClient
from src.ingestion.alpaca_client import AlpacaClient


@dataclass
class ScreenedTicker:
    """Result of screening a TrendingTicker against fundamental criteria."""
    ticker:        str
    company_name:  str
    sector:        str
    market_cap:    float
    avg_volume_30d: float
    latest_price:  float
    passed:        bool
    fail_reasons:  list[str]
    trending_data: TrendingTicker


class StockScreener:
    """
    Validates trending tickers against market cap, volume, price,
    and Alpaca tradability criteria.

    All 4 checks must pass for a ticker to be considered eligible.
    Failed checks are captured in fail_reasons — no exceptions raised.
    """

    def __init__(
        self,
        polygon_client: Optional[PolygonClient] = None,
        alpaca_client:  Optional[AlpacaClient]  = None,
    ) -> None:
        self.config  = get_config()
        self._polygon = polygon_client or PolygonClient()
        self._alpaca  = alpaca_client  or AlpacaClient()

        # Config thresholds with safe fallback defaults
        disc = getattr(self.config, "discovery", None)
        self._min_market_cap  = getattr(disc, "min_market_cap",  500_000_000) if disc else 500_000_000
        self._min_avg_volume  = getattr(disc, "min_avg_volume",  500_000)     if disc else 500_000
        self._min_price       = getattr(disc, "min_price",       5.0)         if disc else 5.0

        # Per-day metric cache: market cap / volume / price barely change intraday,
        # so we fetch each ticker from Polygon at most ONCE per day. This is the
        # main defence against Polygon free-tier 429s — the scheduler re-scans the
        # same trending tickers every 30 min, and without this every scan re-hit
        # the API. Keyed by ticker; file rotates by date.
        storage = getattr(self.config.data, "storage_path", "./data")
        self._cache_path = Path(storage) / "discovery" / f"screen_cache_{date.today().isoformat()}.json"
        self._metric_cache: dict[str, dict] = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        try:
            if self._cache_path.exists():
                return json.loads(self._cache_path.read_text())
        except Exception as exc:
            logger.debug(f"[StockScreener] Could not load screen cache: {exc}")
        return {}

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._metric_cache))
        except Exception as exc:
            logger.debug(f"[StockScreener] Could not save screen cache: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def screen(self, candidates: list[TrendingTicker]) -> list[ScreenedTicker]:
        """
        Run all 4 checks on each candidate.

        Args:
            candidates: List of TrendingTicker from TrendScanner.

        Returns:
            All results (both passed and failed).
        """
        results: list[ScreenedTicker] = []
        for tt in candidates:
            result = self._screen_one(tt)
            results.append(result)
        return results

    # ── Private screening methods ─────────────────────────────────────────────

    def _screen_one(self, tt: TrendingTicker) -> ScreenedTicker:
        """Screen a single TrendingTicker against all 4 criteria."""
        fail_reasons: list[str] = []

        # Fundamentals/price come from cache when available (0 Polygon calls),
        # otherwise 2 Polygon calls (details + 30d bars) with free-tier pacing.
        m = self._get_metrics(tt.ticker)
        market_cap   = m["market_cap"]
        avg_volume   = m["avg_volume"]
        latest_price = m["latest_price"]
        sector       = m["sector"]
        company_name = m["company_name"]

        # Check 1: Market cap
        if market_cap < self._min_market_cap:
            fail_reasons.append(
                f"Market cap ${market_cap:,.0f} < minimum ${self._min_market_cap:,.0f}"
            )
        # Check 2: Volume
        if avg_volume < self._min_avg_volume:
            fail_reasons.append(
                f"Avg volume {avg_volume:,.0f} < minimum {self._min_avg_volume:,.0f}"
            )
        # Check 3: Price
        if latest_price < self._min_price:
            fail_reasons.append(
                f"Price ${latest_price:.2f} < minimum ${self._min_price:.2f}"
            )

        # Check 4: Alpaca tradability (no Polygon call)
        tradeable, trade_reason = self._check_tradeable_on_alpaca(tt.ticker)
        if not tradeable:
            fail_reasons.append(trade_reason)

        passed = len(fail_reasons) == 0

        return ScreenedTicker(
            ticker         = tt.ticker,
            company_name   = company_name if company_name != "Unknown" else tt.company_name,
            sector         = sector if sector != "Unknown" else tt.sector,
            market_cap     = market_cap,
            avg_volume_30d = avg_volume,
            latest_price   = latest_price,
            passed         = passed,
            fail_reasons   = fail_reasons,
            trending_data  = tt,
        )

    def _get_metrics(self, ticker: str) -> dict:
        """
        Return {market_cap, avg_volume, latest_price, sector, company_name} for a
        ticker, served from the per-day cache when present (no Polygon calls).

        On a cache miss, makes 2 Polygon calls (ticker details + a single 30-day
        daily-bars fetch reused for both volume and last price), paces them for
        the free tier, and caches the result for the rest of the day.
        """
        if ticker in self._metric_cache:
            return self._metric_cache[ticker]

        market_cap, sector, company_name = self._fetch_details(ticker)
        time.sleep(1.0)   # pace between the two Polygon calls
        avg_volume, latest_price = self._fetch_volume_and_price(ticker)

        metrics = {
            "market_cap":   market_cap,
            "avg_volume":   avg_volume,
            "latest_price": latest_price,
            "sector":       sector,
            "company_name": company_name,
        }
        self._metric_cache[ticker] = metrics
        self._save_cache()
        return metrics

    def _fetch_details(self, ticker: str) -> tuple[float, str, str]:
        """Fetch (market_cap, sector, company_name) via one Polygon call."""
        try:
            details      = self._polygon._client.get_ticker_details(ticker)
            market_cap   = float(getattr(details, "market_cap", 0) or 0)
            sector       = getattr(details, "sic_description", None) or "Unknown"
            company_name = getattr(details, "name", None) or "Unknown"
            return (market_cap, sector, company_name)
        except Exception as e:
            logger.warning(f"[StockScreener] Could not fetch details for {ticker}: {e}")
            return (0.0, "Unknown", "Unknown")

    def _fetch_volume_and_price(self, ticker: str) -> tuple[float, float]:
        """
        Fetch (30-day avg volume, latest close) from ONE 30-day daily-bars call.

        Previously volume and price were two separate Polygon calls; combining
        them halves the per-ticker request count against the free-tier limit.
        """
        try:
            end   = date.today()
            start = end - timedelta(days=30)
            df    = self._polygon.fetch_daily_bars(ticker, start, end)
            if df.empty:
                return (0.0, 0.0)
            avg_volume   = float(df["volume"].mean())
            latest_price = float(df["close"].iloc[-1])
            return (avg_volume, latest_price)
        except Exception as e:
            logger.warning(f"[StockScreener] Could not fetch volume/price for {ticker}: {e}")
            return (0.0, 0.0)

    def _check_tradeable_on_alpaca(self, ticker: str) -> tuple[bool, str]:
        """
        Verify the ticker is tradeable on Alpaca (active + tradable).

        Args:
            ticker: Stock symbol.

        Returns:
            (tradeable, reason_string)
        """
        try:
            from alpaca.trading.enums import AssetStatus

            asset = self._alpaca._trading_client.get_asset(ticker)
            if asset.tradable and asset.status == AssetStatus.ACTIVE:
                return (True, "")
            reasons = []
            if not asset.tradable:
                reasons.append("not tradable on Alpaca")
            if asset.status != AssetStatus.ACTIVE:
                reasons.append(f"asset status is '{asset.status}'")
            return (False, "; ".join(reasons))
        except Exception as e:
            logger.warning(f"[StockScreener] Could not verify Alpaca tradability for {ticker}: {e}")
            return (False, "Could not verify Alpaca tradability")
