"""
discovery/stock_screener.py — Validates trending tickers against
fundamental and tradability criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
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
        market_cap    = 0.0
        avg_volume    = 0.0
        latest_price  = 0.0

        # Check 1: Market cap
        market_cap, cap_ok, cap_reason = self._check_market_cap(tt.ticker)
        if not cap_ok:
            fail_reasons.append(cap_reason)

        # Check 2: Volume
        avg_volume, vol_ok, vol_reason = self._check_volume(tt.ticker)
        if not vol_ok:
            fail_reasons.append(vol_reason)

        # Check 3: Price
        latest_price, price_ok, price_reason = self._check_price(tt.ticker)
        if not price_ok:
            fail_reasons.append(price_reason)

        # Check 4: Alpaca tradability
        tradeable, trade_reason = self._check_tradeable_on_alpaca(tt.ticker)
        if not tradeable:
            fail_reasons.append(trade_reason)

        passed = len(fail_reasons) == 0

        return ScreenedTicker(
            ticker         = tt.ticker,
            company_name   = tt.company_name,
            sector         = tt.sector,
            market_cap     = market_cap,
            avg_volume_30d = avg_volume,
            latest_price   = latest_price,
            passed         = passed,
            fail_reasons   = fail_reasons,
            trending_data  = tt,
        )

    def _check_market_cap(self, ticker: str) -> tuple[float, bool, str]:
        """
        Check if market cap meets minimum threshold.

        Args:
            ticker: Stock symbol.

        Returns:
            (market_cap, passed, reason_string)
        """
        try:
            details    = self._polygon._client.get_ticker_details(ticker)
            market_cap = float(getattr(details, "market_cap", 0) or 0)
            if market_cap >= self._min_market_cap:
                return (market_cap, True, "")
            return (
                market_cap,
                False,
                f"Market cap ${market_cap:,.0f} < minimum ${self._min_market_cap:,.0f}",
            )
        except Exception as e:
            logger.warning(f"[StockScreener] Could not fetch market cap for {ticker}: {e}")
            return (0.0, False, "Could not fetch market cap")

    def _check_volume(self, ticker: str) -> tuple[float, bool, str]:
        """
        Check if 30-day average daily volume meets minimum threshold.

        Args:
            ticker: Stock symbol.

        Returns:
            (avg_volume, passed, reason_string)
        """
        try:
            end   = date.today()
            start = end - timedelta(days=30)
            df    = self._polygon.fetch_daily_bars(ticker, start, end)

            if df.empty:
                return (0.0, False, f"No volume data available for {ticker}")

            avg_volume = float(df["volume"].mean())
            if avg_volume >= self._min_avg_volume:
                return (avg_volume, True, "")
            return (
                avg_volume,
                False,
                f"Avg volume {avg_volume:,.0f} < minimum {self._min_avg_volume:,.0f}",
            )
        except Exception as e:
            logger.warning(f"[StockScreener] Could not fetch volume for {ticker}: {e}")
            return (0.0, False, f"Could not fetch volume data: {e}")

    def _check_price(self, ticker: str) -> tuple[float, bool, str]:
        """
        Check if latest close price meets minimum threshold.

        Args:
            ticker: Stock symbol.

        Returns:
            (price, passed, reason_string)
        """
        try:
            end   = date.today()
            start = end - timedelta(days=5)
            df    = self._polygon.fetch_daily_bars(ticker, start, end)

            if df.empty:
                return (0.0, False, f"No price data available for {ticker}")

            price = float(df["close"].iloc[-1])
            if price >= self._min_price:
                return (price, True, "")
            return (
                price,
                False,
                f"Price ${price:.2f} < minimum ${self._min_price:.2f}",
            )
        except Exception as e:
            logger.warning(f"[StockScreener] Could not fetch price for {ticker}: {e}")
            return (0.0, False, f"Could not fetch price data: {e}")

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
