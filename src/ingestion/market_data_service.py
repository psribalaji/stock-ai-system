"""
market_data_service.py — Orchestrates all market data operations.
This is the single entry point for all OHLCV data needs.
Coordinates between PolygonClient (historical) and AlpacaClient (real-time).
"""
from __future__ import annotations
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

from src.config import get_config
from src.ingestion.polygon_client import PolygonClient
from src.ingestion.alpaca_client import AlpacaClient
from src.ingestion.storage import ParquetStore


class MarketDataService:
    """
    Single entry point for all market data.

    Data flow:
      Historical data → PolygonClient → ParquetStore (./data/raw/)
      Real-time data  → AlpacaClient  → in-memory / direct to FeatureEngine
    """

    def __init__(
        self,
        polygon: Optional[PolygonClient] = None,
        alpaca: Optional[AlpacaClient] = None,
        store: Optional[ParquetStore] = None,
    ):
        self.config  = get_config()
        self.polygon = polygon or PolygonClient()
        self.alpaca  = alpaca  or AlpacaClient()
        self.store   = store   or ParquetStore(self.config.data.storage_path)

    # ── HISTORICAL DATA ──────────────────────────────────────────

    def fetch_and_store_historical(
        self,
        tickers: Optional[list[str]] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
        years_back: int = 4,
    ) -> dict[str, dict]:
        """
        Fetch historical OHLCV from Polygon and store as Parquet.
        This is run once during Phase 0 setup, then daily for incremental updates.

        Args:
            tickers:    Symbols to fetch. Defaults to config asset universe.
            start:      Start date. Defaults to years_back years ago.
            end:        End date. Defaults to today.
            years_back: How many years of history to pull on first run.

        Returns:
            Dict of ticker → {rows_fetched, validation_report}
        """
        tickers = tickers or self.config.assets.all_tradeable
        start   = start or (date.today() - timedelta(days=years_back * 365))
        end     = end   or date.today()

        logger.info(
            f"Fetching historical data for {len(tickers)} tickers: "
            f"{start} → {end}"
        )

        results = {}
        for ticker in tickers:
            try:
                df = self.polygon.fetch_daily_bars(ticker, start, end)

                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    results[ticker] = {"rows": 0, "valid": False, "issues": ["No data"]}
                    continue

                # Validate before storing
                validation = self.store.validate_ohlcv(df, ticker)
                if not validation["valid"]:
                    logger.warning(
                        f"Data quality issues for {ticker}: {validation['issues']}"
                    )

                # Store even with warnings — log issues for review
                self.store.save_ohlcv(ticker, df)
                results[ticker] = {
                    "rows":       len(df),
                    "valid":      validation["valid"],
                    "issues":     validation["issues"],
                    "date_range": f"{df['timestamp'].min()} → {df['timestamp'].max()}",
                }

            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = {"rows": 0, "valid": False, "issues": [str(e)]}

            # Polygon free tier: 5 requests/min — pause between tickers
            time.sleep(13)

        self._log_fetch_summary(results)
        return results

    def fetch_incremental_update(
        self,
        tickers: Optional[list[str]] = None,
    ) -> dict[str, int]:
        """
        Fetch only new bars since the last stored date using Alpaca (1 batch
        call for all tickers). Falls back to Polygon per-ticker for any that
        have no local history yet (first-time full backfill).
        """
        tickers = tickers or self.config.assets.all_tradeable
        results = {}

        # ── First-time tickers: full backfill from Polygon ────────────────────
        import time as _time
        new_tickers = [t for t in tickers if self.store.load_ohlcv(t).empty]
        for ticker in new_tickers:
            try:
                result = self.fetch_and_store_historical([ticker])
                results[ticker] = result.get(ticker, {}).get("rows", 0)
                logger.info(f"{ticker}: first-time backfill complete")
            except Exception as e:
                logger.error(f"Backfill failed for {ticker}: {e}")
                results[ticker] = -1
            _time.sleep(13.0)  # Polygon free tier: 5 calls/min

        # ── Incremental tickers: batch fetch via Alpaca ───────────────────────
        existing_tickers = [t for t in tickers if t not in new_tickers]
        if not existing_tickers:
            return results

        # Find the oldest last_date across all tickers — fetch from there
        start_dates = {}
        for ticker in existing_tickers:
            df = self.store.load_ohlcv(ticker)
            last_date = pd.to_datetime(df["timestamp"]).max().date()
            start = last_date + timedelta(days=1)
            if start <= date.today():
                start_dates[ticker] = start

        fetch_tickers = list(start_dates.keys())
        up_to_date    = [t for t in existing_tickers if t not in start_dates]
        for t in up_to_date:
            results[t] = 0
            logger.debug(f"{t} is up to date")

        if fetch_tickers:
            try:
                from src.ingestion.alpaca_client import AlpacaClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                from datetime import datetime as dt

                alpaca  = AlpacaClient()
                # Use earliest start date so all tickers get their missing bars
                min_start = min(start_dates.values())

                # Filter to stock tickers only (Alpaca doesn't serve BTC/SOL via StockBarsRequest)
                stock_tickers = [t for t in fetch_tickers if t not in ("BTC", "SOL")]
                crypto_tickers = [t for t in fetch_tickers if t in ("BTC", "SOL")]

                if stock_tickers:
                    request = StockBarsRequest(
                        symbol_or_symbols=stock_tickers,
                        timeframe=TimeFrame.Day,
                        start=dt.combine(min_start, dt.min.time()),
                        feed=self.config.data.alpaca_feed,
                    )
                    bars = alpaca._data_client.get_stock_bars(request)

                    for ticker in stock_tickers:
                        try:
                            if bars and ticker in bars.data and bars.data[ticker]:
                                records = [
                                    {
                                        "timestamp": bar.timestamp,
                                        "open":      bar.open,
                                        "high":      bar.high,
                                        "low":       bar.low,
                                        "close":     bar.close,
                                        "volume":    bar.volume,
                                        "vwap":      getattr(bar, "vwap", None),
                                        "ticker":    ticker,
                                        "source":    "alpaca",
                                    }
                                    for bar in bars.data[ticker]
                                    if bar.timestamp.date() >= start_dates[ticker]
                                ]
                                if records:
                                    df_new = pd.DataFrame(records)
                                    self.store.save_ohlcv(ticker, df_new)
                                    results[ticker] = len(records)
                                    logger.info(f"{ticker}: +{len(records)} new bars (Alpaca)")
                                else:
                                    results[ticker] = 0
                            else:
                                results[ticker] = 0
                        except Exception as e:
                            logger.error(f"Alpaca parse failed for {ticker}: {e}")
                            results[ticker] = -1

                # Crypto: fall back to Polygon (Alpaca crypto API is separate)
                for ticker in crypto_tickers:
                    try:
                        df = self.polygon.fetch_daily_bars(ticker, start_dates[ticker], date.today())
                        if not df.empty:
                            self.store.save_ohlcv(ticker, df)
                            results[ticker] = len(df)
                            logger.info(f"{ticker}: +{len(df)} new bars (Polygon)")
                        else:
                            results[ticker] = 0
                        _time.sleep(13.0)
                    except Exception as e:
                        logger.error(f"Polygon failed for {ticker}: {e}")
                        results[ticker] = -1

            except Exception as e:
                logger.error(f"Alpaca batch fetch failed: {e} — falling back to Polygon")
                # Full Polygon fallback
                for ticker in fetch_tickers:
                    if ticker in results:
                        continue
                    try:
                        df = self.polygon.fetch_daily_bars(ticker, start_dates[ticker], date.today())
                        if not df.empty:
                            self.store.save_ohlcv(ticker, df)
                            results[ticker] = len(df)
                        else:
                            results[ticker] = 0
                        _time.sleep(13.0)
                    except Exception as e2:
                        logger.error(f"Polygon fallback failed for {ticker}: {e2}")
                        results[ticker] = -1

        return results

    # ── LOADING DATA FOR FEATURE ENGINE ──────────────────────────

    def get_ohlcv_for_signals(
        self,
        tickers: Optional[list[str]] = None,
        lookback_days: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data ready for feature computation.
        Called by FeatureEngine before computing indicators.
        """
        tickers      = tickers      or self.config.assets.all_tradeable
        lookback_days = lookback_days or self.config.signals.lookback_days

        start = date.today() - timedelta(days=lookback_days + 30)  # Buffer for weekends/holidays
        return self.store.load_ohlcv_multi(tickers, start=start)

    def get_latest_prices(
        self,
        tickers: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Get the most recent price for each ticker.
        Used by RiskManager for position sizing.
        """
        tickers = tickers or self.config.assets.all_tradeable
        prices = {}
        for ticker in tickers:
            price = self.alpaca.get_latest_price(ticker)
            if price:
                prices[ticker] = price
        return prices

    # ── S3 SYNC ──────────────────────────────────────────────────

    def sync_to_s3(self) -> None:
        """
        Sync local Parquet files to S3.
        Only runs if config.data.sync_to_s3 = true (Phase 3).
        """
        if not self.config.data.sync_to_s3:
            logger.debug("S3 sync disabled in config — skipping")
            return

        try:
            import boto3
            s3 = boto3.client("s3")
            bucket  = self.config.data.s3_bucket
            prefix  = self.config.data.s3_prefix
            data_path = Path(self.config.data.storage_path)

            synced = 0
            for parquet_file in data_path.rglob("*.parquet"):
                relative = parquet_file.relative_to(data_path)
                s3_key   = f"{prefix}/{relative}".replace("\\", "/")
                s3.upload_file(str(parquet_file), bucket, s3_key)
                synced += 1

            logger.info(f"S3 sync complete: {synced} files → s3://{bucket}/{prefix}")

        except Exception as e:
            logger.error(f"S3 sync failed: {e}")

    # ── VALIDATION ───────────────────────────────────────────────

    def run_data_health_check(
        self,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """
        Run a full data quality check across all stored tickers.
        Call this after the initial historical fetch to confirm data is clean.
        """
        tickers = tickers or self.config.assets.all_tradeable
        report = {}

        for ticker in tickers:
            df = self.store.load_ohlcv(ticker)
            validation = self.store.validate_ohlcv(df, ticker)
            report[ticker] = validation

            if validation["valid"]:
                logger.info(f"✓ {ticker}: {validation['rows']} rows — CLEAN")
            else:
                logger.warning(f"✗ {ticker}: issues found — {validation['issues']}")

        clean = sum(1 for v in report.values() if v.get("valid"))
        logger.info(
            f"Data health check: {clean}/{len(tickers)} tickers clean"
        )
        return report

    def validate_connections(self) -> dict:
        """Check all API connections before starting the scheduler."""
        return {
            "polygon": self.polygon.validate_connection(),
            "alpaca":  self.alpaca.validate_connection(),
        }

    # ── HELPERS ──────────────────────────────────────────────────

    @staticmethod
    def _log_fetch_summary(results: dict) -> None:
        total  = len(results)
        ok     = sum(1 for v in results.values() if v.get("valid"))
        issues = sum(1 for v in results.values() if not v.get("valid"))
        total_rows = sum(v.get("rows", 0) for v in results.values())

        logger.info(
            f"Fetch complete: {ok}/{total} tickers clean | "
            f"{issues} with issues | {total_rows:,} total rows"
        )
