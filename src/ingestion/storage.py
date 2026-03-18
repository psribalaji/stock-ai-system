"""
storage.py — Parquet-based data store.
All read/write operations for OHLCV, signals, news, and audit data go here.
No database required — Parquet files are faster for time-series analytics
and cost a fraction of RDS.
"""
from __future__ import annotations
import os
from pathlib import Path
from datetime import date, datetime
from typing import Optional
import pandas as pd
from loguru import logger


class ParquetStore:
    """
    Handles all local Parquet storage.
    Directory structure:
      data/
        raw/       TICKER_YEAR.parquet  — OHLCV price data
        signals/   YYYY-MM-DD.parquet  — Daily signals
        news/      TICKER_YYYY-MM.parquet — News cache
        audit/     YYYY-MM.parquet     — Trade audit log
    """

    def __init__(self, base_path: str = "./data"):
        self.base = Path(base_path)
        self._init_dirs()

    def _init_dirs(self) -> None:
        for subdir in ["raw", "signals", "news", "audit"]:
            (self.base / subdir).mkdir(parents=True, exist_ok=True)

    # ── OHLCV ────────────────────────────────────────────────────

    def save_ohlcv(self, ticker: str, df: pd.DataFrame) -> Path:
        """
        Save OHLCV data. Merges with existing data to avoid duplicates.
        Files partitioned by ticker (one file per ticker, all years).
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {ticker} — skipping save")
            return Path()

        df = self._normalize_ohlcv(df, ticker)
        path = self.base / "raw" / f"{ticker}.parquet"

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df]).drop_duplicates(
                subset=["timestamp"]
            ).sort_values("timestamp")
        else:
            combined = df

        combined.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Saved {len(combined)} rows for {ticker} → {path}")
        return path

    def load_ohlcv(
        self,
        ticker: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load OHLCV for a ticker with optional date range filter."""
        path = self.base / "raw" / f"{ticker}.parquet"
        if not path.exists():
            logger.warning(f"No OHLCV data found for {ticker}")
            return pd.DataFrame()

        df = pd.read_parquet(path, engine="pyarrow")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Convert nullable Int64 volume to float64 so pandas-ta can handle NA values
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype("float64")

        if start:
            df = df[df["timestamp"].dt.date >= start]
        if end:
            df = df[df["timestamp"].dt.date <= end]

        return df.sort_values("timestamp").reset_index(drop=True)

    def load_ohlcv_multi(
        self,
        tickers: list[str],
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV for multiple tickers at once."""
        return {t: self.load_ohlcv(t, start, end) for t in tickers}

    # ── SIGNALS ──────────────────────────────────────────────────

    def save_signals(self, df: pd.DataFrame, signal_date: Optional[date] = None) -> Path:
        """Save signals for a given date."""
        if df.empty:
            return Path()
        d = signal_date or date.today()
        path = self.base / "signals" / f"{d.isoformat()}.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Saved {len(df)} signals → {path}")
        return path

    def load_signals(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load all signals across a date range."""
        sig_dir = self.base / "signals"
        files = sorted(sig_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            file_date = date.fromisoformat(f.stem)
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue
            dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)

    # ── NEWS ─────────────────────────────────────────────────────

    def save_news(self, ticker: str, df: pd.DataFrame) -> Path:
        """Cache news articles for a ticker, partitioned by month."""
        if df.empty:
            return Path()
        month = datetime.now().strftime("%Y-%m")
        path = self.base / "news" / f"{ticker}_{month}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).drop_duplicates(subset=["id"])
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        return path

    def load_news(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """Load recent news for a ticker."""
        news_dir = self.base / "news"
        files = sorted(news_dir.glob(f"{ticker}_*.parquet"))
        if not files:
            return pd.DataFrame()
        # Load last 2 monthly files to be safe
        dfs = [pd.read_parquet(f) for f in files[-2:]]
        df = pd.concat(dfs).drop_duplicates(subset=["id"])
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df[df["datetime"] >= cutoff]
        return df.sort_values("datetime", ascending=False).reset_index(drop=True)

    # ── AUDIT LOG ────────────────────────────────────────────────

    def save_audit(self, df: pd.DataFrame) -> Path:
        """Append trade records to the monthly audit log."""
        if df.empty:
            return Path()
        month = datetime.now().strftime("%Y-%m")
        path = self.base / "audit" / f"{month}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df]).drop_duplicates(subset=["trade_id"])
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        logger.info(f"Audit log updated → {path}")
        return path

    def load_audit(
        self,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load audit log across months."""
        audit_dir = self.base / "audit"
        files = sorted(audit_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs).drop_duplicates(subset=["trade_id"])
        df["timestamp_submitted"] = pd.to_datetime(df["timestamp_submitted"], utc=True)
        if start:
            df = df[df["timestamp_submitted"].dt.date >= start]
        if end:
            df = df[df["timestamp_submitted"].dt.date <= end]
        return df.sort_values("timestamp_submitted", ascending=False).reset_index(drop=True)

    # ── DATA VALIDATION ──────────────────────────────────────────

    def validate_ohlcv(self, df: pd.DataFrame, ticker: str) -> dict:
        """
        Validate OHLCV data quality.
        Returns a report dict with issues found.
        """
        report = {"ticker": ticker, "rows": len(df), "issues": []}

        if df.empty:
            report["issues"].append("EMPTY: No data")
            report["valid"] = False
            return report

        # Check required columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            report["issues"].append(f"MISSING COLUMNS: {missing_cols}")

        # Check for null values
        null_counts = df[required].isnull().sum()
        nulls = null_counts[null_counts > 0]
        if not nulls.empty:
            report["issues"].append(f"NULL VALUES: {nulls.to_dict()}")

        # Check for zero prices
        zero_prices = (df["close"] == 0).sum()
        if zero_prices > 0:
            report["issues"].append(f"ZERO PRICES: {zero_prices} rows")

        # Check for price spikes (> 3 std deviations)
        if len(df) > 10:
            returns = df["close"].pct_change().dropna()
            std = returns.std()
            spikes = (returns.abs() > 3 * std).sum()
            if spikes > 0:
                report["issues"].append(
                    f"PRICE SPIKES: {spikes} bars > 3σ — manual review suggested"
                )

        # Check for date gaps (weekdays only)
        if "timestamp" in df.columns and len(df) > 1:
            df_sorted = df.sort_values("timestamp")
            dates = pd.to_datetime(df_sorted["timestamp"]).dt.normalize().unique()
            business_days = pd.bdate_range(dates[0], dates[-1])
            missing_days = len(business_days) - len(dates)
            if missing_days > 5:
                report["issues"].append(f"DATE GAPS: ~{missing_days} missing business days")

        report["valid"] = len(report["issues"]) == 0
        return report

    # ── HELPERS ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalize column names and types for consistent storage."""
        # Rename common column variations
        rename_map = {
            "t": "timestamp", "time": "timestamp", "date": "timestamp",
            "o": "open", "h": "high", "l": "low", "c": "close",
            "v": "volume", "vw": "vwap",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure timestamp is UTC datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Add ticker column
        df["ticker"] = ticker

        # Ensure numeric types
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round(0).astype("Int64")

        # Drop duplicates
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        return df

    def get_stats(self) -> dict:
        """Quick summary of what's in the store."""
        raw_files = list((self.base / "raw").glob("*.parquet"))
        signal_files = list((self.base / "signals").glob("*.parquet"))
        audit_files = list((self.base / "audit").glob("*.parquet"))

        tickers = [f.stem for f in raw_files]
        total_rows = sum(
            len(pd.read_parquet(f)) for f in raw_files
        ) if raw_files else 0

        return {
            "tickers_stored": tickers,
            "total_ohlcv_rows": total_rows,
            "signal_files": len(signal_files),
            "audit_files": len(audit_files),
            "storage_path": str(self.base.absolute()),
        }
