"""
discovery/universe_manager.py — Manages the dynamic ticker watchlist.
Persists to data/discovery/watchlist.parquet.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.config import get_config
from src.discovery.stock_screener import ScreenedTicker


# Status values
STATUS_CANDIDATE = "CANDIDATE"
STATUS_WATCHLIST = "WATCHLIST"
STATUS_APPROVED  = "APPROVED"
STATUS_IGNORED   = "IGNORED"
STATUS_EXPIRED   = "EXPIRED"

# Schema columns
_SCHEMA_COLS = [
    "ticker", "company_name", "sector", "status",
    "added_at", "approved_at",
    "mention_spike", "avg_sentiment", "sources",
    "market_cap", "avg_volume", "latest_price",
    "signal_count", "notes",
]


def _empty_df() -> pd.DataFrame:
    """Return empty DataFrame with the correct watchlist schema."""
    return pd.DataFrame(columns=_SCHEMA_COLS)


class UniverseManager:
    """
    Manages the dynamic ticker watchlist.
    Persists discovered candidates to data/discovery/watchlist.parquet.

    Important: auto_approve is ALWAYS False — human approval required
    via the dashboard before any ticker enters the trading pipeline.
    """

    def __init__(self, base_path: str = "./data") -> None:
        self.config = get_config()
        self.watchlist_path = Path(base_path) / "discovery" / "watchlist.parquet"
        self.watchlist_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_candidates(self, screened: list[ScreenedTicker]) -> int:
        """
        Add passed screened tickers to the watchlist as CANDIDATE status.
        Skips if ticker already present with a non-EXPIRED status.

        Args:
            screened: List of ScreenedTicker from StockScreener.

        Returns:
            Count of new tickers added.
        """
        df = self._load()
        now = datetime.now(timezone.utc)
        added = 0

        for st in screened:
            if not st.passed:
                continue

            # Skip if ticker already present with active (non-EXPIRED) status
            existing_mask = (df["ticker"] == st.ticker) & (df["status"] != STATUS_EXPIRED)
            if not df.empty and existing_mask.any():
                logger.debug(f"[UniverseManager] {st.ticker} already in watchlist — skipping")
                continue

            sources_str = ",".join(st.trending_data.sources)
            new_row = {
                "ticker":        st.ticker,
                "company_name":  st.company_name,
                "sector":        st.sector,
                "status":        STATUS_CANDIDATE,
                "added_at":      now,
                "approved_at":   None,
                "mention_spike": st.trending_data.mention_spike,
                "avg_sentiment": st.trending_data.avg_sentiment,
                "sources":       sources_str,
                "market_cap":    st.market_cap,
                "avg_volume":    st.avg_volume_30d,
                "latest_price":  st.latest_price,
                "signal_count":  0,
                "notes":         "",
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            added += 1
            logger.info(f"[UniverseManager] Added candidate: {st.ticker} "
                        f"(spike={st.trending_data.mention_spike:.1f}x)")

        if added > 0:
            self._save(df)

        return added

    def approve(self, ticker: str) -> bool:
        """
        Approve a ticker for the trading universe.
        Sets status=APPROVED and approved_at=now.

        Args:
            ticker: Stock symbol to approve.

        Returns:
            True if found and updated, False otherwise.
        """
        df = self._load()
        if df.empty:
            return False

        mask = df["ticker"] == ticker
        if not mask.any():
            logger.warning(f"[UniverseManager] approve: ticker {ticker} not found")
            return False

        df.loc[mask, "status"]      = STATUS_APPROVED
        df.loc[mask, "approved_at"] = datetime.now(timezone.utc)
        self._save(df)
        logger.info(f"[UniverseManager] Approved: {ticker}")
        return True

    def ignore(self, ticker: str) -> bool:
        """
        Mark a ticker as IGNORED to exclude from future candidates.

        Args:
            ticker: Stock symbol to ignore.

        Returns:
            True if found and updated, False otherwise.
        """
        df = self._load()
        if df.empty:
            return False

        mask = df["ticker"] == ticker
        if not mask.any():
            logger.warning(f"[UniverseManager] ignore: ticker {ticker} not found")
            return False

        df.loc[mask, "status"] = STATUS_IGNORED
        self._save(df)
        logger.info(f"[UniverseManager] Ignored: {ticker}")
        return True

    def expire_old(self, days: int = 14) -> int:
        """
        Expire CANDIDATE rows older than `days` days.

        Args:
            days: Age threshold in days.

        Returns:
            Count of rows expired.
        """
        df = self._load()
        if df.empty:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Ensure added_at is tz-aware
        df["added_at"] = pd.to_datetime(df["added_at"], utc=True)

        mask = (df["status"] == STATUS_CANDIDATE) & (df["added_at"] < cutoff)
        count = int(mask.sum())

        if count > 0:
            df.loc[mask, "status"] = STATUS_EXPIRED
            self._save(df)
            logger.info(f"[UniverseManager] Expired {count} stale candidates")

        return count

    def get_tradeable_universe(self) -> list[str]:
        """
        Return the full tradeable universe: config assets + APPROVED tickers.

        Returns:
            Deduplicated list of ticker symbols.
        """
        base = list(self.config.assets.all_tradeable)
        df   = self._load()

        if not df.empty:
            approved = df[df["status"] == STATUS_APPROVED]["ticker"].tolist()
            base = list(dict.fromkeys(base + approved))  # deduplicate preserving order

        return base

    def get_watchlist(self, status_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Load the watchlist, optionally filtered by status.

        Args:
            status_filter: One of CANDIDATE | WATCHLIST | APPROVED | IGNORED | EXPIRED.
                           None returns all rows.

        Returns:
            DataFrame (empty if file doesn't exist).
        """
        df = self._load()
        if df.empty:
            return df

        if status_filter is not None:
            df = df[df["status"] == status_filter].reset_index(drop=True)

        return df

    def get_stats(self) -> dict:
        """
        Return counts by status and approval rate.

        Returns:
            Dict with keys: total, candidate, watchlist, approved, ignored, expired, approval_rate
        """
        df = self._load()
        if df.empty:
            return {
                "total":         0,
                "candidate":     0,
                "watchlist":     0,
                "approved":      0,
                "ignored":       0,
                "expired":       0,
                "approval_rate": 0.0,
            }

        counts = df["status"].value_counts().to_dict()
        total    = len(df)
        approved = counts.get(STATUS_APPROVED,  0)
        ignored  = counts.get(STATUS_IGNORED,   0)
        decided  = approved + ignored
        approval_rate = (approved / decided) if decided > 0 else 0.0

        return {
            "total":         total,
            "candidate":     counts.get(STATUS_CANDIDATE, 0),
            "watchlist":     counts.get(STATUS_WATCHLIST, 0),
            "approved":      approved,
            "ignored":       ignored,
            "expired":       counts.get(STATUS_EXPIRED,  0),
            "approval_rate": round(approval_rate, 3),
        }

    # ── Internal I/O ──────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        """Load watchlist.parquet or return empty DataFrame with correct schema."""
        if not self.watchlist_path.exists():
            return _empty_df()

        try:
            df = pd.read_parquet(self.watchlist_path, engine="pyarrow")
            # Ensure all schema columns exist
            for col in _SCHEMA_COLS:
                if col not in df.columns:
                    df[col] = None
            return df[_SCHEMA_COLS]
        except Exception as e:
            logger.warning(f"[UniverseManager] Failed to load watchlist: {e}")
            return _empty_df()

    def _save(self, df: pd.DataFrame) -> None:
        """Save DataFrame to watchlist.parquet."""
        try:
            # Ensure the directory exists
            self.watchlist_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(
                self.watchlist_path,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            logger.debug(f"[UniverseManager] Saved watchlist ({len(df)} rows) → {self.watchlist_path}")
        except Exception as e:
            logger.error(f"[UniverseManager] Failed to save watchlist: {e}")
