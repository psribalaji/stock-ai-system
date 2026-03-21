"""
tests/test_discovery.py — Tests for the Dynamic Universe Discovery feature.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trending(
    ticker: str = "ZXYZ",
    mention_spike: float = 5.0,
    sources: list[str] | None = None,
) -> "TrendingTicker":
    from src.discovery.trend_scanner import TrendingTicker
    return TrendingTicker(
        ticker        = ticker,
        company_name  = f"{ticker} Corp",
        sector        = "Technology",
        mention_count = 50,
        mention_spike = mention_spike,
        avg_sentiment = 0.3,
        sources       = sources or ["news"],
        first_seen    = datetime.now(timezone.utc),
        price         = 100.0,
        market_cap    = 1_000_000_000.0,
    )


def _make_screened(ticker: str = "ZXYZ", passed: bool = True) -> "ScreenedTicker":
    from src.discovery.stock_screener import ScreenedTicker
    return ScreenedTicker(
        ticker         = ticker,
        company_name   = f"{ticker} Corp",
        sector         = "Technology",
        market_cap     = 1_000_000_000.0,
        avg_volume_30d = 2_000_000.0,
        latest_price   = 100.0,
        passed         = passed,
        fail_reasons   = [] if passed else ["Low market cap"],
        trending_data  = _make_trending(ticker),
    )


# ── TestTrendScanner ──────────────────────────────────────────────────────────

class TestTrendScanner:

    def test_scan_returns_list(self):
        """scan() should always return a list (even on empty results)."""
        with patch("src.discovery.trend_scanner.TrendScanner._scan_news_velocity", return_value=[]), \
             patch("src.discovery.trend_scanner.TrendScanner._scan_reddit", return_value=[]):
            from src.discovery.trend_scanner import TrendScanner
            scanner = TrendScanner()
            result  = scanner.scan()
            assert isinstance(result, list)

    def test_scan_news_velocity_skips_existing_universe(self):
        """Tickers already in the asset universe should not be returned."""
        from src.discovery.trend_scanner import TrendScanner, TrendingTicker
        from src.config import get_config

        cfg = get_config()
        existing_ticker = cfg.assets.stocks[0]  # e.g. "NVDA"

        with patch("src.discovery.trend_scanner.TrendScanner._scan_news_velocity",
                   return_value=[_make_trending(existing_ticker)]), \
             patch("src.discovery.trend_scanner.TrendScanner._scan_reddit", return_value=[]):
            scanner = TrendScanner()
            result  = scanner.scan()
            tickers = [t.ticker for t in result]
            assert existing_ticker not in tickers

    def test_scan_reddit_returns_empty_when_praw_missing(self):
        """_scan_reddit() should return [] when praw is not importable."""
        from src.discovery.trend_scanner import TrendScanner

        scanner = TrendScanner()
        with patch.dict("sys.modules", {"praw": None}):
            result = scanner._scan_reddit()
            assert result == []

    def test_scan_reddit_filters_blocklist(self):
        """
        Words in the blocklist (e.g. 'CEO', 'BUY') should not appear as tickers
        in Reddit scan output.
        """
        from src.discovery.trend_scanner import TrendScanner, _REDDIT_BLOCKLIST

        scanner = TrendScanner()

        # Simulate praw being available but returning posts with blocklist words
        mock_praw  = MagicMock()
        mock_reddit = MagicMock()
        mock_sub   = MagicMock()

        # Post text that only contains blocklist words
        mock_post  = MagicMock()
        mock_post.title       = "CEO BUY SELL HOLD IMO"
        mock_post.selftext    = "FUD FOMO YOLO"
        mock_post.upvote_ratio = 0.8

        mock_sub.hot.return_value = [mock_post] * 110
        mock_sub.new.return_value = [mock_post] * 50
        mock_reddit.subreddit.return_value = mock_sub
        mock_praw.Reddit.return_value = mock_reddit

        import os
        with patch.dict("sys.modules", {"praw": mock_praw}), \
             patch.dict(os.environ, {
                 "REDDIT_CLIENT_ID":     "test_id",
                 "REDDIT_CLIENT_SECRET": "test_secret",
             }):
            result = scanner._scan_reddit()
            tickers = [t.ticker for t in result]
            for ticker in tickers:
                assert ticker not in _REDDIT_BLOCKLIST, f"{ticker} should be filtered"

    def test_duplicates_deduplicated_across_sources(self):
        """Same ticker from news and Reddit should be merged into one entry."""
        from src.discovery.trend_scanner import TrendScanner

        same_ticker = "ZXYZ"
        news_hit    = _make_trending(same_ticker, sources=["news"])
        reddit_hit  = _make_trending(same_ticker, sources=["reddit"])

        with patch("src.discovery.trend_scanner.TrendScanner._scan_news_velocity",
                   return_value=[news_hit]), \
             patch("src.discovery.trend_scanner.TrendScanner._scan_reddit",
                   return_value=[reddit_hit]):
            scanner = TrendScanner()
            result  = scanner.scan()
            tickers = [t.ticker for t in result]
            assert tickers.count(same_ticker) == 1
            merged = next(t for t in result if t.ticker == same_ticker)
            assert "news" in merged.sources
            assert "reddit" in merged.sources

    def test_scan_limited_to_max_candidates(self):
        """scan() should return at most max_candidates results."""
        from src.discovery.trend_scanner import TrendScanner

        # Create 30 unique non-universe tickers
        candidates = [_make_trending(f"ZZ{i:02d}") for i in range(30)]

        with patch("src.discovery.trend_scanner.TrendScanner._scan_news_velocity",
                   return_value=candidates), \
             patch("src.discovery.trend_scanner.TrendScanner._scan_reddit", return_value=[]):
            scanner = TrendScanner()
            result  = scanner.scan()
            assert len(result) <= 20  # default max_candidates


# ── TestStockScreener ─────────────────────────────────────────────────────────

class TestStockScreener:

    def _make_screener_with_mocks(
        self,
        market_cap:   float = 2_000_000_000.0,
        avg_volume:   float = 1_000_000.0,
        price:        float = 50.0,
        tradeable:    bool  = True,
    ):
        """Build a StockScreener with all external clients mocked."""
        from src.discovery.stock_screener import StockScreener

        # Mock PolygonClient
        mock_poly = MagicMock()

        # Ticker details for market cap
        mock_details         = MagicMock()
        mock_details.market_cap = market_cap
        mock_poly._client.get_ticker_details.return_value = mock_details

        # Daily bars for volume + price check
        bars_df = pd.DataFrame({
            "close":  [price]  * 10,
            "volume": [avg_volume] * 10,
        })
        mock_poly.fetch_daily_bars.return_value = bars_df

        # Mock AlpacaClient
        mock_alpaca = MagicMock()
        mock_asset  = MagicMock()
        mock_asset.tradable = tradeable

        from alpaca.trading.enums import AssetStatus
        mock_asset.status = AssetStatus.ACTIVE if tradeable else "inactive"
        mock_alpaca._trading_client.get_asset.return_value = mock_asset

        screener = StockScreener.__new__(StockScreener)
        screener.config         = MagicMock()
        screener._polygon       = mock_poly
        screener._alpaca        = mock_alpaca
        screener._min_market_cap = 500_000_000
        screener._min_avg_volume = 500_000
        screener._min_price      = 5.0

        return screener

    def test_screen_returns_screened_ticker_for_each_input(self):
        """screen() should return one ScreenedTicker per input TrendingTicker."""
        screener   = self._make_screener_with_mocks()
        candidates = [_make_trending("AAA"), _make_trending("BBB")]
        results    = screener.screen(candidates)
        assert len(results) == 2

    def test_fails_on_low_market_cap(self):
        """Should fail and populate fail_reasons when market cap is too low."""
        screener = self._make_screener_with_mocks(market_cap=100_000_000)
        results  = screener.screen([_make_trending("LOWCAP")])
        assert len(results) == 1
        assert not results[0].passed
        assert any("market cap" in r.lower() for r in results[0].fail_reasons)

    def test_fails_on_low_volume(self):
        """Should fail when avg daily volume is below threshold."""
        screener = self._make_screener_with_mocks(avg_volume=10_000)
        results  = screener.screen([_make_trending("LOWVOL")])
        assert not results[0].passed
        assert any("volume" in r.lower() for r in results[0].fail_reasons)

    def test_fails_on_low_price(self):
        """Should fail when latest price is below threshold."""
        screener = self._make_screener_with_mocks(price=1.0)
        results  = screener.screen([_make_trending("PENNY")])
        assert not results[0].passed
        assert any("price" in r.lower() for r in results[0].fail_reasons)

    def test_passes_when_all_criteria_met(self):
        """Should pass when all 4 criteria are satisfied."""
        screener = self._make_screener_with_mocks(
            market_cap = 2_000_000_000.0,
            avg_volume = 1_000_000.0,
            price      = 50.0,
            tradeable  = True,
        )
        results = screener.screen([_make_trending("GOOD")])
        assert results[0].passed
        assert results[0].fail_reasons == []

    def test_fail_reasons_populated(self):
        """Multiple failures should all appear in fail_reasons."""
        screener = self._make_screener_with_mocks(
            market_cap = 100_000,    # too low
            avg_volume = 100,        # too low
            price      = 0.5,        # too low
            tradeable  = False,
        )
        results = screener.screen([_make_trending("BAD")])
        assert not results[0].passed
        assert len(results[0].fail_reasons) >= 3


# ── TestUniverseManager ───────────────────────────────────────────────────────

class TestUniverseManager:

    def test_add_candidates_stores_to_parquet(self, tmp_path: Path):
        """add_candidates() should persist passed tickers to parquet."""
        from src.discovery.universe_manager import UniverseManager

        manager   = UniverseManager(base_path=str(tmp_path))
        screened  = [_make_screened("ZXYZ", passed=True)]
        added     = manager.add_candidates(screened)

        assert added == 1
        assert (tmp_path / "discovery" / "watchlist.parquet").exists()

        df = manager.get_watchlist()
        assert "ZXYZ" in df["ticker"].values

    def test_add_candidates_skips_duplicates(self, tmp_path: Path):
        """Adding the same ticker twice should only count it once."""
        from src.discovery.universe_manager import UniverseManager

        manager  = UniverseManager(base_path=str(tmp_path))
        screened = [_make_screened("ZXYZ", passed=True)]

        first  = manager.add_candidates(screened)
        second = manager.add_candidates(screened)

        assert first  == 1
        assert second == 0

        df = manager.get_watchlist()
        assert (df["ticker"] == "ZXYZ").sum() == 1

    def test_add_candidates_skips_failed(self, tmp_path: Path):
        """Tickers that failed screening should not be added."""
        from src.discovery.universe_manager import UniverseManager

        manager  = UniverseManager(base_path=str(tmp_path))
        screened = [_make_screened("FAIL", passed=False)]
        added    = manager.add_candidates(screened)

        assert added == 0

    def test_approve_changes_status(self, tmp_path: Path):
        """approve() should set status to APPROVED and set approved_at."""
        from src.discovery.universe_manager import UniverseManager, STATUS_APPROVED

        manager = UniverseManager(base_path=str(tmp_path))
        manager.add_candidates([_make_screened("ZXYZ")])

        result = manager.approve("ZXYZ")
        assert result is True

        df = manager.get_watchlist()
        row = df[df["ticker"] == "ZXYZ"].iloc[0]
        assert row["status"] == STATUS_APPROVED
        assert pd.notna(row["approved_at"])

    def test_approve_returns_false_when_not_found(self, tmp_path: Path):
        """approve() should return False when ticker is not in watchlist."""
        from src.discovery.universe_manager import UniverseManager

        manager = UniverseManager(base_path=str(tmp_path))
        result  = manager.approve("NOTEXIST")
        assert result is False

    def test_ignore_changes_status(self, tmp_path: Path):
        """ignore() should set status to IGNORED."""
        from src.discovery.universe_manager import UniverseManager, STATUS_IGNORED

        manager = UniverseManager(base_path=str(tmp_path))
        manager.add_candidates([_make_screened("ZXYZ")])

        result = manager.ignore("ZXYZ")
        assert result is True

        df = manager.get_watchlist()
        assert df[df["ticker"] == "ZXYZ"].iloc[0]["status"] == STATUS_IGNORED

    def test_expire_old(self, tmp_path: Path):
        """expire_old() should set status=EXPIRED for old CANDIDATE rows."""
        from src.discovery.universe_manager import UniverseManager, STATUS_EXPIRED, _SCHEMA_COLS
        import pyarrow as pa
        import pyarrow.parquet as pq

        manager = UniverseManager(base_path=str(tmp_path))

        # Add a candidate, then manually backdate its added_at
        manager.add_candidates([_make_screened("OLD01")])

        df = manager._load()
        old_time = datetime.now(timezone.utc) - timedelta(days=20)
        df.loc[df["ticker"] == "OLD01", "added_at"] = old_time
        manager._save(df)

        expired = manager.expire_old(days=14)
        assert expired == 1

        df_after = manager.get_watchlist()
        assert df_after[df_after["ticker"] == "OLD01"].iloc[0]["status"] == STATUS_EXPIRED

    def test_expire_old_leaves_recent_candidates(self, tmp_path: Path):
        """expire_old() should not expire recently added candidates."""
        from src.discovery.universe_manager import UniverseManager, STATUS_CANDIDATE

        manager = UniverseManager(base_path=str(tmp_path))
        manager.add_candidates([_make_screened("NEW01")])

        expired = manager.expire_old(days=14)
        assert expired == 0

        df = manager.get_watchlist()
        assert df[df["ticker"] == "NEW01"].iloc[0]["status"] == STATUS_CANDIDATE

    def test_get_tradeable_universe_includes_approved(self, tmp_path: Path):
        """get_tradeable_universe() should include APPROVED tickers."""
        from src.discovery.universe_manager import UniverseManager
        from src.config import get_config

        manager = UniverseManager(base_path=str(tmp_path))
        manager.add_candidates([_make_screened("ZXYZ")])
        manager.approve("ZXYZ")

        universe = manager.get_tradeable_universe()
        assert "ZXYZ" in universe

        # Should also include base config tickers
        cfg = get_config()
        for ticker in cfg.assets.all_tradeable:
            assert ticker in universe

    def test_get_watchlist_empty_when_no_file(self, tmp_path: Path):
        """get_watchlist() should return empty DataFrame when no file exists."""
        from src.discovery.universe_manager import UniverseManager

        manager = UniverseManager(base_path=str(tmp_path))
        df = manager.get_watchlist()

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_stats_returns_expected_keys(self, tmp_path: Path):
        """get_stats() should return a dict with the required keys."""
        from src.discovery.universe_manager import UniverseManager

        manager = UniverseManager(base_path=str(tmp_path))

        stats = manager.get_stats()
        required_keys = {"total", "candidate", "watchlist", "approved", "ignored", "expired", "approval_rate"}
        assert required_keys.issubset(stats.keys())

    def test_get_stats_counts_correctly(self, tmp_path: Path):
        """get_stats() should count statuses correctly."""
        from src.discovery.universe_manager import UniverseManager

        manager = UniverseManager(base_path=str(tmp_path))

        # Add 2 candidates, approve 1, ignore 1
        manager.add_candidates([_make_screened("AAA"), _make_screened("BBB")])
        manager.approve("AAA")
        manager.ignore("BBB")

        stats = manager.get_stats()
        assert stats["total"]    == 2
        assert stats["approved"] == 1
        assert stats["ignored"]  == 1
        assert stats["candidate"] == 0
        assert stats["approval_rate"] == 0.5

    def test_get_watchlist_with_status_filter(self, tmp_path: Path):
        """get_watchlist(status_filter=...) should only return rows of that status."""
        from src.discovery.universe_manager import UniverseManager, STATUS_CANDIDATE, STATUS_APPROVED

        manager = UniverseManager(base_path=str(tmp_path))
        manager.add_candidates([_make_screened("AAA"), _make_screened("BBB")])
        manager.approve("AAA")

        candidates = manager.get_watchlist(status_filter=STATUS_CANDIDATE)
        approved   = manager.get_watchlist(status_filter=STATUS_APPROVED)

        assert len(candidates) == 1
        assert len(approved)   == 1
        assert candidates.iloc[0]["ticker"] == "BBB"
        assert approved.iloc[0]["ticker"]   == "AAA"
