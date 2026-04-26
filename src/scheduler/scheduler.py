"""
scheduler/scheduler.py — APScheduler job runner for the signal pipeline.

Jobs (times are US/Eastern):
  - data_sync:        05:00 daily     — fetch latest OHLCV from Polygon
  - signal_pipeline:  every 5 min     — run DecisionEngine for all tickers
                      (only fires during market hours 09:30–16:00 Mon–Fri)
  - drift_check:      Sunday 20:00    — weekly ModelMonitor drift report
  - recalibrate:      quarterly       — re-run backtests, update baselines

Usage:
    scheduler = TradingScheduler()
    scheduler.start()          # blocking
    scheduler.start_background()  # non-blocking (for tests / embedding)
    scheduler.stop()
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Callable, List, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from zoneinfo import ZoneInfo

from src.config import get_config
from src.ingestion.storage import ParquetStore
from src.execution.decision_engine import DecisionEngine
from src.risk.risk_manager import PortfolioState
from src.monitoring.model_monitor import ModelMonitor, DriftReport


ET = ZoneInfo("America/New_York")

# Market hours (Eastern)
_MARKET_OPEN  = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def _is_weekend() -> bool:
    """Return True if today is Saturday or Sunday (ET)."""
    return datetime.now(ET).weekday() >= 5


def _is_market_hours() -> bool:
    """Return True if current ET time is within regular market hours (Mon–Fri).
    Always returns True when schedule.force_market_hours is enabled (simulation mode)."""
    cfg = get_config()
    if cfg.schedule.force_market_hours:
        return True
    now = datetime.now(ET)
    if now.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    return _MARKET_OPEN <= now.time() <= _MARKET_CLOSE


class TradingScheduler:
    """
    Orchestrates all scheduled jobs for the trading system.

    Args:
        decision_engine: Pre-built DecisionEngine (created fresh if None)
        monitor:         Pre-built ModelMonitor (created fresh if None)
        store:           Pre-built ParquetStore (created fresh if None)
        on_decisions:    Optional callback called with List[TradeDecision]
                         after each pipeline run (use for paper order submission)
        on_drift:        Optional callback called with DriftReport after each drift check
    """

    def __init__(
        self,
        decision_engine: Optional[DecisionEngine] = None,
        monitor: Optional[ModelMonitor] = None,
        store: Optional[ParquetStore] = None,
        executor=None,
        on_decisions: Optional[Callable] = None,
        on_drift: Optional[Callable] = None,
        on_discovery: Optional[Callable] = None,
    ) -> None:
        self.config  = get_config()
        self.engine  = decision_engine or DecisionEngine()
        self.monitor = monitor or ModelMonitor()
        self.store   = store or ParquetStore(self.config.data.storage_path)
        self.executor = executor  # OrderExecutor for position checks
        self.on_decisions = on_decisions
        self.on_drift     = on_drift
        self.on_discovery = on_discovery

        self._scheduler = BackgroundScheduler(timezone=str(ET))
        self._running   = False
        self._register_jobs()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_background(self) -> None:
        """Start scheduler in the background (non-blocking)."""
        if self._running:
            logger.warning("[Scheduler] Already running")
            return
        self._scheduler.start()
        self._running = True
        logger.info("[Scheduler] Started (background)")

    def start(self) -> None:
        """Start scheduler and block until KeyboardInterrupt."""
        self.start_background()
        logger.info("[Scheduler] Running — press Ctrl+C to stop")
        try:
            import time as _time
            while True:
                _time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self) -> None:
        """Shutdown the scheduler gracefully."""
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("[Scheduler] Stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_jobs(self) -> List[str]:
        """Return list of registered job IDs."""
        return [job.id for job in self._scheduler.get_jobs()]

    # ── Job registration ──────────────────────────────────────────────────────

    def _register_jobs(self) -> None:
        cfg = self.config.schedule

        # 1. Daily data sync at 05:00 ET
        self._scheduler.add_job(
            self.job_data_sync,
            CronTrigger(hour=cfg.data_sync_hour, minute=0, timezone=ET),
            id="data_sync",
            name="Daily OHLCV sync",
            max_instances=1,
            misfire_grace_time=300,
        )

        # 2. Signal pipeline every N minutes (market hours only)
        self._scheduler.add_job(
            self.job_signal_pipeline,
            IntervalTrigger(minutes=cfg.signal_interval_min, timezone=ET),
            id="signal_pipeline",
            name="Signal pipeline",
            max_instances=1,
            misfire_grace_time=60,
        )

        # 2b. Position check (trailing stops + take profits) every N minutes
        self._scheduler.add_job(
            self.job_position_check,
            IntervalTrigger(minutes=cfg.position_check_interval_min, timezone=ET),
            id="position_check",
            name="Position check (stops + take-profits)",
            max_instances=1,
            misfire_grace_time=30,
        )

        # 3. Weekly drift check — Sunday at drift_check_hour ET
        self._scheduler.add_job(
            self.job_drift_check,
            CronTrigger(
                day_of_week=cfg.drift_check_day,
                hour=cfg.drift_check_hour,
                minute=0,
                timezone=ET,
            ),
            id="drift_check",
            name="Weekly drift check",
            max_instances=1,
        )

        # 4. Quarterly recalibration — 1st of Jan/Apr/Jul/Oct at 06:00 ET
        months = ",".join(str(m) for m in cfg.recalibration_months)
        self._scheduler.add_job(
            self.job_recalibrate,
            CronTrigger(month=months, day=1, hour=6, minute=0, timezone=ET),
            id="recalibrate",
            name="Quarterly recalibration",
            max_instances=1,
        )

        # 5 & 6. Discovery scan jobs (only if discovery.enabled is True)
        disc_cfg = getattr(self.config, "discovery", None)
        if disc_cfg and getattr(disc_cfg, "enabled", False):
            pre_market_hour  = getattr(disc_cfg, "pre_market_scan_hour", 8)
            scan_interval    = getattr(disc_cfg, "scan_interval_min", 30)

            # Pre-market scan: 8am ET Mon-Fri
            self._scheduler.add_job(
                self.job_discovery_scan,
                CronTrigger(hour=pre_market_hour, minute=0, day_of_week="mon-fri", timezone=ET),
                id="discovery_premarket",
                name="Pre-market discovery scan",
                max_instances=1,
                misfire_grace_time=300,
            )

            # Intraday scan: every N minutes (market hours guard inside job)
            self._scheduler.add_job(
                self.job_discovery_scan,
                IntervalTrigger(minutes=scan_interval, timezone=ET),
                id="discovery_intraday",
                name="Intraday discovery scan",
                max_instances=1,
                misfire_grace_time=60,
            )

        logger.info(f"[Scheduler] Registered jobs: {self.get_jobs()}")

    # ── Jobs ──────────────────────────────────────────────────────────────────

    def job_data_sync(self) -> None:
        """
        Fetch latest OHLCV bars from Polygon for all tradeable assets.
        Skips on weekends.
        """
        now = datetime.now(ET)
        if _is_weekend():
            logger.debug("[Scheduler] data_sync skipped — weekend")
            return

        logger.info("[Scheduler] Running data_sync job")
        from src.discovery.universe_manager import UniverseManager
        tickers = list(dict.fromkeys(
            self.config.assets.all_symbols + UniverseManager().get_tradeable_universe()
        ))

        # OHLCV sync
        try:
            from src.ingestion.market_data_service import MarketDataService
            svc = MarketDataService()
            for ticker in tickers:
                try:
                    df = svc.get_latest_bars(ticker)
                    if not df.empty:
                        self.store.save_ohlcv(ticker, df)
                        logger.debug(f"[Scheduler] Synced {ticker}: {len(df)} bars")
                except Exception as exc:
                    logger.error(f"[Scheduler] data_sync failed for {ticker}: {exc}")
        except Exception as exc:
            logger.error(f"[Scheduler] data_sync job failed: {exc}")

        # News sync (Finnhub — runs after OHLCV, failures don't block pipeline)
        try:
            import time as _time
            from src.ingestion.news_service import NewsService
            news_svc = NewsService()
            for ticker in tickers:
                try:
                    news_df = news_svc.fetch_company_news(ticker, days_back=2)
                    if not news_df.empty:
                        self.store.save_news(ticker, news_df)
                        logger.debug(f"[Scheduler] News synced {ticker}: {len(news_df)} articles")
                    _time.sleep(0.5)  # Finnhub free tier: 60 req/min
                except Exception as exc:
                    logger.warning(f"[Scheduler] News sync failed for {ticker}: {exc}")
        except Exception as exc:
            logger.error(f"[Scheduler] news_sync job failed: {exc}")

    def job_signal_pipeline(self) -> None:
        """
        Run the full signal pipeline for all tradeable tickers.
        Only executes during market hours.
        """
        if not _is_market_hours():
            logger.debug("[Scheduler] signal_pipeline skipped — outside market hours")
            return

        logger.info("[Scheduler] Running signal_pipeline job")

        from src.discovery.universe_manager import UniverseManager
        tickers  = UniverseManager().get_tradeable_universe()
        data_map: dict  = {}
        price_map: dict = {}

        for ticker in tickers:
            df = self.store.load_ohlcv(ticker)
            if df.empty:
                logger.warning(f"[Scheduler] No OHLCV data for {ticker} — skipping")
                continue
            data_map[ticker]  = df
            price_map[ticker] = float(df["close"].iloc[-1])

        if not data_map:
            logger.warning("[Scheduler] signal_pipeline: no data available")
            return

        portfolio = self._build_portfolio_state()

        try:
            decisions = self.engine.decide_all(
                tickers=list(data_map),
                data_map=data_map,
                price_map=price_map,
                portfolio=portfolio,
            )

            if decisions:
                df_out = self.engine.decisions_to_dataframe(decisions)
                self.store.save_signals(df_out)
                logger.info(f"[Scheduler] Pipeline: {len(decisions)} decision(s) saved")

                # Notify for each approved decision
                try:
                    from src.notifications import notify
                    for d in decisions:
                        notify(
                            f"{d.ticker} {d.direction} @ ${d.entry_price:,.2f} "
                            f"({d.strategy}, conf={d.confidence:.2f})",
                            level="trade" if d.direction == "BUY" else "sell",
                            ticker=d.ticker,
                            data={"confidence": d.confidence, "strategy": d.strategy},
                        )
                except Exception:
                    pass

                if self.on_decisions:
                    self.on_decisions(decisions)
            else:
                logger.info("[Scheduler] Pipeline: no approved decisions this cycle")

        except Exception as exc:
            logger.error(f"[Scheduler] signal_pipeline failed: {exc}")

    def job_position_check(self) -> None:
        """
        Fast loop: check trailing stops and take-profit targets for held positions.
        Only executes during market hours. Requires an executor to be injected.
        """
        if not _is_market_hours():
            return

        if self.executor is None:
            logger.debug("[Scheduler] position_check skipped — no executor")
            return

        logger.debug("[Scheduler] Running position_check job")

        try:
            # Get current prices for held tickers only
            price_map: dict = {}
            trailing_stops = getattr(self.executor, "_trailing_stops", {})
            take_profits = getattr(self.executor, "_take_profits", {})
            tickers_to_check = set(trailing_stops.keys()) | set(take_profits.keys())

            if not tickers_to_check:
                return

            # Use live Alpaca prices — Parquet is only updated at 5am and would
            # give yesterday's close, making stop/TP checks meaningless intraday.
            from src.ingestion.alpaca_client import AlpacaClient
            alpaca = AlpacaClient()
            for ticker in tickers_to_check:
                price = alpaca.get_latest_price(ticker)
                if price is not None:
                    price_map[ticker] = price

            if not price_map:
                return

            # Check take-profits first (higher priority)
            tp_triggered = self.executor.check_take_profits(price_map)
            for ticker in tp_triggered:
                logger.info(f"[Scheduler] Take-profit triggered for {ticker}")
                from src.notifications import notify
                price = price_map.get(ticker, 0)
                notify(f"{ticker} take-profit hit @ ${price:,.2f}",
                       level="tp", ticker=ticker)

            # Check trailing stops
            stop_triggered = self.executor.update_trailing_stops(price_map)
            for ticker in stop_triggered:
                logger.info(f"[Scheduler] Trailing stop triggered for {ticker}")
                from src.notifications import notify
                price = price_map.get(ticker, 0)
                notify(f"{ticker} trailing stop triggered @ ${price:,.2f}",
                       level="stop", ticker=ticker)

        except Exception as exc:
            logger.error(f"[Scheduler] position_check failed: {exc}")

    def job_drift_check(self) -> None:
        """
        Run ModelMonitor drift detection across all strategies.
        Logs a full report and calls on_drift callback if provided.
        """
        logger.info("[Scheduler] Running drift_check job")
        try:
            report = self.monitor.check_drift()
            logger.info(f"[Scheduler] Drift check complete:\n{report.summary()}")

            # Notify if drift alerts found
            if report.has_alerts:
                from src.notifications import notify
                for alert in report.alerts:
                    level = "critical" if alert.severity == "CRITICAL" else "warning"
                    notify(f"Drift: {alert.name} — {alert.message}", level=level)

            if self.on_drift:
                self.on_drift(report)

        except Exception as exc:
            logger.error(f"[Scheduler] drift_check failed: {exc}")

    def job_discovery_scan(self) -> None:
        """
        Run the discovery pipeline: scan news/Reddit for trending tickers,
        screen them, add candidates, and expire stale ones.
        Only fires during market hours for intraday trigger.
        """
        if not _is_market_hours():
            # Allow pre-market (8am) trigger to pass; guard only truly off-hours
            now = datetime.now(ET)
            if now.time() < time(7, 30) or now.time() > time(17, 0):
                logger.debug("[Scheduler] discovery_scan skipped — outside trading window")
                return

        logger.info("[Scheduler] Running discovery_scan job")

        try:
            from src.discovery.trend_scanner import TrendScanner
            from src.discovery.stock_screener import StockScreener
            from src.discovery.universe_manager import UniverseManager

            scanner  = TrendScanner()
            screener = StockScreener()
            manager  = UniverseManager()

            candidates = scanner.scan()
            screened   = screener.screen(candidates)
            passed     = [s for s in screened if s.passed]
            added      = manager.add_candidates(passed)
            manager.expire_old(days=14)

            logger.info(
                f"[Discovery] Scan complete: {len(candidates)} trending, "
                f"{len(passed)} passed screening, {added} new candidates added"
            )

            # Notify for new discoveries
            if added > 0:
                from src.notifications import notify
                for s in passed[:5]:  # notify top 5
                    notify(
                        f"{s.ticker} discovered — {s.mention_spike:.1f}x mention spike, "
                        f"sentiment {s.avg_sentiment:+.2f}",
                        level="discovery", ticker=s.ticker,
                    )
                if added > 5:
                    notify(f"...and {added - 5} more new candidates", level="discovery")

            if self.on_discovery:
                self.on_discovery(passed)

        except Exception as exc:
            logger.error(f"[Scheduler] discovery_scan failed: {exc}")

    def job_recalibrate(self) -> None:
        """
        Quarterly recalibration: re-run backtests on fresh data and update
        ModelMonitor baselines.
        """
        logger.info("[Scheduler] Running recalibrate job")
        try:
            from lean.lean_bridge import LEANBridge
            bridge = LEANBridge()
            new_baselines: dict = {}

            for ticker in self.config.assets.stocks[:3]:   # use first 3 stocks as proxy
                df = self.store.load_ohlcv(ticker)
                if df.empty:
                    continue
                for strategy in ["momentum", "trend_following", "volatility_breakout"]:
                    try:
                        result = bridge.run_python_backtest(strategy, df, ticker)
                        if strategy not in new_baselines:
                            new_baselines[strategy] = []
                        new_baselines[strategy].append(result.win_rate)
                    except Exception as exc:
                        logger.warning(f"[Scheduler] Recalibration backtest failed "
                                       f"{strategy}/{ticker}: {exc}")

            # Average win-rates across tickers per strategy
            averaged = {s: float(pd.Series(v).mean())
                        for s, v in new_baselines.items() if v}
            if averaged:
                self.monitor.recalibrate(averaged)
                logger.info(f"[Scheduler] Recalibration complete: {averaged}")
            else:
                logger.warning("[Scheduler] Recalibration: no results — baselines unchanged")

        except Exception as exc:
            logger.error(f"[Scheduler] recalibrate job failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_portfolio_state(self) -> PortfolioState:
        """
        Build a PortfolioState from the audit log.
        Falls back to empty portfolio if no audit data exists.
        """
        try:
            audit = self.store.load_audit()
            total_value = 100_000.0  # paper account default

            if audit.empty:
                return PortfolioState(
                    total_value_usd=total_value,
                    cash_usd=total_value,
                    open_positions=0,
                    crypto_exposure_usd=0.0,
                    daily_pnl_pct=0.0,
                    peak_value_usd=total_value,
                )

            # Count open positions: BUYs without matching SELLs
            open_count = 0
            if "direction" in audit.columns and "ticker" in audit.columns:
                buys  = set(audit[audit["direction"] == "BUY"]["ticker"].unique())
                sells = set(audit[audit["direction"] == "SELL"]["ticker"].unique())
                open_count = len(buys - sells)

            return PortfolioState(
                total_value_usd=total_value,
                cash_usd=total_value,
                open_positions=open_count,
                crypto_exposure_usd=0.0,
                daily_pnl_pct=0.0,
                peak_value_usd=total_value,
            )
        except Exception as exc:
            logger.warning(f"[Scheduler] Could not build portfolio state: {exc} — using default")
            return PortfolioState(
                total_value_usd=100_000.0,
                cash_usd=100_000.0,
                open_positions=0,
                crypto_exposure_usd=0.0,
                daily_pnl_pct=0.0,
                peak_value_usd=100_000.0,
            )
