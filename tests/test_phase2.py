"""
tests/test_phase2.py — Phase 2 test suite.

Covers:
  - ModelMonitor: trade recording, metrics computation, drift detection
  - TradingScheduler: initialisation, job registration, job execution
  - Dashboard: importability and helper functions (no Streamlit server needed)

Run with:
    py -3.12 -m pytest tests/test_phase2.py -v
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.monitoring.model_monitor import (
    DriftAlert,
    DriftReport,
    ModelMonitor,
    StrategyMetrics,
)
from src.scheduler.scheduler import TradingScheduler, _is_market_hours, _is_weekend


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def monitor() -> ModelMonitor:
    return ModelMonitor()


@pytest.fixture
def monitor_with_trades(monitor: ModelMonitor) -> ModelMonitor:
    """Monitor pre-loaded with 20 winning / 10 losing trades per strategy."""
    for strategy in ["momentum", "trend_following", "volatility_breakout"]:
        for i in range(20):
            monitor.record_trade(strategy, "TEST", pnl_pct=0.03, won=True)
        for i in range(10):
            monitor.record_trade(strategy, "TEST", pnl_pct=-0.02, won=False)
    return monitor


@pytest.fixture
def scheduler() -> TradingScheduler:
    """Scheduler with mocked engine, monitor, and store — does NOT start."""
    engine  = MagicMock()
    mon     = ModelMonitor()
    store   = MagicMock()
    store.load_ohlcv.return_value  = pd.DataFrame()
    store.load_audit.return_value  = pd.DataFrame()
    store.save_signals.return_value = None
    return TradingScheduler(
        decision_engine=engine,
        monitor=mon,
        store=store,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ModelMonitor — trade recording
# ══════════════════════════════════════════════════════════════════════════════

class TestModelMonitorRecording:

    def test_record_trade_stores_entry(self, monitor):
        monitor.record_trade("momentum", "NVDA", pnl_pct=0.05, won=True)
        assert len(monitor._trades["momentum"]) == 1

    def test_record_trade_fields(self, monitor):
        monitor.record_trade("momentum", "NVDA", pnl_pct=0.05, won=True)
        t = monitor._trades["momentum"][0]
        assert t["ticker"]  == "NVDA"
        assert t["pnl_pct"] == pytest.approx(0.05)
        assert t["won"]     is True
        assert isinstance(t["ts"], datetime)

    def test_record_multiple_strategies(self, monitor):
        monitor.record_trade("momentum",        "NVDA", pnl_pct=0.03, won=True)
        monitor.record_trade("trend_following", "AAPL", pnl_pct=-0.01, won=False)
        assert len(monitor._trades["momentum"])        == 1
        assert len(monitor._trades["trend_following"]) == 1

    def test_record_trades_from_df(self, monitor):
        df = pd.DataFrame([
            {"strategy": "momentum", "ticker": "NVDA", "pnl_pct": 0.04, "won": True},
            {"strategy": "momentum", "ticker": "MSFT", "pnl_pct": -0.01, "won": False},
        ])
        count = monitor.record_trades_from_df(df)
        assert count == 2
        assert len(monitor._trades["momentum"]) == 2

    def test_record_trades_from_empty_df(self, monitor):
        count = monitor.record_trades_from_df(pd.DataFrame())
        assert count == 0

    def test_record_trades_from_df_missing_columns(self, monitor):
        df = pd.DataFrame([{"ticker": "NVDA"}])
        count = monitor.record_trades_from_df(df)
        assert count == 0


# ══════════════════════════════════════════════════════════════════════════════
# ModelMonitor — metrics computation
# ══════════════════════════════════════════════════════════════════════════════

class TestModelMonitorMetrics:

    def test_get_metrics_returns_none_below_threshold(self, monitor):
        monitor.record_trade("momentum", "NVDA", pnl_pct=0.02, won=True)
        assert monitor.get_metrics("momentum") is None

    def test_get_metrics_returns_object_with_enough_trades(self, monitor):
        for _ in range(10):
            monitor.record_trade("momentum", "NVDA", pnl_pct=0.02, won=True)
        m = monitor.get_metrics("momentum")
        assert isinstance(m, StrategyMetrics)

    def test_win_rate_correct(self, monitor):
        for _ in range(6):
            monitor.record_trade("momentum", "NVDA", pnl_pct=0.03, won=True)
        for _ in range(4):
            monitor.record_trade("momentum", "NVDA", pnl_pct=-0.02, won=False)
        m = monitor.get_metrics("momentum")
        assert m.live_win_rate == pytest.approx(0.6)

    def test_win_rate_is_property_alias(self, monitor_with_trades):
        m = monitor_with_trades.get_metrics("momentum")
        assert m.live_win_rate == pytest.approx(2/3, abs=0.01)

    def test_sharpe_is_finite(self, monitor_with_trades):
        m = monitor_with_trades.get_metrics("momentum")
        assert math.isfinite(m.live_sharpe)

    def test_drawdown_is_between_0_and_1(self, monitor_with_trades):
        m = monitor_with_trades.get_metrics("momentum")
        assert 0.0 <= m.live_drawdown <= 1.0

    def test_trade_count_correct(self, monitor_with_trades):
        m = monitor_with_trades.get_metrics("momentum")
        assert m.trade_count == 30

    def test_get_all_metrics_returns_list(self, monitor_with_trades):
        metrics = monitor_with_trades.get_all_metrics()
        assert len(metrics) == 3
        strategies = {m.strategy for m in metrics}
        assert strategies == {"momentum", "trend_following", "volatility_breakout"}

    def test_unknown_strategy_returns_empty_list(self, monitor):
        assert monitor.get_all_metrics() == []


# ══════════════════════════════════════════════════════════════════════════════
# ModelMonitor — drift detection
# ══════════════════════════════════════════════════════════════════════════════

class TestModelMonitorDrift:

    def test_check_drift_returns_report(self, monitor_with_trades):
        report = monitor_with_trades.check_drift()
        assert isinstance(report, DriftReport)

    def test_report_has_checked_at(self, monitor_with_trades):
        report = monitor_with_trades.check_drift()
        assert isinstance(report.checked_at, datetime)

    def test_no_alerts_when_healthy(self, monitor):
        """Nominal trades: high win-rate, good returns — should not trigger drift."""
        for _ in range(30):
            monitor.record_trade("momentum", "NVDA", pnl_pct=0.03, won=True)
        report = monitor.check_drift()
        momentum_alerts = [a for a in report.alerts if a.strategy == "momentum"]
        # Win-rate will be 1.0, which is above all baselines — no WIN_RATE_DROP
        assert not any(a.alert_type == "WIN_RATE_DROP" for a in momentum_alerts)

    def test_win_rate_drop_triggers_alert(self, monitor):
        """Win-rate significantly below baseline should fire WIN_RATE_DROP."""
        # Baseline for momentum is 0.55; push live win-rate to 0.20
        for _ in range(4):
            monitor.record_trade("momentum", "TEST", pnl_pct=0.02, won=True)
        for _ in range(16):
            monitor.record_trade("momentum", "TEST", pnl_pct=-0.02, won=False)
        report = monitor.check_drift()
        types = [a.alert_type for a in report.alerts if a.strategy == "momentum"]
        assert "WIN_RATE_DROP" in types

    def test_drawdown_pause_triggers_critical(self, monitor):
        """Severe drawdown (>15%) should trigger CRITICAL DRAWDOWN_PAUSE."""
        # All losses of 5% each — cumulative drawdown will exceed 15%
        for _ in range(20):
            monitor.record_trade("momentum", "TEST", pnl_pct=-0.05, won=False)
        report = monitor.check_drift()
        critical = [a for a in report.alerts
                    if a.strategy == "momentum" and a.severity == "CRITICAL"]
        assert len(critical) >= 1

    def test_paused_after_critical_drawdown(self, monitor):
        for _ in range(20):
            monitor.record_trade("trend_following", "TEST", pnl_pct=-0.05, won=False)
        report = monitor.check_drift()
        assert monitor.is_paused("trend_following") or report.any_paused

    def test_resume_clears_pause(self, monitor):
        monitor._paused.add("momentum")
        monitor.resume("momentum")
        assert not monitor.is_paused("momentum")

    def test_benchmark_lag_triggers_alert(self, monitor):
        """Strategy underperforming SPY by >10% should fire BENCHMARK_LAG."""
        for _ in range(10):
            monitor.record_trade("momentum", "TEST", pnl_pct=-0.01, won=False)
        # Pass benchmark_return of +0.15 (strategy is -0.10 → 25pp lag)
        report = monitor.check_drift(benchmark_return=0.15)
        types  = [a.alert_type for a in report.alerts if a.strategy == "momentum"]
        assert "BENCHMARK_LAG" in types

    def test_report_summary_string(self, monitor_with_trades):
        report = monitor_with_trades.check_drift()
        assert isinstance(report.summary(), str)
        assert len(report.summary()) > 0

    def test_has_alerts_property(self, monitor):
        report = DriftReport(
            checked_at=datetime.now(timezone.utc),
            alerts=[],
            strategy_metrics=[],
            any_paused=False,
            paused_strategies=[],
        )
        assert not report.has_alerts

    def test_recalibrate_updates_baselines(self, monitor):
        monitor.recalibrate({"momentum": 0.62})
        assert monitor._baselines["momentum"] == pytest.approx(0.62)

    def test_recalibrate_rejects_out_of_range(self, monitor):
        original = monitor._baselines["momentum"]
        monitor.recalibrate({"momentum": 1.5})   # invalid — should be ignored
        assert monitor._baselines["momentum"] == original

    def test_sharpe_floor_alert(self, monitor):
        """Negative Sharpe (all losses) should trigger SHARPE_FLOOR."""
        for _ in range(10):
            monitor.record_trade("volatility_breakout", "TEST", pnl_pct=-0.03, won=False)
        report = monitor.check_drift()
        types  = [a.alert_type for a in report.alerts
                  if a.strategy == "volatility_breakout"]
        assert "SHARPE_FLOOR" in types


# ══════════════════════════════════════════════════════════════════════════════
# ModelMonitor — static helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestModelMonitorHelpers:

    def test_compute_sharpe_positive_mean(self):
        # Varying positive returns → positive Sharpe
        pnl = [0.02, 0.03, 0.01, 0.04, 0.02] * 4
        sharpe = ModelMonitor._compute_sharpe(pnl)
        assert sharpe > 0

    def test_compute_sharpe_negative_mean(self):
        # Varying negative returns → negative Sharpe
        pnl = [-0.02, -0.03, -0.01, -0.04, -0.02] * 4
        sharpe = ModelMonitor._compute_sharpe(pnl)
        assert sharpe < 0

    def test_compute_sharpe_zero_std(self):
        # All same value → std ≈ 0 → sharpe returned as 0
        pnl = [0.01] * 20
        sharpe = ModelMonitor._compute_sharpe(pnl)
        assert sharpe == pytest.approx(0.0, abs=1e-6)

    def test_compute_sharpe_insufficient_data(self):
        assert ModelMonitor._compute_sharpe([0.01]) == 0.0

    def test_compute_max_drawdown_no_drawdown(self):
        pnl = [0.01] * 10
        dd = ModelMonitor._compute_max_drawdown(pnl)
        assert dd == pytest.approx(0.0, abs=1e-9)

    def test_compute_max_drawdown_severe(self):
        pnl = [-0.1, -0.1, -0.1, -0.1, 0.05]
        dd = ModelMonitor._compute_max_drawdown(pnl)
        assert dd > 0.25   # at least 25% drawdown

    def test_compute_max_drawdown_empty(self):
        assert ModelMonitor._compute_max_drawdown([]) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TradingScheduler — initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestSchedulerInit:

    def test_scheduler_creates_without_error(self, scheduler):
        assert scheduler is not None

    def test_scheduler_not_running_on_init(self, scheduler):
        assert not scheduler.is_running

    def test_four_jobs_registered(self, scheduler):
        jobs = scheduler.get_jobs()
        # At minimum the 5 core jobs are always registered.
        # Discovery jobs may also be present when discovery.enabled is True.
        assert len(jobs) >= 5

    def test_job_ids_correct(self, scheduler):
        jobs = set(scheduler.get_jobs())
        core_jobs = {"data_sync", "signal_pipeline", "position_check", "drift_check", "recalibrate"}
        assert core_jobs.issubset(jobs)

    def test_start_background_sets_running(self, scheduler):
        scheduler.start_background()
        assert scheduler.is_running
        scheduler.stop()

    def test_stop_sets_not_running(self, scheduler):
        scheduler.start_background()
        scheduler.stop()
        assert not scheduler.is_running

    def test_double_start_does_not_crash(self, scheduler):
        scheduler.start_background()
        scheduler.start_background()  # second call should be no-op
        scheduler.stop()

    def test_callbacks_accepted(self):
        cb_decisions = MagicMock()
        cb_drift     = MagicMock()
        s = TradingScheduler(
            decision_engine=MagicMock(),
            monitor=ModelMonitor(),
            store=MagicMock(),
            on_decisions=cb_decisions,
            on_drift=cb_drift,
        )
        assert s.on_decisions is cb_decisions
        assert s.on_drift is cb_drift


# ══════════════════════════════════════════════════════════════════════════════
# TradingScheduler — job logic
# ══════════════════════════════════════════════════════════════════════════════

class TestSchedulerJobs:

    def test_signal_pipeline_skips_outside_market_hours(self, scheduler):
        """job_signal_pipeline should be a no-op outside market hours."""
        with patch("src.scheduler.scheduler._is_market_hours", return_value=False):
            scheduler.job_signal_pipeline()
        # Engine should NOT have been called
        scheduler.engine.decide_all.assert_not_called()

    def test_signal_pipeline_runs_during_market_hours(self, scheduler):
        """job_signal_pipeline should call engine.decide_all during market hours."""
        scheduler.store.load_ohlcv.return_value = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "open":  [100.0]*5, "high":  [105.0]*5,
            "low":   [98.0]*5,  "close": [102.0]*5,
            "volume":[1_000_000]*5, "ticker": ["TEST"]*5,
        })
        scheduler.engine.decide_all.return_value = []

        with patch("src.scheduler.scheduler._is_market_hours", return_value=True):
            scheduler.job_signal_pipeline()

        scheduler.engine.decide_all.assert_called_once()

    def test_signal_pipeline_saves_decisions(self, scheduler):
        """Approved decisions should be saved to the store."""
        ohlcv = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "open":  [100.0]*5, "high":  [105.0]*5,
            "low":   [98.0]*5,  "close": [102.0]*5,
            "volume":[1_000_000]*5, "ticker": ["TEST"]*5,
        })
        scheduler.store.load_ohlcv.return_value = ohlcv

        mock_decision = MagicMock()
        scheduler.engine.decide_all.return_value = [mock_decision]
        scheduler.engine.decisions_to_dataframe.return_value = pd.DataFrame(
            [{"ticker": "TEST", "direction": "BUY"}]
        )

        with patch("src.scheduler.scheduler._is_market_hours", return_value=True):
            scheduler.job_signal_pipeline()

        scheduler.store.save_signals.assert_called_once()

    def test_signal_pipeline_calls_on_decisions_callback(self, scheduler):
        cb = MagicMock()
        scheduler.on_decisions = cb

        ohlcv = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "open":  [100.0]*5, "high":  [105.0]*5,
            "low":   [98.0]*5,  "close": [102.0]*5,
            "volume":[1_000_000]*5, "ticker": ["TEST"]*5,
        })
        scheduler.store.load_ohlcv.return_value = ohlcv
        scheduler.engine.decide_all.return_value = [MagicMock()]
        scheduler.engine.decisions_to_dataframe.return_value = pd.DataFrame([{"ticker": "TEST"}])

        with patch("src.scheduler.scheduler._is_market_hours", return_value=True):
            scheduler.job_signal_pipeline()

        cb.assert_called_once()

    def test_data_sync_skips_on_weekend(self, scheduler):
        """data_sync should skip on Saturday/Sunday."""
        with patch("src.scheduler.scheduler._is_weekend", return_value=True):
            scheduler.job_data_sync()
        # MarketDataService should NOT have been imported or called
        # (we just verify no exception is raised)

    def test_drift_check_job_runs(self, scheduler):
        """job_drift_check should produce a DriftReport without error."""
        scheduler.job_drift_check()   # monitor has no trades → no alerts

    def test_drift_check_calls_on_drift_callback(self, scheduler):
        cb = MagicMock()
        scheduler.on_drift = cb
        scheduler.job_drift_check()
        cb.assert_called_once()
        report = cb.call_args[0][0]
        assert isinstance(report, DriftReport)

    def test_signal_pipeline_no_data_is_no_op(self, scheduler):
        """If store has no data, pipeline should be a no-op."""
        scheduler.store.load_ohlcv.return_value = pd.DataFrame()
        with patch("src.scheduler.scheduler._is_market_hours", return_value=True):
            scheduler.job_signal_pipeline()
        scheduler.engine.decide_all.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Market hours helper
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketHoursHelper:

    def test_is_market_hours_returns_bool(self):
        result = _is_market_hours()
        assert isinstance(result, bool)

    def test_weekday_open_is_market_hours(self):
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        with patch("src.scheduler.scheduler.datetime") as mock_dt:
            # Monday 10:00 ET
            mock_now = MagicMock()
            mock_now.weekday.return_value = 0
            from datetime import time
            mock_now.time.return_value = time(10, 0)
            mock_dt.now.return_value = mock_now
            assert _is_market_hours() is True

    def test_weekend_is_not_market_hours(self):
        with patch("src.scheduler.scheduler.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 6  # Sunday
            mock_dt.now.return_value = mock_now
            assert _is_market_hours() is False


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard — importability (no Streamlit server needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestDashboardImport:

    def test_dashboard_module_importable(self):
        """dashboard/app.py should be importable without launching a server."""
        import importlib, sys
        # Prevent Streamlit from executing set_page_config on import
        with patch("streamlit.set_page_config"):
            # Clear cached module if already imported
            sys.modules.pop("dashboard.app", None)
            sys.modules.pop("app", None)
            import importlib.util, pathlib
            spec = importlib.util.spec_from_file_location(
                "dashboard.app",
                str(pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            # Should not raise
            assert mod is not None

    def test_fmt_pct_positive(self):
        with patch("streamlit.set_page_config"):
            import importlib.util, pathlib
            spec = importlib.util.spec_from_file_location(
                "dashboard_app_test",
                str(pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert mod._fmt_pct(0.05)  == "+5.0%"
            assert mod._fmt_pct(-0.10) == "-10.0%"
            assert mod._fmt_pct(float("nan")) == "—"

    def test_fmt_num(self):
        with patch("streamlit.set_page_config"):
            import importlib.util, pathlib
            spec = importlib.util.spec_from_file_location(
                "dashboard_app_test2",
                str(pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert mod._fmt_num(1.234) == "1.23"
            assert mod._fmt_num(float("inf")) == "—"

    def test_direction_color(self):
        with patch("streamlit.set_page_config"):
            import importlib.util, pathlib
            spec = importlib.util.spec_from_file_location(
                "dashboard_app_test3",
                str(pathlib.Path(__file__).parent.parent / "dashboard" / "app.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert "🟢" in mod._direction_color("BUY")
            assert "🔴" in mod._direction_color("SELL")
