"""
test_phase3.py — Tests for Phase 3: Live Trading.

Covers:
  - LiveTrader pre-flight checklist (all checks pass / each check fails)
  - S3Sync upload, download, sync, restore, disabled behaviour
  - CloudWatchReporter metric publishing (all push methods)

All AWS calls are mocked — no real AWS credentials required.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

# ── LiveTrader ────────────────────────────────────────────────────────────────

from src.live.live_trader import LiveTrader, PreflightReport, PreflightCheck


def _make_alpaca(portfolio_value=50_000.0, trading_blocked=False, validate_ok=True):
    """Factory: mock AlpacaClient."""
    client = MagicMock()
    client.validate_connection.return_value = validate_ok
    client.get_account.return_value = {
        "portfolio_value": portfolio_value,
        "cash": portfolio_value * 0.3,
        "buying_power": portfolio_value * 0.5,
        "equity": portfolio_value,
        "trading_blocked": trading_blocked,
    }
    return client


def _make_s3(reachable=True):
    """Factory: mock S3Sync."""
    s3 = MagicMock()
    s3.validate_connection.return_value = reachable
    return s3


def _live_config_patch():
    """Patch get_config to return live mode with safe risk settings."""
    from src.config import AppConfig, TradingConfig, RiskConfig, DataConfig
    cfg = AppConfig(
        trading=TradingConfig(mode="live"),
        risk=RiskConfig(
            max_position_pct=0.05,
            stop_loss_pct=0.07,
            min_confidence=0.60,
            max_open_positions=5,
        ),
        data=DataConfig(sync_to_s3=False),
    )
    return cfg


class TestPreflightReport:
    def test_all_passed_true_when_all_pass(self):
        report = PreflightReport(checks=[
            PreflightCheck("a", True, "ok"),
            PreflightCheck("b", True, "ok"),
        ])
        assert report.all_passed is True

    def test_all_passed_false_when_any_fail(self):
        report = PreflightReport(checks=[
            PreflightCheck("a", True, "ok"),
            PreflightCheck("b", False, "fail"),
        ])
        assert report.all_passed is False

    def test_failures_filters_correctly(self):
        report = PreflightReport(checks=[
            PreflightCheck("a", True, "ok"),
            PreflightCheck("b", False, "fail"),
            PreflightCheck("c", False, "fail2"),
        ])
        assert len(report.failures) == 2

    def test_summary_contains_check_names(self):
        report = PreflightReport(checks=[
            PreflightCheck("config_mode", True, "ok"),
            PreflightCheck("api_keys", False, "missing"),
        ])
        summary = report.summary()
        assert "config_mode" in summary
        assert "api_keys" in summary

    def test_summary_shows_pass_fail(self):
        report = PreflightReport(checks=[
            PreflightCheck("x", True, "good"),
            PreflightCheck("y", False, "bad"),
        ])
        summary = report.summary()
        assert "PASS" in summary
        assert "FAIL" in summary


class TestLiveTraderConfigCheck:
    def test_passes_when_mode_is_live(self):
        with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
            trader = LiveTrader(alpaca_client=_make_alpaca(), s3_sync=None)
            check = trader._check_config_mode()
        assert check.passed is True

    def test_fails_when_mode_is_paper(self):
        from src.config import AppConfig, TradingConfig
        cfg = AppConfig(trading=TradingConfig(mode="paper"))
        with patch("src.live.live_trader.reload_config", return_value=cfg):
            trader = LiveTrader(alpaca_client=_make_alpaca())
            check = trader._check_config_mode()
        assert check.passed is False
        assert "paper" in check.message


class TestLiveTraderApiKeyCheck:
    def test_passes_with_valid_keys(self):
        with patch("src.secrets.Secrets.alpaca_api_key", return_value="ABCDEF123456"):
            with patch("src.secrets.Secrets.alpaca_secret_key", return_value="XYZXYZ987654"):
                trader = LiveTrader()
                check = trader._check_api_keys()
        assert check.passed is True

    def test_fails_with_placeholder_key(self):
        with patch("src.secrets.Secrets.alpaca_api_key", return_value="your_key_here"):
            with patch("src.secrets.Secrets.alpaca_secret_key", return_value="valid_secret_key"):
                trader = LiveTrader()
                check = trader._check_api_keys()
        assert check.passed is False

    def test_fails_with_short_key(self):
        with patch("src.secrets.Secrets.alpaca_api_key", return_value="abc"):
            with patch("src.secrets.Secrets.alpaca_secret_key", return_value="validkey123"):
                trader = LiveTrader()
                check = trader._check_api_keys()
        assert check.passed is False

    def test_fails_when_secret_raises(self):
        with patch("src.secrets.Secrets.alpaca_api_key", side_effect=ValueError("missing")):
            trader = LiveTrader()
            check = trader._check_api_keys()
        assert check.passed is False


class TestLiveTraderAlpacaConnectionCheck:
    def test_passes_when_connection_ok(self):
        trader = LiveTrader(alpaca_client=_make_alpaca(validate_ok=True))
        check = trader._check_alpaca_connection()
        assert check.passed is True

    def test_fails_when_connection_fails(self):
        trader = LiveTrader(alpaca_client=_make_alpaca(validate_ok=False))
        check = trader._check_alpaca_connection()
        assert check.passed is False

    def test_fails_when_no_client_injected(self):
        trader = LiveTrader(alpaca_client=None)
        check = trader._check_alpaca_connection()
        assert check.passed is False

    def test_fails_when_alpaca_raises(self):
        client = MagicMock()
        client.validate_connection.side_effect = ConnectionError("timeout")
        trader = LiveTrader(alpaca_client=client)
        check = trader._check_alpaca_connection()
        assert check.passed is False


class TestLiveTraderPortfolioCheck:
    def test_passes_above_minimum(self):
        trader = LiveTrader(alpaca_client=_make_alpaca(portfolio_value=10_000))
        check = trader._check_portfolio_value()
        assert check.passed is True

    def test_fails_below_minimum(self):
        trader = LiveTrader(alpaca_client=_make_alpaca(portfolio_value=500))
        check = trader._check_portfolio_value()
        assert check.passed is False

    def test_passes_exactly_at_minimum(self):
        trader = LiveTrader(alpaca_client=_make_alpaca(portfolio_value=LiveTrader.MIN_PORTFOLIO_VALUE))
        check = trader._check_portfolio_value()
        assert check.passed is True

    def test_fails_with_no_client(self):
        trader = LiveTrader(alpaca_client=None)
        check = trader._check_portfolio_value()
        assert check.passed is False


class TestLiveTraderAccountBlockedCheck:
    def test_passes_when_not_blocked(self):
        trader = LiveTrader(alpaca_client=_make_alpaca(trading_blocked=False))
        check = trader._check_account_not_blocked()
        assert check.passed is True

    def test_fails_when_blocked(self):
        trader = LiveTrader(alpaca_client=_make_alpaca(trading_blocked=True))
        check = trader._check_account_not_blocked()
        assert check.passed is False

    def test_fails_with_no_client(self):
        trader = LiveTrader(alpaca_client=None)
        check = trader._check_account_not_blocked()
        assert check.passed is False


class TestLiveTraderRiskConfigCheck:
    def test_passes_with_safe_config(self):
        with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
            trader = LiveTrader()
            check = trader._check_risk_config()
        assert check.passed is True

    def test_fails_when_position_size_too_large(self):
        from src.config import AppConfig, RiskConfig, TradingConfig
        cfg = AppConfig(
            trading=TradingConfig(mode="live"),
            risk=RiskConfig(max_position_pct=0.20),
        )
        with patch("src.live.live_trader.reload_config", return_value=cfg):
            trader = LiveTrader()
            check = trader._check_risk_config()
        assert check.passed is False
        assert "max_position_pct" in check.message

    def test_fails_when_min_confidence_too_low(self):
        from src.config import AppConfig, RiskConfig, TradingConfig
        cfg = AppConfig(
            trading=TradingConfig(mode="live"),
            risk=RiskConfig(min_confidence=0.30),
        )
        with patch("src.live.live_trader.reload_config", return_value=cfg):
            trader = LiveTrader()
            check = trader._check_risk_config()
        assert check.passed is False


class TestLiveTraderS3Check:
    def test_skipped_when_s3_disabled(self):
        with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
            trader = LiveTrader()
            check = trader._check_s3_if_enabled()
        assert check.passed is True
        assert "disabled" in check.message

    def test_passes_when_s3_reachable(self):
        from src.config import AppConfig, TradingConfig, DataConfig
        cfg = AppConfig(
            trading=TradingConfig(mode="live"),
            data=DataConfig(sync_to_s3=True),
        )
        with patch("src.live.live_trader.reload_config", return_value=cfg):
            trader = LiveTrader(s3_sync=_make_s3(reachable=True))
            check = trader._check_s3_if_enabled()
        assert check.passed is True

    def test_fails_when_s3_not_reachable(self):
        from src.config import AppConfig, TradingConfig, DataConfig
        cfg = AppConfig(
            trading=TradingConfig(mode="live"),
            data=DataConfig(sync_to_s3=True),
        )
        with patch("src.live.live_trader.reload_config", return_value=cfg):
            trader = LiveTrader(s3_sync=_make_s3(reachable=False))
            check = trader._check_s3_if_enabled()
        assert check.passed is False


class TestLiveTraderGoLive:
    def _all_pass_trader(self):
        with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
            with patch("src.secrets.Secrets.alpaca_api_key", return_value="VALIDKEY123456"):
                with patch("src.secrets.Secrets.alpaca_secret_key", return_value="VALIDSECRET9876"):
                    trader = LiveTrader(alpaca_client=_make_alpaca(), s3_sync=None)
                    return trader

    def test_go_live_raises_if_any_check_fails(self):
        trader = LiveTrader(alpaca_client=None)
        with pytest.raises(RuntimeError, match="Cannot go live"):
            with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
                trader.go_live()

    def test_get_live_status_returns_mode(self):
        with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
            trader = LiveTrader(alpaca_client=_make_alpaca())
            status = trader.get_live_status()
        assert status["mode"] == "live"
        assert status["is_live"] is True
        assert status["is_paper"] is False

    def test_get_live_status_includes_portfolio_value(self):
        with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
            trader = LiveTrader(alpaca_client=_make_alpaca(portfolio_value=99_000))
            status = trader.get_live_status()
        assert status["portfolio_value"] == 99_000.0

    def test_get_live_status_handles_missing_client(self):
        with patch("src.live.live_trader.reload_config", return_value=_live_config_patch()):
            trader = LiveTrader(alpaca_client=None)
            status = trader.get_live_status()
        assert status["portfolio_value"] is None


# ── S3Sync ────────────────────────────────────────────────────────────────────

from src.ingestion.s3_sync import S3Sync


def _make_s3sync(sync_enabled=True, bucket="test-bucket", prefix="market-data"):
    """Factory: S3Sync with mocked boto3 client."""
    from src.config import AppConfig, DataConfig
    cfg = AppConfig(data=DataConfig(sync_to_s3=sync_enabled, s3_bucket=bucket, s3_prefix=prefix))

    sync = S3Sync(bucket=bucket, prefix=prefix)
    sync.config = cfg

    mock_client = MagicMock()
    sync._client = mock_client
    return sync, mock_client


class TestS3SyncIsEnabled:
    def test_enabled_when_config_true(self):
        sync, _ = _make_s3sync(sync_enabled=True)
        assert sync.is_enabled() is True

    def test_disabled_when_config_false(self):
        sync, _ = _make_s3sync(sync_enabled=False)
        assert sync.is_enabled() is False


class TestS3SyncValidateConnection:
    def test_returns_true_when_bucket_exists(self):
        sync, mock_client = _make_s3sync()
        mock_client.head_bucket.return_value = {}
        assert sync.validate_connection() is True

    def test_returns_false_when_bucket_missing(self):
        sync, mock_client = _make_s3sync()
        mock_client.head_bucket.side_effect = Exception("NoSuchBucket")
        assert sync.validate_connection() is False


class TestS3SyncUpload:
    def test_upload_calls_boto3(self, tmp_path):
        f = tmp_path / "TEST.parquet"
        f.write_bytes(b"parquet data")

        sync, mock_client = _make_s3sync(sync_enabled=True)
        uri = sync.upload_file(f)

        assert uri.startswith("s3://")
        mock_client.upload_file.assert_called_once()

    def test_upload_noop_when_disabled(self, tmp_path):
        f = tmp_path / "TEST.parquet"
        f.write_bytes(b"data")

        sync, mock_client = _make_s3sync(sync_enabled=False)
        uri = sync.upload_file(f)

        assert uri == ""
        mock_client.upload_file.assert_not_called()

    def test_upload_skips_missing_file(self):
        sync, mock_client = _make_s3sync(sync_enabled=True)
        uri = sync.upload_file(Path("/nonexistent/file.parquet"))
        assert uri == ""
        mock_client.upload_file.assert_not_called()

    def test_upload_uses_custom_key(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"x")
        sync, mock_client = _make_s3sync()
        sync.upload_file(f, s3_key="custom/key.parquet")
        args = mock_client.upload_file.call_args
        assert args[0][2] == "custom/key.parquet"


class TestS3SyncDownload:
    def test_download_calls_boto3(self, tmp_path):
        dest = tmp_path / "output.parquet"
        sync, mock_client = _make_s3sync()
        result = sync.download_file("market-data/TEST.parquet", dest)
        assert result is True
        mock_client.download_file.assert_called_once()

    def test_download_returns_false_on_error(self, tmp_path):
        dest = tmp_path / "output.parquet"
        sync, mock_client = _make_s3sync()
        mock_client.download_file.side_effect = Exception("NoSuchKey")
        result = sync.download_file("bad/key.parquet", dest)
        assert result is False


class TestS3SyncDirectory:
    def test_sync_uploads_parquet_files(self, tmp_path):
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "NVDA.parquet").write_bytes(b"data")
        (tmp_path / "raw" / "AAPL.parquet").write_bytes(b"data")

        sync, mock_client = _make_s3sync(sync_enabled=True)
        # Simulate files not yet on S3 (head_object raises)
        mock_client.head_object.side_effect = Exception("NotFound")
        mock_client.upload_file.return_value = None

        uploaded = sync.sync_directory(tmp_path)
        assert len(uploaded) == 2
        assert mock_client.upload_file.call_count == 2

    def test_sync_noop_when_disabled(self, tmp_path):
        (tmp_path / "NVDA.parquet").write_bytes(b"data")
        sync, mock_client = _make_s3sync(sync_enabled=False)
        uploaded = sync.sync_directory(tmp_path)
        assert uploaded == []
        mock_client.upload_file.assert_not_called()

    def test_sync_returns_empty_for_missing_dir(self):
        sync, _ = _make_s3sync(sync_enabled=True)
        result = sync.sync_directory(Path("/nonexistent/dir"))
        assert result == []

    def test_sync_skips_non_parquet_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("ignore me")
        (tmp_path / "data.parquet").write_bytes(b"parquet")

        sync, mock_client = _make_s3sync(sync_enabled=True)
        mock_client.head_object.side_effect = Exception("NotFound")
        uploaded = sync.sync_directory(tmp_path)
        assert len(uploaded) == 1


class TestS3SyncListRemote:
    def test_list_remote_returns_keys(self):
        sync, mock_client = _make_s3sync()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "market-data/raw/NVDA.parquet"}, {"Key": "market-data/raw/AAPL.parquet"}]},
        ]
        keys = sync.list_remote("market-data")
        assert len(keys) == 2
        assert "market-data/raw/NVDA.parquet" in keys

    def test_list_remote_handles_empty_bucket(self):
        sync, mock_client = _make_s3sync()
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]
        keys = sync.list_remote("market-data")
        assert keys == []


# ── CloudWatchReporter ────────────────────────────────────────────────────────

from src.monitoring.cloudwatch_reporter import CloudWatchReporter


def _make_reporter(is_live=False):
    """Factory: CloudWatchReporter with mocked boto3 client."""
    from src.config import AppConfig, TradingConfig
    cfg = AppConfig(trading=TradingConfig(mode="live" if is_live else "paper"))

    mock_cw = MagicMock()
    reporter = CloudWatchReporter(client=mock_cw)
    reporter.config = cfg
    reporter._environment = "live" if is_live else "paper"
    return reporter, mock_cw


class TestCloudWatchReporterPutMetric:
    def test_put_metric_calls_boto3(self):
        reporter, mock_cw = _make_reporter()
        result = reporter.put_metric("TestMetric", 42.0, unit="Count")
        assert result is True
        mock_cw.put_metric_data.assert_called_once()

    def test_put_metric_uses_stockai_namespace(self):
        reporter, mock_cw = _make_reporter()
        reporter.put_metric("X", 1.0)
        call_kwargs = mock_cw.put_metric_data.call_args[1]
        assert call_kwargs["Namespace"] == "StockAI"

    def test_put_metric_includes_environment_dimension(self):
        reporter, mock_cw = _make_reporter(is_live=False)
        reporter.put_metric("X", 1.0)
        call_kwargs = mock_cw.put_metric_data.call_args[1]
        dims = call_kwargs["MetricData"][0]["Dimensions"]
        env_dim = next((d for d in dims if d["Name"] == "Environment"), None)
        assert env_dim is not None
        assert env_dim["Value"] == "paper"

    def test_put_metric_returns_false_on_boto3_error(self):
        reporter, mock_cw = _make_reporter()
        mock_cw.put_metric_data.side_effect = Exception("AccessDenied")
        result = reporter.put_metric("X", 1.0)
        assert result is False


class TestCloudWatchReporterPortfolioMetrics:
    def test_pushes_portfolio_value(self):
        reporter, mock_cw = _make_reporter()
        account = {"portfolio_value": 100_000.0, "cash": 20_000.0, "buying_power": 40_000.0, "equity": 100_000.0}
        result = reporter.push_portfolio_metrics(account)
        assert result is True
        call_kwargs = mock_cw.put_metric_data.call_args[1]
        names = [m["MetricName"] for m in call_kwargs["MetricData"]]
        assert "PortfolioValue" in names
        assert "CashBalance" in names

    def test_returns_false_for_empty_account(self):
        reporter, mock_cw = _make_reporter()
        result = reporter.push_portfolio_metrics({})
        assert result is False


class TestCloudWatchReporterSignalMetrics:
    def test_pushes_signal_counts(self):
        reporter, mock_cw = _make_reporter()
        result = reporter.push_signal_metrics(signals_generated=10, signals_approved=3)
        assert result is True
        call_kwargs = mock_cw.put_metric_data.call_args[1]
        names = [m["MetricName"] for m in call_kwargs["MetricData"]]
        assert "SignalsGenerated" in names
        assert "SignalsApproved" in names

    def test_values_match_input(self):
        reporter, mock_cw = _make_reporter()
        reporter.push_signal_metrics(7, 2)
        metrics = mock_cw.put_metric_data.call_args[1]["MetricData"]
        gen = next(m for m in metrics if m["MetricName"] == "SignalsGenerated")
        appr = next(m for m in metrics if m["MetricName"] == "SignalsApproved")
        assert gen["Value"] == 7.0
        assert appr["Value"] == 2.0


class TestCloudWatchReporterRiskMetrics:
    def test_pushes_risk_metrics(self):
        reporter, mock_cw = _make_reporter()
        result = reporter.push_risk_metrics(
            drawdown_pct=0.05,
            open_positions=3,
            circuit_breaker_tripped=False,
        )
        assert result is True
        names = [m["MetricName"] for m in mock_cw.put_metric_data.call_args[1]["MetricData"]]
        assert "DrawdownPct" in names
        assert "OpenPositions" in names
        assert "CircuitBreakerTripped" in names

    def test_drawdown_converted_to_percentage(self):
        reporter, mock_cw = _make_reporter()
        reporter.push_risk_metrics(0.08, 2, False)
        metrics = mock_cw.put_metric_data.call_args[1]["MetricData"]
        dd = next(m for m in metrics if m["MetricName"] == "DrawdownPct")
        assert abs(dd["Value"] - 8.0) < 0.01

    def test_circuit_breaker_tripped_is_one(self):
        reporter, mock_cw = _make_reporter()
        reporter.push_risk_metrics(0.02, 1, circuit_breaker_tripped=True)
        metrics = mock_cw.put_metric_data.call_args[1]["MetricData"]
        cb = next(m for m in metrics if m["MetricName"] == "CircuitBreakerTripped")
        assert cb["Value"] == 1.0

    def test_circuit_breaker_not_tripped_is_zero(self):
        reporter, mock_cw = _make_reporter()
        reporter.push_risk_metrics(0.0, 0, circuit_breaker_tripped=False)
        metrics = mock_cw.put_metric_data.call_args[1]["MetricData"]
        cb = next(m for m in metrics if m["MetricName"] == "CircuitBreakerTripped")
        assert cb["Value"] == 0.0


class TestCloudWatchReporterDriftAlerts:
    def _alert(self, severity):
        a = MagicMock()
        a.severity = severity
        return a

    def test_counts_warnings_and_criticals(self):
        reporter, mock_cw = _make_reporter()
        alerts = [self._alert("WARNING"), self._alert("WARNING"), self._alert("CRITICAL")]
        reporter.push_drift_alerts(alerts)
        metrics = mock_cw.put_metric_data.call_args[1]["MetricData"]
        warnings  = next(m for m in metrics if m["MetricName"] == "DriftWarnings")
        criticals = next(m for m in metrics if m["MetricName"] == "DriftCriticals")
        assert warnings["Value"] == 2.0
        assert criticals["Value"] == 1.0

    def test_empty_alerts_pushes_zeros(self):
        reporter, mock_cw = _make_reporter()
        reporter.push_drift_alerts([])
        metrics = mock_cw.put_metric_data.call_args[1]["MetricData"]
        for m in metrics:
            assert m["Value"] == 0.0


class TestCloudWatchReporterDailyPnl:
    def test_pushes_positive_pnl(self):
        reporter, mock_cw = _make_reporter()
        result = reporter.push_daily_pnl(1500.0)
        assert result is True
        metric = mock_cw.put_metric_data.call_args[1]["MetricData"][0]
        assert metric["MetricName"] == "DailyPnL"
        assert metric["Value"] == 1500.0

    def test_pushes_negative_pnl(self):
        reporter, mock_cw = _make_reporter()
        reporter.push_daily_pnl(-300.0)
        metric = mock_cw.put_metric_data.call_args[1]["MetricData"][0]
        assert metric["Value"] == -300.0


class TestCloudWatchReporterBatching:
    def test_batches_more_than_20_metrics(self):
        reporter, mock_cw = _make_reporter()
        # Build 25 metric dicts
        dims = {"Environment": "paper"}
        metrics = [reporter._build_metric_datum(f"M{i}", float(i), "Count", dims) for i in range(25)]
        reporter._put_metric_data(metrics)
        # Should be called twice: batch of 20, then batch of 5
        assert mock_cw.put_metric_data.call_count == 2

    def test_empty_metrics_list_is_noop(self):
        reporter, mock_cw = _make_reporter()
        result = reporter._put_metric_data([])
        assert result is True
        mock_cw.put_metric_data.assert_not_called()
