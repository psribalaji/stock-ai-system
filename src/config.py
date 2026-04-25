"""
config.py — Central configuration loader.
Reads config.yaml and exposes typed config objects to all modules.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional


# ── Pydantic models for typed config ─────────────────────────────

class TradingConfig(BaseModel):
    mode: str = "paper"
    broker: str = "alpaca"
    timezone: str = "America/New_York"

class AssetsConfig(BaseModel):
    stocks: List[str] = []
    crypto: List[str] = []
    watchlist: List[str] = []

    @property
    def all_tradeable(self) -> List[str]:
        return self.stocks + self.crypto

    @property
    def all_symbols(self) -> List[str]:
        return self.stocks + self.crypto + self.watchlist

class RiskConfig(BaseModel):
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.25
    max_crypto_pct: float = 0.10
    max_open_positions: int = 5
    min_confidence: float = 0.60
    stop_loss_pct: float = 0.07
    trailing_stop_atr_mult: float = 2.0
    reward_risk_ratio: float = 2.0
    daily_loss_limit: float = 0.02
    max_drawdown_pct: float = 0.15
    max_portfolio_corr: float = 0.70

class SignalsConfig(BaseModel):
    lookback_days: int = 252
    confidence_lookback: int = 60
    min_volume_ratio: float = 1.5
    rsi_oversold: int = 35
    rsi_overbought: int = 70
    ma_short: int = 50
    ma_long: int = 200

class BacktestConfig(BaseModel):
    train_start: str = "2015-01-01"
    train_end: str = "2022-12-31"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: str = "2025-12-31"
    slippage_pct: float = 0.0008
    commission: float = 0.0

class QualityGateConfig(BaseModel):
    min_cagr: float = 0.15
    min_sharpe: float = 1.0
    max_drawdown: float = 0.25
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.5
    min_calmar: float = 0.5
    min_trades: int = 30

class DriftConfig(BaseModel):
    sharpe_floor: float = 0.5
    win_rate_drop: float = 0.15
    drawdown_pause: float = 0.15
    benchmark_lag: float = 0.10
    check_frequency: str = "weekly"

class DataConfig(BaseModel):
    storage_path: str = "./data"
    s3_bucket: str = "stock-ai-system-data"
    s3_prefix: str = "market-data"
    sync_to_s3: bool = False
    s3_role_arn: str = ""
    polygon_timeframe: str = "day"
    alpaca_feed: str = "iex"

class ScheduleConfig(BaseModel):
    data_sync_hour: int = 5
    signal_interval_min: int = 30
    position_check_interval_min: int = 5
    force_market_hours: bool = False
    drift_check_day: str = "sun"
    drift_check_hour: int = 20
    recalibration_months: List[int] = [1, 4, 7, 10]

class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_path: str = "./logs"
    rotation: str = "1 week"
    retention: str = "1 month"

class MLEnsembleConfig(BaseModel):
    enabled: bool = False
    min_training_trades: int = 300
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    model_path: str = "./data/ml_ensemble_model.pkl"

class DiscoveryConfig(BaseModel):
    enabled:              bool       = True
    scan_interval_min:    int        = 30
    pre_market_scan_hour: int        = 8
    min_market_cap:       float      = 500_000_000.0
    min_avg_volume:       float      = 500_000.0
    min_price:            float      = 5.0
    max_candidates:       int        = 20
    mention_spike_factor: float      = 3.0
    reddit_subreddits:    List[str]  = ["wallstreetbets", "investing", "stocks", "stockmarket"]
    news_lookback_hours:  int        = 48
    auto_approve:         bool       = False  # ALWAYS False — human approval required

class AppConfig(BaseModel):
    trading: TradingConfig = Field(default_factory=TradingConfig)
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    signals: SignalsConfig = Field(default_factory=SignalsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ml_ensemble: MLEnsembleConfig = Field(default_factory=MLEnsembleConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)

    @property
    def is_live(self) -> bool:
        return self.trading.mode == "live"

    @property
    def is_paper(self) -> bool:
        return self.trading.mode == "paper"


# ── Loader ───────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> AppConfig:
    """Load and cache config. Call get_config() from anywhere."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)


def reload_config() -> AppConfig:
    """Force reload — clears cache. Use when config changes at runtime."""
    get_config.cache_clear()
    return get_config()
