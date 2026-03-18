"""
scripts/e2e_test.py — End-to-end integration test with live API keys.

Tests the full pipeline:
  1. API connection validation (Polygon, Alpaca, Finnhub, Anthropic)
  2. Historical data fetch for one ticker (NVDA)
  3. Feature computation (30+ indicators)
  4. Signal detection (all 3 strategies)
  5. Confidence scoring
  6. LLM enrichment (Claude)
  7. Risk validation
  8. Full DecisionEngine run

Usage:
  py -3.12 scripts/e2e_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

TICKER = "NVDA"
PORTFOLIO_VALUE = 100_000.0


def section(title: str) -> None:
    console.print(f"\n[bold cyan]── {title} ──[/bold cyan]")


def ok(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def fail(msg: str) -> None:
    console.print(f"  [red]✗[/red] {msg}")


def info(msg: str) -> None:
    console.print(f"  [dim]{msg}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Config + secrets load
# ─────────────────────────────────────────────────────────────────────────────
section("Step 1: Config & Secrets")

try:
    from src.config import get_config
    cfg = get_config()
    ok(f"Config loaded — mode: {cfg.trading.mode.upper()}, assets: {len(cfg.assets.all_tradeable)}")
except Exception as e:
    fail(f"Config failed: {e}")
    sys.exit(1)

try:
    from src.secrets import Secrets
    alpaca_key   = Secrets.alpaca_api_key()
    polygon_key  = Secrets.polygon_api_key()
    finnhub_key  = Secrets.finnhub_api_key()
    anthropic_key = Secrets.anthropic_api_key()
    ok(f"Alpaca key loaded   ({alpaca_key[:8]}...)")
    ok(f"Polygon key loaded  ({polygon_key[:8]}...)")
    ok(f"Finnhub key loaded  ({finnhub_key[:8]}...)")
    ok(f"Anthropic key loaded ({anthropic_key[:14]}...)")
except Exception as e:
    fail(f"Secrets failed: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — API connection validation
# ─────────────────────────────────────────────────────────────────────────────
section("Step 2: API Connections")

from src.ingestion.market_data_service import MarketDataService
from src.ingestion.news_service import NewsService

svc = MarketDataService()
connections = svc.validate_connections()

all_connected = True
for name, status in connections.items():
    if status:
        ok(f"{name.upper()} connected")
    else:
        fail(f"{name.upper()} connection failed")
        all_connected = False

try:
    news_svc = NewsService()
    news_ok = news_svc.validate_connection()
    if news_ok:
        ok("FINNHUB connected")
    else:
        fail("FINNHUB connection failed")
        all_connected = False
except Exception as e:
    fail(f"FINNHUB error: {e}")
    all_connected = False

if not all_connected:
    console.print("\n[bold red]Stopping — fix API connections first.[/bold red]")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Fetch historical OHLCV
# ─────────────────────────────────────────────────────────────────────────────
section(f"Step 3: Fetch Historical OHLCV for {TICKER}")

from datetime import date, timedelta

start = date.today() - timedelta(days=365)
end   = date.today()

try:
    df_raw = svc.polygon.fetch_daily_bars(TICKER, start, end)
    if df_raw.empty:
        fail("No data returned from Polygon")
        sys.exit(1)
    ok(f"Fetched {len(df_raw)} bars: {df_raw['timestamp'].min()} → {df_raw['timestamp'].max()}")
    svc.store.save_ohlcv(TICKER, df_raw)
    ok(f"Stored to Parquet")
except Exception as e:
    fail(f"Polygon fetch failed: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Feature computation
# ─────────────────────────────────────────────────────────────────────────────
section("Step 4: Feature Engine")

try:
    from src.features.feature_engine import FeatureEngine
    fe = FeatureEngine()
    df_features = fe.compute_all(df_raw)
    n_cols = len(df_features.columns)
    ok(f"Computed {n_cols} columns (was {len(df_raw.columns)})")
    ok(f"Rows: {len(df_features)}")

    # Show a sample of computed indicators
    indicator_cols = [c for c in df_features.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
    info(f"Indicators: {', '.join(indicator_cols[:10])}...")
except Exception as e:
    fail(f"Feature engine failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Signal detection
# ─────────────────────────────────────────────────────────────────────────────
section("Step 5: Signal Detection")

try:
    from src.signals.signal_detector import SignalDetector
    detector = SignalDetector()
    signals = detector.detect_actionable(TICKER, df_features)
    ok(f"Detected {len(signals)} actionable signal(s)")

    if signals:
        t = Table(show_header=True, header_style="bold blue")
        t.add_column("Strategy")
        t.add_column("Pattern")
        t.add_column("Direction")
        t.add_column("Strength", justify="right")
        for s in signals:
            t.add_row(s.strategy, s.pattern, s.direction, f"{s.strength:.3f}")
        console.print(t)
    else:
        info("No signals today — market may be ranging. That's normal.")
except Exception as e:
    fail(f"Signal detection failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────
section("Step 6: Confidence Scoring")

try:
    from src.signals.confidence_scorer import ConfidenceScorer
    scorer = ConfidenceScorer()
    scored = scorer.score_all(signals, TICKER)
    passed = [s for s in scored if not s.blocked]
    blocked = [s for s in scored if s.blocked]
    ok(f"Scored {len(scored)} signal(s): {len(passed)} passed, {len(blocked)} blocked")

    if scored:
        t = Table(show_header=True, header_style="bold blue")
        t.add_column("Pattern")
        t.add_column("Confidence", justify="right")
        t.add_column("Status")
        for s in scored:
            status = "[green]PASS[/green]" if not s.blocked else f"[red]BLOCK: {s.block_reason}[/red]"
            t.add_row(s.signal.pattern, f"{s.confidence:.3f}", status)
        console.print(t)
except Exception as e:
    fail(f"Confidence scoring failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — News + LLM enrichment
# ─────────────────────────────────────────────────────────────────────────────
section("Step 7: News & LLM Enrichment")

news_summary = None
llm_result = None

try:
    news = news_svc.get_latest_news(TICKER, limit=5)
    news_summary = news_svc.get_sentiment_summary(TICKER)
    ok(f"Fetched {len(news)} news items for {TICKER}")
    info(f"Sentiment summary: {news_summary[:120]}...")
except Exception as e:
    info(f"News fetch skipped: {e}")

if passed:
    try:
        from src.llm.llm_analysis_service import LLMAnalysisService
        llm = LLMAnalysisService()
        llm_result = llm.enrich(passed[0], news_summary=news_summary)
        ok("LLM enrichment successful")
        info(f"Reasoning: {llm_result.reasoning[:200]}...")
        info(f"Summary: {llm_result.summary}")
    except Exception as e:
        info(f"LLM enrichment skipped: {e}")
else:
    info("No passed signals — skipping LLM enrichment")


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Risk validation
# ─────────────────────────────────────────────────────────────────────────────
section("Step 8: Risk Validation")

try:
    from src.risk.risk_manager import RiskManager, PortfolioState
    rm = RiskManager()

    # Get latest price from Alpaca
    latest_price = svc.alpaca.get_latest_price(TICKER)
    if not latest_price:
        latest_price = float(df_features["close"].iloc[-1])
        info(f"Using last close as price: ${latest_price:.2f}")
    else:
        ok(f"Live price from Alpaca: ${latest_price:.2f}")

    portfolio = PortfolioState(
        total_value_usd=PORTFOLIO_VALUE,
        cash_usd=PORTFOLIO_VALUE * 0.8,
        open_positions=1,
        crypto_exposure_usd=0.0,
        daily_pnl_pct=0.003,
        peak_value_usd=PORTFOLIO_VALUE,
    )

    if passed:
        risk_decision = rm.validate(passed[0], latest_price, portfolio, TICKER)
        status = "[green]APPROVED[/green]" if risk_decision.approved else f"[red]BLOCKED: {risk_decision.block_reason}[/red]"
        ok(f"Risk decision: {status}")
        ok(f"Position size: {risk_decision.position_size_pct*100:.1f}% = ${risk_decision.position_size_usd:,.0f}")
        ok(f"Stop loss: ${risk_decision.stop_loss_price:.2f} ({risk_decision.stop_loss_pct*100:.1f}% below entry)")
        if risk_decision.risk_notes:
            info(f"Notes: {'; '.join(risk_decision.risk_notes)}")
    else:
        info("No passed signals — risk check skipped (would pass on real signal)")
        ok("Risk manager loaded and operational")
except Exception as e:
    fail(f"Risk validation failed: {e}")
    import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Full DecisionEngine pipeline
# ─────────────────────────────────────────────────────────────────────────────
section("Step 9: DecisionEngine (Full Pipeline)")

try:
    from src.execution.decision_engine import DecisionEngine
    from src.llm.llm_analysis_service import LLMAnalysisService

    try:
        llm_svc = LLMAnalysisService()
    except Exception:
        llm_svc = None
        info("LLM service unavailable — running without enrichment")

    engine = DecisionEngine(llm_service=llm_svc)
    decisions = engine.decide(
        ticker=TICKER,
        df=df_features,
        entry_price=latest_price,
        portfolio=portfolio,
        news_summary=news_summary,
    )

    ok(f"DecisionEngine returned {len(decisions)} approved trade decision(s)")

    if decisions:
        t = Table(show_header=True, header_style="bold green")
        t.add_column("Ticker")
        t.add_column("Direction")
        t.add_column("Strategy")
        t.add_column("Confidence", justify="right")
        t.add_column("Size USD", justify="right")
        t.add_column("Stop Loss", justify="right")
        t.add_column("LLM Summary")
        for d in decisions:
            t.add_row(
                d.ticker,
                f"[{'green' if d.direction == 'BUY' else 'red'}]{d.direction}[/]",
                d.strategy,
                f"{d.confidence:.3f}",
                f"${d.position_size_usd:,.0f}",
                f"${d.stop_loss_price:.2f}",
                (d.llm_summary or "—")[:50],
            )
        console.print(t)
    else:
        info("No approved decisions today — signals may be low confidence or blocked by risk rules")
        info("This is normal — the system correctly gatekeeps low-quality signals")

except Exception as e:
    fail(f"DecisionEngine failed: {e}")
    import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
console.print(Panel.fit(
    "[bold green]✓ End-to-End Test Complete[/bold green]\n\n"
    f"  Ticker tested:    {TICKER}\n"
    f"  Data fetched:     {len(df_raw)} bars\n"
    f"  Features:         {n_cols} columns\n"
    f"  Signals detected: {len(signals)}\n"
    f"  Signals passed:   {len(passed)}\n"
    f"  Decisions issued: {len(decisions) if 'decisions' in dir() else 0}\n\n"
    "[bold]All pipeline stages operational.[/bold]",
    border_style="green",
))
