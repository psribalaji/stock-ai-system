"""
scripts/setup.py — One-time Phase 0 setup script.
Run this ONCE after cloning the repo to:
  1. Validate all API connections
  2. Fetch 4 years of historical data for all tickers
  3. Run data health check
  4. Print a summary

Usage (Windows):
  python scripts/setup.py

Requirements: .env file must exist with all API keys set.
"""
import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

from src.config import get_config
from src.ingestion.market_data_service import MarketDataService
from src.ingestion.news_service import NewsService

console = Console()


def run_setup():
    config = get_config()

    console.print(Panel.fit(
        "[bold cyan]STOCKAI — Phase 0 Setup[/bold cyan]\n"
        "Initializing data pipeline...",
        border_style="cyan"
    ))

    # ── Step 1: Validate connections ──────────────────────────────
    console.print("\n[bold]Step 1/4:[/bold] Validating API connections...")
    svc = MarketDataService()
    connections = svc.validate_connections()

    conn_table = Table(show_header=True, header_style="bold blue")
    conn_table.add_column("Service",    style="cyan")
    conn_table.add_column("Status",     style="green")
    conn_table.add_column("Notes")

    for service, ok in connections.items():
        status = "✓ CONNECTED" if ok else "✗ FAILED"
        style  = "green" if ok else "red"
        conn_table.add_row(
            service.upper(),
            f"[{style}]{status}[/{style}]",
            "Ready" if ok else "Check .env file"
        )

    # News
    try:
        news_svc = NewsService()
        news_ok  = news_svc.validate_connection()
        conn_table.add_row(
            "FINNHUB",
            "[green]✓ CONNECTED[/green]" if news_ok else "[red]✗ FAILED[/red]",
            "Ready" if news_ok else "Check FINNHUB_API_KEY"
        )
    except Exception as e:
        conn_table.add_row("FINNHUB", "[red]✗ FAILED[/red]", str(e))
        news_ok = False

    console.print(conn_table)

    failed = [k for k, v in connections.items() if not v]
    if failed:
        console.print(
            f"\n[bold red]⚠ Setup cannot continue — fix connections: {failed}[/bold red]"
        )
        console.print("\nMake sure your [bold].env[/bold] file exists with these keys:")
        console.print("  ALPACA_API_KEY=your_key")
        console.print("  ALPACA_SECRET_KEY=your_secret")
        console.print("  POLYGON_API_KEY=your_key")
        console.print("  FINNHUB_API_KEY=your_key")
        console.print("  ANTHROPIC_API_KEY=your_key")
        sys.exit(1)

    # ── Step 2: Fetch historical data ─────────────────────────────
    tickers = config.assets.all_tradeable
    console.print(
        f"\n[bold]Step 2/4:[/bold] Fetching 4 years of historical data "
        f"for {len(tickers)} tickers..."
    )
    console.print(f"  Tickers: [cyan]{', '.join(tickers)}[/cyan]")
    console.print("  This will take 2–5 minutes on first run.\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching data...", total=len(tickers))
        results = {}
        for ticker in tickers:
            progress.update(task, description=f"Fetching {ticker}...")
            result = svc.fetch_and_store_historical(
                tickers=[ticker],
                years_back=4,
            )
            results.update(result)
            progress.advance(task)

    # ── Step 3: Data health check ─────────────────────────────────
    console.print("\n[bold]Step 3/4:[/bold] Running data health check...")
    health = svc.run_data_health_check()

    health_table = Table(show_header=True, header_style="bold blue")
    health_table.add_column("Ticker",  style="cyan")
    health_table.add_column("Rows",    style="green", justify="right")
    health_table.add_column("Status")
    health_table.add_column("Issues")

    for ticker, report in health.items():
        status_str = "[green]✓ CLEAN[/green]" if report.get("valid") else "[red]✗ ISSUES[/red]"
        issues_str = ", ".join(report.get("issues", [])) or "—"
        health_table.add_row(
            ticker,
            str(report.get("rows", 0)),
            status_str,
            issues_str[:60],
        )

    console.print(health_table)

    # ── Step 4: Print storage summary ─────────────────────────────
    console.print("\n[bold]Step 4/4:[/bold] Storage summary")
    stats = svc.store.get_stats()

    summary_table = Table(show_header=False)
    summary_table.add_column("Key",   style="bold")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_row("Storage path",     stats["storage_path"])
    summary_table.add_row("Tickers stored",   str(len(stats["tickers_stored"])))
    summary_table.add_row("Total OHLCV rows", f"{stats['total_ohlcv_rows']:,}")
    summary_table.add_row("Signal files",     str(stats["signal_files"]))
    summary_table.add_row("Trading mode",     config.trading.mode.upper())

    console.print(summary_table)

    # ── Done ──────────────────────────────────────────────────────
    clean = sum(1 for v in health.values() if v.get("valid"))
    total = len(health)

    console.print(Panel.fit(
        f"[bold green]✓ Phase 0 Setup Complete[/bold green]\n\n"
        f"  {clean}/{total} tickers have clean data\n"
        f"  {stats['total_ohlcv_rows']:,} OHLCV rows stored\n\n"
        f"[bold]Next step:[/bold] Run the feature engine\n"
        f"  [cyan]python scripts/run_features.py[/cyan]",
        border_style="green"
    ))


if __name__ == "__main__":
    run_setup()
