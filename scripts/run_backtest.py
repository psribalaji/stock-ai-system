#!/usr/bin/env python3
"""
scripts/run_backtest.py — CLI entry point for the walk-forward backtester.

Usage examples:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2025-01-01 --end 2026-01-01
    python scripts/run_backtest.py --tickers NVDA AAPL MSFT --capital 50000
    python scripts/run_backtest.py --start 2025-06-01 --save
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward backtester for stock-ai-system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tickers", nargs="+", metavar="TICKER",
                   help="Tickers to backtest (default: all tradeable stocks from config)")
    p.add_argument("--start", metavar="YYYY-MM-DD",
                   help="Start date (default: earliest available data)")
    p.add_argument("--end", metavar="YYYY-MM-DD",
                   help="End date (default: latest available data)")
    p.add_argument("--capital", type=float, default=100_000.0,
                   help="Initial capital in USD (default: 100000)")
    p.add_argument("--save", action="store_true",
                   help="Save equity curve, trades, and summary to data/backtest/")
    p.add_argument("--trades", action="store_true",
                   help="Print individual trade list after summary")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import after sys.path is set
    from loguru import logger
    from src.backtesting.backtester import Backtester

    logger.info("=" * 55)
    logger.info("  Stock AI — Walk-Forward Backtester")
    logger.info("=" * 55)

    bt     = Backtester()
    result = bt.run(
        tickers=args.tickers,
        start=args.start or "",
        end=args.end or "",
        initial_capital=args.capital,
    )

    print(result.summary)

    if args.trades and result.trades:
        print(f"\n{'─'*55}")
        print(f"  Trade Log ({len(result.trades)} trades)")
        print(f"{'─'*55}")
        header = f"{'Ticker':<6} {'Entry':>10} {'Exit':>10} {'Strategy':<20} {'PnL':>10}  Exit Reason"
        print(header)
        print("─" * len(header))
        for t in sorted(result.trades, key=lambda x: x.entry_date):
            pnl_str = f"${t.pnl:>8.2f}"
            print(
                f"{t.ticker:<6} {str(t.entry_date):>10} {str(t.exit_date):>10} "
                f"{t.strategy:<20} {pnl_str}  {t.exit_reason}"
            )

    if args.save:
        result.save()
        print("  Results saved to data/backtest/\n")

    # Exit code: 0 if quality gate passes, 1 if fails
    sys.exit(0 if result.quality_gate.get("overall_pass") else 1)


if __name__ == "__main__":
    main()
