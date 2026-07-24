"""
scripts/prune_discovery.py — Demote all approved discovery tickers back to CANDIDATE,
except for a curated keep-list. Run on the server to stop noise signal generation.

Usage:
    python3 scripts/prune_discovery.py           # dry-run: shows what would change
    python3 scripts/prune_discovery.py --apply   # apply the changes
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.discovery.universe_manager import UniverseManager, STATUS_APPROVED, STATUS_CANDIDATE

# Tickers to KEEP as approved — current positions + highest-quality names only.
# Everything else goes back to CANDIDATE (can be re-approved via dashboard).
KEEP = {
    # Current open positions — never demote these
    "GL", "TD",
    # High-conviction large caps already in discovery with proven track record
    "GOOG", "SHOP", "UBER", "COIN", "DDOG", "PYPL", "RBLX",
    "ADBE", "DOCU", "OKTA", "TEAM", "AXON", "PANW", "CRSP",
    "MRVL", "MU", "AMAT", "LRCX", "NXPI", "TXN",
    "WMT", "COST", "TGT", "NKE", "MCD", "CMG",
    "MA", "MS", "CME", "IBM",
    "SPOT", "RDDT", "SNAP",
    "MELI", "BABA", "NIO",
    "NVTS", "SMCI", "DELL", "HPE",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    args = parser.parse_args()

    um = UniverseManager()
    df = um._load()

    approved = df[df["status"] == STATUS_APPROVED]["ticker"].tolist()
    to_demote = [t for t in approved if t not in KEEP]
    to_keep   = [t for t in approved if t in KEEP]

    print(f"Total APPROVED:   {len(approved)}")
    print(f"Keeping:          {len(to_keep)}  {sorted(to_keep)}")
    print(f"Demoting to CANDIDATE: {len(to_demote)}")
    print()

    if not args.apply:
        print("DRY-RUN — pass --apply to make changes")
        print(f"\nWould demote: {sorted(to_demote)}")
        return

    # Demote by writing CANDIDATE status directly into the parquet
    import pandas as pd
    mask = df["ticker"].isin(to_demote) & (df["status"] == STATUS_APPROVED)
    df.loc[mask, "status"] = STATUS_CANDIDATE
    um._save(df)

    print(f"Done. Demoted {mask.sum()} tickers to CANDIDATE.")
    print(f"Tradeable universe now: {len(to_keep)} discovery tickers + config tickers")


if __name__ == "__main__":
    main()
