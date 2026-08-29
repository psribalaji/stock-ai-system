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
    # Cybersecurity
    "PANW", "CRWD", "AXON", "ZS", "OKTA",
    # Cloud / SaaS
    "DDOG", "TEAM", "WDAY", "SNOW", "VEEV", "MDB", "GTLB",
    "TTD", "ADBE", "DOCU", "TWLO",
    # Fintech / Payments
    "COIN", "PYPL", "AFRM", "HOOD", "SOFI", "MA",
    # Consumer tech / Entertainment
    "SHOP", "UBER", "ABNB", "DASH", "DUOL", "RBLX",
    "SPOT", "NFLX", "RDDT", "SNAP",
    # Semis / Hardware
    "MRVL", "MU", "AMAT", "LRCX", "NXPI", "TXN", "NVTS", "SMCI",
    "DELL", "HPE", "SNDK", "SNPS", "KLAC",
    # Energy / Industrial
    "VRT", "VST", "FSLR", "GE", "CEG",
    # Healthcare / Biotech
    "CRSP", "NVO", "INTU",
    # International / Macro
    "GOOG", "MELI", "BABA", "NIO",
    # Finance / Diversified
    "MS", "CME", "IBM",
    # Consumer staples / Retail
    "WMT", "COST", "TGT", "NKE", "MCD", "CMG", "SBUX", "LULU",
    # Other quality large-caps in watchlist
    "OXY", "IONQ", "RKLB", "MSTR", "RIOT", "MARA",
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
