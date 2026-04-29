"""
download_data.py — Download 1 Year of Nifty 50 & BankNifty Historical Data
============================================================================

Fetches 365 days of 30-minute OHLCV candle data for both Nifty 50 and
BankNifty using the Upstox API via src/fetcher.get_historical_data() and
saves them to:

  • data/nifty_historical.csv
  • data/banknifty_historical.csv

Usage:
    python download_data.py
"""

import os
import sys
from datetime import datetime, timedelta

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fetcher import get_historical_data


def main() -> None:
    today = datetime.now()
    from_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    os.makedirs("data", exist_ok=True)

    # ── Nifty 50 ────────────────────────────────────────────────────
    print(f"Downloading Nifty 50 data from {from_date} to {to_date}...")

    nifty_df = get_historical_data(
        instrument_key="NSE_INDEX|Nifty 50",
        interval="30minute",
        from_date=from_date,
        to_date=to_date,
    )

    nifty_path = os.path.join("data", "nifty_historical.csv")
    nifty_df.to_csv(nifty_path, index=False)
    print(f"Saved {len(nifty_df)} rows to {nifty_path}")

    # ── BankNifty ───────────────────────────────────────────────────
    print(f"\nDownloading BankNifty data from {from_date} to {to_date}...")

    banknifty_df = get_historical_data(
        instrument_key="NSE_INDEX|Nifty Bank",
        interval="30minute",
        from_date=from_date,
        to_date=to_date,
    )

    banknifty_path = os.path.join("data", "banknifty_historical.csv")
    banknifty_df.to_csv(banknifty_path, index=False)
    print(f"Saved {len(banknifty_df)} rows to {banknifty_path}")


if __name__ == "__main__":
    main()
