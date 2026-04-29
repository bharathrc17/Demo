"""
src/fetcher.py — Live & Historical OHLCV Data Fetcher (Upstox API v2)
=======================================================================

This module connects to the Upstox API v2 via the upstox-python-sdk and
provides functions for retrieving OHLCV candle data for Nifty 50 and
BankNifty:

  • get_historical_data()  — fetch an arbitrary date range, automatically
                              chunking requests into ≤30-day windows to
                              respect the Upstox per-request limit.
  • get_live_candles()     — convenience wrapper that returns the last *n*
                              candles up to the current moment.
  • get_live_candles_multi_timeframe()
                            — fetches both 5min and 15min candles for
                              trend detection (EMA alignment check).
  • get_nearest_expiry()   — fetches option contracts from Upstox and
                              returns the nearest Thursday expiry date.

All API calls are wrapped in try/except and raise a custom ``FetchError``
on failure.  A 0.5-second sleep is inserted between consecutive API calls
to stay within rate limits.

Logs are written to ``logs/fetcher.log`` via Python's built-in logging
module (rotating file handler, 5 MB × 3 backups).

Usage:
    from src.fetcher import get_historical_data, get_live_candles

    # Nifty 50
    df = get_historical_data(
        instrument_key="NSE_INDEX|Nifty 50",
        interval="30minute",
        from_date="2025-01-01",
        to_date="2025-03-31",
    )

    # BankNifty
    df = get_historical_data(
        instrument_key="NSE_INDEX|Nifty Bank",
        interval="30minute",
        from_date="2025-01-01",
        to_date="2025-03-31",
    )

    live = get_live_candles(instrument_key="NSE_INDEX|Nifty 50", interval="30minute", n=150)

    multi = get_live_candles_multi_timeframe(instrument_key="NSE_INDEX|Nifty Bank")
    # multi == {"5min": df_5min, "15min": df_15min}

    expiry = get_nearest_expiry("NIFTY")      # "01-May-2026"
    expiry = get_nearest_expiry("BANKNIFTY")   # "01-May-2026"
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Optional

import pandas as pd
import pytz
import upstox_client
from upstox_client.rest import ApiException

# ---------------------------------------------------------------------------
# Resolve project root so relative paths work from any cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config import Config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — dedicated fetcher.log with rotation
# ---------------------------------------------------------------------------
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logger = logging.getLogger("fetcher")
logger.setLevel(logging.DEBUG)

_fh = RotatingFileHandler(
    filename=os.path.join(_LOG_DIR, "fetcher.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_fh.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)

_fmt = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)

if not logger.handlers:
    logger.addHandler(_fh)
    logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_DAYS_PER_REQUEST = 30
_API_CALL_DELAY = 0.5
_API_VERSION = "v2"
_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# Mapping from user-friendly index names to Upstox instrument keys
_INDEX_INSTRUMENT_MAP = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class FetchError(Exception):
    """
    Raised when an API call to Upstox fails irrecoverably.

    Attributes
    ----------
    message : str
        Human-readable error description.
    original_exception : Exception | None
        The underlying exception that triggered this error.
    """

    def __init__(self, message: str,
                 original_exception: Optional[Exception] = None) -> None:
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"FetchError(message={self.message!r}, "
            f"original_exception={self.original_exception!r})"
        )


# ---------------------------------------------------------------------------
# Upstox API client setup
# ---------------------------------------------------------------------------

def _get_api_instance() -> upstox_client.HistoryApi:
    """
    Return a configured ``upstox_client.HistoryApi`` instance.

    Authentication is set via the access token from Config. The Upstox
    historical candle endpoints technically don't require auth, but we
    configure it anyway for consistency and in case of future changes.

    Raises
    ------
    FetchError
        If the access token is not set.
    """
    access_token = Config.UPSTOX_ACCESS_TOKEN
    if not access_token:
        msg = "UPSTOX_ACCESS_TOKEN must be set in the environment or .env file."
        logger.error(msg)
        raise FetchError(msg)

    configuration = upstox_client.Configuration()
    configuration.access_token = access_token
    api_client = upstox_client.ApiClient(configuration)
    return upstox_client.HistoryApi(api_client)


def _get_options_api_instance() -> upstox_client.OptionsApi:
    """
    Return a configured ``upstox_client.OptionsApi`` instance.

    Used by ``get_nearest_expiry()`` to fetch option contracts.

    Raises
    ------
    FetchError
        If the access token is not set.
    """
    access_token = Config.UPSTOX_ACCESS_TOKEN
    if not access_token:
        msg = "UPSTOX_ACCESS_TOKEN must be set in the environment or .env file."
        logger.error(msg)
        raise FetchError(msg)

    configuration = upstox_client.Configuration()
    configuration.access_token = access_token
    api_client = upstox_client.ApiClient(configuration)
    return upstox_client.OptionsApi(api_client)


def _generate_date_chunks(
    from_date: datetime,
    to_date: datetime,
    max_days: int = _MAX_DAYS_PER_REQUEST,
) -> list[tuple[datetime, datetime]]:
    """
    Split ``[from_date, to_date]`` into sub-ranges of at most *max_days*.

    Returns
    -------
    list of (datetime, datetime)
        Ordered list of ``(chunk_start, chunk_end)`` tuples.
    """
    chunks: list[tuple[datetime, datetime]] = []
    current_start = from_date

    while current_start <= to_date:
        current_end = min(current_start + timedelta(days=max_days - 1), to_date)
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)

    logger.debug(
        "Date range %s → %s split into %d chunk(s) of ≤%d days.",
        from_date.date(), to_date.date(), len(chunks), max_days,
    )
    return chunks


def _parse_candles(raw_candles: list[list]) -> pd.DataFrame:
    """
    Convert the Upstox raw candle list into a normalised DataFrame.

    Each candle is a list: [timestamp, open, high, low, close, volume, oi].
    We keep only the first 6 elements (drop OI).

    Returns
    -------
    pd.DataFrame
        Columns: date, open, high, low, close, volume.
    """
    if not raw_candles:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    records = []
    for candle in raw_candles:
        records.append({
            "date": candle[0],
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": int(candle[5]),
        })

    df = pd.DataFrame(records, columns=_OHLCV_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_historical_data(
    instrument_key: str = Config.INSTRUMENT_KEY,
    interval: str = "30minute",
    from_date: str = "",
    to_date: str = "",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV candle data from Upstox.

    Automatically chunks the requested range into ≤30-day windows and
    concatenates the results.  Works for both Nifty 50 and BankNifty.

    Parameters
    ----------
    instrument_key : str
        Upstox instrument key, e.g. ``"NSE_INDEX|Nifty 50"`` or
        ``"NSE_INDEX|Nifty Bank"``.
    interval : str
        Candle interval — ``"1minute"``, ``"5minute"``, ``"15minute"``,
        ``"30minute"``, ``"day"``, ``"week"``, ``"month"``.
    from_date : str
        Start date in ``"YYYY-MM-DD"`` format.  Defaults to 30 days ago.
    to_date : str
        End date in ``"YYYY-MM-DD"`` format.  Defaults to today.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``open``, ``high``, ``low``, ``close``,
        ``volume``.  Sorted by ``date`` ascending, duplicates removed.

    Raises
    ------
    FetchError
        On any Upstox API or network error.
    """
    today = datetime.now()

    if not to_date:
        to_dt = today
    else:
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")

    if not from_date:
        from_dt = to_dt - timedelta(days=_MAX_DAYS_PER_REQUEST)
    else:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")

    logger.info(
        "Fetching historical data | instrument=%s | interval=%s | %s → %s",
        instrument_key, interval, from_dt.date(), to_dt.date(),
    )

    chunks = _generate_date_chunks(from_dt, to_dt)
    api_instance = _get_api_instance()
    all_frames: list[pd.DataFrame] = []

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        chunk_from_str = chunk_start.strftime("%Y-%m-%d")
        chunk_to_str = chunk_end.strftime("%Y-%m-%d")

        logger.debug(
            "  Chunk %d/%d: %s → %s",
            idx, len(chunks), chunk_from_str, chunk_to_str,
        )

        try:
            response = api_instance.get_historical_candle_data1(
                instrument_key=instrument_key,
                interval=interval,
                to_date=chunk_to_str,
                from_date=chunk_from_str,
                api_version=_API_VERSION,
            )

            candles = response.data.candles if response.data and response.data.candles else []
            chunk_df = _parse_candles(candles)
            all_frames.append(chunk_df)
            logger.debug("    ↳ received %d candle(s).", len(chunk_df))

        except ApiException as exc:
            msg = (
                f"Upstox API error for chunk {chunk_from_str}→{chunk_to_str}: "
                f"status={exc.status}, reason={exc.reason}"
            )
            logger.exception(msg)
            raise FetchError(msg, original_exception=exc) from exc

        except Exception as exc:
            msg = f"Unexpected error for chunk {chunk_from_str}→{chunk_to_str}: {exc}"
            logger.exception(msg)
            raise FetchError(msg, original_exception=exc) from exc

        # Rate-limit pause (skip after last chunk)
        if idx < len(chunks):
            time.sleep(_API_CALL_DELAY)

    # Combine all chunks
    if not all_frames:
        logger.warning("No data returned for %s (%s).", instrument_key, interval)
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    df = pd.concat(all_frames, ignore_index=True)
    df.sort_values("date", inplace=True)
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Historical data ready — %d rows, %s → %s.",
        len(df), df["date"].iloc[0], df["date"].iloc[-1],
    )
    return df


def get_live_candles(
    instrument_key: str = Config.INSTRUMENT_KEY,
    interval: str = "30minute",
    n: int = 150,
) -> pd.DataFrame:
    """
    Return the most recent *n* candles up to the current moment.

    Calculates ``from_date`` automatically based on the requested number
    of candles and the interval, fetches slightly more than needed to
    account for weekends and holidays, then trims to the last *n* rows.

    Works for both Nifty 50 (``"NSE_INDEX|Nifty 50"``) and BankNifty
    (``"NSE_INDEX|Nifty Bank"``).

    Parameters
    ----------
    instrument_key : str
        Upstox instrument key, e.g. ``"NSE_INDEX|Nifty 50"`` or
        ``"NSE_INDEX|Nifty Bank"``.
    interval : str
        Candle interval.
    n : int
        Number of candles to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``open``, ``high``, ``low``, ``close``,
        ``volume``.  Exactly *n* rows (or fewer if not enough history).

    Raises
    ------
    FetchError
        On any Upstox API or network error.
    """
    logger.info(
        "Fetching last %d live candles | instrument=%s | interval=%s",
        n, instrument_key, interval,
    )

    # Estimate calendar days needed for n trading candles
    interval_minutes_map = {
        "1minute": 1,
        "5minute": 5,
        "15minute": 15,
        "30minute": 30,
        "day": 375,
        "week": 375 * 5,
        "month": 375 * 22,
    }
    candle_minutes = interval_minutes_map.get(interval, 30)
    trading_minutes_per_day = 375  # NSE: 09:15 – 15:30

    candles_per_day = max(trading_minutes_per_day // candle_minutes, 1)
    # 2x safety margin for weekends / holidays
    days_needed = max((n // candles_per_day + 1) * 2, 7)

    today = datetime.now()
    from_dt = today - timedelta(days=days_needed)

    to_date_str = today.strftime("%Y-%m-%d")
    from_date_str = from_dt.strftime("%Y-%m-%d")

    df = get_historical_data(
        instrument_key=instrument_key,
        interval=interval,
        from_date=from_date_str,
        to_date=to_date_str,
    )

    # Trim to the last n rows
    if len(df) > n:
        df = df.tail(n).reset_index(drop=True)

    logger.info("Live candles returned — %d rows.", len(df))
    return df


def _resample_candles(
    df_1min: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV candles to a coarser timeframe.

    Parameters
    ----------
    df_1min : pd.DataFrame
        Must have columns: ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume`` with ``date`` as timezone-aware datetimes.
    rule : str
        Pandas offset alias, e.g. ``"5min"`` or ``"15min"``.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with the same columns.
    """
    if df_1min.empty:
        return pd.DataFrame(columns=_OHLCV_COLUMNS)

    resampled = df_1min.resample(rule, on="date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()

    return resampled


def get_live_candles_multi_timeframe(
    instrument_key: str = Config.INSTRUMENT_KEY,
    n_1min: int = 75,
) -> dict[str, pd.DataFrame]:
    """
    Fetch 1-minute candles and resample to 5-minute and 15-minute for
    multi-timeframe trend detection (EMA alignment check).

    Upstox free API only supports ``1minute``, ``30minute``, ``day``,
    ``week``, ``month`` intervals.  This function works around the
    missing 5min/15min intervals by fetching raw 1-minute data and
    resampling with pandas.

    Works for both Nifty 50 (``"NSE_INDEX|Nifty 50"``) and BankNifty
    (``"NSE_INDEX|Nifty Bank"``).

    Parameters
    ----------
    instrument_key : str
        Upstox instrument key, e.g. ``"NSE_INDEX|Nifty 50"`` or
        ``"NSE_INDEX|Nifty Bank"``.
    n_1min : int
        Number of 1-minute candles to fetch (default 75 = 75 minutes
        of data).  These are resampled into 5min and 15min candles.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{"5min": df_5min, "15min": df_15min}``
        Each DataFrame has columns: ``date``, ``open``, ``high``,
        ``low``, ``close``, ``volume``.

    Raises
    ------
    FetchError
        On any Upstox API or network error.
    """
    logger.info(
        "Fetching multi-timeframe candles | instrument=%s | "
        "fetching %d × 1min → resample to 5min + 15min",
        instrument_key, n_1min,
    )

    # Fetch raw 1-minute candles (supported by Upstox free API)
    df_1min = get_live_candles(
        instrument_key=instrument_key,
        interval="1minute",
        n=n_1min,
    )

    # Resample to 5-minute and 15-minute candles
    df_5min = _resample_candles(df_1min, "5min")
    df_15min = _resample_candles(df_1min, "15min")

    logger.info(
        "Multi-timeframe resample complete — "
        "1min: %d rows → 5min: %d rows, 15min: %d rows.",
        len(df_1min), len(df_5min), len(df_15min),
    )

    return {"5min": df_5min, "15min": df_15min}


def _calculate_expiry_manually(index_name_upper: str) -> str:
    """
    Calculate the nearest weekly expiry date using Python datetime.

    Fallback used when the Upstox Options API is unavailable (e.g. 401).

    Rules
    -----
    - NIFTY   weekly expiry = nearest Thursday  (weekday 3)
    - BANKNIFTY weekly expiry = nearest Wednesday (weekday 2)
    - If today IS the expiry day and current time > 15:30 IST,
      advance to NEXT week's expiry day.

    Parameters
    ----------
    index_name_upper : str
        ``"NIFTY"`` or ``"BANKNIFTY"`` (already uppercased).

    Returns
    -------
    str
        Expiry date formatted as ``"DD-Mon-YYYY"``.
    """
    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    today = now_ist.date()

    # NIFTY → Thursday (weekday 3), BANKNIFTY → Wednesday (weekday 2)
    if index_name_upper == "BANKNIFTY":
        expiry_weekday = 2  # Wednesday
    else:
        expiry_weekday = 3  # Thursday

    # Days until the next expiry weekday (0 = today is that day)
    days_ahead = (expiry_weekday - today.weekday()) % 7

    if days_ahead == 0:
        # Today IS the expiry day
        market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        if now_ist > market_close:
            # Market closed — use next week's expiry
            days_ahead = 7

    expiry_date = today + timedelta(days=days_ahead)
    result = expiry_date.strftime("%d-%b-%Y")

    logger.info(
        "Using calculated expiry (API unavailable): %s",
        result,
    )
    return result


def get_nearest_expiry(index_name: str) -> str:
    """
    Return the nearest weekly expiry date for an index.

    First attempts to fetch option contracts from the Upstox Options API.
    If the API call fails (e.g. 401 Unauthorized), falls back to
    calculating the expiry date manually:

    - **NIFTY** weekly expiry → nearest Thursday
    - **BANKNIFTY** weekly expiry → nearest Wednesday
    - If today IS the expiry day and time > 15:30 IST → next week.

    Parameters
    ----------
    index_name : str
        ``"NIFTY"`` or ``"BANKNIFTY"``.

    Returns
    -------
    str
        Nearest expiry date string, e.g. ``"01-May-2026"``.

    Raises
    ------
    FetchError
        If the index name is invalid.
    """
    index_name_upper = index_name.upper()
    instrument_key = _INDEX_INSTRUMENT_MAP.get(index_name_upper)

    if instrument_key is None:
        valid = ", ".join(_INDEX_INSTRUMENT_MAP.keys())
        msg = f"Invalid index_name '{index_name}'. Must be one of: {valid}"
        logger.error(msg)
        raise FetchError(msg)

    logger.info(
        "Fetching nearest expiry | index=%s | instrument_key=%s",
        index_name_upper, instrument_key,
    )

    # ── Attempt 1: Upstox Options API ───────────────────────────────
    try:
        options_api = _get_options_api_instance()
        response = options_api.get_option_contracts(instrument_key)

        if not response.data:
            logger.warning(
                "No option contracts returned for %s — falling back to manual calculation.",
                instrument_key,
            )
            return _calculate_expiry_manually(index_name_upper)

        # Collect unique expiry dates from the contracts
        expiry_dates: set[str] = set()
        for contract in response.data:
            if hasattr(contract, "expiry") and contract.expiry:
                expiry_dates.add(str(contract.expiry))

        if not expiry_dates:
            logger.warning(
                "No expiry dates found in contracts for %s — falling back to manual calculation.",
                instrument_key,
            )
            return _calculate_expiry_manually(index_name_upper)

        # Parse expiry dates — Upstox returns them in YYYY-MM-DD format
        today = datetime.now().date()
        parsed_expiries: list[datetime] = []

        for exp_str in expiry_dates:
            try:
                exp_dt = datetime.strptime(exp_str[:10], "%Y-%m-%d")
                # Only consider future or today expiries
                if exp_dt.date() >= today:
                    parsed_expiries.append(exp_dt)
            except ValueError:
                logger.debug("Skipping unparseable expiry: %s", exp_str)
                continue

        if not parsed_expiries:
            logger.warning(
                "No future expiry dates found for %s — falling back to manual calculation.",
                instrument_key,
            )
            return _calculate_expiry_manually(index_name_upper)

        # Sort and pick the nearest one
        parsed_expiries.sort()
        nearest = parsed_expiries[0]

        # Format as "DD-Mon-YYYY" (e.g. "01-May-2026")
        result = nearest.strftime("%d-%b-%Y")

        logger.info(
            "Nearest expiry for %s: %s (from %d future expiries found).",
            index_name_upper, result, len(parsed_expiries),
        )
        return result

    except FetchError:
        # FetchError from _get_options_api_instance (no token) → fallback
        logger.warning(
            "FetchError during options API call — falling back to manual calculation.",
        )
        return _calculate_expiry_manually(index_name_upper)

    except ApiException as exc:
        logger.warning(
            "Upstox Options API error (status=%s, reason=%s) — "
            "falling back to manual expiry calculation.",
            exc.status, exc.reason,
        )
        return _calculate_expiry_manually(index_name_upper)

    except Exception as exc:
        logger.warning(
            "Unexpected error fetching option contracts: %s — "
            "falling back to manual expiry calculation.",
            exc,
        )
        return _calculate_expiry_manually(index_name_upper)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("  Nifty 50 / BankNifty Data Fetcher (Upstox v2) -- Smoke Test")
    print("=" * 64)

    # -- Test 1: get_historical_data -- Nifty 50 (last 30 days) ------
    print("\n[1] Testing get_historical_data -- Nifty 50 (last 30 days, 30min)...")
    try:
        today = datetime.now()
        from_dt = today - timedelta(days=30)
        hist_nifty = get_historical_data(
            instrument_key="NSE_INDEX|Nifty 50",
            interval="30minute",
            from_date=from_dt.strftime("%Y-%m-%d"),
            to_date=today.strftime("%Y-%m-%d"),
        )
        print(f"    [OK] Shape : {hist_nifty.shape}")
        print(f"    [OK] Cols  : {list(hist_nifty.columns)}")
        print(hist_nifty.head(2).to_string(index=False))
    except FetchError as e:
        print(f"    [FAIL] FetchError: {e.message}")

    # -- Test 2: get_historical_data -- BankNifty (last 30 days) -----
    print("\n[2] Testing get_historical_data -- BankNifty (last 30 days, 30min)...")
    try:
        hist_bank = get_historical_data(
            instrument_key="NSE_INDEX|Nifty Bank",
            interval="30minute",
            from_date=from_dt.strftime("%Y-%m-%d"),
            to_date=today.strftime("%Y-%m-%d"),
        )
        print(f"    [OK] Shape : {hist_bank.shape}")
        print(f"    [OK] Cols  : {list(hist_bank.columns)}")
        print(hist_bank.head(2).to_string(index=False))
    except FetchError as e:
        print(f"    [FAIL] FetchError: {e.message}")

    # -- Test 3: get_live_candles -- Nifty 50 (n=150) ----------------
    print("\n[3] Testing get_live_candles -- Nifty 50 (n=150, 30min)...")
    try:
        live_df = get_live_candles(
            instrument_key="NSE_INDEX|Nifty 50",
            interval="30minute",
            n=150,
        )
        print(f"    [OK] Shape : {live_df.shape}")
        print(f"    [OK] Cols  : {list(live_df.columns)}")
        print(live_df.head(2).to_string(index=False))
    except FetchError as e:
        print(f"    [FAIL] FetchError: {e.message}")

    # -- Test 4: get_live_candles_multi_timeframe -- BankNifty --------
    print("\n[4] Testing get_live_candles_multi_timeframe -- BankNifty...")
    try:
        multi = get_live_candles_multi_timeframe(
            instrument_key="NSE_INDEX|Nifty Bank",
            n_1min=75,
        )
        print(f"    [OK] 5min shape  : {multi['5min'].shape}")
        print(f"    [OK] 15min shape : {multi['15min'].shape}")
    except FetchError as e:
        print(f"    [FAIL] FetchError: {e.message}")

    # -- Test 5: get_nearest_expiry ----------------------------------
    print("\n[5] Testing get_nearest_expiry -- NIFTY...")
    try:
        exp_nifty = get_nearest_expiry("NIFTY")
        print(f"    [OK] Nearest NIFTY expiry: {exp_nifty}")
    except FetchError as e:
        print(f"    [FAIL] FetchError: {e.message}")

    print("\n[6] Testing get_nearest_expiry -- BANKNIFTY...")
    try:
        exp_bank = get_nearest_expiry("BANKNIFTY")
        print(f"    [OK] Nearest BANKNIFTY expiry: {exp_bank}")
    except FetchError as e:
        print(f"    [FAIL] FetchError: {e.message}")

    print("\n" + "=" * 64)
    print("  Smoke test complete.")
    print("=" * 64)
