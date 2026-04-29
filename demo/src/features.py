"""
src/features.py — Stateless Technical Indicator Feature Engineering
=====================================================================

This module takes a raw OHLCV DataFrame and returns it with all technical
indicator columns appended.  Works identically for Nifty 50 and BankNifty
-- any DataFrame with [date, open, high, low, close, volume] columns.

Functions:
    compute_features(df)
        Append 25 indicator columns to an OHLCV DataFrame.  Used for both
        offline training and live prediction.

    compute_adx(df) -> float
        Standalone ADX computation.  Returns the latest ADX(14) value for
        signal grading (trend strength assessment).

    compute_ema_trend(df_5min, df_15min) -> str
        Multi-timeframe EMA alignment check.  Returns "BULLISH",
        "BEARISH", or "MIXED" based on EMA9/EMA21 relationship across
        5-minute and 15-minute candles.

Indicators computed (using pandas_ta):
    RSI, MACD (line/signal/hist), Bollinger Bands (upper/mid/lower/width),
    ATR, EMA (9/21/50), EMA crossover flag, OBV, Stochastic (%K/%D),
    ADX, VWAP, VWAP distance, candle body size, upper/lower shadow,
    volume ratio, hour-of-day, day-of-week.

Usage:
    from src.features import compute_features, compute_adx, compute_ema_trend
    enriched_df = compute_features(ohlcv_df)
    adx_value   = compute_adx(ohlcv_df)
    trend       = compute_ema_trend(df_5min, df_15min)
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a raw OHLCV DataFrame with 20+ technical indicator columns.

    This function is **stateless** and **deterministic** — given the same
    input it always produces the same output.  No external state is read
    or mutated.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume``.  The ``date`` column should be parseable
        by ``pd.to_datetime``.

    Returns
    -------
    pd.DataFrame
        A copy of the input with indicator columns appended.  NaN values
        from indicator warm-up periods are forward/backward filled so
        that no rows are lost.

    Raises
    ------
    KeyError
        If any required OHLCV column is missing from the input.
    """
    # Work on a copy so the caller's DataFrame is never mutated.
    df = df.copy()

    # ------------------------------------------------------------------
    # Ensure correct dtypes
    # ------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], utc=True)
    # Strip timezone → naive datetime (pandas_ta VWAP chokes on tz-aware)
    df["date"] = df["date"].dt.tz_localize(None)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)

    # ------------------------------------------------------------------
    # 1. RSI (14)
    # ------------------------------------------------------------------
    df["rsi"] = ta.rsi(df["close"], length=14)

    # ------------------------------------------------------------------
    # 2. MACD (12, 26, 9)  →  macd, macd_signal, macd_hist
    # ------------------------------------------------------------------
    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd_df.iloc[:, 0].values
    df["macd_signal"] = macd_df.iloc[:, 1].values
    df["macd_hist"] = macd_df.iloc[:, 2].values

    # ------------------------------------------------------------------
    # 3. Bollinger Bands (20, 2)  →  bb_upper, bb_mid, bb_lower, bb_width
    # ------------------------------------------------------------------
    bb_df = ta.bbands(df["close"], length=20, std=2.0)
    df["bb_lower"] = bb_df.iloc[:, 0].values
    df["bb_mid"] = bb_df.iloc[:, 1].values
    df["bb_upper"] = bb_df.iloc[:, 2].values
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ------------------------------------------------------------------
    # 4. ATR (14)
    # ------------------------------------------------------------------
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # ------------------------------------------------------------------
    # 5. EMA (9, 21, 50)
    # ------------------------------------------------------------------
    df["ema9"] = ta.ema(df["close"], length=9)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema50"] = ta.ema(df["close"], length=50)

    # ------------------------------------------------------------------
    # 6. EMA crossover signal:  1 if ema9 > ema21, else 0
    # ------------------------------------------------------------------
    df["ema_cross"] = (df["ema9"] > df["ema21"]).astype(int)

    # ------------------------------------------------------------------
    # 7. OBV (On-Balance Volume)
    # ------------------------------------------------------------------
    df["obv"] = ta.obv(df["close"], df["volume"])

    # ------------------------------------------------------------------
    # 8. Stochastic Oscillator (14, 3, 3)  →  stoch_k, stoch_d
    # ------------------------------------------------------------------
    stoch_df = ta.stoch(df["high"], df["low"], df["close"],
                        k=14, d=3, smooth_k=3)
    df["stoch_k"] = stoch_df.iloc[:, 0].values
    df["stoch_d"] = stoch_df.iloc[:, 1].values

    # ------------------------------------------------------------------
    # 9. ADX (14)
    # ------------------------------------------------------------------
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx_df.iloc[:, 0].values

    # ------------------------------------------------------------------
    # 10. VWAP (session-based via pandas_ta)
    # pandas_ta requires a DatetimeIndex to compute VWAP.
    # Set date as index → compute → reset index back to column.
    # ------------------------------------------------------------------
    df = df.set_index("date")
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df = df.reset_index()

    # ------------------------------------------------------------------
    # 11. Distance from VWAP:  (close - vwap) / vwap
    # ------------------------------------------------------------------
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["vwap"]

    # ------------------------------------------------------------------
    # 12. Candle body size:  abs(close - open)
    # ------------------------------------------------------------------
    df["body_size"] = (df["close"] - df["open"]).abs()

    # ------------------------------------------------------------------
    # 13. Upper shadow:  high - max(open, close)
    # ------------------------------------------------------------------
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)

    # ------------------------------------------------------------------
    # 14. Lower shadow:  min(open, close) - low
    # ------------------------------------------------------------------
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # ------------------------------------------------------------------
    # 15. Volume ratio:  volume / 20-period rolling mean of volume
    # ------------------------------------------------------------------
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

    # ------------------------------------------------------------------
    # 16. Hour of day (0–23) from date column
    # ------------------------------------------------------------------
    df["hour"] = df["date"].dt.hour

    # ------------------------------------------------------------------
    # 17. Day of week (0=Monday … 4=Friday)
    # ------------------------------------------------------------------
    df["dow"] = df["date"].dt.dayofweek

    # ------------------------------------------------------------------
    # Fill NaN from indicator warm-up periods instead of dropping rows.
    # Forward-fill first, then backward-fill any remaining leading NaNs.
    # ------------------------------------------------------------------
    df = df.ffill().bfill()
    df.reset_index(drop=True, inplace=True)

    return df


# ── Convenience: ordered list of feature column names for the model ──
FEATURE_COLUMNS: list[str] = [
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "atr",
    "ema9", "ema21", "ema50", "ema_cross",
    "obv",
    "stoch_k", "stoch_d",
    "adx",
    "vwap", "vwap_dist",
    "body_size", "upper_shadow", "lower_shadow",
    "vol_ratio",
    "hour", "dow",
]


# ----------------------------------------------------------------------
# Standalone ADX for signal grading
# ----------------------------------------------------------------------

def compute_adx(df: pd.DataFrame, length: int = 14) -> float:
    """
    Compute the latest ADX value from an OHLCV DataFrame.

    This is a convenience function used by the signal engine to grade
    the strength of a trend independently of the full feature pipeline.
    Works identically for Nifty 50 and BankNifty data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``high``, ``low``, ``close``.
        Should have at least ``2 * length`` rows for a meaningful result.
    length : int
        ADX look-back period (default 14).

    Returns
    -------
    float
        The most recent ADX value.  Returns ``0.0`` if the DataFrame is
        too short or the computation yields NaN.
    """
    if len(df) < length + 1:
        return 0.0

    work = df.copy()
    for col in ("high", "low", "close"):
        work[col] = work[col].astype(float)

    adx_df = ta.adx(work["high"], work["low"], work["close"], length=length)
    if adx_df is None or adx_df.empty:
        return 0.0

    # First column of ta.adx output is ADX itself
    latest = adx_df.iloc[:, 0].iloc[-1]
    if pd.isna(latest):
        return 0.0

    return float(latest)


# ----------------------------------------------------------------------
# Multi-timeframe EMA trend alignment
# ----------------------------------------------------------------------

def compute_ema_trend(
    df_5min: pd.DataFrame,
    df_15min: pd.DataFrame,
) -> str:
    """
    Determine trend alignment across 5-minute and 15-minute timeframes.

    Computes EMA9 and EMA21 on the ``close`` column of each DataFrame,
    checks whether EMA9 > EMA21 (bullish) or EMA9 < EMA21 (bearish)
    at the most recent candle, and returns a consensus label.

    Works identically for Nifty 50 and BankNifty data.

    Parameters
    ----------
    df_5min : pd.DataFrame
        5-minute OHLCV candles.  Must contain a ``close`` column.
    df_15min : pd.DataFrame
        15-minute OHLCV candles.  Must contain a ``close`` column.

    Returns
    -------
    str
        ``"BULLISH"``  — EMA9 > EMA21 on **both** timeframes.
        ``"BEARISH"``  — EMA9 < EMA21 on **both** timeframes.
        ``"MIXED"``    — the two timeframes disagree, or data is
                         insufficient to compute EMAs.
    """
    def _ema_bias(df: pd.DataFrame) -> str:
        """Return 'BULL', 'BEAR', or 'MIXED' for a single timeframe."""
        if df is None or df.empty or len(df) < 21:
            return "MIXED"

        close = df["close"].astype(float)
        ema9 = ta.ema(close, length=9)
        ema21 = ta.ema(close, length=21)

        if ema9 is None or ema21 is None:
            return "MIXED"

        latest_9 = ema9.iloc[-1]
        latest_21 = ema21.iloc[-1]

        if pd.isna(latest_9) or pd.isna(latest_21):
            return "MIXED"

        if latest_9 > latest_21:
            return "BULL"
        elif latest_9 < latest_21:
            return "BEAR"
        else:
            return "MIXED"

    bias_5 = _ema_bias(df_5min)
    bias_15 = _ema_bias(df_15min)

    if bias_5 == "BULL" and bias_15 == "BULL":
        return "BULLISH"
    elif bias_5 == "BEAR" and bias_15 == "BEAR":
        return "BEARISH"
    else:
        return "MIXED"


# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Resolve project root so config imports work when run directly
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _PROJECT_ROOT)

    NIFTY_PATH = os.path.join(_PROJECT_ROOT, "data", "nifty_historical.csv")
    BANKNIFTY_PATH = os.path.join(_PROJECT_ROOT, "data", "banknifty_historical.csv")

    print("=" * 60)
    print("  Feature Engineering -- Smoke Test")
    print("=" * 60)

    # -- Test 1: compute_features on Nifty ---------------------------
    print("\n[1] compute_features -- Nifty 50...")
    if not os.path.exists(NIFTY_PATH):
        print(f"    [SKIP] File not found: {NIFTY_PATH}")
    else:
        raw_df = pd.read_csv(NIFTY_PATH)
        print(f"    Raw shape: {raw_df.shape}")
        enriched_df = compute_features(raw_df)
        print(f"    Enriched shape: {enriched_df.shape}")
        print(f"    Feature columns ({len(FEATURE_COLUMNS)}):")
        for col in FEATURE_COLUMNS:
            present = "[OK]" if col in enriched_df.columns else "[MISS]"
            print(f"      {present} {col}")
        nan_total = enriched_df[FEATURE_COLUMNS].isna().sum().sum()
        print(f"    NaN total: {nan_total}")

    # -- Test 2: compute_features on BankNifty -----------------------
    print("\n[2] compute_features -- BankNifty...")
    if not os.path.exists(BANKNIFTY_PATH):
        print(f"    [SKIP] File not found: {BANKNIFTY_PATH}")
    else:
        raw_bank = pd.read_csv(BANKNIFTY_PATH)
        print(f"    Raw shape: {raw_bank.shape}")
        enriched_bank = compute_features(raw_bank)
        print(f"    Enriched shape: {enriched_bank.shape}")
        nan_total = enriched_bank[FEATURE_COLUMNS].isna().sum().sum()
        print(f"    NaN total: {nan_total}")

    # -- Test 3: compute_adx ----------------------------------------
    print("\n[3] compute_adx...")
    if os.path.exists(NIFTY_PATH):
        raw_df = pd.read_csv(NIFTY_PATH)
        adx_val = compute_adx(raw_df)
        print(f"    [OK] Nifty ADX(14) = {adx_val:.2f}")
    if os.path.exists(BANKNIFTY_PATH):
        raw_bank = pd.read_csv(BANKNIFTY_PATH)
        adx_val_bank = compute_adx(raw_bank)
        print(f"    [OK] BankNifty ADX(14) = {adx_val_bank:.2f}")

    # -- Test 4: compute_ema_trend -----------------------------------
    print("\n[4] compute_ema_trend...")
    # Create synthetic 5min/15min data from the tail of Nifty data
    if os.path.exists(NIFTY_PATH):
        raw_df = pd.read_csv(NIFTY_PATH)
        # Simulate: use last 75 rows as "5min" and last 25 as "15min"
        df_5 = raw_df.tail(75).reset_index(drop=True)
        df_15 = raw_df.tail(25).reset_index(drop=True)
        trend = compute_ema_trend(df_5, df_15)
        print(f"    [OK] Nifty EMA trend = {trend}")
    if os.path.exists(BANKNIFTY_PATH):
        raw_bank = pd.read_csv(BANKNIFTY_PATH)
        df_5b = raw_bank.tail(75).reset_index(drop=True)
        df_15b = raw_bank.tail(25).reset_index(drop=True)
        trend_b = compute_ema_trend(df_5b, df_15b)
        print(f"    [OK] BankNifty EMA trend = {trend_b}")

    print("\n" + "=" * 60)
    print("  Smoke test complete.")
    print("=" * 60)

