"""
src/signal_engine.py — Dual-Index Signal Prediction & Filtering Engine
========================================================================

Core decision-maker for both Nifty 50 and BankNifty.  Loads per-index
XGBoost models, fetches live data, computes features, applies grading
rules (ADX + confidence + EMA trend alignment), and returns a fully
populated signal dict ready for Telegram delivery.

Grading rules:
    Grade A : confidence > 72 % AND ADX > 25
    Grade B : confidence 65-72 %
    Avoid   : below 65 % -> return None

Trend filtering:
    MIXED on Grade A -> downgrade to Grade B
    MIXED on Grade B -> return None

Cooldown:
    Separate 30-min cooldown per index.
    No signals during 09:15-09:30 IST (opening-bell blackout).
    No signals after 15:15 IST (end-of-day cutoff).

Usage:
    from src.signal_engine import run_signal
    result = run_signal("NSE_INDEX|Nifty 50", "NIFTY")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config import Config  # noqa: E402
from src.features import FEATURE_COLUMNS, compute_adx, compute_ema_trend, compute_features  # noqa: E402
from src.fetcher import (  # noqa: E402
    FetchError,
    get_live_candles,
    get_live_candles_multi_timeframe,
    get_nearest_expiry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NIFTY_MODEL_PATH = os.path.join(_PROJECT_ROOT, Config.NIFTY_MODEL_PATH)
BANKNIFTY_MODEL_PATH = os.path.join(_PROJECT_ROOT, Config.BANKNIFTY_MODEL_PATH)
FEATURE_COLS_PATH = os.path.join(_PROJECT_ROOT, "models", "feature_columns.json")
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")

IST = pytz.timezone("Asia/Kolkata")

# Opening-bell blackout (09:15 - 09:30 IST)
BLACKOUT_START_HOUR, BLACKOUT_START_MIN = 9, 15
BLACKOUT_END_HOUR, BLACKOUT_END_MIN = 9, 30

# End-of-day cutoff — no signals after 15:15 IST
EOD_CUTOFF_HOUR, EOD_CUTOFF_MIN = 15, 15

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("signal_engine")
logger.setLevel(logging.DEBUG)

_fh = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "signals.log"),
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
# Load models & feature columns at module import time
# ---------------------------------------------------------------------------
_models: dict[str, object] = {}
_feature_columns: list[str] = []

# -- Nifty model --
try:
    if os.path.exists(NIFTY_MODEL_PATH):
        _models["NIFTY"] = joblib.load(NIFTY_MODEL_PATH)
        logger.info("Nifty model loaded from %s", NIFTY_MODEL_PATH)
    else:
        logger.warning("Nifty model not found at %s", NIFTY_MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load Nifty model: %s", exc)

# -- BankNifty model --
try:
    if os.path.exists(BANKNIFTY_MODEL_PATH):
        _models["BANKNIFTY"] = joblib.load(BANKNIFTY_MODEL_PATH)
        logger.info("BankNifty model loaded from %s", BANKNIFTY_MODEL_PATH)
    else:
        logger.warning("BankNifty model not found at %s", BANKNIFTY_MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load BankNifty model: %s", exc)

# -- Feature columns --
try:
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
            _feature_columns = json.load(f)
        logger.info("Feature columns loaded (%d cols) from %s", len(_feature_columns), FEATURE_COLS_PATH)
    else:
        logger.warning("Feature columns file not found at %s — using FEATURE_COLUMNS from features.py", FEATURE_COLS_PATH)
        _feature_columns = list(FEATURE_COLUMNS)
except Exception as exc:
    logger.error("Failed to load feature columns: %s — using defaults", exc)
    _feature_columns = list(FEATURE_COLUMNS)

# ---------------------------------------------------------------------------
# Per-index cooldown state
# ---------------------------------------------------------------------------
_last_signal_times: dict[str, datetime] = {}


# ---------------------------------------------------------------------------
# Time-window helpers
# ---------------------------------------------------------------------------

def _now_ist() -> datetime:
    """Return the current time in IST, timezone-aware."""
    return datetime.now(IST)


def _is_in_blackout(now: datetime) -> bool:
    """True during the 09:15-09:30 IST opening-bell blackout."""
    blackout_start = now.replace(hour=BLACKOUT_START_HOUR, minute=BLACKOUT_START_MIN, second=0, microsecond=0)
    blackout_end = now.replace(hour=BLACKOUT_END_HOUR, minute=BLACKOUT_END_MIN, second=0, microsecond=0)
    return blackout_start <= now < blackout_end


def _is_past_cutoff(now: datetime) -> bool:
    """True after the 15:15 IST end-of-day signal cutoff."""
    cutoff = now.replace(hour=EOD_CUTOFF_HOUR, minute=EOD_CUTOFF_MIN, second=0, microsecond=0)
    return now >= cutoff


def _is_in_cooldown(index_name: str, now: datetime) -> bool:
    """True if fewer than SIGNAL_COOLDOWN_MINUTES since last signal for this index."""
    last = _last_signal_times.get(index_name)
    if last is None:
        return False
    elapsed = (now - last).total_seconds() / 60.0
    return elapsed < Config.SIGNAL_COOLDOWN_MINUTES


# ---------------------------------------------------------------------------
# Strike & option-price helpers
# ---------------------------------------------------------------------------

def _calculate_strike(current_price: float, index_name: str) -> int:
    """Round current_price to the nearest strike for the given index."""
    if index_name == "BANKNIFTY":
        return int(round(current_price / 100) * 100)
    else:  # NIFTY
        return int(round(current_price / 50) * 50)


def _estimate_option_price(current_price: float, index_name: str) -> float:
    """Rough ATM option premium estimate."""
    if index_name == "BANKNIFTY":
        return current_price * 0.005
    else:  # NIFTY
        return current_price * 0.006


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_signal(instrument_key: str, index_name: str) -> Optional[dict]:
    """
    End-to-end signal pipeline for a single index.

    Parameters
    ----------
    instrument_key : str
        ``"NSE_INDEX|Nifty 50"`` or ``"NSE_INDEX|Nifty Bank"``.
    index_name : str
        ``"NIFTY"`` or ``"BANKNIFTY"``.

    Returns
    -------
    dict or None
        Fully populated signal dict if an actionable signal is found,
        otherwise ``None``.
    """
    now = _now_ist()
    timestamp_str = now.isoformat()

    # -- Guard: model loaded? ------------------------------------------------
    model = _models.get(index_name)
    if model is None:
        logger.error("No model loaded for %s -- cannot predict.", index_name)
        return None

    if not _feature_columns:
        logger.error("Feature columns not loaded -- cannot predict.")
        return None

    # -- Guard: blackout window ----------------------------------------------
    if _is_in_blackout(now):
        logger.debug("Opening-bell blackout -- skipping %s.", index_name)
        return None

    # -- Guard: end-of-day cutoff --------------------------------------------
    if _is_past_cutoff(now):
        logger.debug("Past EOD cutoff (15:15 IST) -- skipping %s.", index_name)
        return None

    # -- Guard: cooldown -----------------------------------------------------
    if _is_in_cooldown(index_name, now):
        logger.debug("Cooldown active for %s -- skipping.", index_name)
        return None

    try:
        # ==============================================================
        # 1. Fetch live 30-min candles
        # ==============================================================
        df = get_live_candles(instrument_key=instrument_key, interval="30minute", n=150)
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for %s (%d rows).", index_name, len(df))
            return None

        current_price = float(df["close"].iloc[-1])

        # ==============================================================
        # 2. Compute features
        # ==============================================================
        enriched = compute_features(df)
        if enriched.empty:
            logger.warning("compute_features returned empty for %s.", index_name)
            return None

        # ==============================================================
        # 3. Model prediction
        # ==============================================================
        missing = [c for c in _feature_columns if c not in enriched.columns]
        if missing:
            logger.error("Missing feature columns for %s: %s", index_name, missing)
            return None

        X_latest = enriched[_feature_columns].iloc[[-1]]
        proba = model.predict_proba(X_latest)
        prob_buy = float(proba[0][1])
        prob_sell = 1.0 - prob_buy

        # ==============================================================
        # 4. Direction & confidence
        # ==============================================================
        if prob_buy >= prob_sell:
            direction = "CE"
            confidence = prob_buy
        else:
            direction = "PE"
            confidence = prob_sell

        # ==============================================================
        # 5. Minimum confidence gate
        # ==============================================================
        if confidence < Config.CONFIDENCE_THRESHOLD_B:
            logger.info(
                "%s | confidence %.1f%% below %.0f%% threshold -- no signal.",
                index_name, confidence * 100, Config.CONFIDENCE_THRESHOLD_B * 100,
            )
            return None

        # ==============================================================
        # 6. ADX (trend strength)
        # ==============================================================
        adx_val = compute_adx(enriched)

        # ==============================================================
        # 7. EMA trend (multi-timeframe)
        # ==============================================================
        try:
            multi = get_live_candles_multi_timeframe(instrument_key=instrument_key, n_1min=75)
            trend_label = compute_ema_trend(multi["5min"], multi["15min"])
        except FetchError as exc:
            logger.warning("Multi-TF fetch failed for %s: %s -- treating as MIXED.", index_name, exc)
            trend_label = "MIXED"

        # ==============================================================
        # 8. Grading
        # ==============================================================
        if confidence > Config.CONFIDENCE_THRESHOLD_A and adx_val > Config.ADX_THRESHOLD:
            grade = "A"
        elif confidence >= Config.CONFIDENCE_THRESHOLD_B:
            grade = "B"
        else:
            return None

        # -- Trend filtering -----------------------------------------------
        if trend_label == "MIXED" and grade == "A":
            grade = "B"
            logger.info("%s | Grade A downgraded to B (trend MIXED).", index_name)
        elif trend_label == "MIXED" and grade == "B":
            logger.info("%s | Grade B with MIXED trend -- skipping.", index_name)
            return None

        # ==============================================================
        # 9. Strike & option price
        # ==============================================================
        strike = _calculate_strike(current_price, index_name)
        opt_price = _estimate_option_price(current_price, index_name)

        entry_low = round(opt_price - 1, 2)
        entry_high = round(opt_price + 1, 2)
        stop_loss = round(entry_low * 0.82, 2)
        target = round(entry_high * 1.38, 2)

        # ==============================================================
        # 10. Expiry
        # ==============================================================
        expiry = get_nearest_expiry(index_name)

        # ==============================================================
        # 11. Build signal dict
        # ==============================================================
        ml_score = int(confidence * 100)
        confidence_label = "High Confidence" if grade == "A" else "Moderate Confidence"

        if trend_label == "BULLISH":
            trend_str = "BULLISH (5m + 15m aligned)"
        elif trend_label == "BEARISH":
            trend_str = "BEARISH (5m + 15m aligned)"
        else:
            trend_str = "MIXED"

        signal = {
            "index_name": index_name,
            "direction": direction,
            "strike": strike,
            "expiry": expiry,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop_loss": stop_loss,
            "target": target,
            "rr_ratio": "1:2.0",
            "ml_score": ml_score,
            "grade": grade,
            "confidence_label": confidence_label,
            "trend": trend_str,
            "current_price": current_price,
            "timestamp": timestamp_str,
        }

        # Update cooldown
        _last_signal_times[index_name] = now

        logger.info(
            "SIGNAL | %s | %s %d %s | Grade %s | ML %d%% | ADX %.1f | %s | price=%.2f",
            index_name, index_name, strike, direction, grade,
            ml_score, adx_val, trend_str, current_price,
        )

        return signal

    except FetchError as exc:
        logger.error("FetchError in run_signal(%s): %s", index_name, exc.message)
        return None

    except Exception as exc:
        logger.exception("Unexpected error in run_signal(%s): %s", index_name, exc)
        return None


def reset_cooldown(index_name: Optional[str] = None) -> None:
    """
    Clear cooldown timer(s).

    Parameters
    ----------
    index_name : str or None
        If given, clears only that index.  If None, clears all.
    """
    if index_name:
        _last_signal_times.pop(index_name, None)
        logger.info("Cooldown reset for %s.", index_name)
    else:
        _last_signal_times.clear()
        logger.info("All cooldown timers reset.")
