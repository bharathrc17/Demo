"""
config.py — Central Configuration for the Nifty 50 & BankNifty Signal Bot
===========================================================================

This module stores all configurable parameters used across the project:

  - Upstox API credentials (api_key, api_secret, access_token, redirect_uri)
  - Telegram bot token and chat ID for alert delivery
  - Trading parameters (instrument keys, candle interval)
  - Model paths and thresholds for both Nifty and BankNifty
  - Signal grading parameters (ADX, confidence thresholds)
  - Scheduler intervals (data-fetch frequency, signal-run cadence)
  - Logging configuration (log level, log file path)

Values are loaded from environment variables (via python-dotenv) so that
secrets never appear in source control.

Usage:
    from config import Config
    api_key = Config.UPSTOX_API_KEY
"""

import os
from dotenv import load_dotenv

# Load .env file if present (ignored in production containers)
load_dotenv()


class Config:
    """
    Centralised configuration namespace.

    All values default to None / sensible fallbacks and are expected to be
    overridden via environment variables or a .env file.
    """

    # -- Upstox API ----------------------------------------------------------
    UPSTOX_API_KEY: str = os.getenv("UPSTOX_API_KEY", "")
    UPSTOX_API_SECRET: str = os.getenv("UPSTOX_API_SECRET", "")
    UPSTOX_ACCESS_TOKEN: str = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    UPSTOX_REDIRECT_URI: str = os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1")

    # -- Instrument Settings -------------------------------------------------
    # Nifty 50 instrument key for Upstox
    INSTRUMENT_KEY: str = os.getenv("INSTRUMENT_KEY", "NSE_INDEX|Nifty 50")
    CANDLE_INTERVAL: str = os.getenv("CANDLE_INTERVAL", "5minute")

    # Per-index instrument keys
    NIFTY_INSTRUMENT_KEY: str = "NSE_INDEX|Nifty 50"
    BANKNIFTY_INSTRUMENT_KEY: str = "NSE_INDEX|Nifty Bank"

    # -- Model Paths ---------------------------------------------------------
    NIFTY_MODEL_PATH: str = os.getenv("NIFTY_MODEL_PATH", "models/xgb_nifty.pkl")
    BANKNIFTY_MODEL_PATH: str = os.getenv("BANKNIFTY_MODEL_PATH", "models/xgb_banknifty.pkl")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/xgb_signal_model.json")
    FEATURE_COLS_PATH: str = os.getenv("FEATURE_COLS_PATH", "models/feature_columns.json")

    # -- Signal Grading Thresholds -------------------------------------------
    CONFIDENCE_THRESHOLD_A: float = float(os.getenv("CONFIDENCE_THRESHOLD_A", "0.72"))
    CONFIDENCE_THRESHOLD_B: float = float(os.getenv("CONFIDENCE_THRESHOLD_B", "0.65"))
    ADX_THRESHOLD: float = float(os.getenv("ADX_THRESHOLD", "25"))
    SIGNAL_COOLDOWN_MINUTES: int = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", "30"))

    # -- Legacy threshold (kept for backward compat) -------------------------
    PREDICTION_THRESHOLD: float = float(os.getenv("PREDICTION_THRESHOLD", "0.6"))

    # -- Telegram Notifications ----------------------------------------------
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # -- Scheduler Settings --------------------------------------------------
    FETCH_INTERVAL_MINUTES: int = int(os.getenv("FETCH_INTERVAL_MINUTES", "5"))

    # -- Logging -------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/bot.log")

    # -- Data Storage --------------------------------------------------------
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
