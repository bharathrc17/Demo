"""
src/notifier.py — Telegram Alert Notifier (python-telegram-bot 20.7, async)
=============================================================================

Delivers formatted BUY/SELL (CE/PE) signal alerts and status messages to
Telegram.  Supports both Nifty 50 and BankNifty with dynamic index names.

Public API (all async):
    send_signal_alert(signal_dict)  -- formatted CE/PE alert
    send_error_alert(message)       -- plain error alert
    send_startup()                  -- bot-started notification
    _send_message(text)             -- low-level send (also used by main.py)

Credentials are read from config.py (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
All send attempts are logged to logs/notifier.log.

Usage:
    import asyncio
    from src.notifier import send_signal_alert, send_startup

    asyncio.run(send_startup())
    asyncio.run(send_signal_alert(signal_dict))
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from telegram import Bot
from telegram.error import TelegramError

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config import Config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — dedicated notifier.log with rotation
# ---------------------------------------------------------------------------
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logger = logging.getLogger("notifier")
logger.setLevel(logging.DEBUG)

_fh = RotatingFileHandler(
    filename=os.path.join(_LOG_DIR, "notifier.log"),
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
# Telegram bot instance
# ---------------------------------------------------------------------------
_BOT_TOKEN: str = Config.TELEGRAM_BOT_TOKEN
_CHAT_ID: str = Config.TELEGRAM_CHAT_ID


def _get_bot() -> Bot:
    """Return a telegram.Bot instance configured with the token from Config."""
    if not _BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in config / .env")
    return Bot(token=_BOT_TOKEN)


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

def _format_signal_message(s: dict) -> str:
    """
    Build the signal alert message from a signal dict produced by
    ``run_signal()`` in signal_engine.py.

    CE signals use a green circle, PE signals use a red circle.
    """
    emoji = "\U0001f7e2" if s["direction"] == "CE" else "\U0001f534"  # green / red circle

    return (
        f"{emoji} NEW SIGNAL DETECTED\n"
        f"{s['index_name']} {s['strike']} {s['direction']} | {s['grade']} Setup\n"
        f"Expiry: {s['expiry']} (Verified from Live Instrument Master)\n"
        f"Time: {s['timestamp']} IST\n"
        f"\n"
        f"ENTRY: \u20b9{s['entry_low']:.2f} - \u20b9{s['entry_high']:.2f}\n"
        f"STOP LOSS: \u20b9{s['stop_loss']:.2f}\n"
        f"TARGET: \u20b9{s['target']:.2f}\n"
        f"R:R Ratio: {s['rr_ratio']}\n"
        f"\n"
        f"ML Score: {s['ml_score']}% | {s['confidence_label']}\n"
        f"Trend: {s['trend']}\n"
        f"\n"
        f"Checklist:\n"
        f"\u2610 EMA Aligned\n"
        f"\u2610 Breakout\n"
        f"\u2610 Retest\n"
        f"\u2610 Confirmed"
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def send_signal_alert(signal_dict: dict) -> bool:
    """
    Send a formatted CE/PE signal alert to Telegram.

    Parameters
    ----------
    signal_dict : dict
        Signal dict from ``signal_engine.run_signal()``.

    Returns
    -------
    bool
        True if the message was sent successfully.
    """
    message = _format_signal_message(signal_dict)
    context = (
        f"signal={signal_dict['index_name']} "
        f"{signal_dict['strike']} {signal_dict['direction']}"
    )
    return await _send_message(message, context=context)


async def send_error_alert(message: str) -> bool:
    """
    Send a plain error / warning alert to Telegram.

    Parameters
    ----------
    message : str
        The error description to send.

    Returns
    -------
    bool
        True if delivered successfully.
    """
    text = f"\U0001f527 Bot Error Alert\n\n{message}"
    return await _send_message(text, context="error_alert")


async def send_startup() -> bool:
    """
    Send a one-time startup notification so the user knows the bot is
    live and monitoring both Nifty 50 and BankNifty.

    Returns
    -------
    bool
        True if delivered successfully.
    """
    text = (
        "\U0001f916 Signal Bot started -- monitoring Nifty 50 & BankNifty "
        "every 5 min during market hours (9:15-3:30 IST)"
    )
    return await _send_message(text, context="startup")


# ---------------------------------------------------------------------------
# Internal send helper
# ---------------------------------------------------------------------------

async def _send_message(text: str, context: str = "") -> bool:
    """
    Low-level helper that sends *text* to TELEGRAM_CHAT_ID via the Bot API.

    All Telegram errors are caught, logged, and result in a False return
    so the caller (and the bot's main loop) never crashes.

    Parameters
    ----------
    text : str
        The message body to send.
    context : str
        A short label for log readability (e.g. "signal=NIFTY", "startup").

    Returns
    -------
    bool
        True on success, False on any failure.
    """
    if not _CHAT_ID:
        logger.error("TELEGRAM_CHAT_ID is not set -- cannot send message.")
        return False

    try:
        bot = _get_bot()
        await bot.send_message(chat_id=_CHAT_ID, text=text)
        logger.info("Message sent [%s] -> chat %s", context, _CHAT_ID)
        return True

    except TelegramError as exc:
        logger.error("Telegram API error [%s]: %s", context, exc)
        return False

    except ValueError as exc:
        logger.error("Configuration error [%s]: %s", context, exc)
        return False

    except Exception as exc:
        logger.exception("Unexpected error sending message [%s]: %s", context, exc)
        return False
