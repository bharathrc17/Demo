"""
src/scheduler.py — APScheduler Job Orchestrator (Dual-Index)
=============================================================

Schedules periodic jobs for the Nifty 50 & BankNifty signal bot:

  1. run_cycle (every 5 min, Mon-Fri, 9:15-15:30 IST):
     Runs run_signal() for both Nifty and BankNifty, sends alerts on hit.

  2. retrain_job (every Sunday at midnight IST):
     Runs train_model.py offline via subprocess.

All times are strictly bound to IST (Asia/Kolkata).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config import Config  # noqa: E402
from src.notifier import _send_message, send_error_alert, send_signal_alert  # noqa: E402
from src.signal_engine import run_signal  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("scheduler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)

IST = pytz.timezone("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

async def run_cycle() -> None:
    """
    Fires every 5 minutes on weekdays.

    Runs the full signal pipeline for both Nifty 50 and BankNifty.
    Sends a Telegram alert if an actionable signal is generated.
    """
    now = datetime.now(IST)
    time_val = now.hour * 100 + now.minute

    # Strictly restrict to 09:15 - 15:30 IST market hours
    if not (915 <= time_val <= 1530):
        logger.debug("Outside market hours (%d) -- skipping cycle.", time_val)
        return

    logger.info("Starting run cycle...")

    # -- Nifty 50 -----------------------------------------------------------
    try:
        nifty_signal = run_signal(Config.NIFTY_INSTRUMENT_KEY, "NIFTY")
        if nifty_signal is not None:
            await send_signal_alert(nifty_signal)
        else:
            logger.info("No signal for NIFTY.")
    except Exception as e:
        err_msg = f"Exception in run_cycle (NIFTY): {e}"
        logger.exception(err_msg)
        await send_error_alert(err_msg)

    # -- BankNifty ----------------------------------------------------------
    try:
        banknifty_signal = run_signal(Config.BANKNIFTY_INSTRUMENT_KEY, "BANKNIFTY")
        if banknifty_signal is not None:
            await send_signal_alert(banknifty_signal)
        else:
            logger.info("No signal for BANKNIFTY.")
    except Exception as e:
        err_msg = f"Exception in run_cycle (BANKNIFTY): {e}"
        logger.exception(err_msg)
        await send_error_alert(err_msg)


async def retrain_job() -> None:
    """
    Fires every Sunday at midnight IST.
    Runs train_model.py offline via subprocess and reports the result.
    """
    logger.info("Starting weekly model retrain job...")
    try:
        script_path = os.path.join(_PROJECT_ROOT, "train_model.py")

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("Retrain stdout:\n%s", result.stdout[-1000:])
            logger.info("Models retrained successfully.")
            await _send_message("Model retrained successfully (Nifty + BankNifty)")
        else:
            err_msg = f"Model retrain failed (code {result.returncode}):\n{result.stderr[-500:]}"
            logger.error(err_msg)
            await send_error_alert(err_msg)

    except Exception as e:
        err_msg = f"Exception in retrain_job: {e}"
        logger.exception(err_msg)
        await send_error_alert(err_msg)


# ---------------------------------------------------------------------------
# Scheduler initialisation
# ---------------------------------------------------------------------------

def start_scheduler() -> AsyncIOScheduler:
    """
    Creates, configures, and starts the AsyncIOScheduler.

    Returns
    -------
    AsyncIOScheduler
        The active scheduler instance.
    """
    scheduler = AsyncIOScheduler(timezone=IST)

    # Job 1: Every 5 minutes, Mon-Fri, during trading hours
    scheduler.add_job(
        run_cycle,
        "cron",
        day_of_week="mon-fri",
        hour="9-15",
        minute="*/5",
        id="run_cycle_job",
        replace_existing=True,
    )

    # Job 2: Every Sunday at midnight
    scheduler.add_job(
        retrain_job,
        "cron",
        day_of_week="sun",
        hour=0,
        minute=0,
        id="retrain_job",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started successfully.")

    return scheduler
