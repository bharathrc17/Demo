"""
main.py — Entry Point for the Nifty 50 & BankNifty Signal Bot
===============================================================

Top-level orchestrator that connects the scheduler and Telegram notifier.

Execution sequence:
  1. Verify both model files exist (xgb_nifty.pkl, xgb_banknifty.pkl).
  2. Send a Telegram startup notification.
  3. Start the AsyncIOScheduler (5-min cycles + weekly retrain).
  4. Keep the event loop alive indefinitely.
  5. On KeyboardInterrupt (Ctrl+C), send "Bot stopped" and shut down.

Usage:
    python main.py
"""

import asyncio
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from src.notifier import _send_message, send_startup  # noqa: E402
from src.scheduler import start_scheduler  # noqa: E402

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
NIFTY_MODEL = os.path.join(_PROJECT_ROOT, "models", "xgb_nifty.pkl")
BANKNIFTY_MODEL = os.path.join(_PROJECT_ROOT, "models", "xgb_banknifty.pkl")

# ---------------------------------------------------------------------------
# Global logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def _check_models() -> bool:
    """
    Verify both model files exist on disk.
    Returns True if both are present, False otherwise.
    """
    missing = []
    if not os.path.exists(NIFTY_MODEL):
        missing.append(NIFTY_MODEL)
    if not os.path.exists(BANKNIFTY_MODEL):
        missing.append(BANKNIFTY_MODEL)

    if missing:
        logger.error("=" * 60)
        logger.error("  MISSING MODEL FILE(S):")
        for m in missing:
            logger.error("    - %s", m)
        logger.error("")
        logger.error("  Run 'python train_model.py' first to train both models.")
        logger.error("=" * 60)
        return False

    logger.info("Both model files found -- ready to start.")
    return True


async def main() -> None:
    """Main asynchronous execution loop."""
    logger.info("Starting Nifty 50 & BankNifty Signal Bot...")

    # 1. Check model files
    if not _check_models():
        print("\n  [ERROR] Model files missing. Run: python train_model.py\n")
        sys.exit(1)

    # 2. Send startup notification to Telegram
    await send_startup()

    # 3. Start the job scheduler
    scheduler = start_scheduler()

    # 4. Keep the process alive
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Main loop cancelled.")
    finally:
        scheduler.shutdown()
        logger.info("Scheduler shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")

        try:
            asyncio.run(
                _send_message("Bot stopped (KeyboardInterrupt)", context="shutdown")
            )
        except Exception as e:
            logger.error("Failed to send shutdown message: %s", e)

        sys.exit(0)
