"""Simple utility to wait for database availability."""

from __future__ import annotations

import logging
import sys
import time

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from .db import create_all, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [wait_for_db] %(message)s")
logger = logging.getLogger(__name__)


def wait(max_retries: int = 30, delay: float = 2.0) -> int:
    for attempt in range(1, max_retries + 1):
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful.")
            create_all()
            return 0
        except OperationalError as exc:
            logger.warning("Database unavailable (attempt %s/%s): %s", attempt, max_retries, exc)
            time.sleep(delay)
    logger.error("Database connection failed after %s attempts.", max_retries)
    return 1


if __name__ == "__main__":
    sys.exit(wait())
