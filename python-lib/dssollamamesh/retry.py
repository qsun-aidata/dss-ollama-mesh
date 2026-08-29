"""Timing policy for retrying transient failures.

Deliberately free of third-party imports: the "is this error transient?" decision
is openai-specific and lives in client.py, while everything here — backoff, jitter,
the wall-clock budget — stays importable and testable on its own.
"""

import logging
import random
import time

from dssollamamesh.constants import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_JITTER_RATIO,
    RETRY_MAX_DELAY,
    RETRY_TOTAL_BUDGET,
)

logger = logging.getLogger(__name__)


def next_delay(attempt, rand=random.random):
    """Capped exponential backoff with +/-RETRY_JITTER_RATIO of jitter."""
    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
    spread = delay * RETRY_JITTER_RATIO
    return max(0.0, delay - spread + 2 * spread * rand())


def call_with_retry(
    fn, what, semaphore, retryable, now=time.monotonic, sleep=time.sleep, deadline=None
):
    """Run a call under the semaphore with exponential backoff on retryable errors.

    The semaphore is held only for the duration of each attempt, not during the
    sleep between attempts, so other threads can proceed while one backs off.
    Retries stop at MAX_RETRIES or once RETRY_TOTAL_BUDGET seconds have elapsed,
    whichever comes first; the last transient error is then re-raised. The budget
    only prevents a *new* attempt from starting — an attempt already in flight is
    bounded by the client's REQUEST_TIMEOUT, so one call can overshoot the budget
    by at most that much.
    """
    deadline = deadline if deadline is not None else now() + RETRY_TOTAL_BUDGET
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        with semaphore:
            try:
                return fn()
            except Exception as err:
                if not retryable(err):
                    raise
                last_err = err

        if attempt >= MAX_RETRIES:
            break
        delay = next_delay(attempt)
        if deadline - now() <= delay:
            logger.warning(
                "%s failed (%s), giving up: %.0fs retry budget exhausted after %d attempt(s)",
                what,
                type(last_err).__name__,
                RETRY_TOTAL_BUDGET,
                attempt + 1,
            )
            break
        logger.warning(
            "%s failed (%s), retry %d/%d in %.1fs",
            what,
            type(last_err).__name__,
            attempt + 1,
            MAX_RETRIES,
            delay,
        )
        sleep(delay)

    raise last_err
