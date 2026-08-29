"""Per-(base_url, model) concurrency limits, shared process-wide.

A local 7B model and a remote cloud model should not compete for the same slots,
so each pair gets its own semaphore.
"""

import logging
import threading

from dssollamamesh.constants import DEFAULT_MAX_CONCURRENT

logger = logging.getLogger(__name__)

_SEMAPHORES = {}
_SEM_LOCK = threading.Lock()
MAX_CONFIGURED_CONCURRENT = 64


def get_semaphore(base_url, model, limit):
    """Return the semaphore for a (base_url, model) pair, creating it on first use."""
    key = (base_url, model)
    with _SEM_LOCK:
        entry = _SEMAPHORES.get(key)
        if entry is None:
            sem = threading.Semaphore(limit)
            _SEMAPHORES[key] = (sem, limit)
            logger.info("Concurrency limit for %s@%s = %d", model, base_url, limit)
            return sem

        sem, established = entry
        if limit != established:
            # First connection to claim the pair wins: a live semaphore cannot be
            # resized without losing track of the slots already handed out. Say so
            # loudly, because the UI still shows the value that is being ignored.
            logger.warning(
                "Concurrency limit for %s@%s is already %d (set by an earlier connection); "
                "ignoring the configured value %d",
                model,
                base_url,
                established,
                limit,
            )
        return sem


def resolve_concurrency(config):
    raw = config.get("max_concurrency")
    if raw is None or raw == "":
        return DEFAULT_MAX_CONCURRENT
    if isinstance(raw, bool):
        raise ValueError("max_concurrency must be an integer from 1 to 64")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError("max_concurrency must be an integer from 1 to 64")
    try:
        value = int(raw)
    except (TypeError, ValueError) as err:
        raise ValueError("max_concurrency must be an integer from 1 to 64") from err
    if value == 0:
        return DEFAULT_MAX_CONCURRENT
    if value < 1 or value > MAX_CONFIGURED_CONCURRENT:
        raise ValueError("max_concurrency must be an integer from 1 to 64")
    return value
