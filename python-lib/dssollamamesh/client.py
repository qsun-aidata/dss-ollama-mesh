"""OpenAI client factory and the transient-error policy for Ollama calls.

Ollama exposes an OpenAI-compatible HTTP API. We use the official openai Python
client with a custom base_url rather than calling REST directly. Everything that
needs the openai package lives here; the import-light pieces sit next door in
concurrency.py, retry.py, and util.py so they stay testable on their own.
"""

import ipaddress
import logging
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from dssollamamesh import retry
from dssollamamesh.constants import DEFAULT_API_KEY, DEFAULT_BASE_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

# Transient server-side conditions: rate limiting and gateway/overload errors.
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# An Ollama build older than the one that learned `stream_options` rejects the
# field outright instead of ignoring it. That is a permanent 4xx, so is_retryable
# says no — but the same prompt still succeeds without streaming.
STREAM_UNSUPPORTED_STATUS = (400, 422)


def resolve_api_key(config):
    """Use a configured key when present; otherwise the Ollama placeholder."""
    return config.get("api_key") or DEFAULT_API_KEY


def _is_loopback_host(hostname):
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def resolve_base_url(config):
    """Validate and return the configured Ollama endpoint.

    The value is validated before constructing the OpenAI client. Passing
    ``None`` through can let the SDK fall back to its own endpoint, and an API
    key on a remote ``http://`` URL would be sent in clear text. Ollama's local
    default remains HTTP; remote authenticated endpoints must use HTTPS.
    """
    value = config["base_url"] if "base_url" in config else DEFAULT_BASE_URL
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Ollama Base URL must be a non-empty http(s) URL")
    value = value.strip()
    if any(char.isspace() for char in value):
        raise ValueError("Ollama Base URL must not contain whitespace")

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as err:
        raise ValueError("Ollama Base URL is malformed") from err

    valid_port = port is None or 1 <= port <= 65535
    if parsed.scheme not in ("http", "https") or not hostname or not valid_port:
        raise ValueError("Ollama Base URL must contain a valid http(s) host and port")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError("Ollama Base URL must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama Base URL must not contain a query string or fragment")

    if parsed.scheme == "http" and not _is_loopback_host(hostname):
        if config.get("api_key"):
            raise ValueError("HTTPS is required for a remote Ollama endpoint when API Key is set")
        logger.warning(
            "Remote Ollama endpoint uses HTTP without an API key; prompts and images are "
            "sent unencrypted"
        )
    return value.rstrip("/")


def make_client(config):
    """Build an OpenAI client pointed at Ollama's compatible endpoint."""
    base_url = resolve_base_url(config)
    return OpenAI(
        base_url=base_url,
        api_key=resolve_api_key(config),
        # We implement our own retry with backoff; disable the SDK's built-in retry
        # to avoid double-retries and to respect our concurrency semaphore.
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    )


def is_retryable(err):
    """Only retry transient failures — not 4xx client errors like bad model names."""
    if isinstance(err, APIStatusError):
        return err.status_code in RETRYABLE_STATUS
    return isinstance(err, (APITimeoutError, APIConnectionError))


def is_stream_unsupported(err):
    """True when the server rejected the streaming request itself, not the work.

    Used only before any output has been yielded, to decide whether retrying the
    same prompt non-streamed is worth one attempt. A genuinely bad request (an
    unknown model, say) also lands here and simply fails again on the retry.
    """
    return isinstance(err, APIStatusError) and err.status_code in STREAM_UNSUPPORTED_STATUS


def call_with_retry(fn, what, semaphore, deadline=None):
    """Run fn under the semaphore, retrying transient Ollama failures with backoff."""
    return retry.call_with_retry(fn, what, semaphore, retryable=is_retryable, deadline=deadline)
