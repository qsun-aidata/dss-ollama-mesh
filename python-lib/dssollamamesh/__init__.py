"""Shared helpers for the dss-ollama-mesh plugin.

Plugin components (llm.py) stay thin; reusable logic lives here so it can be
imported as ``from dssollamamesh import ...`` inside DSS code envs.

Only the names in ``_CLIENT_EXPORTS`` pull in the openai package, and they do so
lazily, so the message/retry/streaming logic can be imported and tested without it.
"""

from dssollamamesh.concurrency import get_semaphore, resolve_concurrency
from dssollamamesh.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    ENABLE_STREAMING_DEFAULT,
)
from dssollamamesh.messages import build_chat_kwargs
from dssollamamesh.streaming import accumulate_tool_calls, finalize_tool_calls

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "ENABLE_STREAMING_DEFAULT",
    "accumulate_tool_calls",
    "build_chat_kwargs",
    "call_with_retry",
    "finalize_tool_calls",
    "get_semaphore",
    "is_retryable",
    "is_stream_unsupported",
    "make_client",
    "resolve_api_key",
    "resolve_base_url",
    "resolve_concurrency",
]

_CLIENT_EXPORTS = frozenset({
    "call_with_retry",
    "is_retryable",
    "is_stream_unsupported",
    "make_client",
    "resolve_api_key",
    "resolve_base_url",
})


def __getattr__(name):
    if name in _CLIENT_EXPORTS:
        from dssollamamesh import client as client_mod

        return getattr(client_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
