"""Default configuration for Ollama LLM connections.

Tunable via llm.json connection parameters unless noted as module-level only.
"""

# --- Connection defaults (overridden per model in llm.json) ---

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CHAT_MODEL = "llama3.1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Local models are usually VRAM-bound; cloud endpoints can raise this in the UI.
DEFAULT_MAX_CONCURRENT = 2

# OpenAI SDK requires a non-empty key; local Ollama ignores it. See resolve_api_key().
DEFAULT_API_KEY = "ollama"

# --- Retry / timeout (module-level; not exposed in llm.json) ---

MAX_RETRIES = 6

# A local Ollama that is merely busy recovers in seconds, so start small and grow.
RETRY_BASE_DELAY = 2.0  # seconds; doubled each attempt (2, 4, 8, ...)
RETRY_MAX_DELAY = 60.0  # per-attempt ceiling, so backoff cannot run away
RETRY_JITTER_RATIO = 0.25  # +/-25%, so requests that fail together do not retry in lockstep

# Wall-clock ceiling for one logical call, covering every attempt and every sleep.
# Without it, MAX_RETRIES attempts at REQUEST_TIMEOUT each could pin a DSS worker
# thread for well over an hour.
RETRY_TOTAL_BUDGET = 900.0

REQUEST_TIMEOUT = 600.0  # large models on CPU can be slow

# --- Streaming default when enable_streaming is unset in connection config ---

ENABLE_STREAMING_DEFAULT = True
