# dss-ollama-mesh LLM handler (python-llms/ollama-mesh/llm.py)
#
# DSS discovers subclasses of BaseLLM / BaseEmbeddingModel in this file.
# Register capabilities in the connection UI:
#   - "Chat completion (multimodal)"  -> OllamaLLM
#   - "Text embedding"                -> OllamaEmbeddingModel
#
# Dataiku integration pitfalls:
#   1. process_stream must be a generator (body contains yield) or DSS raises TypeError
#   2. toolCalls must be emitted as a separate chunk — footer chunks are ignored
#   3. assistant tool_calls must be preserved when replaying history
#   4. DSS tool results use {'role':'tool', 'toolOutputs':[{'callId','output'},...]}
#      and may bundle several parallel results in one message — split into OpenAI tool messages
#
# TODO: `trace` is accepted but no spans are emitted, so LLM Mesh tracing shows
# nothing for this provider. Documented under "Known limitations" in the README.
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from dataiku.llm.python import BaseLLM

try:
    from dataiku.llm.python import BaseEmbeddingModel
except ImportError:
    # Older DSS builds expose the embedding base class under custom.*
    from dataiku.llm.python.custom.base_embedding_model import BaseEmbeddingModel

if TYPE_CHECKING:
    # Annotation-only, so they never run at import time. These paths have moved
    # between DSS versions, and a plugin module that fails to import disappears
    # from the connection UI entirely — only the two base classes are worth that
    # risk.
    from dataiku.llm.python.types import (
        CompletionResponse,
        CompletionSettings,
        SingleCompletionQuery,
    )
    from dataikuapi.dss.llm_tracing import SpanBuilder

from dssollamamesh import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    ENABLE_STREAMING_DEFAULT,
    accumulate_tool_calls,
    build_chat_kwargs,
    call_with_retry,
    finalize_tool_calls,
    get_semaphore,
    is_retryable,
    is_stream_unsupported,
    make_client,
    resolve_concurrency,
)
from dssollamamesh.constants import RETRY_TOTAL_BUDGET
from dssollamamesh.messages import validate_tool_calls

logger = logging.getLogger(__name__)


def _configure_connection(handler, config, default_model):
    """Shared wiring for chat and embedding handlers from connection config."""
    handler.model = config.get("model", default_model)
    handler.client = make_client(config)
    base_url = str(handler.client.base_url).rstrip("/")
    handler.semaphore = get_semaphore(base_url, handler.model, resolve_concurrency(config))


class OllamaLLM(BaseLLM):
    """Chat completion (multimodal) with streaming and tool calling."""

    def __init__(self):
        pass

    def set_config(self, config: dict, plugin_config: dict) -> None:
        _configure_connection(self, config, DEFAULT_CHAT_MODEL)
        streaming = config.get("enable_streaming")
        self.streaming = ENABLE_STREAMING_DEFAULT if streaming is None else bool(streaming)

    def _build_kwargs(self, query, settings):
        return build_chat_kwargs(self.model, query, settings)

    def process(self, query: SingleCompletionQuery, settings: CompletionSettings,
                trace: SpanBuilder) -> CompletionResponse:
        """Non-streaming completion; also used as the stream-failure fallback."""
        return self._process_with_deadline(query, settings, trace)

    def _process_with_deadline(self, query, settings, trace, deadline=None):
        kwargs = self._build_kwargs(query, settings)
        response = call_with_retry(
            lambda: self.client.chat.completions.create(**kwargs),
            what="chat(%s)" % self.model,
            semaphore=self.semaphore,
            deadline=deadline,
        )

        message = response.choices[0].message
        logger.debug(
            "RAW_MESSAGE role=assistant content_len=%d tool_calls=%d",
            len(message.content or ""),
            len(getattr(message, "tool_calls", None) or []),
        )

        result = {"text": message.content or "", "estimatedCost": 0.0}

        if getattr(message, "tool_calls", None):
            tool_calls = [
                {
                    "id": tc.id or "call_%d" % index,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for index, tc in enumerate(message.tool_calls)
            ]
            result["toolCalls"] = validate_tool_calls(tool_calls, settings)

        usage = getattr(response, "usage", None)
        if usage:
            result["promptTokens"] = usage.prompt_tokens
            result["completionTokens"] = usage.completion_tokens or 0
        return result

    def process_stream(self, query: SingleCompletionQuery, settings: CompletionSettings,
                       trace: SpanBuilder):
        """Streaming completion. Must be a generator — see module header pitfall #1."""
        deadline = time.monotonic() + RETRY_TOTAL_BUDGET
        if not self.streaming:
            yield from self._fallback_stream(query, settings, trace, deadline)
            return

        kwargs = self._build_kwargs(query, settings)
        kwargs["stream"] = True
        # Ask Ollama to attach usage on the final stream chunk when supported.
        kwargs["stream_options"] = {"include_usage": True}

        prompt_tokens = None
        completion_tokens = None
        # Tool call name/arguments arrive in fragments keyed by tc.index.
        acc = {}
        # Once any text chunk is yielded we cannot safely restart the stream.
        started = False

        try:
            # The semaphore is held for the whole stream on purpose: Ollama keeps this
            # request's slot busy until we finish reading it, so releasing early would
            # let more work reach the server than the limit allows. The `as stream`
            # context manager closes the HTTP response even when DSS abandons this
            # generator part-way through.
            with self.semaphore, self.client.chat.completions.create(**kwargs) as stream:
                for event in stream:
                    usage = getattr(event, "usage", None)
                    if usage:
                        prompt_tokens = usage.prompt_tokens
                        completion_tokens = usage.completion_tokens
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    if not delta:
                        continue

                    if delta.content:
                        started = True
                        yield {"chunk": {"text": delta.content}}

                    accumulate_tool_calls(acc, getattr(delta, "tool_calls", None))
        except Exception as err:
            # Retry only when no output was sent yet: once text is on the wire the
            # stream cannot be restarted. Worth one non-streaming attempt if the
            # error was transient, or if the server rejected the streaming request
            # itself — an Ollama older than `stream_options` answers 400 for it.
            if started or not (is_retryable(err) or is_stream_unsupported(err)):
                raise
            if time.monotonic() >= deadline:
                raise
            logger.warning(
                "Streaming failed (%s), falling back to non-streaming", type(err).__name__
            )
            yield from self._fallback_stream(query, settings, trace, deadline)
            return

        # Pitfall #2: toolCalls in footer are ignored — emit as their own chunk.
        if acc:
            tool_calls = finalize_tool_calls(acc)
            tool_calls = validate_tool_calls(tool_calls, settings)
            logger.debug(
                "STREAM_TOOL_CALLS count=%d names=%s",
                len(tool_calls),
                [t["function"]["name"] for t in tool_calls],
            )
            yield {"chunk": {"toolCalls": tool_calls}}

        footer = {"estimatedCost": 0.0}
        if prompt_tokens is not None:
            footer["promptTokens"] = prompt_tokens
            footer["completionTokens"] = completion_tokens or 0
        yield {"footer": footer}

    def _fallback_stream(self, query, settings, trace, deadline=None):
        """Wrap a one-shot process() result in the chunk/footer shape DSS expects."""
        result = self._process_with_deadline(query, settings, trace, deadline)

        text = result.get("text") or ""
        if text:
            yield {"chunk": {"text": text}}
        if result.get("toolCalls"):
            yield {"chunk": {"toolCalls": result["toolCalls"]}}

        footer = {"estimatedCost": result.get("estimatedCost", 0.0)}
        if "promptTokens" in result:
            footer["promptTokens"] = result["promptTokens"]
            footer["completionTokens"] = result["completionTokens"]
        yield {"footer": footer}


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """Text embedding via Ollama's OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self):
        pass

    def set_config(self, config: dict, plugin_config: dict) -> None:
        _configure_connection(self, config, DEFAULT_EMBEDDING_MODEL)

    def _embed(self, texts):
        response = call_with_retry(
            lambda: self.client.embeddings.create(model=self.model, input=texts),
            what="embeddings(%s)" % self.model,
            semaphore=self.semaphore,
        )
        return [item.embedding for item in response.data]

    def process(self, query, settings, trace):
        # DSS sends either a batch {"queries": [...]} or a single {"text": "..."}.
        if "queries" in query:
            texts = [q.get("text", "") for q in query["queries"]]
            embeddings = self._embed(texts)
            return {"responses": [{"embedding": e} for e in embeddings]}

        if "text" in query:
            return {"embedding": self._embed([query["text"]])[0]}

        raise ValueError("Unexpected embedding query shape, keys: %s" % list(query.keys()))
