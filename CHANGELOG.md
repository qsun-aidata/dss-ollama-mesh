# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security and maintenance

- Validate Ollama endpoints and image URIs before making outbound requests.
- Require HTTPS for remote endpoints when an API Key is configured.
- Reject undeclared or malformed tool calls and avoid logging exception bodies.
- Bound runtime concurrency settings and share the retry deadline with streaming
  fallback.
- Harden the release workflow and validate tag/version consistency.

## [0.1.0] - 2026-08-29

First public release.

### Added

- Custom LLM connection for the Dataiku LLM Mesh backed by Ollama's
  OpenAI-compatible API, with two capabilities: `OllamaLLM` for chat completion
  (multimodal) and `OllamaEmbeddingModel` for text embedding.
- Streaming chat with tool calling. Tool-call fragments are stitched back
  together across stream deltas and emitted as their own chunk.
- Multimodal input: inline images have their MIME type sniffed from the payload,
  since DSS does not supply one.
- Generation settings passed through to Ollama — temperature, max output tokens,
  top-p, stop sequences, seed, JSON response format, and top-k (via `extra_body`,
  which the OpenAI chat schema has no field for).
- Per-`(base_url, model)` concurrency limits, so a local 7B model and a cloud
  endpoint do not compete for the same slots.
- Retry with capped exponential backoff and ±25% jitter on 429, 5xx, timeout and
  connection errors, bounded by a wall-clock budget. The concurrency slot is
  released while backing off.
- Automatic fallback to a non-streaming request when streaming fails before any
  output has been sent — including when the server rejects `stream_options`
  outright, which is how older Ollama builds answer.
- `make dist` builds the installable plugin zip from committed files only.

### Known limitations

- The `trace` (`SpanBuilder`) argument is accepted but no spans are emitted, so
  DSS LLM Mesh tracing shows nothing for this provider.
- A second connection configuring a different concurrency limit for a
  `(base_url, model)` pair already in use keeps the established limit and logs a
  warning; a live semaphore cannot be resized safely.

[Unreleased]: https://github.com/qsun-aidata/dss-ollama-mesh/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/qsun-aidata/dss-ollama-mesh/releases/tag/v0.1.0
