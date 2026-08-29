# Security Policy

## Supported versions

This project is pre-1.0. Only the latest release receives fixes.

## Reporting a vulnerability

Please report security issues privately to **contact@qsun.fr** rather than
opening a public issue. Include what you did, what happened, and the plugin and
DSS versions involved.

Expect an acknowledgement within a week. If the report is confirmed, a fix and a
release will follow, and you will be credited unless you prefer otherwise.

## What this plugin does and does not handle

Worth knowing before reporting, and worth knowing as an operator:

- **The plugin forwards prompts, tool outputs, and images to whatever endpoint
  the connection's Base URL names.** It does not inspect or redact prompt and
  tool-output content. Point it only at Ollama instances you trust.
- **The Base URL is an outbound-network boundary.** It accepts HTTP(S) only;
  remote HTTP endpoints with an API Key are rejected, while unauthenticated
  remote HTTP remains possible for controlled networks. Use HTTPS for remote
  deployments.
- **`IMAGE_URI` values are forwarded to Ollama after scheme validation.** Ollama
  or a proxy may fetch them server-side, so use trusted URLs and apply network
  egress controls where needed.
- **Tool calls are model output.** The plugin rejects tool names that were not
  declared in the request, but operators should still treat models and Ollama
  endpoints as untrusted when tools can cause side effects.
- **A local Ollama does not authenticate requests.** Do not expose port 11434 to
  the internet without a reverse proxy and authentication in front of it. Use the
  connection's API Key field for the bearer token in that setup.
- **The API key is held by the OpenAI client only.** The plugin does not log or
  copy it elsewhere; the client retains it for authenticated requests.
- **Debug logging records summaries, never content** — message counts, tool-call
  names, content lengths. Raising the log level adds metadata to DSS logs, but
  will not write prompt or response bodies.

Reports about a misconfigured Ollama that was deliberately exposed are an
operational matter rather than a plugin vulnerability, but tell us anyway if the
plugin made that mistake easy to make.
