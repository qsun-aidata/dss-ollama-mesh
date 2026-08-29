# Contributing

Thanks for taking an interest. This is a small plugin with a deliberately small
surface, and the main thing to preserve is that almost all of it can be tested
without a Dataiku instance.

## Setting up

```bash
git clone https://github.com/qsun-aidata/dss-ollama-mesh
cd dss-ollama-mesh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest -q       # no DSS and no Ollama required
ruff check .
```

`pyproject.toml` puts `python-lib/` on the import path, so there is nothing else
to configure — if `pytest` needs `PYTHONPATH` set by hand, something regressed.

## How the code is arranged

| Path | Role |
|---|---|
| `python-lib/dssollamamesh/` | All reusable logic. Imports no DSS modules. |
| `python-llms/ollama-mesh/llm.py` | Thin DSS adapter: the classes DSS discovers. |
| `code-env/python/spec/requirements.txt` | The plugin's runtime dependency, installed by DSS. |

Three invariants hold the design together. Breaking one is fine if you mean to,
but please say so in the PR:

1. **`python-lib/dssollamamesh/` never imports `dataiku` or `dataikuapi`.** That
   is what lets the test suite run anywhere.
2. **Only `client.py` imports `openai`**, and `__init__.py` exposes it lazily
   through a module-level `__getattr__`. The message, retry, and streaming logic
   stays importable with no SDK installed.
3. **`llm.py` stays thin.** New behaviour belongs in `python-lib/`, where it can
   be tested; `llm.py` should only wire it to DSS's interface.

## Four DSS details that cost real debugging time

They are also in the `llm.py` module header, and worth reading before changing
the streaming path:

1. `process_stream` must be a generator function — its body must contain `yield`,
   or DSS raises `TypeError`.
2. `toolCalls` must be emitted as their own chunk. DSS ignores them in the footer.
3. Assistant `tool_calls` must be preserved when replaying history, or the model
   repeats calls it already made.
4. DSS tool results arrive as `{'role': 'tool', 'toolOutputs': [...]}` and may
   bundle several parallel results into one message.

## Testing against a real DSS

The unit tests cover conversion, retry, and streaming logic. They cannot cover
DSS's own contract. For changes to `llm.py`, please also run through:

```bash
make dist    # builds dist/dss-ollama-mesh-<version>.zip from committed files
```

then install that zip in DSS, build the code env, create a Custom LLM connection,
and exercise whatever you changed in a Prompt Studio.

## Pull requests

- Keep `ruff check .` and `pytest -q` green; CI runs both on Python 3.8, 3.11
  and 3.12.
- Add tests for anything in `python-lib/`. If a change genuinely cannot be tested
  without DSS, say why in the PR.
- Match the surrounding comment style: comments here explain *why*, and the
  existing ones are worth imitating.
- Note user-visible changes in `CHANGELOG.md` under `## [Unreleased]`.
