import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


class Semaphore:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Stream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *args):
        return False


class FakeCompletions:
    def __init__(self):
        self.calls = []
        self.response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return Stream([
                SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="answer", tool_calls=None)
                        )
                    ],
                ),
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
                    choices=[],
                ),
            ])
        return self.response


class FakeEmbeddings:
    def create(self, **kwargs):
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index)])
                for index, _ in enumerate(kwargs["input"])
            ]
        )


@pytest.fixture
def adapter(monkeypatch):
    dataiku = types.ModuleType("dataiku")
    llm_package = types.ModuleType("dataiku.llm")
    python_package = types.ModuleType("dataiku.llm.python")
    python_package.BaseLLM = type("BaseLLM", (), {})
    python_package.BaseEmbeddingModel = type("BaseEmbeddingModel", (), {})
    llm_package.python = python_package
    dataiku.llm = llm_package
    monkeypatch.setitem(sys.modules, "dataiku", dataiku)
    monkeypatch.setitem(sys.modules, "dataiku.llm", llm_package)
    monkeypatch.setitem(sys.modules, "dataiku.llm.python", python_package)

    spec = importlib.util.spec_from_file_location(
        "dss_ollama_mesh_test_adapter", ROOT / "python-llms/ollama-mesh/llm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_adapter(adapter):
    completions = FakeCompletions()
    fake_client = SimpleNamespace(
        base_url="https://ollama.example/v1/",
        chat=SimpleNamespace(completions=completions),
        embeddings=FakeEmbeddings(),
    )
    adapter.make_client = lambda config: fake_client
    adapter.get_semaphore = lambda *args: Semaphore()
    adapter.call_with_retry = lambda fn, what, semaphore, deadline=None: fn()
    return fake_client, completions


def test_chat_adapter_process_and_stream(adapter):
    fake_client, completions = configure_adapter(adapter)
    chat = adapter.OllamaLLM()
    chat.set_config({"base_url": "https://ollama.example/v1", "model": "test"}, {})
    query = {"messages": [{"role": "user", "content": "hello"}]}

    assert chat.process(query, {}, None)["text"] == "answer"
    stream_output = list(chat.process_stream(query, {}, None))
    assert stream_output == [
        {"chunk": {"text": "answer"}},
        {"footer": {"estimatedCost": 0.0, "promptTokens": 3, "completionTokens": 1}},
    ]
    assert completions.calls[0]["model"] == "test"
    assert completions.calls[-1]["stream_options"] == {"include_usage": True}


def test_embedding_adapter_supports_batch_queries(adapter):
    configure_adapter(adapter)
    embedding = adapter.OllamaEmbeddingModel()
    embedding.set_config({"base_url": "https://ollama.example/v1", "model": "embed"}, {})

    assert embedding.process({"queries": [{"text": "a"}, {"text": "b"}]}, {}, None) == {
        "responses": [{"embedding": [0.0]}, {"embedding": [1.0]}]
    }


def test_chat_adapter_rejects_undeclared_tool_calls(adapter):
    _, completions = configure_adapter(adapter)
    completions.response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="c1",
                    function=SimpleNamespace(name="delete_all", arguments="{}"),
                )],
            )
        )],
        usage=None,
    )
    chat = adapter.OllamaLLM()
    chat.set_config({"base_url": "https://ollama.example/v1", "model": "test"}, {})

    with pytest.raises(ValueError, match="undeclared"):
        chat.process({"messages": [{"role": "user", "content": "hello"}]}, {}, None)
