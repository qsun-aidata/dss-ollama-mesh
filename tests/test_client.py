import pytest

openai = pytest.importorskip("openai")

from dssollamamesh import client  # noqa: E402
from dssollamamesh.constants import DEFAULT_API_KEY  # noqa: E402


class FakeRequest:
    """Duck-typed stand-in — openai 1.x uses httpx, 3.x uses httpx2."""


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}
        self.request = FakeRequest()


def status_error(code):
    return openai.APIStatusError("boom", response=FakeResponse(code), body=None)


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_status_codes_are_retryable(code):
    assert client.is_retryable(status_error(code)) is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(code):
    assert client.is_retryable(status_error(code)) is False


def test_network_errors_are_retryable():
    assert client.is_retryable(openai.APITimeoutError(request=FakeRequest())) is True
    assert client.is_retryable(openai.APIConnectionError(request=FakeRequest())) is True


@pytest.mark.parametrize("code", [400, 422])
def test_stream_rejection_triggers_non_streaming_fallback(code):
    # An Ollama older than stream_options answers 400: never retryable, but the
    # same prompt still works unstreamed.
    err = status_error(code)
    assert client.is_stream_unsupported(err) is True
    assert client.is_retryable(err) is False


@pytest.mark.parametrize("code", [401, 404, 429, 500])
def test_other_statuses_are_not_stream_rejections(code):
    assert client.is_stream_unsupported(status_error(code)) is False


def test_network_errors_are_not_stream_rejections():
    assert client.is_stream_unsupported(openai.APITimeoutError(request=FakeRequest())) is False
    assert client.is_stream_unsupported(ValueError("boom")) is False


def test_unrelated_exceptions_are_not_retryable():
    assert client.is_retryable(ValueError("bad model name")) is False


def test_resolve_api_key_uses_placeholder_when_unset():
    assert client.resolve_api_key({"api_key": "sk-real"}) == "sk-real"
    assert client.resolve_api_key({"api_key": ""}) == DEFAULT_API_KEY
    assert client.resolve_api_key({}) == DEFAULT_API_KEY


def test_make_client_disables_sdk_retry_and_points_at_ollama():
    c = client.make_client({"base_url": "http://ollama:11434/v1"})
    assert str(c.base_url).rstrip("/") == "http://ollama:11434/v1"
    assert c.max_retries == 0


@pytest.mark.parametrize(
    "base_url",
    [None, "", "ftp://ollama/v1", "http://user:pass@ollama/v1", "https://ollama/v1?token=x"],
)
def test_invalid_base_urls_are_rejected(base_url):
    with pytest.raises(ValueError):
        client.make_client({"base_url": base_url, "api_key": "configured"})


def test_remote_http_with_api_key_is_rejected():
    with pytest.raises(ValueError, match="HTTPS"):
        client.make_client({"base_url": "http://ollama:11434/v1", "api_key": "configured"})


def test_local_http_with_api_key_is_allowed():
    c = client.make_client({"base_url": "http://127.0.0.1:11434/v1", "api_key": "configured"})
    assert str(c.base_url).rstrip("/") == "http://127.0.0.1:11434/v1"


def test_remote_https_with_api_key_is_allowed():
    c = client.make_client({"base_url": "https://ollama.example/v1", "api_key": "configured"})
    assert str(c.base_url).rstrip("/") == "https://ollama.example/v1"
