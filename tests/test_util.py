import base64

import pytest

from dssollamamesh.util import (
    DEFAULT_IMAGE_MIME,
    get_any,
    get_first_set,
    guess_image_mime,
    validate_image_uri,
)


def b64(raw):
    return base64.b64encode(raw + b"\x00" * 32).decode()


def test_get_any_returns_first_non_empty():
    d = {"a": "", "b": None, "c": "value", "d": "other"}
    assert get_any(d, "a", "b", "c", "d") == "value"
    assert get_any(d, "a", "b") is None
    assert get_any({}, "missing") is None


def test_get_first_set_keeps_meaningful_falsy_values():
    # The distinction that matters for generation settings: temperature=0 is set.
    assert get_first_set({"temperature": 0}, "temperature") == 0
    assert get_first_set({"a": False}, "a") is False
    assert get_first_set({"a": ""}, "a") == ""
    assert get_any({"temperature": 0}, "temperature") is None


def test_get_first_set_skips_absent_and_null_keys():
    d = {"a": None, "b": 0.5}
    assert get_first_set(d, "a", "b") == 0.5
    assert get_first_set(d, "a") is None
    assert get_first_set({}, "missing") is None


def test_guess_image_mime_recognizes_common_formats():
    assert guess_image_mime(b64(b"\x89PNG\r\n\x1a\n")) == "image/png"
    assert guess_image_mime(b64(b"\xff\xd8\xff\xe0")) == "image/jpeg"
    assert guess_image_mime(b64(b"GIF89a")) == "image/gif"
    assert guess_image_mime(b64(b"BM\x00\x00")) == "image/bmp"
    assert guess_image_mime(b64(b"RIFF\x00\x00\x00\x00WEBP")) == "image/webp"


def test_guess_image_mime_falls_back_to_jpeg():
    assert guess_image_mime(b64(b"not an image")) == DEFAULT_IMAGE_MIME
    assert guess_image_mime("") == DEFAULT_IMAGE_MIME
    assert guess_image_mime(None) == DEFAULT_IMAGE_MIME
    assert guess_image_mime("!!!not base64!!!") == DEFAULT_IMAGE_MIME


def test_validate_image_uri_allows_http_s_and_query_strings():
    uri = "https://cdn.example/image.png?signature=x"
    assert validate_image_uri(uri) == uri


@pytest.mark.parametrize(
    "uri",
    [
        "file:///etc/passwd",
        "gopher://127.0.0.1:6379/",
        "data:image/png;base64,AAAA",
        "https://user:pass@example/image.png",
        "https://example/image.png#fragment",
        "https://example:99999/image.png",
    ],
)
def test_validate_image_uri_rejects_unsafe_forms(uri):
    with pytest.raises(ValueError):
        validate_image_uri(uri)
