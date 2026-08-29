import logging

import pytest

from dssollamamesh.concurrency import get_semaphore, resolve_concurrency
from dssollamamesh.constants import DEFAULT_MAX_CONCURRENT


def test_same_pair_returns_the_same_semaphore():
    a = get_semaphore("http://a/v1", "llama3.1", 2)
    b = get_semaphore("http://a/v1", "llama3.1", 2)
    assert a is b


def test_different_pairs_get_independent_semaphores():
    local = get_semaphore("http://local/v1", "llama3.1", 2)
    cloud = get_semaphore("http://cloud/v1", "llama3.1", 8)
    other_model = get_semaphore("http://local/v1", "qwen2.5", 2)
    assert local is not cloud
    assert local is not other_model


def test_conflicting_limit_is_ignored_but_warned_about(caplog):
    first = get_semaphore("http://conflict/v1", "llama3.1", 2)
    with caplog.at_level(logging.WARNING, logger="dssollamamesh.concurrency"):
        second = get_semaphore("http://conflict/v1", "llama3.1", 16)
    assert second is first
    assert "already 2" in caplog.text
    assert "16" in caplog.text


def test_resolve_concurrency_defaults():
    assert resolve_concurrency({"max_concurrency": 8}) == 8
    assert resolve_concurrency({"max_concurrency": "4"}) == 4
    assert resolve_concurrency({}) == DEFAULT_MAX_CONCURRENT
    assert resolve_concurrency({"max_concurrency": 0}) == DEFAULT_MAX_CONCURRENT
    assert resolve_concurrency({"max_concurrency": None}) == DEFAULT_MAX_CONCURRENT


@pytest.mark.parametrize("value", [-1, 65, "not-an-int", 2.5, True])
def test_resolve_concurrency_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="1 to 64"):
        resolve_concurrency({"max_concurrency": value})
