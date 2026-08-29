import logging

import pytest

from dssollamamesh import constants, retry


class FakeSemaphore:
    """Records how often the retry loop holds a slot, and that it never nests."""

    def __init__(self):
        self.acquisitions = 0
        self.held = False

    def __enter__(self):
        assert not self.held, "semaphore held across a backoff sleep"
        self.held = True
        self.acquisitions += 1
        return self

    def __exit__(self, *exc):
        self.held = False
        return False


class Clock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def always_retryable(_err):
    return True


def never_retryable(_err):
    return False


def test_returns_first_success_without_retrying():
    sem = FakeSemaphore()
    clock = Clock()
    result = retry.call_with_retry(
        lambda: "ok", "chat", sem, always_retryable, now=clock.now, sleep=clock.sleep
    )
    assert result == "ok"
    assert sem.acquisitions == 1
    assert clock.slept == []


def test_retries_until_success():
    sem = FakeSemaphore()
    clock = Clock()
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("busy")
        return "ok"

    result = retry.call_with_retry(
        flaky, "chat", sem, always_retryable, now=clock.now, sleep=clock.sleep
    )
    assert result == "ok"
    assert sem.acquisitions == 3
    assert len(clock.slept) == 2


def test_non_retryable_error_is_raised_immediately():
    sem = FakeSemaphore()
    clock = Clock()

    def boom():
        raise ValueError("unknown model")

    with pytest.raises(ValueError):
        retry.call_with_retry(
            boom, "chat", sem, never_retryable, now=clock.now, sleep=clock.sleep
        )
    assert sem.acquisitions == 1
    assert clock.slept == []


def test_gives_up_after_max_retries():
    sem = FakeSemaphore()
    clock = Clock()

    def boom():
        raise RuntimeError("busy")

    with pytest.raises(RuntimeError):
        retry.call_with_retry(
            boom, "chat", sem, always_retryable, now=clock.now, sleep=clock.sleep
        )
    assert sem.acquisitions == constants.MAX_RETRIES + 1
    # No sleep after the final attempt.
    assert len(clock.slept) == constants.MAX_RETRIES


def test_total_budget_cuts_retries_short():
    sem = FakeSemaphore()
    clock = Clock()

    def slow_and_broken():
        # Each attempt burns most of the budget, as a REQUEST_TIMEOUT would.
        clock.t += constants.RETRY_TOTAL_BUDGET / 2
        raise RuntimeError("timeout")

    with pytest.raises(RuntimeError):
        retry.call_with_retry(
            slow_and_broken, "chat", sem, always_retryable, now=clock.now, sleep=clock.sleep
        )
    # The budget stops new attempts starting; it cannot cut an in-flight one
    # short — REQUEST_TIMEOUT does that — so a single attempt may overshoot it.
    # Without the budget this would run all 7 attempts, ~3150s of fake time.
    assert sem.acquisitions == 2
    assert clock.t < constants.RETRY_TOTAL_BUDGET * 1.5


def test_existing_deadline_is_respected():
    sem = FakeSemaphore()
    clock = Clock()

    def boom():
        raise RuntimeError("secret-prompt")

    with pytest.raises(RuntimeError):
        retry.call_with_retry(
            boom,
            "chat",
            sem,
            always_retryable,
            now=clock.now,
            sleep=clock.sleep,
            deadline=1.0,
        )
    assert sem.acquisitions == 1
    assert clock.slept == []


def test_retry_logs_error_type_without_exception_body(caplog):
    sem = FakeSemaphore()
    clock = Clock()

    def boom():
        raise RuntimeError("secret-prompt")

    with caplog.at_level(logging.WARNING, logger="dssollamamesh.retry"):
        with pytest.raises(RuntimeError):
            retry.call_with_retry(
                boom,
                "chat",
                sem,
                always_retryable,
                now=clock.now,
                sleep=clock.sleep,
                deadline=1.0,
            )
    assert "secret-prompt" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_next_delay_grows_and_is_capped():
    lo = retry.next_delay(0, rand=lambda: 0.0)
    hi = retry.next_delay(0, rand=lambda: 1.0)
    assert lo < hi
    assert lo >= constants.RETRY_BASE_DELAY * (1 - constants.RETRY_JITTER_RATIO)
    assert hi <= constants.RETRY_BASE_DELAY * (1 + constants.RETRY_JITTER_RATIO)

    capped = retry.next_delay(30, rand=lambda: 1.0)
    assert capped <= constants.RETRY_MAX_DELAY * (1 + constants.RETRY_JITTER_RATIO)


def test_next_delay_jitter_spreads_values():
    delays = {retry.next_delay(3) for _ in range(50)}
    assert len(delays) > 1
