import pytest
from ratelimit import TokenBucket


def test_burst_then_deny():
    clock = [0.0]
    bucket = TokenBucket(3, 1.0, clock=lambda: clock[0])
    # starts full: 3 tokens available
    assert bucket.allow() is True
    assert bucket.allow() is True
    assert bucket.allow() is True
    # bucket empty, next call denied
    assert bucket.allow() is False


def test_capped_refill():
    clock = [0.0]
    bucket = TokenBucket(3, 1.0, clock=lambda: clock[0])
    bucket.allow()
    bucket.allow()
    bucket.allow()
    # advance far past capacity
    clock[0] = 100.0
    assert bucket.allow() is True
    assert bucket.allow() is True
    assert bucket.allow() is True
    # refill is capped at capacity=3, so 4th is denied
    assert bucket.allow() is False


def test_denied_call_advances_clock():
    clock = [0.0]
    bucket = TokenBucket(1, 1.0, clock=lambda: clock[0])
    assert bucket.allow() is True
    assert bucket.allow() is False  # denied; clock still advances _last
    clock[0] = 1.0
    # 1 token refilled since last (denied) call
    assert bucket.allow() is True


def test_invalid_capacity():
    with pytest.raises(ValueError):
        TokenBucket(0, 1.0)


def test_invalid_refill_rate():
    with pytest.raises(ValueError):
        TokenBucket(1, 0.0)


def test_allow_tokens_less_than_one():
    bucket = TokenBucket(5, 1.0)
    with pytest.raises(ValueError):
        bucket.allow(0)


def test_allow_tokens_exceeds_capacity():
    bucket = TokenBucket(5, 1.0)
    with pytest.raises(ValueError):
        bucket.allow(6)
