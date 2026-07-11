from ratelimit import TokenBucket


def test_starts_full():
    clock = [0.0]
    bucket = TokenBucket(5, 1.0, clock=lambda: clock[0])
    # all 5 slots available at start
    for _ in range(5):
        assert bucket.allow() is True
    assert bucket.allow() is False


def test_partial_refill():
    clock = [0.0]
    bucket = TokenBucket(10, 2.0, clock=lambda: clock[0])
    for _ in range(10):
        bucket.allow()
    clock[0] = 1.0
    # 2 tokens/s * 1s = 2 tokens refilled
    assert bucket.allow() is True
    assert bucket.allow() is True
    assert bucket.allow() is False
