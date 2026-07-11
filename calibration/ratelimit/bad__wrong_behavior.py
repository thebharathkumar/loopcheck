from ratelimit import TokenBucket


def test_bucket_starts_empty():
    # wrong: spec says bucket starts FULL, not empty
    bucket = TokenBucket(5, 1.0)
    assert bucket.allow() is False


def test_refill_no_cap():
    # wrong: ignores that refill is capped at capacity
    clock = [0.0]
    bucket = TokenBucket(3, 1.0, clock=lambda: clock[0])
    bucket.allow()
    bucket.allow()
    bucket.allow()
    clock[0] = 100.0
    # incorrectly expects 100 tokens to be available (no cap)
    for _ in range(10):
        assert bucket.allow() is True
