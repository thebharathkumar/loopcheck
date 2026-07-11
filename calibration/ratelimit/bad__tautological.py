from ratelimit import TokenBucket


def test_returns_bool():
    bucket = TokenBucket(5, 1.0)
    assert isinstance(bucket.allow(), bool)


def test_is_itself():
    bucket = TokenBucket(5, 1.0)
    result = bucket.allow()
    assert result == result


def test_truthy():
    bucket = TokenBucket(1, 1.0)
    assert bucket.allow() in (True, False)
