from ratelimit import TokenBucket


def test_allow_call():
    bucket = TokenBucket(5, 1.0)
    bucket.allow()


def test_multiple_calls():
    bucket = TokenBucket(3, 1.0)
    bucket.allow()
    bucket.allow()
    bucket.allow()
    bucket.allow()


def test_construction():
    TokenBucket(10, 2.0)
