import pytest

from targets.pricing.module import price_order
from targets.ratelimit.module import TokenBucket
from targets.slugify.module import slugify


def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"


def test_slugify_unicode():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_slugify_empty_and_symbols():
    assert slugify("") == ""
    assert slugify("!!!") == ""


def test_slugify_truncates_without_trailing_hyphen():
    assert slugify("aa bb cc", max_length=5) == "aa-bb"
    with pytest.raises(ValueError):
        slugify("x", max_length=0)


def test_pricing_tiers():
    assert price_order(5, 10.0) == 50.00          # 0% tier
    assert price_order(10, 10.0) == 95.00         # 5% tier boundary
    assert price_order(50, 10.0) == 450.00        # 10% tier boundary
    assert price_order(100, 10.0) == 850.00       # 15% tier boundary


def test_pricing_vip_and_cap():
    assert price_order(5, 10.0, "vip") == 47.50   # 0+5 = 5%
    assert price_order(100, 10.0, "vip") == 800.00  # 15+5 = 20% (cap)


def test_pricing_rounding_half_up():
    assert price_order(1, 0.125, "vip") == 0.12   # 0.125*0.95=0.11875 -> 0.12


def test_pricing_errors():
    with pytest.raises(ValueError):
        price_order(-1, 10.0)
    with pytest.raises(ValueError):
        price_order(1, 10.0, "gold")


def test_bucket_burst_then_deny():
    t = [0.0]
    b = TokenBucket(capacity=3, refill_rate=1.0, clock=lambda: t[0])
    assert [b.allow(), b.allow(), b.allow(), b.allow()] == [True, True, True, False]


def test_bucket_refills_capped():
    t = [0.0]
    b = TokenBucket(capacity=2, refill_rate=1.0, clock=lambda: t[0])
    b.allow()
    b.allow()
    t[0] = 10.0  # refill far beyond capacity
    assert b.allow() and b.allow() and not b.allow()


def test_bucket_oversized_request():
    b = TokenBucket(capacity=2, refill_rate=1.0)
    with pytest.raises(ValueError):
        b.allow(3)
