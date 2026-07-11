from pricing import price_order


def test_wrong_discount_at_10():
    # asserts 10% discount at qty=10, but spec says 5%
    assert price_order(10, 10.0) == 90.00


def test_vip_no_cap():
    # asserts 25% discount (15%+10%) at qty 100 vip — ignores the 20% cap
    assert price_order(100, 10.0, "vip") == 750.00


def test_zero_no_discount():
    # qty 0 should return 0.0; this wrongly expects a result > 0
    assert price_order(0, 10.0) > 0
