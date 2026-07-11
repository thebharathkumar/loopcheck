from pricing import price_order


def test_standard_100_items():
    assert price_order(100, 10.0) == 850.00


def test_vip_100_items():
    assert price_order(100, 10.0, "vip") == 800.00


def test_small_order_no_discount():
    assert price_order(10, 10.0) == 95.00


def test_mid_tier_discount():
    assert price_order(50, 10.0) == 450.00
