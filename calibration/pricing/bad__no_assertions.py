from pricing import price_order


def test_standard_order():
    price_order(10, 10.0)


def test_vip_order():
    price_order(100, 10.0, "vip")


def test_zero_quantity():
    price_order(0, 5.0)
