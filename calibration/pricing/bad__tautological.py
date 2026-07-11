from pricing import price_order


def test_returns_float():
    assert isinstance(price_order(10, 10.0), float)


def test_is_itself():
    assert price_order(50, 10.0) == price_order(50, 10.0)


def test_positive_result():
    result = price_order(10, 10.0)
    assert result >= 0
