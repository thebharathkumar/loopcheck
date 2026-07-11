import pytest
from pricing import price_order


def test_no_discount_tier_boundary_low():
    # qty 9: 0% discount
    assert price_order(9, 10.0) == 90.00


def test_discount_5pct_tier_boundary_low():
    # qty 10: 5% discount starts here
    assert price_order(10, 10.0) == 95.00


def test_discount_5pct_tier_boundary_high():
    # qty 49: still 5%
    assert price_order(49, 10.0) == 465.50


def test_discount_10pct_tier_boundary_low():
    # qty 50: 10% discount starts here
    assert price_order(50, 10.0) == 450.00


def test_discount_10pct_tier_boundary_high():
    # qty 99: still 10%
    assert price_order(99, 10.0) == 891.00


def test_discount_15pct_tier_boundary():
    # qty 100: 15% discount starts here
    assert price_order(100, 10.0) == 850.00


def test_vip_adds_5pct():
    # qty 100: 15% + 5% = 20% (at cap)
    assert price_order(100, 10.0, "vip") == 800.00


def test_vip_cap_at_20pct():
    # qty 100 vip: discount capped at 20%, not 25%
    assert price_order(100, 10.0, "vip") == 800.00
    # vip at qty 50: 10% + 5% = 15% (under cap)
    assert price_order(50, 10.0, "vip") == 425.00


def test_zero_quantity():
    assert price_order(0, 10.0) == 0.0


def test_invalid_quantity():
    with pytest.raises(ValueError):
        price_order(-1, 10.0)


def test_invalid_customer_type():
    with pytest.raises(ValueError):
        price_order(10, 10.0, "premium")
