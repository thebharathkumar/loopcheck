from decimal import ROUND_HALF_UP, Decimal

_TIERS = [(100, Decimal("0.15")), (50, Decimal("0.10")), (10, Decimal("0.05")), (0, Decimal("0"))]
_VIP_BONUS = Decimal("0.05")
_MAX_DISCOUNT = Decimal("0.20")


def price_order(quantity: int, unit_price: float, customer_type: str = "standard") -> float:
    """Total price after tiered quantity discount. See SPEC.md."""
    if quantity < 0:
        raise ValueError("quantity must be >= 0")
    if customer_type not in ("standard", "vip"):
        raise ValueError(f"unknown customer_type: {customer_type}")
    discount = next(d for threshold, d in _TIERS if quantity >= threshold)
    if customer_type == "vip":
        discount = min(discount + _VIP_BONUS, _MAX_DISCOUNT)
    total = Decimal(quantity) * Decimal(str(unit_price)) * (1 - discount)
    return float(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
