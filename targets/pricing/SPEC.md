# price_order(quantity, unit_price, customer_type="standard") -> float

Total order price after tiered quantity discounts, rounded half-up to cents.

- Tiers by quantity: 0–9 → 0%, 10–49 → 5%, 50–99 → 10%, 100+ → 15%. Boundaries inclusive.
- customer_type "vip" adds 5 percentage points, capped at 20% total discount.
- Rounding is decimal half-up (0.11875 → 0.12), not banker's rounding.
- quantity < 0 raises ValueError; customer_type other than "standard"/"vip" raises ValueError.
- quantity 0 returns 0.0.
