# TokenBucket(capacity, refill_rate, clock=time.monotonic)

Token-bucket rate limiter. `allow(tokens=1) -> bool` consumes tokens if available.

- Starts full (capacity tokens). Allows bursts up to capacity, then denies.
- Refills at refill_rate tokens/second based on elapsed clock time; never exceeds capacity.
- Denied calls still advance the refill clock (no token loss).
- `clock` is injectable for deterministic tests.
- capacity < 1 or refill_rate <= 0 raises ValueError at construction.
- allow(tokens) with tokens < 1 raises ValueError; tokens > capacity raises ValueError.
