import time
from collections.abc import Callable


class TokenBucket:
    """Token-bucket rate limiter with injectable clock. See SPEC.md."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1 or refill_rate <= 0:
            raise ValueError("capacity must be >= 1 and refill_rate > 0")
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def allow(self, tokens: int = 1) -> bool:
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        if tokens > self.capacity:
            raise ValueError("request exceeds bucket capacity")
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_rate)
        self._last = now
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False
