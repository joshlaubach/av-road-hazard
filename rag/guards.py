"""Limits that keep a public endpoint from being abused: request rate, simultaneous work, and spend.

All three are held in memory, so they reset when the server restarts and are per process. The spend
guard is a second line of defense: the real cap is the spending limit set in the provider's console.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

# US dollars per million tokens (input, output) for the models this project may call. A model
# that is not listed is priced at the highest listed rate, so a new model never looks free.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
DEFAULT_PRICE = max(PRICES.values())
SECONDS_PER_DAY = 86_400
SECONDS_PER_MONTH = 30 * SECONDS_PER_DAY


class BudgetExceeded(RuntimeError):
    """The daily or monthly spend limit was reached; the message is safe to show."""


class RateLimited(RuntimeError):
    """Too many requests; `retry_after` is the number of seconds to wait."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("too many requests")
        self.retry_after = retry_after


class Busy(RuntimeError):
    """Too many requests are being worked on at once."""


@dataclass
class RateLimiter:
    """At most `limit` events per `window_s` seconds for each key (a session id or an address)."""

    limit: int
    window_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _events: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        """Forget every count (for tests, which share one client address)."""
        with self._lock:
            self._events.clear()

    def check(self, key: str) -> None:
        now = self.clock()
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - self.window_s:
                events.popleft()
            if len(events) >= self.limit:
                raise RateLimited(max(1, int(events[0] + self.window_s - now) + 1))
            events.append(now)
            if len(self._events) > 10_000:  # forget keys that have gone quiet
                for stale in [
                    k for k, v in self._events.items() if not v or v[-1] <= now - self.window_s
                ]:
                    del self._events[stale]


class Slots:
    """A fixed number of simultaneous jobs; asking for one when none is free fails at once."""

    def __init__(self, count: int) -> None:
        self._semaphore = threading.BoundedSemaphore(count)

    def __enter__(self) -> None:
        if not self._semaphore.acquire(blocking=False):
            raise Busy("the server is busy")

    def __exit__(self, *_: object) -> None:
        self._semaphore.release()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, DEFAULT_PRICE)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class SpendGuard:
    """Adds up what model calls cost and refuses new ones past a daily or monthly limit.

    Costs are estimates from the token counts the API reports, at the list prices above. Calls
    that are in flight when the limit is crossed still finish, so a limit can be overshot by a few
    calls at most (`Slots` bounds how many).
    """

    def __init__(
        self,
        daily_usd: float,
        monthly_usd: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.daily_usd = daily_usd
        self.monthly_usd = monthly_usd
        self._clock = clock
        self._lock = threading.Lock()
        self._spent: deque[tuple[float, float]] = deque()  # (time, dollars)

    def _window(self, seconds: float) -> float:
        now = self._clock()
        return sum(usd for t, usd in self._spent if t > now - seconds)

    def spent_today(self) -> float:
        with self._lock:
            return self._window(SECONDS_PER_DAY)

    def spent_this_month(self) -> float:
        with self._lock:
            return self._window(SECONDS_PER_MONTH)

    def check(self) -> None:
        """Raise BudgetExceeded if a new call should not start."""
        with self._lock:
            if self._window(SECONDS_PER_DAY) >= self.daily_usd:
                raise BudgetExceeded("the daily limit for answers has been reached")
            if self._window(SECONDS_PER_MONTH) >= self.monthly_usd:
                raise BudgetExceeded("the monthly limit for answers has been reached")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        now = self._clock()
        with self._lock:
            self._spent.append((now, cost_usd(model, input_tokens, output_tokens)))
            while self._spent and self._spent[0][0] <= now - SECONDS_PER_MONTH:
                self._spent.popleft()
