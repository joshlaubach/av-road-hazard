from __future__ import annotations

import pytest

from rag.guards import (
    BudgetExceeded,
    Busy,
    RateLimited,
    RateLimiter,
    Slots,
    SpendGuard,
    cost_usd,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_rate_limiter_allows_the_limit_then_says_when_to_retry() -> None:
    clock = Clock()
    limiter = RateLimiter(limit=3, window_s=60, clock=clock)
    for _ in range(3):
        limiter.check("a")
    with pytest.raises(RateLimited) as caught:
        limiter.check("a")
    assert 1 <= caught.value.retry_after <= 61
    limiter.check("b")  # another key has its own allowance
    clock.now += 61
    limiter.check("a")  # the window has passed


def test_slots_refuse_instead_of_queueing_and_free_up_on_exit() -> None:
    slots = Slots(1)
    with slots, pytest.raises(Busy), slots:
        pass
    with slots:
        pass


def test_known_models_are_priced_and_unknown_ones_at_the_highest_rate() -> None:
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(2.0)
    assert cost_usd("some-new-model", 1_000_000, 0) == pytest.approx(4.0)


def test_the_spend_guard_stops_calls_at_the_daily_limit_and_reopens_next_day() -> None:
    clock = Clock()
    guard = SpendGuard(daily_usd=1.0, monthly_usd=10.0, clock=clock)
    guard.check()
    guard.record("claude-opus-5-5", 250_000, 0)  # $1.00
    with pytest.raises(BudgetExceeded, match="daily"):
        guard.check()
    clock.now += 86_401
    guard.check()
    assert guard.spent_today() == pytest.approx(0.0)
    assert guard.spent_this_month() == pytest.approx(1.0)


def test_the_spend_guard_stops_calls_at_the_monthly_limit() -> None:
    clock = Clock()
    guard = SpendGuard(daily_usd=100.0, monthly_usd=1.0, clock=clock)
    guard.record("claude-opus-5-5", 250_000, 0)
    clock.now += 86_401 * 2  # a new day, same month
    with pytest.raises(BudgetExceeded, match="monthly"):
        guard.check()
