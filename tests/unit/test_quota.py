from datetime import datetime

import pytest

from mosaic.llm.quota import PACIFIC, PoolRule, QuotaExhausted, QuotaTracker, next_reset


class FakeClock:
    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


START = datetime(2026, 9, 25, 10, 0, tzinfo=PACIFIC).timestamp()


def make_tracker(clock: FakeClock) -> QuotaTracker:
    return QuotaTracker(
        {
            "lite": (["lite-a", "lite-b"], 2, 5),
            "flash": (["flash-a", "flash-b"], 1, 2),
        },
        clock=clock,
        sleep=clock.sleep,
    )


LITE = [PoolRule("lite")]


def test_uses_first_model_until_its_minute_is_full():
    clock = FakeClock(START)
    t = make_tracker(clock)
    assert t.try_acquire(LITE)[0] == "lite-a"
    assert t.try_acquire(LITE)[0] == "lite-a"
    assert t.try_acquire(LITE)[0] == "lite-b"


def test_reports_wait_when_all_models_are_busy_this_minute():
    clock = FakeClock(START)
    t = make_tracker(clock)
    for _ in range(4):
        t.try_acquire(LITE)
    model, wait = t.try_acquire(LITE)
    assert model is None
    assert 0 < wait <= 60


def test_acquire_sleeps_through_the_minute_window():
    clock = FakeClock(START)
    t = make_tracker(clock)
    for _ in range(4):
        t.try_acquire(LITE)
    assert t.acquire(LITE) == "lite-a"
    assert clock.now >= START + 60


def test_daily_limit_exhausts_then_resets_at_pacific_midnight():
    clock = FakeClock(START)
    t = make_tracker(clock)
    for _ in range(10):  # 5 per day on each of two models
        clock.now += 61
        assert t.acquire(LITE)
    with pytest.raises(QuotaExhausted):
        t.acquire(LITE)
    clock.now = next_reset(clock.now) + 1
    assert t.acquire(LITE) == "lite-a"


def test_falls_back_from_flash_to_lite():
    clock = FakeClock(START)
    t = make_tracker(clock)
    route = [PoolRule("flash"), PoolRule("lite")]
    t.mark_rate_limited("flash-a", daily=True)
    t.mark_rate_limited("flash-b", daily=True)
    assert t.acquire(route) == "lite-a"


def test_reserve_keeps_flash_for_other_roles():
    clock = FakeClock(START)
    t = make_tracker(clock)
    strategist = [PoolRule("flash", min_fraction_left=0.5), PoolRule("lite")]
    reviewer = [PoolRule("flash"), PoolRule("lite")]
    clock.now += 61
    assert t.acquire(strategist) == "flash-a"  # 4 of 4 left
    clock.now += 61
    assert t.acquire(strategist) == "flash-a"  # 3 of 4 left
    clock.now += 61
    assert t.acquire(strategist) == "lite-a"  # 2 of 4 left, at the reserve
    assert t.acquire(reviewer) == "flash-b"


def test_per_minute_block_expires():
    clock = FakeClock(START)
    t = make_tracker(clock)
    t.mark_rate_limited("lite-a", daily=False, retry_after=30)
    assert t.try_acquire(LITE)[0] == "lite-b"
    clock.now += 31
    assert t.try_acquire(LITE)[0] == "lite-a"


def test_refund_returns_the_slot():
    clock = FakeClock(START)
    t = make_tracker(clock)
    t.try_acquire(LITE)
    t.refund("lite-a")
    assert t.snapshot()["lite-a"]["used_today"] == 0


def test_runs_left_estimate():
    clock = FakeClock(START)
    t = make_tracker(clock)
    assert t.runs_left_estimate(calls_per_run=5) == 2


def test_circuit_breaker_backs_off_longer_and_resets_on_success():
    clock = FakeClock(START)
    t = make_tracker(clock)
    assert t.note_failure("lite-a") == 30
    assert t.note_failure("lite-a") == 120
    assert t.try_acquire(LITE)[0] == "lite-b"  # lite-a is set aside
    clock.now += 121
    assert t.try_acquire(LITE)[0] == "lite-a"
    t.note_success("lite-a")
    assert t.note_failure("lite-a") == 30  # back to the shortest pause


def test_a_job_deadline_stops_model_calls():
    import pytest

    from mosaic.llm.quota import DeadlineTracker, JobTimeout

    clock = FakeClock(START)
    tracker = DeadlineTracker(make_tracker(clock), deadline=START + 10, clock=clock)
    assert tracker.acquire(LITE)  # time left: works as usual
    assert tracker.snapshot()  # everything else passes through to the shared tracker
    clock.now = START + 11
    with pytest.raises(JobTimeout):
        tracker.acquire(LITE)
