"""Per-model request accounting for the Gemini free tier.

Each model has its own per-minute and per-day limit. The tracker counts requests,
picks the first model on a route that has room, and records models that Google
has rate-limited. Daily counts reset at midnight Pacific time, when Google resets
free-tier quotas.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
MINUTE = 60.0


class QuotaExhausted(RuntimeError):
    """Every model on the route is out of quota for the day."""


@dataclass(frozen=True)
class PoolRule:
    """Use `pool` only while more than `min_fraction_left` of its daily total is left."""

    pool: str
    min_fraction_left: float = 0.0


@dataclass
class ModelState:
    name: str
    pool: str
    rpm: int
    rpd: int
    minute: deque[float] = field(default_factory=deque)
    day_key: str = ""
    day_count: int = 0
    blocked_until: float = 0.0


def day_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, PACIFIC).date().isoformat()


def next_reset(ts: float) -> float:
    now = datetime.fromtimestamp(ts, PACIFIC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


class QuotaTracker:
    def __init__(
        self,
        pools: dict[str, tuple[Iterable[str], int, int]],
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`pools` maps a pool name to (model names in fallback order, rpm, rpd)."""
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._models: dict[str, ModelState] = {}
        self._pools: dict[str, list[str]] = {}
        for pool, (names, rpm, rpd) in pools.items():
            names = list(names)
            self._pools[pool] = names
            for name in names:
                self._models[name] = ModelState(name=name, pool=pool, rpm=max(rpm, 1), rpd=rpd)

    # ---- internal helpers (call with the lock held) ----

    def _refresh(self, state: ModelState, now: float) -> None:
        while state.minute and now - state.minute[0] >= MINUTE:
            state.minute.popleft()
        key = day_key(now)
        if state.day_key != key:
            state.day_key = key
            state.day_count = 0
            if state.blocked_until and state.blocked_until <= now:
                state.blocked_until = 0.0

    def _wait_for(self, state: ModelState, now: float) -> float | None:
        """Seconds until the model can take a request, or None if it's done for the day."""
        self._refresh(state, now)
        if state.day_count >= state.rpd:
            return None
        if state.blocked_until > now:
            if state.blocked_until >= next_reset(now):
                return None
            return state.blocked_until - now
        if len(state.minute) >= state.rpm:
            return MINUTE - (now - state.minute[0])
        return 0.0

    def _pool_fraction_left(self, pool: str, now: float) -> float:
        total = used = 0
        for name in self._pools.get(pool, []):
            state = self._models[name]
            self._refresh(state, now)
            total += state.rpd
            used += min(state.day_count, state.rpd)
        return 0.0 if total == 0 else (total - used) / total

    # ---- public API ----

    def try_acquire(self, rules: Iterable[PoolRule]) -> tuple[str | None, float | None]:
        """Reserve a request slot on the first model with room.

        Returns (model, None) on success, or (None, wait_seconds) when every candidate
        is only busy for the current minute, or (None, None) when all are done for the day.
        """
        with self._lock:
            now = self._clock()
            shortest: float | None = None
            for rule in rules:
                if self._pool_fraction_left(rule.pool, now) <= rule.min_fraction_left:
                    continue
                for name in self._pools.get(rule.pool, []):
                    state = self._models[name]
                    wait = self._wait_for(state, now)
                    if wait == 0.0:
                        state.minute.append(now)
                        state.day_count += 1
                        return name, None
                    if wait is not None:
                        shortest = wait if shortest is None else min(shortest, wait)
            return None, shortest

    def acquire(self, rules: Iterable[PoolRule], max_wait: float = 120.0) -> str:
        """Block until a model has room, or raise QuotaExhausted."""
        rules = list(rules)
        waited = 0.0
        while True:
            model, wait = self.try_acquire(rules)
            if model:
                return model
            if wait is None or waited + wait > max_wait:
                raise QuotaExhausted("All models on this route are out of quota for now.")
            self._sleep(wait + 0.05)
            waited += wait + 0.05

    def mark_rate_limited(
        self, model: str, *, daily: bool, retry_after: float | None = None
    ) -> None:
        """Record a 429 from Google so the model is skipped until it recovers."""
        with self._lock:
            now = self._clock()
            state = self._models[model]
            if daily:
                state.blocked_until = next_reset(now)
            else:
                state.blocked_until = now + (retry_after or MINUTE)

    def refund(self, model: str) -> None:
        """Give back a slot for a request that never reached Google."""
        with self._lock:
            state = self._models[model]
            if state.minute:
                state.minute.pop()
            state.day_count = max(0, state.day_count - 1)

    def snapshot(self) -> dict[str, dict[str, int | str]]:
        with self._lock:
            now = self._clock()
            out = {}
            for name, state in self._models.items():
                self._refresh(state, now)
                out[name] = {
                    "pool": state.pool,
                    "used_today": state.day_count,
                    "limit_today": state.rpd,
                    "used_this_minute": len(state.minute),
                }
            return out

    def pool_remaining(self, pool: str) -> int:
        with self._lock:
            now = self._clock()
            remaining = 0
            for name in self._pools.get(pool, []):
                state = self._models[name]
                self._refresh(state, now)
                if state.blocked_until < next_reset(now):
                    remaining += max(0, state.rpd - state.day_count)
            return remaining

    def runs_left_estimate(self, calls_per_run: int = 8) -> int:
        """Rough number of live runs left today. Lite does most of the work."""
        return self.pool_remaining("lite") // max(calls_per_run, 1)
