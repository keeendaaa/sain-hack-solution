import asyncio
import time
from dataclasses import dataclass

from app.errors import ResourceExhausted


@dataclass(frozen=True, slots=True)
class Reservation:
    stage: str
    amount: int


class Ledger:
    def __init__(self, budget: int = 145000, extras: int = 3):
        self.budget = budget
        self.charged = 0
        self.known_usage = 0
        self.calls = 0
        self.unknown_usage_calls = 0
        self.overruns = 0
        self.extras = 0
        self.max_extras = extras
        self.stage_charged = {"planner": 0, "planner_repair": 0, "generation": 0}
        self.stage_limits = {"planner": 11000, "planner_repair": 4000, "generation": 129600}
        self.stage_calls = {"planner": 0, "planner_repair": 0, "generation": 0}
        self.lock = asyncio.Lock()

    async def reserve(self, stage: str, amount: int, *, extra: bool = False) -> Reservation:
        async with self.lock:
            if self.overruns:
                raise ResourceExhausted("tokenizer_calibration_required")
            if stage not in self.stage_limits or amount <= 0:
                raise ResourceExhausted("invalid_reservation")
            if extra and self.extras >= self.max_extras:
                raise ResourceExhausted("extra_calls_exhausted")
            if stage == "planner" and self.stage_calls[stage] >= 2:
                raise ResourceExhausted("planner_calls_exhausted")
            if stage == "planner_repair" and self.stage_calls[stage] >= 1:
                raise ResourceExhausted("planner_repair_exhausted")
            if (
                self.charged + amount > self.budget
                or self.stage_charged[stage] + amount > self.stage_limits[stage]
            ):
                raise ResourceExhausted("token_budget_exhausted")
            if extra:
                self.extras += 1
            self.charged += amount
            self.calls += 1
            self.stage_charged[stage] += amount
            self.stage_calls[stage] += 1
            return Reservation(stage, amount)

    async def settle(self, reservation: Reservation, actual: int | None) -> None:
        async with self.lock:
            if actual is None:
                self.unknown_usage_calls += 1
                return  # A timeout is not a free request.
            delta = actual - reservation.amount
            self.charged += delta
            self.stage_charged[reservation.stage] += delta
            self.known_usage += actual
            if delta > 0:
                self.overruns += 1  # Stop future calls; tokenizer assumption was low.
                self.budget = self.charged

    def report(self) -> dict:
        return {
            "charged_estimate": self.charged,
            "known_usage": self.known_usage,
            "calls": self.calls,
            "unknown_usage_calls": self.unknown_usage_calls,
            "tokenizer_overruns": self.overruns,
            "extra_calls": self.extras,
            "stages": dict(self.stage_charged),
        }


class RunResources:
    def __init__(self, seconds: float = 2520, budget: int = 145000, extras: int = 3):
        self.deadline = time.monotonic() + seconds
        self.ledger = Ledger(budget, extras)
        self.backoff_used = 0.0
        self.backoff_lock = asyncio.Lock()

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def require_time(self, seconds: float = 0) -> None:
        if self.remaining() <= seconds:
            raise ResourceExhausted("deadline_exhausted")

    async def backoff(self, seconds: float) -> None:
        async with self.backoff_lock:
            if seconds < 0 or self.backoff_used + seconds > 35:
                raise ResourceExhausted("backoff_exhausted")
            self.require_time(seconds + 1)
            self.backoff_used += seconds
        await asyncio.sleep(seconds)


class RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60 / per_minute
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self, resources: RunResources) -> None:
        async with self.lock:
            delay = max(0.0, self.next_start - time.monotonic())
            resources.require_time(delay + 1)
            if delay:
                await asyncio.sleep(delay)
            resources.require_time()
            self.next_start = time.monotonic() + self.interval
