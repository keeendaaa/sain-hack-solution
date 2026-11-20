import asyncio

import pytest

from app.errors import ResourceExhausted
from app.registry import RunRegistry
from app.resources import Ledger, RunResources


async def test_timeout_keeps_full_reservation():
    ledger = Ledger(budget=5000)
    reservation = await ledger.reserve("generation", 3600)
    await ledger.settle(reservation, None)
    assert ledger.charged == 3600
    assert ledger.unknown_usage_calls == 1
    with pytest.raises(ResourceExhausted):
        await ledger.reserve("generation", 3600)


async def test_known_usage_refunds_unused_output_and_limits_extras():
    ledger = Ledger(extras=1)
    first = await ledger.reserve("generation", 3600)
    await ledger.settle(first, 700)
    assert ledger.charged == ledger.known_usage == 700
    await ledger.reserve("generation", 3600, extra=True)
    with pytest.raises(ResourceExhausted):
        await ledger.reserve("generation", 100, extra=True)


async def test_planner_stage_cap_is_shared_between_calls():
    ledger = Ledger()
    await ledger.reserve("planner", 6000)
    with pytest.raises(ResourceExhausted):
        await ledger.reserve("planner", 6000)
    await ledger.reserve("planner", 5000)
    with pytest.raises(ResourceExhausted):
        await ledger.reserve("planner", 1)


async def test_deadline_cannot_be_extended_by_backoff():
    resources = RunResources(seconds=0.01)
    with pytest.raises(ResourceExhausted):
        await resources.backoff(5)
    assert resources.backoff_used == 0


async def test_singleflight_and_authorization_scoping():
    registry = RunRegistry()
    starts = 0
    started = asyncio.Event()
    finish = asyncio.Event()

    async def compute():
        nonlocal starts
        starts += 1
        started.set()
        await finish.wait()
        return b'{"answers":[]}'

    key = registry.key("code", "Bearer one")
    assert key != registry.key("code", "Bearer two")
    assert "one" not in key and "code" not in key
    task1 = asyncio.create_task(registry.run(key, compute))
    await started.wait()
    task2 = asyncio.create_task(registry.run(key, compute))
    with pytest.raises(ResourceExhausted):
        await registry.run(registry.key("other", "Bearer one"), compute)
    finish.set()
    assert await task1 == await task2 == await registry.run(key, compute)
    assert starts == 1
    await registry.close()


async def test_disconnected_waiter_does_not_cancel_the_only_attempt():
    registry = RunRegistry()
    event = asyncio.Event()

    async def compute():
        await event.wait()
        return b"ok"

    key = registry.key("x", "Bearer local")
    waiter = asyncio.create_task(registry.run(key, compute))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    event.set()
    assert await registry.run(key, compute) == b"ok"
    await registry.close()
