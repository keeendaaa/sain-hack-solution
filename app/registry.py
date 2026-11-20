import asyncio
import hashlib
import hmac
import secrets
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from app.errors import ResourceExhausted


class RunRegistry:
    def __init__(self, entries: int = 2, max_bytes: int = 64 * 1024 * 1024, ttl: float = 3600):
        self.salt = secrets.token_bytes(32)
        self.entries, self.max_bytes, self.ttl = entries, max_bytes, ttl
        self.cache: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self.active: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()

    def key(self, code: str, authorization: str) -> str:
        return hmac.new(self.salt, (authorization + "\0" + code).encode(), hashlib.sha256).hexdigest()

    def _prune(self):
        now = time.monotonic()
        for key in list(self.cache):
            if now - self.cache[key][0] > self.ttl:
                del self.cache[key]
        while len(self.cache) > self.entries or sum(len(v[1]) for v in self.cache.values()) > self.max_bytes:
            self.cache.popitem(last=False)

    async def run(self, key: str, compute: Callable[[], Awaitable[bytes]]) -> bytes:
        async with self.lock:
            self._prune()
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key][1]
            task = self.active.get(key)
            if task is None:
                if self.active:
                    raise ResourceExhausted("another_run_active")
                task = asyncio.create_task(self._compute(key, compute))
                # Consume background exceptions even if all HTTP clients disconnect.
                task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
                self.active[key] = task
        return await asyncio.shield(task)

    async def _compute(self, key: str, compute: Callable[[], Awaitable[bytes]]) -> bytes:
        payload = None
        try:
            payload = await compute()
            return payload
        finally:
            async with self.lock:
                self.active.pop(key, None)
                if payload is not None:
                    self.cache[key] = (time.monotonic(), payload)
                    self._prune()

    async def close(self) -> None:
        tasks = list(self.active.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
