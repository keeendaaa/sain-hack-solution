import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.errors import ContractError, NotConfigured, OfflineMode, UpstreamAuthenticationError, UpstreamError
from app.resources import RunResources


def validate_origin(url: str) -> str:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise NotConfigured("https_origin_required")
    return url.rstrip("/")


async def read_bounded(response: httpx.Response, maximum: int) -> bytes:
    data = bytearray()
    async for chunk in response.aiter_bytes():
        if len(data) + len(chunk) > maximum:
            raise UpstreamError("response_too_large")
        data.extend(chunk)
    return bytes(data)


class DatasetClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings, self.client = settings, client

    async def fetch(self, code: str, authorization: str, resources: RunResources) -> dict:
        if self.settings.dataset_path:
            path: Path = self.settings.dataset_path
            try:
                if path.stat().st_size > self.settings.max_dataset_bytes:
                    raise ContractError("local_dataset_too_large")
                value = json.loads(await asyncio.to_thread(path.read_text))
            except (OSError, ValueError):
                raise ContractError("local_dataset_invalid") from None
            if not isinstance(value, dict):
                raise ContractError("dataset_not_object")
            return value
        if not self.settings.allow_network:
            raise OfflineMode()
        api = validate_origin(self.settings.hack_api_url)
        deadline = min(resources.deadline, time.monotonic() + self.settings.fetch_seconds)
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    async with self.client.stream(
                        "GET",
                        api + "/case/4/getData",
                        params={"dataset_code": code},
                        headers={"Authorization": authorization},
                        follow_redirects=False,
                        timeout=httpx.Timeout(remaining, connect=min(10, remaining)),
                    ) as response:
                        if response.status_code in (401, 403):
                            raise UpstreamAuthenticationError()
                        if response.status_code == 429 or response.status_code >= 500:
                            raise httpx.TransportError("transient")
                        if response.status_code != 200:
                            raise UpstreamError("dataset_http_error")
                        raw = await read_bounded(response, self.settings.max_dataset_bytes)
                value = json.loads(raw)
                del raw
                if not isinstance(value, dict):
                    raise ContractError("dataset_not_object")
                return value
            except (httpx.TransportError, TimeoutError):
                delay = 5 * (2**attempt)
                if attempt == 2 or time.monotonic() + delay >= deadline:
                    break
                await resources.backoff(delay)
            except ValueError:
                raise ContractError("invalid_dataset_json") from None
        raise UpstreamError("dataset_unavailable")
