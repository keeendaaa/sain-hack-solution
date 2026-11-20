import asyncio
import json
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from app.config import Settings
from app.errors import (
    InvalidModelOutput,
    NotConfigured,
    OfflineMode,
    ResourceExhausted,
    TransientLLMError,
    UpstreamError,
)
from app.network import read_bounded, validate_origin
from app.resources import RateLimiter, RunResources
from app.tokens import TokenCounter


def retry_delay(value: str | None) -> float:
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError):
                pass
    return 5.0


class LLMGateway:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, counter: TokenCounter):
        self.settings, self.client, self.counter = settings, client, counter
        self.limiter = RateLimiter(settings.requests_per_minute)
        self.semaphore = asyncio.Semaphore(settings.concurrency)
        self.authentication_failed = False

    async def call(
        self,
        messages: list[dict[str, str]],
        resources: RunResources,
        *,
        stage: str,
        input_cap: int,
        output_cap: int,
        timeout: float,
        extra: bool = False,
    ) -> dict:
        if not self.settings.allow_network:
            raise OfflineMode()
        if self.authentication_failed:
            raise NotConfigured("llm_authentication_failed")
        if not self.settings.llm_model or not self.settings.llm_api_key:
            raise NotConfigured("llm_configuration_missing")
        origin = validate_origin(self.settings.llm_base_url)
        input_tokens = self.counter.messages(messages)
        if input_tokens > input_cap:
            raise ResourceExhausted("prompt_too_large")
        resources.require_time()
        reservation = None
        usage: int | None = None
        try:
            async with asyncio.timeout(resources.remaining()):
                async with self.semaphore:
                    await self.limiter.acquire(resources)
                    resources.require_time()
                    reservation = await resources.ledger.reserve(
                        stage, input_tokens + output_cap, extra=extra
                    )
                    async with asyncio.timeout(min(timeout, resources.remaining())):
                        async with self.client.stream(
                            "POST",
                            origin + "/chat/completions",
                            follow_redirects=False,
                            headers={"Authorization": "Bearer " + self.settings.llm_api_key},
                            json={
                                "model": self.settings.llm_model,
                                "messages": messages,
                                "max_tokens": output_cap,
                                "response_format": {"type": "json_object"},
                            },
                            timeout=httpx.Timeout(min(timeout, resources.remaining()), connect=10),
                        ) as response:
                            if response.status_code == 429 or response.status_code >= 500:
                                raise TransientLLMError(retry_delay(response.headers.get("Retry-After")))
                            if response.status_code in (401, 403):
                                self.authentication_failed = True
                                raise UpstreamError("llm_authentication_failed")
                            if response.status_code != 200:
                                raise UpstreamError("llm_http_error")
                            raw = await read_bounded(response, 2 * 1024 * 1024)
                        try:
                            body = json.loads(raw)
                            if not isinstance(body, dict):
                                raise InvalidModelOutput("invalid_completion_envelope")
                            measured = body.get("usage", {})
                            if not isinstance(measured, dict):
                                measured = {}
                            prompt, completion = (
                                measured.get("prompt_tokens"),
                                measured.get("completion_tokens"),
                            )
                            if (
                                type(prompt) is int
                                and type(completion) is int
                                and prompt >= 0
                                and completion >= 0
                            ):
                                usage = prompt + completion
                            choice = body["choices"][0]
                            if choice["finish_reason"] != "stop":
                                raise InvalidModelOutput("incomplete_completion")
                            content = choice["message"]["content"]
                            value = json.loads(content)
                            if not isinstance(value, dict):
                                raise InvalidModelOutput("model_output_not_object")
                            return value
                        except (ValueError, KeyError, TypeError, IndexError):
                            raise InvalidModelOutput("invalid_completion_json") from None
        except (TimeoutError, httpx.TransportError):
            raise TransientLLMError() from None
        finally:
            if reservation is not None:
                await resources.ledger.settle(reservation, usage)
