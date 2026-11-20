import asyncio

import httpx
import pytest

from app.config import Settings
from app.errors import InvalidModelOutput, OfflineMode, ResourceExhausted, TransientLLMError, UpstreamError
from app.llm import LLMGateway
from app.network import DatasetClient
from app.resources import RunResources


def network_settings(**changes):
    return Settings(
        allow_network=True,
        llm_base_url="https://proxy.invalid/v1",
        llm_api_key="test-proxy-secret",
        llm_model="test-model",
        **changes,
    )


async def test_network_disabled_before_transport_is_used(counter):
    def forbidden(request):
        raise AssertionError("No HTTP request should be sent")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        gateway = LLMGateway(Settings(), client, counter)
        with pytest.raises(OfflineMode):
            await gateway.call(
                [], RunResources(), stage="generation", input_cap=2000, output_cap=1600, timeout=1
            )


async def test_timeout_is_billed_conservatively(counter):
    async def slow(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={})

    resources = RunResources(seconds=10)
    messages = [{"role": "user", "content": "test"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        gateway = LLMGateway(network_settings(), client, counter)
        with pytest.raises(TransientLLMError):
            await gateway.call(
                messages, resources, stage="generation", input_cap=2000, output_cap=1600, timeout=0.005
            )
    assert resources.ledger.charged == counter.messages(messages) + 1600
    assert resources.ledger.unknown_usage_calls == 1


@pytest.mark.parametrize(
    "response",
    [
        {
            "choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
        {"choices": [{"finish_reason": "stop", "message": {"content": "not-json"}}]},
        [],
    ],
)
async def test_bad_completions_do_not_become_success(counter, response):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))
    ) as client:
        gateway = LLMGateway(network_settings(), client, counter)
        resources = RunResources()
        with pytest.raises(InvalidModelOutput):
            await gateway.call(
                [{"role": "user", "content": "test"}],
                resources,
                stage="generation",
                input_cap=2000,
                output_cap=1600,
                timeout=1,
            )
        assert resources.ledger.charged > 0


async def test_redirect_does_not_forward_stage_token_to_other_origin():
    seen = []

    def response(request):
        seen.append(request)
        return httpx.Response(302, headers={"Location": "https://attacker.invalid/collect"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        loader = DatasetClient(Settings(allow_network=True), client)
        with pytest.raises(UpstreamError):
            await loader.fetch("private-code", "Bearer private-stage-token", RunResources())
    assert len(seen) == 1
    assert seen[0].url.host == "hack-api.be-healthy.tech"


async def test_oversized_prompt_never_reaches_proxy(counter):
    seen = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: seen.append(r))) as client:
        gateway = LLMGateway(network_settings(), client, counter)
        with pytest.raises(ResourceExhausted):
            await gateway.call(
                [{"role": "user", "content": "x " * 5000}],
                RunResources(),
                stage="generation",
                input_cap=2000,
                output_cap=1600,
                timeout=1,
            )
    assert not seen


async def test_real_usage_overrun_stops_further_dispatch(counter):
    def response(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 10000, "completion_tokens": 10000},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        gateway = LLMGateway(network_settings(), client, counter)
        gateway.limiter.interval = 0
        resources = RunResources()
        messages = [{"role": "user", "content": "test"}]
        await gateway.call(
            messages, resources, stage="generation", input_cap=2000, output_cap=1600, timeout=1
        )
        with pytest.raises(ResourceExhausted):
            await gateway.call(
                messages, resources, stage="generation", input_cap=2000, output_cap=1600, timeout=1
            )
    assert resources.ledger.overruns == 1


async def test_rate_limit_is_exposed_as_bounded_retry_not_hidden_http_retry(counter):
    sent = []

    def response(request):
        sent.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"})

    resources = RunResources(seconds=10)
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        gateway = LLMGateway(network_settings(), client, counter)
        with pytest.raises(TransientLLMError) as error:
            await gateway.call(
                [{"role": "user", "content": "test"}],
                resources,
                stage="generation",
                input_cap=2000,
                output_cap=1600,
                timeout=1,
            )
        assert error.value.retry_after == 120
    assert len(sent) == 1
    assert resources.ledger.unknown_usage_calls == 1
    with pytest.raises(ResourceExhausted):
        await resources.backoff(error.value.retry_after)
