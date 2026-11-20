import json
import logging

import httpx
from fastapi.testclient import TestClient

import main
from app.config import Settings
from app.contracts import ContractAdapter
from app.llm import LLMGateway
from app.network import DatasetClient
from app.service import RunService
from tests.conftest import fake_completion


def test_health_and_local_run_never_send_anything(settings, counter, monkeypatch):
    monkeypatch.setattr(main, "TokenCounter", lambda *args: counter)
    sent = []

    def forbidden(request):
        sent.append(request)
        raise AssertionError("Unexpected outbound request")

    with TestClient(main.create_app(settings, httpx.MockTransport(forbidden))) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and not health["ready_for_online"]
        response = client.post(
            "/run",
            headers={"Authorization": "Bearer local"},
            json={"case": 4, "dataset_code": "synthetic-case4-demo"},
        )
        assert response.status_code == 200
        assert len(response.json()["answers"]) == 6
        assert not sent


def test_errors_are_sanitized_and_unknown_official_contract_is_blocked(counter, monkeypatch):
    monkeypatch.setattr(main, "TokenCounter", lambda *args: counter)
    with TestClient(main.create_app(Settings())) as client:
        assert client.post("/run", json={"case": 4, "dataset_code": "secret"}).status_code == 401
        response = client.post(
            "/run",
            headers={"Authorization": "Bearer local"},
            json={"case": "private-value", "dataset_code": "private-code"},
        )
        assert response.status_code == 422
        assert "private" not in response.text
        response = client.post(
            "/run", headers={"Authorization": "Bearer local"}, json={"case": 4, "dataset_code": "secret"}
        )
        assert response.status_code == 503 and response.json() == {"error": "not_configured"}


def test_full_99_record_run_with_mocked_services_and_no_reference_leak(
    online_envelope, verified_test_profile, counter, monkeypatch, caplog
):
    monkeypatch.setattr(main, "TokenCounter", lambda *args: counter)
    seen = []
    token = "private-stage-token-marker"
    code = "private-dataset-code-marker"

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            assert request.headers["Authorization"] == "Bearer " + token
            assert request.url.params["dataset_code"] == code
            return httpx.Response(200, json=online_envelope)
        assert request.url.host == "proxy.invalid"
        assert request.headers["Authorization"] == "Bearer private-llm-key"
        assert "DO_NOT_PASS_TO_PRODUCTION" not in request.content.decode()
        return httpx.Response(200, json=fake_completion(json.loads(request.content), counter))

    cfg = Settings(
        allow_network=True,
        contract_profile=verified_test_profile,
        llm_base_url="https://proxy.invalid/v1",
        llm_api_key="private-llm-key",
        llm_model="mock-model",
    )
    with caplog.at_level(logging.INFO):
        with TestClient(main.create_app(cfg, httpx.MockTransport(handler))) as client:
            client.app.state.service.gateway.limiter.interval = 0
            response = client.post(
                "/run", headers={"Authorization": "Bearer " + token}, json={"case": 4, "dataset_code": code}
            )
            assert response.status_code == 200, response.text
            before = len(seen)
            repeated = client.post(
                "/run", headers={"Authorization": "Bearer " + token}, json={"case": 4, "dataset_code": code}
            )
            assert repeated.content == response.content
            assert len(seen) == before
    body = response.json()
    rows = body["answers"]
    assert len(rows) == 99
    assert sum(r["refused"] for r in rows) == 39
    assert [r["request_id"] for r in rows] == [
        r["request_id"] for r in online_envelope["dataset"]["requests"]
    ]
    by_group = {}
    for record, answer in zip(online_envelope["dataset"]["requests"], rows, strict=True):
        by_group.setdefault(record["user_id"], []).append(answer)
    assert all(
        len({json.dumps(a["references"], sort_keys=True) for a in trio}) == 1 for trio in by_group.values()
    )
    assert all(len({a["text"] for a in trio}) == 3 for trio in by_group.values())
    assert sum(r.method == "GET" for r in seen) == 1
    assert len(seen) == 35  # 1 fetch, 1 planner, 33 generation, no unnecessary repairs
    assert not any(
        secret in caplog.text for secret in (token, code, "private-llm-key", "DO_NOT_PASS_TO_PRODUCTION")
    )


async def test_failed_group_isolated_and_repair_budget_bounded(settings, counter, online_envelope, tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(online_envelope))
    # Local source with an explicitly verified TEST-only profile for the mock proxy.
    from app.config import ROOT

    profile = json.loads((ROOT / "contracts/profiles/local_example.json").read_text())
    profile["verified_official"] = True
    for key in ("dataset_schema", "answers_schema"):
        profile[key] = str((ROOT / "contracts/profiles" / profile[key]).resolve())
    profile_path = tmp_path / "test-profile.json"
    profile_path.write_text(json.dumps(profile))
    cfg = settings.model_copy(
        update={
            "dataset_path": path,
            "allow_network": True,
            "contract_profile": profile_path,
            "llm_base_url": "https://proxy.invalid/v1",
            "llm_api_key": "test",
            "llm_model": "mock",
        }
    )
    calls = []

    def handler(request):
        body = json.loads(request.content)
        data = json.loads(body["messages"][-1]["content"])
        if "group_key" in data and data["group_key"] in {"G0000", "G0001", "G0002", "G0003", "G0004"}:
            calls.append(data["group_key"])
            return httpx.Response(
                200, json={"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
            )
        return httpx.Response(200, json=fake_completion(body, counter))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = ContractAdapter(profile_path)
        gateway = LLMGateway(cfg, client, counter)
        gateway.limiter.interval = 0
        service = RunService(cfg, adapter, DatasetClient(cfg, client), gateway, counter)
        payload, report = await service.execute("synthetic-case4-demo", "Bearer local")
    assert len(json.loads(payload)["answers"]) == 99
    assert report["metrics"]["fallback_groups"] == 5
    assert report["llm"]["extra_calls"] == 3
    assert len(calls) == 8
    assert report["llm"]["charged_estimate"] <= 145000


async def test_unexpected_worker_error_does_not_drop_other_records(
    settings, counter, adapter, dataset, monkeypatch
):
    from app.config import ROOT
    from app.retrieval import Retriever

    index = Retriever(dataset, counter, ROOT / "config/sleep_terms.json")
    original = index.evidence

    def one_broken_group(group, plan=None):
        if group.key == "G0000":
            raise ValueError("sensitive internal message must not escape")
        return original(group, plan)

    monkeypatch.setattr(index, "evidence", one_broken_group)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
        gateway = LLMGateway(settings, client, counter)
        service = RunService(settings, adapter, DatasetClient(settings, client), gateway, counter)
        monkeypatch.setattr(service, "index", lambda dataset: index)
        payload, report = await service.execute("synthetic-case4-demo", "Bearer local")
    assert len(json.loads(payload)["answers"]) == 6
    assert report["metrics"]["validation_errors"]["group_worker_failed"] == 1
    assert "sensitive internal message" not in payload.decode()


def test_bad_proxy_credentials_do_not_trigger_repair_retries(
    verified_test_profile, counter, monkeypatch, envelope
):
    monkeypatch.setattr(main, "TokenCounter", lambda *args: counter)
    generation = []
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=envelope)
        posts.append(request)
        data = json.loads(json.loads(request.content)["messages"][-1]["content"])
        if "group_key" in data:
            generation.append(data["group_key"])
        return httpx.Response(401)

    cfg = Settings(
        allow_network=True,
        contract_profile=verified_test_profile,
        llm_base_url="https://proxy.invalid/v1",
        llm_api_key="invalid-mock-key",
        llm_model="mock",
    )
    with TestClient(main.create_app(cfg, httpx.MockTransport(handler))) as client:
        client.app.state.service.gateway.limiter.interval = 0
        response = client.post(
            "/run",
            headers={"Authorization": "Bearer local"},
            json={"case": 4, "dataset_code": "synthetic-case4-demo"},
        )
    assert response.status_code == 200
    assert not generation
    assert len(posts) == 1  # Failed planner authentication stops all calls with that credential.
