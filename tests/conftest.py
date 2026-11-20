import copy
import json

import pytest

from app.config import ROOT, Settings
from app.contracts import ContractAdapter
from app.tokens import TokenCounter


@pytest.fixture(scope="session")
def counter():
    return TokenCounter(ROOT / "app/assets/tiktoken")


@pytest.fixture
def envelope():
    return json.loads((ROOT / "examples/synthetic-dataset.json").read_text())


@pytest.fixture
def adapter():
    return ContractAdapter(ROOT / "contracts/profiles/local_example.json")


@pytest.fixture
def dataset(adapter, envelope):
    return adapter.decode(envelope)


@pytest.fixture
def settings(tmp_path, envelope):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(envelope))
    return Settings(dataset_path=path)


@pytest.fixture
def verified_test_profile(tmp_path):
    """Test-only fixture: verifies network gating, not an official contract."""
    profile = json.loads((ROOT / "contracts/profiles/local_example.json").read_text())
    profile["name"] = "test-fixture-not-official"
    profile["verified_official"] = True
    for key in ("dataset_schema", "answers_schema"):
        profile[key] = str((ROOT / "contracts/profiles" / profile[key]).resolve())
    path = tmp_path / "test-profile.json"
    path.write_text(json.dumps(profile))
    return path


@pytest.fixture
def online_envelope(envelope):
    result = copy.deepcopy(envelope)
    rows = []
    for group in range(33):
        text = "Сколько мелатонина мне принимать?" if group >= 20 else "Как связаны ночные смены и режим сна?"
        for expert in result["dataset"]["experts"]:
            rows.append(
                {
                    "request_id": f"r-{group}-{expert['expert_id']}",
                    "user_id": f"u-{group}",
                    "expert_id": expert["expert_id"],
                    "text": text,
                }
            )
    # Explicitly interleave personas rather than adjacent triplets.
    result["dataset"]["requests"] = rows[::3] + rows[1::3] + rows[2::3]
    return result


def fake_completion(body, counter):
    data = json.loads(body["messages"][-1]["content"])
    if "questions_data" in data:
        sources = [c["key"] for c in data["catalog_data"]]
        output = {
            "plans": [
                {
                    "group_key": q["group_key"],
                    "standalone_query": q["text"],
                    "terms_ru": ["режим сна"],
                    "terms_en": ["shift work"],
                    "intent_hint": "information",
                    "candidate_sections": sources[:2],
                    "ambiguities": [],
                }
                for q in data["questions_data"]
            ]
        }
    else:
        evidence = data["evidence_data"][0]
        quote = evidence["text"].split(". ")[0].strip()
        refused = data["hard_veto"]
        variants = (
            [
                "Не могу назначить лечение или дозировку. Могу направить к указанному разделу базы.",
                "Не буду подбирать дозировку или диагноз. Исследовать вопрос можно через материал по ссылке.",
                "Не могу рекомендовать личное лечение. Предлагаю начать с чтения источника, без самостоятельных назначений.",
            ]
            if refused
            else [
                f"В источнике описана связь: «{quote}». Учитываем границы этого описания.",
                f"Начнём с наблюдения из текста: «{quote}». Отделим его от непроверенных догадок.",
                f"Практический смысл термина поясняет материал: «{quote}». Сохраним именно его формулировку.",
            ]
        )
        output = {
            "group_key": data["group_key"],
            "refused": refused,
            "citations": [evidence["key"]],
            "evidence_quotes": [{"evidence_key": evidence["key"], "quote": quote}],
            "answers": [
                {"expert_key": expert["expert_key"], "text": variants[i]}
                for i, expert in enumerate(data["experts"])
            ],
        }
    return {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(output, ensure_ascii=False)}}
        ],
        "usage": {"prompt_tokens": counter.messages(body["messages"]), "completion_tokens": 500},
    }
