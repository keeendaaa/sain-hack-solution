from dataclasses import replace

import pytest

from app.config import ROOT
from app.fallback import fallback_answers
from app.grouping import group_requests
from app.models import Candidate, PersonaText, Plan, Quote, Section
from app.planner import QueryPlanner
from app.policy import assess, unsafe_output
from app.retrieval import Retriever
from app.validation import validate_candidate


@pytest.fixture
def index(dataset, counter):
    return Retriever(dataset, counter, ROOT / "config/sleep_terms.json")


def test_content_hash_changes_for_equal_length_corpus(dataset, counter, index):
    section = dataset.sections[0]
    changed = replace(section, text=section.text.replace("термин", "пример"))
    assert len(changed.text) == len(section.text)
    second = Retriever(
        replace(dataset, sections=(changed, *dataset.sections[1:])), counter, ROOT / "config/sleep_terms.json"
    )
    assert second.hash != index.hash


def test_unknown_and_duplicate_plans_cannot_replace_other_groups(dataset, index):
    groups = group_requests(dataset)
    row = {
        "group_key": groups[0].key,
        "standalone_query": "test",
        "terms_ru": [],
        "terms_en": [],
        "intent_hint": "information",
        "candidate_sections": ["S999999", "S0000"],
        "ambiguities": [],
    }
    planner = object.__new__(QueryPlanner)
    plans = planner.decode({"plans": [row]}, groups, index)
    assert plans[groups[0].key].candidate_sections == ["S0000"]
    assert planner.decode({"plans": [row, row]}, groups, index) == {}
    assert planner.decode({"plans": [{**row, "group_key": "foreign"}]}, groups, index) == {}


def test_semantic_channel_can_recover_source_outside_bm25_top12(dataset, counter):
    distractors = tuple(
        Section(f"S{i:04d}", f"d{i}", "intro", "Кошка", "Кошка", "Кошка кошка кошка.") for i in range(20)
    )
    target = Section("S9999", "target", "intro", "Неочевидная тема", "Описание", "Alternate vocabulary only.")
    index = Retriever(
        replace(dataset, sections=(*distractors, target)), counter, ROOT / "config/sleep_terms.json"
    )
    assert target.key not in [r.section.key for r in index.bm25("кошка")[:12]]
    plan = Plan(
        group_key="G0000",
        standalone_query="кошка",
        terms_ru=[],
        terms_en=[],
        intent_hint="information",
        candidate_sections=[target.key],
        ambiguities=[],
    )
    result = [r.section.key for r in index.rank("кошка", plan)]
    assert target.key in result
    assert any(key != target.key for key in result)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Сколько мелатонина мне принимать?", True),
        ("Подбери дозировку для моего друга", True),
        ("Поставь мне диагноз", True),
        ("Назначь лечение", True),
        ("Как в исследованиях объясняют механизм мелатонина?", False),
        ("Что такое диагноз?", False),
    ],
)
def test_boundary_checks_intent_not_just_drug_words(dataset, text, expected):
    group = replace(group_requests(dataset)[0], text=text)
    assert assess(group).hard_veto is expected


def test_adjacent_negation_is_not_cut_to_fit_window(dataset, counter):
    text = "Свет " + "исследование " * 110 + ". Однако эффект не обнаружен."
    source = replace(dataset.sections[0], text=text)
    index = Retriever(replace(dataset, sections=(source,)), counter, ROOT / "config/sleep_terms.json")
    group = replace(group_requests(dataset)[0], text="Свет исследование")
    evidence = index.evidence(group)
    assert not evidence or "не обнаружен" in evidence[0].text


def candidate_for(group, evidence, refused=False):
    e = evidence[0]
    return Candidate(
        group_key=group.key,
        refused=refused,
        citations=[e.key],
        evidence_quotes=[Quote(evidence_key=e.key, quote=e.text.strip())],
        answers=[
            PersonaText(expert_key=expert.key, text=f"Объяснение номер {i}: {e.text.strip()}")
            for i, expert in enumerate(group.experts)
        ],
    )


def test_quote_and_citation_identity_are_validated(dataset, index):
    group = group_requests(dataset)[0]
    evidence = index.evidence(group)
    c = candidate_for(group, evidence)
    errors, _ = validate_candidate(c, group, evidence, assess(group))
    assert not errors
    c.citations = ["E99"]
    errors, _ = validate_candidate(c, group, evidence, assess(group))
    assert "citation_identity" in errors
    c.citations = [evidence[0].key]
    c.evidence_quotes[0].quote = "Этого текста нет в источнике"
    assert "unsupported_quote" in validate_candidate(c, group, evidence, assess(group))[0]


def test_hard_veto_cannot_be_removed_by_llm(dataset, index):
    group = replace(group_requests(dataset)[0], text="Подбери дозировку")
    evidence = index.evidence(group_requests(dataset)[0])
    c = candidate_for(group, evidence)
    assert "hard_veto_conflict" in validate_candidate(c, group, evidence, assess(group))[0]


def test_identical_personas_are_rejected(dataset, index):
    group = group_requests(dataset)[0]
    evidence = index.evidence(group)
    c = candidate_for(group, evidence)
    for a in c.answers:
        a.text = "Одинаковый текст для всех экспертов без различий."
    assert "identical_personas" in validate_candidate(c, group, evidence, assess(group))[0]


def test_fallback_never_invents_a_source_or_uses_false_for_missing_evidence(dataset, index):
    group = replace(group_requests(dataset)[0], text="Совершенно посторонняя неизвестная тема")
    answers = fallback_answers(group, (), assess(group), index.fallback_section)
    assert all(a.refused and a.citations for a in answers)
    assert all(a.citations[0].pair_key in {s.pair_key for s in dataset.sections} for a in answers)
    assert all(not unsafe_output(a.text) for a in answers)
    assert len({a.text for a in answers}) == 3


def test_negated_prescription_is_a_refusal_not_an_assignment():
    assert not unsafe_output("Не назначаю лечение и не прописываю препараты.")
    assert unsafe_output("Назначаю лечение.")
    assert unsafe_output("Не волнуйтесь, назначаю лечение.")


def test_fallback_voices_follow_roles_even_when_experts_are_reordered(dataset, index):
    group = group_requests(dataset)[0]
    changed = replace(group, experts=tuple(reversed(group.experts)))
    a = fallback_answers(group, (), assess(group), index.fallback_section)
    b = fallback_answers(changed, (), assess(changed), index.fallback_section)
    assert {x.expert_id: x.text for x in a} == {x.expert_id: x.text for x in b}


def test_parallel_retrieval_uses_thread_safe_stemmers(index):
    from concurrent.futures import ThreadPoolExecutor

    queries = [
        "ночные смены циркадные ритмы",
        "sleep daytime lighting",
        "мелатонин механизм исследования",
    ] * 40
    expected = [index.tokens(q) for q in queries]
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(index.tokens, queries))
    assert actual == expected
