import copy

import pytest

from app.errors import ContractError, NotConfigured
from app.fallback import fallback_answers
from app.grouping import group_requests
from app.models import identity
from app.policy import assess


def test_example_profile_is_never_claimed_official(adapter):
    with pytest.raises(NotConfigured):
        adapter.require_official()


def test_opaque_identifiers_do_not_collapse(adapter, envelope):
    rows = envelope["dataset"]["requests"]
    rows[0]["request_id"] = 1
    rows[1]["request_id"] = "1"
    dataset = adapter.decode(envelope)
    assert identity(dataset.requests[0].request_id) != identity(dataset.requests[1].request_id)
    for invalid in (True, None, [], {}):
        with pytest.raises(ValueError):
            identity(invalid)


def test_duplicate_and_wrong_parent_ids_rejected(adapter, envelope):
    envelope["dataset"]["requests"][1]["request_id"] = envelope["dataset"]["requests"][0]["request_id"]
    with pytest.raises(ContractError):
        adapter.decode(envelope)


def test_section_ids_may_repeat_across_articles(dataset):
    assert dataset.sections[0].section_id == dataset.sections[1].section_id
    assert dataset.sections[0].pair_key != dataset.sections[1].pair_key


def test_reference_and_dev_attack_labels_never_enter_normalized_data(adapter, envelope):
    envelope["reference"] = {"SECRET_REFERENCE_MARKER": "ANSWER"}
    for row in envelope["dataset"]["requests"]:
        row["is_attack"] = True
        row["gold_articles"] = ["forbidden-leak"]
    assert "SECRET_REFERENCE_MARKER" not in repr(adapter.decode(envelope))
    assert "is_attack" not in repr(adapter.decode(envelope))


def test_grouping_interleaved_99_records(adapter, online_envelope):
    dataset = adapter.decode(online_envelope)
    groups = group_requests(dataset)
    assert len(groups) == 33
    assert all(len(g.records) == 3 for g in groups)
    assert sum(assess(g).hard_veto for g in groups) == 13


def test_context_and_user_are_part_of_group_identity(adapter, envelope):
    template = envelope["dataset"]["requests"][0]
    rows = []
    for i, (user, context) in enumerate((("a", {}), ("b", {}), ("a", {"topic": "night"}))):
        row = copy.deepcopy(template)
        row.update(request_id=i, user_id=user, context=context)
        rows.append(row)
    envelope["dataset"]["requests"] = rows
    assert len(group_requests(adapter.decode(envelope))) == 3


def test_encode_preserves_input_order_and_requires_full_coverage(adapter, dataset):
    answers = []
    for group in group_requests(dataset):
        answers.extend(fallback_answers(group, (), assess(group), dataset.sections[0]))
    result = adapter.encode(dataset, list(reversed(answers)))
    assert [a["request_id"] for a in result["answers"]] == [r.request_id for r in dataset.requests]
    assert adapter.answers_validator.is_valid(result)
    with pytest.raises(ContractError):
        adapter.encode(dataset, answers[:-1])
