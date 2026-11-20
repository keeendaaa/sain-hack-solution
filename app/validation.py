import re

from app.models import Answer, Candidate, Evidence, Group, Risk, identity
from app.policy import contains_refusal, normalize, unsafe_output


def normalized_quote(text: str) -> str:
    # Whitespace/NFC only: punctuation, negation and numbers remain significant.
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", text).split())


def validate_candidate(
    candidate: Candidate, group: Group, evidence: tuple[Evidence, ...], risk: Risk
) -> tuple[tuple[str, ...], list[Answer]]:
    errors: list[str] = []
    pool = {e.key: e for e in evidence}
    if candidate.group_key != group.key:
        errors.append("group_identity")
    expected = {e.key for e in group.experts}
    provided = [a.expert_key for a in candidate.answers]
    if len(set(provided)) != len(provided) or set(provided) != expected:
        errors.append("expert_coverage")
    if len(set(candidate.citations)) != len(candidate.citations) or any(
        k not in pool for k in candidate.citations
    ):
        errors.append("citation_identity")
    quote_keys = set()
    for quote in candidate.evidence_quotes:
        e = pool.get(quote.evidence_key)
        if (
            not e
            or quote.evidence_key not in candidate.citations
            or normalized_quote(quote.quote) not in normalized_quote(e.text)
        ):
            errors.append("unsupported_quote")
        else:
            quote_keys.add(quote.evidence_key)
    if set(candidate.citations) != quote_keys:
        errors.append("citation_without_quote")
    if risk.hard_veto and not candidate.refused:
        errors.append("hard_veto_conflict")
    seen = set()
    for answer in candidate.answers:
        if unsafe_output(answer.text):
            errors.append("unsafe_text")
        if candidate.refused != contains_refusal(answer.text):
            errors.append("refusal_text_conflict")
        text = normalize(answer.text)
        for expert in group.experts:
            # A different name/header alone is not a different voice.
            text = text.replace(normalize(expert.role), "")
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        if text in seen:
            errors.append("identical_personas")
        seen.add(text)
    if errors:
        return tuple(dict.fromkeys(errors)), []
    sections = tuple(pool[key].section for key in candidate.citations)
    if len({s.pair_key for s in sections}) != len(sections):
        return ("duplicate_source_pair",), []
    by_expert = {a.expert_key: a.text for a in candidate.answers}
    expert_keys = {identity(e.expert_id): e.key for e in group.experts}
    return (), [
        Answer(
            r.request_id,
            r.expert_id,
            by_expert[expert_keys[identity(r.expert_id)]],
            candidate.refused,
            sections,
        )
        for r in group.records
    ]


def assert_group_invariants(groups: tuple[Group, ...], answers: list[Answer]) -> None:
    from app.errors import ContractError

    by_id = {identity(a.request_id): a for a in answers}
    for group in groups:
        refs = set()
        for record in group.records:
            a = by_id.get(identity(record.request_id))
            if not a:
                raise ContractError("missing_group_answer")
            refs.add(tuple(s.pair_key for s in a.citations))
        if len(refs) != 1:
            raise ContractError("group_citation_mismatch")
