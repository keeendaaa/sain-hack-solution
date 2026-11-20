"""Explicit, replaceable field mapping; no guessed official contract."""

import json
from pathlib import Path
from typing import Any

from jsonschema import validators
from pydantic import BaseModel, ConfigDict, Field

from app.errors import ContractError, NotConfigured
from app.models import Answer, Dataset, Expert, RequestRecord, Section, canonical, identity

MISSING = object()


def pointer(document: Any, path: str, default: Any = MISSING) -> Any:
    if path == "":
        return document
    if not path.startswith("/"):
        raise ContractError("invalid_mapping")
    try:
        for component in path[1:].split("/"):
            key = component.replace("~1", "/").replace("~0", "~")
            document = document[int(key)] if isinstance(document, list) else document[key]
        return document
    except (KeyError, IndexError, TypeError, ValueError):
        if default is not MISSING:
            return default
        raise ContractError("missing_mapped_field") from None


def set_pointer(document: dict, path: str, value: Any) -> None:
    if not path.startswith("/") or path == "/":
        raise ContractError("invalid_output_mapping")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
    for part in parts[:-1]:
        child = document.setdefault(part, {})
        if not isinstance(child, dict):
            raise ContractError("overlapping_output_mapping")
        document = child
    if parts[-1] in document:
        raise ContractError("overlapping_output_mapping")
    document[parts[-1]] = value


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    verified_official: bool = False
    dataset_schema: str | None = None
    answers_schema: str | None = None
    dataset: str = "/dataset"
    dataset_id: str = "/meta/dataset_id"
    articles: str = "/articles"
    article_id: str = "/article_id"
    article_title: str = "/title"
    sections: str = "/sections"
    section_id: str = "/section_id"
    section_heading: str = "/heading"
    section_text: str = "/text"
    experts: str = "/experts"
    expert_id: str = "/expert_id"
    expert_role: str = "/role"
    expert_style: str = "/style"
    requests: str = "/requests"
    request_id: str = "/request_id"
    request_user: str = "/user_id"
    request_expert: str = "/expert_id"
    request_text: str = "/text"
    request_context: str | None = None
    output_answers: str = "/answers"
    output_request_id: str = "/request_id"
    output_expert_id: str | None = None
    output_text: str = "/text"
    output_refused: str = "/refused"
    output_references: str = "/references"
    reference_article: str = "/article_id"
    reference_section: str = "/section_id"
    output_dataset_id: str | None = None
    output_constants: dict[str, Any] = Field(default_factory=dict)


def _safe_schema(path: Path):
    try:
        schema = json.loads(path.read_text())
    except (OSError, ValueError):
        raise NotConfigured("schema_missing") from None

    def check(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {"$ref", "$dynamicRef"} and (not isinstance(v, str) or not v.startswith("#")):
                    raise NotConfigured("bundle_schema_references_locally")
                check(v)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(schema)
    try:
        cls = validators.validator_for(schema)
        cls.check_schema(schema)
        return cls(schema)
    except Exception:
        raise NotConfigured("invalid_schema") from None


def _list(value: Any) -> list:
    if not isinstance(value, list):
        raise ContractError("expected_array")
    return value


def _string(value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ContractError("expected_text")
    return value


class ContractAdapter:
    def __init__(self, path: Path, max_sections: int = 5000, max_requests: int = 300):
        try:
            self.profile = Profile.model_validate_json(path.read_text())
        except (OSError, ValueError):
            raise NotConfigured("profile_missing_or_invalid") from None
        self.dataset_validator = (
            _safe_schema(path.parent / self.profile.dataset_schema) if self.profile.dataset_schema else None
        )
        self.answers_validator = (
            _safe_schema(path.parent / self.profile.answers_schema) if self.profile.answers_schema else None
        )
        if self.profile.verified_official and not (self.dataset_validator and self.answers_validator):
            raise NotConfigured("official_schemas_required")
        self.max_sections = max_sections
        self.max_requests = max_requests

    def require_official(self) -> None:
        if not self.profile.verified_official:
            raise NotConfigured("official_contract_not_verified")

    def decode(self, envelope: dict) -> Dataset:
        p = self.profile
        raw = pointer(envelope, p.dataset)
        # Validators operate on dataset (not getData's reference/envelope).
        if self.dataset_validator and not self.dataset_validator.is_valid(raw):
            raise ContractError("dataset_schema_failed")
        try:
            dataset_id = pointer(raw, p.dataset_id)
            identity(dataset_id)
            sections: list[Section] = []
            pairs: set[tuple[str, str]] = set()
            article_ids: set[str] = set()
            for article in _list(pointer(raw, p.articles)):
                aid = pointer(article, p.article_id)
                ai = identity(aid)
                if ai in article_ids:
                    raise ContractError("duplicate_article")
                article_ids.add(ai)
                title = _string(pointer(article, p.article_title))
                for section in _list(pointer(article, p.sections)):
                    sid = pointer(section, p.section_id)
                    pair = (ai, identity(sid))
                    if pair in pairs:
                        raise ContractError("duplicate_section_pair")
                    pairs.add(pair)
                    sections.append(
                        Section(
                            f"S{len(sections):04d}",
                            aid,
                            sid,
                            title,
                            _string(pointer(section, p.section_heading)),
                            _string(pointer(section, p.section_text)),
                        )
                    )
                    if len(sections) > self.max_sections:
                        raise ContractError("too_many_sections")
            if not sections:
                raise ContractError("empty_corpus")
            experts: list[Expert] = []
            expert_ids: set[str] = set()
            for item in _list(pointer(raw, p.experts)):
                eid = pointer(item, p.expert_id)
                key = identity(eid)
                if key in expert_ids:
                    raise ContractError("duplicate_expert")
                expert_ids.add(key)
                experts.append(
                    Expert(
                        f"X{len(experts) + 1:02d}",
                        eid,
                        _string(pointer(item, p.expert_role)),
                        _string(pointer(item, p.expert_style)),
                    )
                )
            if not 1 <= len(experts) <= 3:
                raise ContractError("unsupported_expert_count")
            records: list[RequestRecord] = []
            request_ids: set[str] = set()
            for item in _list(pointer(raw, p.requests)):
                rid = pointer(item, p.request_id)
                ri = identity(rid)
                if ri in request_ids:
                    raise ContractError("duplicate_request")
                request_ids.add(ri)
                uid, eid = pointer(item, p.request_user), pointer(item, p.request_expert)
                identity(uid)
                if identity(eid) not in expert_ids:
                    raise ContractError("unknown_expert")
                context = pointer(item, p.request_context, {}) if p.request_context else {}
                records.append(
                    RequestRecord(rid, uid, eid, _string(pointer(item, p.request_text)), canonical(context))
                )
                if len(records) > self.max_requests:
                    raise ContractError("too_many_requests")
            return Dataset(dataset_id, tuple(sections), tuple(experts), tuple(records))
        except (ValueError, TypeError, RecursionError):
            raise ContractError("invalid_dataset") from None

    def encode(self, dataset: Dataset, answers: list[Answer]) -> dict:
        p = self.profile
        by_id = {identity(a.request_id): a for a in answers}
        expected = {identity(r.request_id) for r in dataset.requests}
        if len(by_id) != len(answers) or set(by_id) != expected:
            raise ContractError("answer_coverage")
        corpus = {s.pair_key for s in dataset.sections}
        result = json.loads(canonical(p.output_constants))
        serialized = []
        for record in dataset.requests:
            a = by_id[identity(record.request_id)]
            if identity(a.expert_id) != identity(record.expert_id) or not a.citations:
                raise ContractError("invalid_answer_identity")
            item: dict = {}
            set_pointer(item, p.output_request_id, a.request_id)
            if p.output_expert_id:
                set_pointer(item, p.output_expert_id, a.expert_id)
            set_pointer(item, p.output_text, _string(a.text))
            if type(a.refused) is not bool:
                raise ContractError("invalid_refused")
            set_pointer(item, p.output_refused, a.refused)
            refs = []
            for citation in a.citations:
                if citation.pair_key not in corpus:
                    raise ContractError("unknown_reference")
                ref: dict = {}
                set_pointer(ref, p.reference_article, citation.article_id)
                set_pointer(ref, p.reference_section, citation.section_id)
                refs.append(ref)
            if len({s.pair_key for s in a.citations}) != len(refs):
                raise ContractError("duplicate_reference")
            set_pointer(item, p.output_references, refs)
            serialized.append(item)
        set_pointer(result, p.output_answers, serialized)
        if p.output_dataset_id:
            set_pointer(result, p.output_dataset_id, dataset.dataset_id)
        if self.answers_validator and not self.answers_validator.is_valid(result):
            raise ContractError("answers_schema_failed")
        return result
