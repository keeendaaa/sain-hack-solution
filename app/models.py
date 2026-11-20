import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

OpaqueId = str | int


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value: OpaqueId) -> str:
    if type(value) not in (str, int) or (isinstance(value, str) and not value):
        raise ValueError("invalid opaque identifier")
    return f"{type(value).__name__}:{canonical(value)}"


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Section:
    key: str
    article_id: OpaqueId
    section_id: OpaqueId
    article_title: str
    heading: str
    text: str

    @property
    def pair_key(self) -> tuple[str, str]:
        return identity(self.article_id), identity(self.section_id)


@dataclass(frozen=True, slots=True)
class Expert:
    key: str
    expert_id: OpaqueId
    role: str
    style: str


@dataclass(frozen=True, slots=True)
class RequestRecord:
    request_id: OpaqueId
    user_id: OpaqueId
    expert_id: OpaqueId
    text: str
    context_json: str = "{}"


@dataclass(frozen=True, slots=True)
class Dataset:
    dataset_id: OpaqueId
    sections: tuple[Section, ...]
    experts: tuple[Expert, ...]
    requests: tuple[RequestRecord, ...]


@dataclass(frozen=True, slots=True)
class Group:
    key: str
    text: str
    context_json: str
    records: tuple[RequestRecord, ...]
    experts: tuple[Expert, ...]


@dataclass(frozen=True, slots=True)
class Risk:
    hard_veto: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Evidence:
    key: str
    section: Section
    start: int
    end: int
    score: float

    @property
    def text(self) -> str:
        return self.section.text[self.start : self.end]


@dataclass(frozen=True, slots=True)
class Answer:
    request_id: OpaqueId
    expert_id: OpaqueId
    text: str
    refused: bool
    citations: tuple[Section, ...]
    fallback: bool = False


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Plan(StrictModel):
    group_key: StrictStr
    standalone_query: StrictStr = Field(min_length=1, max_length=2000)
    terms_ru: list[StrictStr] = Field(max_length=12)
    terms_en: list[StrictStr] = Field(max_length=12)
    intent_hint: StrictStr = Field(max_length=100)
    candidate_sections: list[StrictStr] = Field(max_length=8)
    ambiguities: list[StrictStr] = Field(max_length=8)


class PlanBatch(StrictModel):
    plans: list[Plan]


class Quote(StrictModel):
    evidence_key: StrictStr
    quote: StrictStr = Field(min_length=8, max_length=1600)


class PersonaText(StrictModel):
    expert_key: StrictStr
    text: StrictStr = Field(min_length=12, max_length=5000)


class Candidate(StrictModel):
    group_key: StrictStr
    refused: StrictBool
    citations: list[StrictStr] = Field(min_length=1, max_length=2)
    evidence_quotes: list[Quote] = Field(min_length=1, max_length=4)
    answers: list[PersonaText] = Field(min_length=1, max_length=3)


@dataclass(slots=True)
class RunMetrics:
    groups: int = 0
    planner_valid: int = 0
    planner_degraded: int = 0
    fallback_groups: int = 0
    extra_generation_calls: int = 0
    validation_errors: dict[str, int] = field(default_factory=dict)

    def error(self, name: str) -> None:
        self.validation_errors[name] = self.validation_errors.get(name, 0) + 1
