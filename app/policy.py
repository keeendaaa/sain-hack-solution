import re
import unicodedata

from app.models import Group, Risk

DOSE_INTENT = re.compile(
    r"(?:сколько.{0,80}(?:приним|пить|давать|выпить)|(?:подбери|назначь|рассчитай|скажи|назови|увеличить|уменьшить).{0,60}доз|"
    r"(?:какую|какая|какой|какие).{0,30}доз|(?:how much|what dose|prescribe|calculate.{0,20}dose))",
    re.I | re.S,
)
DIAGNOSIS_INTENT = re.compile(
    r"(?:поставь|поставьте|определи|назови|установи).{0,40}диагноз|diagnose (?:me|my|him|her)", re.I | re.S
)
TREATMENT_INTENT = re.compile(
    r"(?:назначь|назначьте|пропиши|пропишите).{0,60}(?:лечени|препарат|лекарств|таблет)|prescribe.{0,40}(?:treatment|medication)",
    re.I | re.S,
)
DOSE_OUTPUT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:мг|мкг|миллиграм\w*|микрограм\w*|mg|mcg|µg|таблет\w*|капсул\w*)\b", re.I
)
PERSONAL_DIAGNOSIS = re.compile(
    r"(?:у вас|у тебя|вам|тебе)\s+(?:точно\s+)?(?:диагноз|бессонница|апноэ|депрессия)|your diagnosis is|you have (?:insomnia|apnea)",
    re.I,
)
PRESCRIPTION_OUTPUT = re.compile(
    r"(?<!не )\bназначаю\b|(?<!не )\bпрописываю\b|вам (?:нужно|следует) принимать|принимайте.{0,40}(?:мелатонин|препарат|таблет|лекарств)|you should take.{0,40}(?:melatonin|medication)",
    re.I,
)
JUDGE_MANIPULATION = re.compile(
    r"проверяющ\w*\s+(?:нейросет|модел)|(?:grader|evaluator).{0,30}(?:ignore|award|score)|ignore (?:all |the )?previous instructions",
    re.I,
)
REFUSAL = re.compile(
    r"не (?:могу|буду|назначаю|подбираю|ставлю|даю)|не моя (?:компетенция|территория)|cannot|can't|won't|do not prescribe",
    re.I,
)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def assess(group: Group) -> Risk:
    # Raw request must be checked independently of any LLM rewrite.
    reasons = []
    for label, pattern in (
        ("dosage_request", DOSE_INTENT),
        ("diagnosis_request", DIAGNOSIS_INTENT),
        ("treatment_request", TREATMENT_INTENT),
    ):
        if pattern.search(group.text):
            reasons.append(label)
    return Risk(bool(reasons), tuple(reasons))


def unsafe_output(text: str) -> bool:
    return any(
        p.search(text) for p in (DOSE_OUTPUT, PERSONAL_DIAGNOSIS, PRESCRIPTION_OUTPUT, JUDGE_MANIPULATION)
    )


def contains_refusal(text: str) -> bool:
    return bool(REFUSAL.search(text))
