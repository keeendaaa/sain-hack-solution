import re

from app.models import Answer, Evidence, Expert, Group, Risk, Section, identity
from app.policy import unsafe_output


def voice_index(expert: Expert, ordinal: int) -> int:
    role = expert.role.lower()
    if any(word in role for word in ("биохак", "эксперимент", "experiment", "biohack")):
        return 1
    if any(word in role for word in ("нутри", "nutrition")):
        return 2
    if any(word in role for word in ("сомнолог", "клини", "clinic", "somnolog")):
        return 0
    return ordinal % 3


def definition(group: Group, evidence: tuple[Evidence, ...], risk: Risk) -> tuple[str, Section] | None:
    """A narrow, deterministic extractive path; never treats any matching text as an answer."""
    if risk.hard_veto:
        return None
    match = re.fullmatch(r"\s*(?:что такое|what is)\s+([\w -]{2,60})[?!.]*\s*", group.text, re.I)
    if not match:
        return None
    term = match.group(1).strip().lower()
    for e in evidence:
        if e.score <= 0 or unsafe_output(e.text):
            continue
        if re.match(re.escape(term) + r"\s*(?:—|–|-|это\b|is\b)", e.text.strip(), re.I):
            return e.text.strip(), e.section
    return None


def fallback_answers(
    group: Group,
    evidence: tuple[Evidence, ...],
    risk: Risk,
    navigation: Section,
    known_llm_refuse: bool = False,
) -> list[Answer]:
    extracted = None if known_llm_refuse else definition(group, evidence, risk)
    if extracted:
        quote, section = extracted
        refused = False
        variants = [
            f"В источнике дано такое определение: «{quote}» Это описание из указанного раздела; его границы важно сохранить.",
            f"Начнём с определения из базы: «{quote}» Именно этот смысл используем дальше, без добавления неподтверждённых объяснений.",
            f"Простыми словами, опора здесь на формулировку статьи: «{quote}» Её можно прочитать целиком по ссылке ниже.",
        ]
    else:
        section = next(
            (
                e.section
                for e in evidence
                if e.score > 0 and not unsafe_output(e.section.heading + " " + e.section.article_title)
            ),
            navigation,
        )
        refused = True
        # The title is navigation, not a fabricated recommendation/evidence claim.
        topic = section.heading if not unsafe_output(section.heading) else "материал базы о сне"
        if risk.hard_veto or known_llm_refuse:
            variants = [
                f"Не могу поставить диагноз, назначить лечение или подобрать дозировку. Такие решения требуют обращения к врачу. Могу предложить безопасное чтение: раздел «{topic}». Это навигация по базе, а не назначение.",
                f"Не буду превращать предположение в медицинский эксперимент: диагнозы, лечение и дозировки здесь не подбираю. Лучше обсудить их с врачом. А исследовать тему в рамках базы можно с раздела «{topic}» по ссылке.",
                f"Не могу рекомендовать лечение, диагноз или дозировку. Для этого нужен врач. Давайте ограничимся тем, что доступно для чтения: «{topic}». Ссылка помогает найти материал, а не заменяет личную консультацию.",
            ]
        else:
            variants = [
                f"Не могу дать достаточно обоснованный ответ по доступным фрагментам. Не буду додумывать выводы. Для дальнейшего чтения могу направить к разделу «{topic}»; он не подтверждает ответ на весь исходный вопрос.",
                f"Не буду выдавать догадку за результат: для уверенного ответа здесь недостаточно оснований. Можно начать изучение с раздела «{topic}», но пока не превращать его в рекомендацию по вашему вопросу.",
                f"Не могу сейчас надёжно ответить по этой базе. Лучше остановиться на понятном следующем шаге — прочитать раздел «{topic}» по ссылке. Это направление для чтения, а не обещание, что вопрос уже решён.",
            ]
    by_expert = {
        identity(expert.expert_id): variants[voice_index(expert, i)] for i, expert in enumerate(group.experts)
    }
    return [
        Answer(
            record.request_id,
            record.expert_id,
            by_expert[identity(record.expert_id)],
            refused,
            (section,),
            True,
        )
        for record in group.records
    ]
