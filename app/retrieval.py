import json
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import snowballstemmer

from app.errors import ContractError
from app.models import Dataset, Evidence, Group, Plan, Section, digest
from app.policy import unsafe_output
from app.tokens import TokenCounter

WORDS = re.compile(r"[a-zа-яё0-9]+", re.I)
STOP = {
    "и",
    "в",
    "на",
    "по",
    "с",
    "к",
    "о",
    "а",
    "как",
    "что",
    "это",
    "ли",
    "the",
    "a",
    "of",
    "in",
    "to",
    "is",
    "and",
}


@dataclass(frozen=True, slots=True)
class RankedSection:
    section: Section
    score: float


class Retriever:
    VERSION = "bm25-rrf-v1"

    def __init__(self, dataset: Dataset, counter: TokenCounter, terms_path: Path):
        self.sections = dataset.sections
        self.by_key = {s.key: s for s in self.sections}
        self.counter = counter
        self._stem_lock = threading.Lock()
        self.ru = snowballstemmer.stemmer("russian")
        self.en = snowballstemmer.stemmer("english")
        self.terms: list[list[str]] = json.loads(terms_path.read_text())
        self.term_tokens = [set(self.tokens(" ".join(cluster))) for cluster in self.terms]
        self.docs = [
            Counter(self.tokens(f"{s.article_title} {s.article_title} {s.heading} {s.heading} {s.text}"))
            for s in self.sections
        ]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.average = sum(self.lengths) / max(1, len(self.lengths))
        frequency: Counter = Counter()
        for doc in self.docs:
            frequency.update(doc.keys())
        self.idf = {w: math.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5)) for w, df in frequency.items()}
        self.grams = [self.chargrams(s.article_title + " " + s.heading + " " + s.text) for s in self.sections]
        self.articles: dict[str, list[Section]] = {}
        article_lookup = {}
        for section in self.sections:
            article_identity = section.pair_key[0]
            if article_identity not in article_lookup:
                article_lookup[article_identity] = f"A{len(article_lookup):04d}"
            self.articles.setdefault(article_lookup[article_identity], []).append(section)
        self.hash = digest(
            {
                "version": self.VERSION,
                "terms": self.terms,
                "sections": sorted(
                    [(s.pair_key, s.article_title, s.heading, s.text) for s in self.sections],
                    key=lambda x: x[0],
                ),
            }
        )
        safe = [
            s
            for s in self.sections
            if not unsafe_output(s.article_title + " " + s.heading)
            and any(not unsafe_output(chunk[2]) for chunk in self.sentences(s.text))
        ]
        if not safe:
            raise ContractError("no_safe_navigation_section")
        self.fallback_section = min(
            safe,
            key=lambda s: (
                -int(bool(re.search(r"обзор|введение|общие|overview|introduction|summary", s.heading, re.I))),
                -int(bool(re.search(r"сон|sleep", s.article_title, re.I))),
                s.pair_key,
            ),
        )

    def tokens(self, text: str) -> list[str]:
        # Snowball stemmers contain mutable cursor state; evidence workers use threads.
        with self._stem_lock:
            result = []
            for word in WORDS.findall(text.lower().replace("ё", "е")):
                if word in STOP:
                    continue
                result.append(self.ru.stemWord(word) if re.search("[а-я]", word) else self.en.stemWord(word))
            return result

    @staticmethod
    def chargrams(text: str) -> set[str]:
        words = WORDS.findall(text.lower().replace("ё", "е"))
        return {word[i : i + 3] for word in words for i in range(max(0, len(word) - 2))}

    def expanded(self, query: str) -> str:
        qt = set(self.tokens(query))
        additions = [
            " ".join(cluster)
            for cluster, terms in zip(self.terms, self.term_tokens, strict=True)
            if qt & terms
        ]
        return query + " " + " ".join(additions)

    def bm25(self, query: str) -> list[RankedSection]:
        words = set(self.tokens(query))
        scores = []
        for section, doc, length in zip(self.sections, self.docs, self.lengths, strict=True):
            score = sum(
                self.idf.get(w, 0)
                * doc[w]
                * 2.5
                / (doc[w] + 1.5 * (0.25 + 0.75 * length / max(1, self.average)))
                for w in words
                if doc[w]
            )
            if score > 0:
                scores.append(RankedSection(section, score))
        return sorted(scores, key=lambda d: (-d.score, d.section.key))

    def rank(self, original: str, plan: Plan | None = None) -> list[RankedSection]:
        rewritten = " ".join([plan.standalone_query, *plan.terms_ru, *plan.terms_en]) if plan else original
        lexical = self.bm25(original)
        expanded = self.bm25(self.expanded(rewritten))
        grams = self.chargrams(original)
        typo = sorted(
            [
                (s.key, len(grams & g) / max(1, len(grams | g)))
                for s, g in zip(self.sections, self.grams, strict=True)
            ],
            key=lambda p: (-p[1], p[0]),
        )
        semantic: list[str] = []
        if plan:
            for key in plan.candidate_sections:
                if key in self.by_key:
                    semantic.append(key)
                elif key in self.articles:
                    article_keys = {s.key for s in self.articles[key]}
                    preferred = [r.section.key for r in expanded if r.section.key in article_keys]
                    semantic.extend((preferred or [s.key for s in self.articles[key]])[:2])
        channels = [
            [r.section.key for r in lexical[:12]],
            [r.section.key for r in expanded[:12]],
            [k for k, score in typo[:12] if score > 0],
            list(dict.fromkeys(semantic)),
        ]
        scores: dict[str, float] = defaultdict(float)
        for channel in channels:
            for i, key in enumerate(channel):
                scores[key] += 1 / (60 + i + 1)
        # First candidate of each nonempty channel is retained, including semantic.
        # Protect the semantic channel early when the final prompt must fit fewer windows.
        selected = list(dict.fromkeys(channels[i][0] for i in (0, 3, 1, 2) if channels[i]))
        for key in sorted(scores, key=lambda k: (-scores[k], k)):
            if key not in selected:
                selected.append(key)
            if len(selected) >= 8:
                break
        return [RankedSection(self.by_key[k], scores[k]) for k in selected]

    @staticmethod
    def sentences(text: str) -> list[tuple[int, int, str]]:
        bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text)] + [len(text)]
        return [(a, b, text[a:b]) for a, b in zip(bounds, bounds[1:], strict=False) if text[a:b].strip()]

    def evidence(self, group: Group, plan: Plan | None = None) -> tuple[Evidence, ...]:
        query = (
            group.text
            if not plan
            else " ".join([group.text, plan.standalone_query, *plan.terms_ru, *plan.terms_en])
        )
        terms = set(self.tokens(self.expanded(query)))
        result = []
        for ranked in self.rank(group.text, plan):
            section = ranked.section
            sentences = self.sentences(section.text)
            if not sentences:
                continue
            best = max(range(len(sentences)), key=lambda i: len(terms & set(self.tokens(sentences[i][2]))))
            lo = max(0, best - 1)
            hi = min(len(sentences), best + 2)
            start, end = sentences[lo][0], sentences[hi - 1][1]
            # Preserve adjacent caveats. Never crop inside a sentence to meet a cap.
            if self.counter.count(section.text[start:end]) > 150:
                start, end, _ = sentences[best]
                adjacent = " ".join(s[2] for s in sentences[lo:hi])
                if re.search(r"\b(?:not|however|only|не|однако|только|огранич)\b", adjacent, re.I):
                    start, end = sentences[lo][0], sentences[hi - 1][1]
            # A caveat may need a larger window. Drop another whole source at packing,
            # rather than turning "no effect" into "effect" by trimming this one.
            if self.counter.count(section.text[start:end]) > 300:
                continue
            result.append(Evidence(f"E{len(result):02d}", section, start, end, ranked.score))
            if len(result) == 4:
                break
        return tuple(result)

    def catalog(self, article_level: bool = False) -> list[dict]:
        if article_level:
            return [
                {
                    "key": key,
                    "title": sections[0].article_title,
                    "headings": [s.heading for s in sections],
                    "excerpt": self.sentences(sections[0].text)[0][2][:240],
                }
                for key, sections in self.articles.items()
            ]
        return [
            {
                "key": s.key,
                "article": s.article_title,
                "heading": s.heading,
                "excerpt": self.sentences(s.text)[0][2][:160],
            }
            for s in self.sections
        ]
