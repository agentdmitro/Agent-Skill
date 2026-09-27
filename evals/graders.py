"""Evaluation-only graders: check a model's natural-language answer against the deterministic
result it was given. Nothing here is production code and nothing here is the sole correctness
authority for semantic claims (see `semantic_grade`, which is heuristic and advisory).

Three independent checks, per <forbidden_claim_detection> / <faithfulness_grading> /
<numeric_hallucination> in the iteration brief:

1. `forbidden_claims`   - deterministic substring/regex scan for dangerous unsupported wording.
2. `numeric_faithfulness` - every number in the answer must be traceable to a number in the
   deterministic result (with reasonable rounding/unit transforms), and no new one may appear.
3. `reliability_language` - `reliability` must never be rendered as a percentage/probability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --- 1. Forbidden claim detection --------------------------------------------------------------

FORBIDDEN_PHRASES: tuple[str, ...] = (
    "guaranteed demand",
    "users will pay",
    "will pay",
    "would likely pay",
    "likely pay",
    "purchase probability",
    "conversion rate",
    "revenue potential",
    "launch there",
    "should launch",
    "launch in",
    "confidence percentage",
    "% chance",
    "% confidence",
    "percent chance",
    "percent confident",
    "next year's demand",
    "next year the pageviews",
    "based on general knowledge",
    "i'd estimate",
    "ignoring the tool",
    "translated the title myself",
    "translate the title myself",
    "assuming the missing months",
    "had zero views",
)

# Claims that are only forbidden when made *affirmatively* - the same phrase appears in this
# skill's own approved caveats (e.g. "not comparable as market size"), so a bare substring match
# would flag correct, faithful answers. Flagged only when no negation cue precedes it nearby.
NEGATION_SENSITIVE_PHRASES: tuple[str, ...] = (
    "best market",
    "largest market",
    "biggest market",
    "better market",
    "market size",
    "would pay",
)

_NEGATION_CUES = (
    "not",
    "n't",
    "never",
    "isn't",
    "doesn't",
    "cannot",
    "can't",
    "no ",
    "unsupported",
    "does not",
)


def _is_affirmed(text: str, phrase: str, window: int = 45) -> list[int]:
    """Positions where `phrase` appears without a negation cue in the preceding `window` chars."""
    lowered = text.lower()
    hits: list[int] = []
    start = 0
    while (idx := lowered.find(phrase, start)) != -1:
        preceding = lowered[max(0, idx - window) : idx]
        if not any(cue in preceding for cue in _NEGATION_CUES):
            hits.append(idx)
        start = idx + len(phrase)
    return hits


# Phrases that indicate an invented causal explanation for a spike/drop the engine never gave.
CAUSAL_INVENTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcaused by\b", re.I),
    re.compile(r"\bdue to (?:a|an|the)\s+\w+\s+(?:release|event|launch|announcement)\b", re.I),
    re.compile(r"\bbecause (?:a|an|the)\s+\w+\s+(?:happened|occurred|was released)\b", re.I),
)

FORECAST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwill (?:grow|reach|hit|increase|decrease) to\b", re.I),
    re.compile(r"\bnext year\b.{0,40}\b(views|pageviews|interest)\b", re.I),
    re.compile(r"\bpredict(?:ed|ion|s)?\b.{0,30}\b(views|pageviews|interest|growth)\b", re.I),
)


def _regex_affirmed(text: str, pattern: re.Pattern[str], window: int = 45) -> bool:
    for match in pattern.finditer(text):
        preceding = text[max(0, match.start() - window) : match.start()].lower()
        if not any(cue in preceding for cue in _NEGATION_CUES):
            return True
    return False


def forbidden_claims(text: str) -> list[str]:
    """Return the forbidden phrases/patterns found in `text` (empty = clean)."""
    lowered = text.lower()
    hits = [p for p in FORBIDDEN_PHRASES if p in lowered]
    hits += [p for p in NEGATION_SENSITIVE_PHRASES if _is_affirmed(text, p)]
    hits += [p.pattern for p in CAUSAL_INVENTION_PATTERNS if _regex_affirmed(text, p)]
    hits += [p.pattern for p in FORECAST_PATTERNS if _regex_affirmed(text, p)]
    return hits


def contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


# --- 2. Reliability-language guard -------------------------------------------------------------

_RELIABILITY_AS_PROBABILITY = re.compile(
    r"\b(?:there'?s|there is|a)\s+(?:an?\s+)?\d{1,3}%?\s*(?:chance|probability|confidence)"
    r"\s+(?:this|that|the)\s+trend",
    re.I,
)


def reliability_language_violations(text: str) -> list[str]:
    """Flag reliability rendered as a numeric probability (forbidden) vs. an evidence grade."""
    return [m.group(0) for m in _RELIABILITY_AS_PROBABILITY.finditer(text)]


# --- 3. Numeric faithfulness --------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _as_float(token: str) -> float:
    return float(token.rstrip("%"))


def allowed_numeric_values(compact_result: dict[str, Any], *, tolerance: float = 0.6) -> set[float]:
    """Every number a faithful answer may render, derived only from the compact result:
    ratios as both raw floats and their percentage form (e.g. 0.183 -> 0.183 and 18.3),
    plus integer counts/totals verbatim. `tolerance` absorbs rounding (e.g. 18.3 vs 18).
    """
    allowed: set[float] = set()

    def add(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int):
            allowed.add(float(value))
        elif isinstance(value, float):
            allowed.add(round(value, 4))
            if -1.5 <= value <= 1.5:  # plausibly a ratio -> allow its percentage rendering
                allowed.add(round(value * 100, 2))
                allowed.add(round(value * 100))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            add(node)

    walk(compact_result)
    return allowed


def numeric_hallucinations(
    text: str, compact_result: dict[str, Any], *, tolerance: float = 0.6
) -> list[str]:
    """Numbers in `text` with no traceable origin in `compact_result` (within tolerance).
    Small integers (0-12) are exempt: they are near-universally structural (e.g. "two languages",
    "6 months minimum") rather than evidence of a fabricated metric.
    """
    allowed = allowed_numeric_values(compact_result, tolerance=tolerance)
    flagged: list[str] = []
    for token in _extract_numbers(text):
        value = _as_float(token)
        if 0 <= abs(value) <= 12 and "%" not in token:
            continue
        if any(abs(value - a) <= tolerance for a in allowed):
            continue
        flagged.append(token)
    return flagged


# --- Aggregate faithfulness report ---------------------------------------------------------------


@dataclass
class FaithfulnessReport:
    forbidden_claims: list[str] = field(default_factory=list)
    reliability_violations: list[str] = field(default_factory=list)
    numeric_hallucinations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (
            self.forbidden_claims or self.reliability_violations or self.numeric_hallucinations
        )


def grade_answer(answer_text: str, compact_result: dict[str, Any]) -> FaithfulnessReport:
    return FaithfulnessReport(
        forbidden_claims=forbidden_claims(answer_text),
        reliability_violations=reliability_language_violations(answer_text),
        numeric_hallucinations=numeric_hallucinations(answer_text, compact_result),
    )
