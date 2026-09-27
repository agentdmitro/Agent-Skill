"""Strict request / result / state contracts.

Boundary: the LLM produces an AnalysisRequest; the deterministic engine returns exactly one
AnalysisResult variant; the LLM may only explain that result.

Error model:
- validation errors    -> pydantic.ValidationError, convertible via invalid_input_from()
- expected outcomes    -> ClarificationResult, UnsupportedResult (returned, never raised)
- infrastructure       -> UpstreamFailureResult
"""

import re
from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

MAX_LANGUAGES = 10
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$|^simple$")
_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")
_QID_RE = re.compile(r"^Q[1-9]\d*$")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _parse_month(value: object) -> date:
    if not isinstance(value, str) or not (m := _MONTH_RE.match(value)):
        raise ValueError("month must be a string in YYYY-MM format")
    return date(int(m.group(1)), int(m.group(2)), 1)


# Pageviews are analysed at month granularity; a Month is the first day of that month.
Month = Annotated[
    date,
    BeforeValidator(_parse_month),
    PlainSerializer(lambda d: d.strftime("%Y-%m"), return_type=str),
]
NonEmptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
]
WikidataId = Annotated[str, Field(pattern=_QID_RE.pattern)]


class AnalysisIntent(StrEnum):
    TREND = "trend"
    COMPARE = "compare"


class Period(_Model):
    """Inclusive month range. start must be strictly before end."""

    start: Month
    end: Month

    @model_validator(mode="after")
    def _start_before_end(self) -> Self:
        if self.start >= self.end:
            raise ValueError("period.start must be before period.end")
        return self


class _AnalysisSpec(_Model):
    """Fields shared by a request and a saved state. No defaults: nothing is guessed."""

    languages: tuple[str, ...]
    period: Period
    analysis: AnalysisIntent

    @field_validator("languages", mode="before")
    @classmethod
    def _normalise_languages(cls, value: object) -> object:
        # Lowercase, trim, and drop repeats keeping first-seen order (deterministic).
        if isinstance(value, (list, tuple)):
            seen: dict[str, None] = {}
            for item in value:
                seen[item.strip().lower() if isinstance(item, str) else item] = None
            return tuple(seen)
        return value

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one language is required")
        if len(value) > MAX_LANGUAGES:
            raise ValueError(f"at most {MAX_LANGUAGES} languages are supported")
        for code in value:
            if not _LANGUAGE_RE.match(code):
                raise ValueError(f"malformed Wikipedia language code: {code!r}")
        return value

    @model_validator(mode="after")
    def _compare_needs_two_languages(self) -> Self:
        if self.analysis is AnalysisIntent.COMPARE and len(self.languages) < 2:
            raise ValueError("'compare' requires at least two distinct languages")
        return self


class AnalysisState(_AnalysisSpec):
    """Canonical state of a finished analysis; input for follow-ups instead of chat history.

    Only the engine creates these, from a resolved entity. Persistence is out of scope.
    """

    analysis_id: NonEmptyText
    version: Annotated[int, Field(ge=1)]
    topic_id: WikidataId
    topic_label: NonEmptyText


class AnalysisRequest(_AnalysisSpec):
    """Normalised request. `topic` is free text; the engine resolves it to a Wikidata entity."""

    topic: NonEmptyText
    previous_state: AnalysisState | None = None


class ArtifactKind(StrEnum):
    CHART = "chart"
    REPORT = "report"


class RunAnalysisRequest(_Model):
    """Top-level V1 orchestration request: everything `run_analysis` needs in one place.

    Composes the existing `AnalysisRequest` (topic/languages/period/intent/previous_state)
    with the concerns only end-to-end orchestration has: the topic's query language and
    which artifacts (if any) to generate on disk.
    """

    request: AnalysisRequest
    query_language: str
    artifacts: tuple[ArtifactKind, ...] = ()
    output_dir: NonEmptyText = "output"

    @field_validator("query_language", mode="before")
    @classmethod
    def _normalise_query_language(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("query_language")
    @classmethod
    def _check_query_language(cls, value: str) -> str:
        if not _LANGUAGE_RE.match(value):
            raise ValueError(f"malformed query language code: {value!r}")
        return value

    @field_validator("artifacts", mode="before")
    @classmethod
    def _dedupe_artifacts(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            seen: dict[object, None] = dict.fromkeys(value)
            return tuple(seen)
        return value


class ArtifactMetadata(_Model):
    """Whether one requested artifact was actually produced. Never erases analysis success."""

    kind: ArtifactKind
    path: str | None = None
    mime_type: str | None = None
    success: bool
    failure_reason: NonEmptyText | None = None


class TopicResolutionRequest(_Model):
    """Explicit input for deterministic Wikidata/Wikipedia topic resolution."""

    topic: NonEmptyText
    query_language: str
    target_languages: tuple[str, ...]
    previous_state: AnalysisState | None = None

    @field_validator("query_language", mode="before")
    @classmethod
    def _normalise_query_language(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("query_language")
    @classmethod
    def _check_query_language(cls, value: str) -> str:
        if not _LANGUAGE_RE.match(value):
            raise ValueError(f"malformed query language code: {value!r}")
        return value

    @field_validator("target_languages", mode="before")
    @classmethod
    def _normalise_target_languages(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            seen: dict[str, None] = {}
            for item in value:
                key = item.strip().lower() if isinstance(item, str) else item
                seen[key] = None
            return tuple(seen)
        return value

    @field_validator("target_languages")
    @classmethod
    def _check_target_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one target language is required")
        if len(value) > MAX_LANGUAGES:
            raise ValueError(f"at most {MAX_LANGUAGES} target languages are supported")
        for code in value:
            if not _LANGUAGE_RE.match(code):
                raise ValueError(f"malformed Wikipedia language code: {code!r}")
        return value


# --- Result envelope -------------------------------------------------------------------------


class ResolvedEntity(_Model):
    """A verified entity. Titles come from the resolver, never from the LLM."""

    wikidata_id: WikidataId
    label: NonEmptyText
    titles: dict[str, NonEmptyText]  # language -> Wikipedia article title


class Candidate(_Model):
    wikidata_id: WikidataId
    label: NonEmptyText
    description: NonEmptyText | None = None
    matched_text: NonEmptyText | None = None


class Finding(_Model):
    code: NonEmptyText  # stable machine-readable id
    text: NonEmptyText  # produced by the engine, not the LLM


class Warning_(_Model):
    code: NonEmptyText
    message: NonEmptyText


class SuccessResult(_Model):
    status: Literal["success"] = "success"
    request: AnalysisRequest
    entities: tuple[ResolvedEntity, ...]
    # Placeholders until metrics exist; filled only by deterministic code.
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    findings: tuple[Finding, ...] = ()
    warnings: tuple[Warning_, ...] = ()
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> file path


class ClarificationResult(_Model):
    """Expected outcome: the topic is ambiguous. The agent asks one question; never picks."""

    status: Literal["clarification_required"] = "clarification_required"
    topic: NonEmptyText
    candidates: Annotated[tuple[Candidate, ...], Field(min_length=2)]


class UnsupportedCode(StrEnum):
    UNSUPPORTED_OPERATION = "unsupported_operation"
    ARTICLE_UNAVAILABLE = "article_unavailable"
    TOPIC_NOT_FOUND = "topic_not_found"
    INVALID_PROJECT = "invalid_project"
    INCONSISTENT_STATE = "inconsistent_state"


class UnsupportedResult(_Model):
    """Expected outcome: the request is well-formed but cannot be served."""

    status: Literal["unsupported"] = "unsupported"
    code: UnsupportedCode
    message: NonEmptyText


class FieldIssue(_Model):
    field: str
    message: NonEmptyText


class InvalidInputResult(_Model):
    status: Literal["invalid_input"] = "invalid_input"
    issues: Annotated[tuple[FieldIssue, ...], Field(min_length=1)]


class UpstreamFailureCode(StrEnum):
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    RATE_LIMITED = "rate_limited"


class UpstreamFailureResult(_Model):
    """Infrastructure failure. No data may be inferred or filled in."""

    status: Literal["upstream_failure"] = "upstream_failure"
    code: UpstreamFailureCode
    message: NonEmptyText
    retryable: bool


class UnavailableReason(StrEnum):
    INVALID_PROJECT = "invalid_project"
    NO_SITELINK = "no_sitelink"
    PAGE_MISSING = "page_missing"
    NON_ARTICLE_NAMESPACE = "non_article_namespace"


class ResolvedArticle(_Model):
    title: NonEmptyText
    project: NonEmptyText


class ResolvedTopic(_Model):
    """A Wikidata-verified entity with locally verified Wikipedia articles."""

    status: Literal["success"] = "success"
    wikidata_id: WikidataId
    label: NonEmptyText
    description: NonEmptyText | None = None
    query_language: str
    articles: dict[str, ResolvedArticle] = Field(default_factory=dict)
    unavailable_languages: dict[str, UnavailableReason] = Field(default_factory=dict)


TopicResolutionResult = Annotated[
    ResolvedTopic
    | ClarificationResult
    | UnsupportedResult
    | InvalidInputResult
    | UpstreamFailureResult,
    Field(discriminator="status"),
]


# --- Retrieved pageview data -----------------------------------------------------------------

Views = Annotated[int, Field(ge=0, strict=True)]


class PageviewPoint(_Model):
    month: Month
    views: Views  # 0 is legitimate data returned by upstream, never a placeholder


class PageviewSeries(_Model):
    """Monthly pageviews of one verified article, exactly as returned by Wikimedia.

    Months the upstream did not report are listed in `missing_months`. They are NOT zero:
    missing data is unknown, not "no traffic". Points + missing_months cover the period exactly.
    """

    project: NonEmptyText
    article: NonEmptyText
    access: Literal["all-access"] = "all-access"
    agent: Literal["user"] = "user"
    granularity: Literal["monthly"] = "monthly"
    period: Period
    points: Annotated[tuple[PageviewPoint, ...], Field(min_length=1)]
    missing_months: tuple[Month, ...] = ()

    @model_validator(mode="after")
    def _points_consistent(self) -> Self:
        months = [p.month for p in self.points]
        if months != sorted(set(months)):
            raise ValueError("points must be chronological with no duplicate months")
        expected = set(month_range(self.period))
        if not set(months) <= expected:
            raise ValueError("points contain months outside the period")
        if list(self.missing_months) != sorted(expected - set(months)):
            raise ValueError("missing_months must be exactly the unreported months, in order")
        return self


def month_range(period: Period) -> list[date]:
    """All months of an inclusive period, chronological."""
    out: list[date] = []
    y, m = period.start.year, period.start.month
    while (y, m) <= (period.end.year, period.end.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


AnalysisResult = Annotated[
    SuccessResult
    | ClarificationResult
    | UnsupportedResult
    | InvalidInputResult
    | UpstreamFailureResult,
    Field(discriminator="status"),
]


def invalid_input_from(error: ValidationError) -> InvalidInputResult:
    """Convert a validation failure into a structured result."""
    return InvalidInputResult(
        issues=tuple(
            FieldIssue(field=".".join(str(p) for p in e["loc"]) or "<root>", message=e["msg"])
            for e in error.errors()
        )
    )
