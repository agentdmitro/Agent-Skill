"""Deterministic cross-language comparison of already-analyzed Wikipedia pageview series.

This module never fetches data and never recomputes trend/reliability statistics; it
consumes :class:`~wikipedia_interest.analysis.AnalysisReport` values produced by
Iteration 4 and compares them along three separate axes that must never collapse into
one score:

1. absolute attention   (how many pageviews an article receives in each edition)
2. trend                (how interest is changing within each edition)
3. reliability          (how strongly the data supports that trend conclusion)

Raw Wikipedia pageviews are not a market-size measure (language populations, Wikipedia
usage habits, and article coverage differ by edition). Nothing here infers market size,
revenue potential, or purchase intent. `candidate_classification` means only "this
language edition's Wikipedia interest signal may be worth further validation" - never a
launch or market recommendation.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wikipedia_interest.analysis import (
    AnalysisReport,
    OutlierDependency,
    RecentConfirmation,
    Reliability,
    TrendDirection,
)
from wikipedia_interest.contracts import NonEmptyText, Period, WikidataId

MIN_LANGUAGES_FOR_COMPARISON = 2


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UnavailabilityCategory(StrEnum):
    """Whether retrying might help. Domain unavailability never will; upstream failure might."""

    DOMAIN_UNAVAILABLE = "domain_unavailable"
    UPSTREAM_FAILURE = "upstream_failure"


class LanguageUnavailability(_Model):
    language: str
    category: UnavailabilityCategory
    reason: NonEmptyText
    retryable: bool


class LanguageAnalysis(_Model):
    """One language edition's verified topic identity plus its Iteration 4 analysis."""

    language: str
    topic_id: WikidataId
    analysis: AnalysisReport


def _dedupe_by_language(items: object) -> object:
    """Keep first-seen order per language, mirroring the Iteration 1 request dedup."""
    if not isinstance(items, (list, tuple)):
        return items
    seen: dict[str, object] = {}
    for item in items:
        language: object = (
            item.get("language") if isinstance(item, dict) else getattr(item, "language", None)
        )
        if isinstance(language, str):
            seen.setdefault(language.strip().lower(), item)
    return tuple(seen.values())


class ComparisonInput(_Model):
    """Pure, deterministic input: already-analyzed languages, never raw Wikimedia payloads."""

    period: Period
    analyses: tuple[LanguageAnalysis, ...] = ()
    unavailable: tuple[LanguageUnavailability, ...] = ()

    @field_validator("analyses", mode="before")
    @classmethod
    def _dedupe_analyses(cls, value: object) -> object:
        return _dedupe_by_language(value)

    @field_validator("unavailable", mode="before")
    @classmethod
    def _dedupe_unavailable(cls, value: object) -> object:
        return _dedupe_by_language(value)

    @model_validator(mode="after")
    def _no_overlap(self) -> ComparisonInput:
        analyzed = {a.language for a in self.analyses}
        unavailable = {u.language for u in self.unavailable}
        overlap = analyzed & unavailable
        if overlap:
            raise ValueError(
                f"language(s) {sorted(overlap)} cannot be both analyzed and unavailable"
            )
        return self


class ComparisonRejectionCode(StrEnum):
    TOPIC_MISMATCH = "topic_mismatch"
    PERIOD_MISMATCH = "period_mismatch"
    INSUFFICIENT_LANGUAGES = "insufficient_languages"


class ComparisonRejected(_Model):
    """Expected outcome: the input cannot be compared as given. Never raised, always returned."""

    status: Literal["rejected"] = "rejected"
    code: ComparisonRejectionCode
    message: NonEmptyText


class CandidateClassification(StrEnum):
    """ "Worth further validation" signal strength, never a market or launch recommendation."""

    STRONG_CANDIDATE = "strong_candidate"
    POSSIBLE_CANDIDATE = "possible_candidate"
    INCONCLUSIVE = "inconclusive"
    WEAK_CURRENT_SIGNAL = "weak_current_signal"


class LanguageComparisonSummary(_Model):
    """Everything a cheap LLM needs to explain one language's result without recomputing math."""

    language: str
    article_title: NonEmptyText
    total_views: int = Field(ge=0)
    mean_monthly_views: float | None = Field(default=None, ge=0)
    median_monthly_views: float | None = Field(default=None, ge=0)
    comparison_pageview_share: float | None = Field(default=None, ge=0, le=1)
    trend_direction: TrendDirection
    normalized_trend_slope: float | None = None
    period_change: float | None = None
    recent_change: float | None = None
    recent_confirmation: RecentConfirmation
    reliability: Reliability
    coverage_ratio: float = Field(ge=0, le=1)
    observed_months: int = Field(ge=0)
    missing_months: int = Field(ge=0)
    candidate_classification: CandidateClassification
    evidence_codes: tuple[str, ...] = ()


class ComparisonFinding(_Model):
    """A compact aggregate statement, never a per-pair explosion. `metric` names the basis."""

    code: NonEmptyText
    languages: Annotated[tuple[str, ...], Field(min_length=1)]
    metric: NonEmptyText | None = None
    detail: NonEmptyText


class ComparisonReport(_Model):
    status: Literal["success"] = "success"
    topic_id: WikidataId
    period: Period
    languages: tuple[LanguageComparisonSummary, ...]
    unavailable: tuple[LanguageUnavailability, ...] = ()
    findings: tuple[ComparisonFinding, ...] = ()


ComparisonResult = Annotated[ComparisonReport | ComparisonRejected, Field(discriminator="status")]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def classify_candidate(report: AnalysisReport) -> tuple[CandidateClassification, tuple[str, ...]]:
    """Explicit, testable rules. See module docstring: this is an evidence signal, not a verdict.

    STRONG_CANDIDATE requires: growing trend, high or medium reliability, recent confirmation
    not contradicting growth, and outlier dependency that is not high.
    POSSIBLE_CANDIDATE: a positive (growing, or stable-with-recent-uptick) signal exists but
    reliability, recent confirmation, or outlier dependency weakens it.
    WEAK_CURRENT_SIGNAL: a reliable decline, or a reliable stable trend with no recent growth.
    INCONCLUSIVE: unclear/insufficient trend evidence, or an unreliable decline/stable trend.
    """
    direction = report.trend.direction
    reliability = report.trend.reliability
    recent = report.metrics.recent_confirmation
    dependency = report.metrics.outlier_dependency

    if direction in (TrendDirection.INSUFFICIENT_DATA, TrendDirection.UNCLEAR):
        return CandidateClassification.INCONCLUSIVE, ("insufficient_or_unclear_trend",)
    if reliability is Reliability.INSUFFICIENT:
        return CandidateClassification.INCONCLUSIVE, ("insufficient_reliability_evidence",)

    if direction is TrendDirection.GROWING:
        if recent is RecentConfirmation.CONTRADICTED:
            return CandidateClassification.POSSIBLE_CANDIDATE, ("recent_trend_contradicts_growth",)
        if dependency is OutlierDependency.HIGH:
            return CandidateClassification.POSSIBLE_CANDIDATE, ("outlier_dependent_growth",)
        if reliability in (Reliability.HIGH, Reliability.MEDIUM):
            return CandidateClassification.STRONG_CANDIDATE, ("growing_with_supported_reliability",)
        return CandidateClassification.POSSIBLE_CANDIDATE, ("growing_low_reliability",)

    if direction is TrendDirection.DECLINING:
        if reliability in (Reliability.HIGH, Reliability.MEDIUM):
            return CandidateClassification.WEAK_CURRENT_SIGNAL, ("reliable_decline",)
        return CandidateClassification.INCONCLUSIVE, ("unreliable_decline",)

    # STABLE
    if recent is RecentConfirmation.CONFIRMED:
        return CandidateClassification.POSSIBLE_CANDIDATE, ("stable_with_recent_uptick",)
    if reliability in (Reliability.HIGH, Reliability.MEDIUM):
        return CandidateClassification.WEAK_CURRENT_SIGNAL, ("reliable_stable_no_growth",)
    return CandidateClassification.INCONCLUSIVE, ("unreliable_stable",)


def _summarize(language_analysis: LanguageAnalysis, total_of_all: int) -> LanguageComparisonSummary:
    report = language_analysis.analysis
    classification, codes = classify_candidate(report)
    share = None if total_of_all == 0 else _round(report.metrics.total_views / total_of_all)
    return LanguageComparisonSummary(
        language=language_analysis.language,
        article_title=report.article.title,
        total_views=report.metrics.total_views,
        mean_monthly_views=report.metrics.mean_monthly_views,
        median_monthly_views=report.metrics.median_monthly_views,
        comparison_pageview_share=share,
        trend_direction=report.trend.direction,
        normalized_trend_slope=report.metrics.normalized_trend_slope,
        period_change=report.metrics.period_change,
        recent_change=report.metrics.recent_change,
        recent_confirmation=report.metrics.recent_confirmation,
        reliability=report.trend.reliability,
        coverage_ratio=report.data_quality.coverage_ratio,
        observed_months=report.data_quality.observed_months,
        missing_months=report.data_quality.missing_months,
        candidate_classification=classification,
        evidence_codes=codes,
    )


def _top(
    summaries: tuple[LanguageComparisonSummary, ...],
    key: Callable[[LanguageComparisonSummary], float | None],
    predicate: Callable[[LanguageComparisonSummary], bool] | None = None,
) -> LanguageComparisonSummary | None:
    """Deterministic highest-first pick by `key`: ties break on ascending language code."""
    candidates = [
        s for s in summaries if key(s) is not None and (predicate is None or predicate(s))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: (-(key(s) or 0.0), s.language))


def _findings(summaries: tuple[LanguageComparisonSummary, ...]) -> tuple[ComparisonFinding, ...]:
    findings: list[ComparisonFinding] = []

    reliable_grower = _top(
        summaries,
        lambda s: s.normalized_trend_slope,
        lambda s: (
            s.trend_direction is TrendDirection.GROWING
            and s.reliability in (Reliability.HIGH, Reliability.MEDIUM)
        ),
    )
    if reliable_grower is not None:
        findings.append(
            ComparisonFinding(
                code="strongest_reliable_growth",
                languages=(reliable_grower.language,),
                metric="normalized_trend_slope",
                detail=(
                    f"{reliable_grower.language} has the strongest normalized robust trend "
                    "slope among languages with growing, reliable evidence."
                ),
            )
        )

    fastest_grower = _top(
        summaries,
        lambda s: s.normalized_trend_slope,
        lambda s: s.trend_direction is TrendDirection.GROWING,
    )
    if (
        fastest_grower is not None
        and reliable_grower is not None
        and fastest_grower.language != reliable_grower.language
    ):
        findings.append(
            ComparisonFinding(
                code="higher_growth_but_lower_reliability",
                languages=(fastest_grower.language,),
                metric="normalized_trend_slope",
                detail=(
                    f"{fastest_grower.language} shows a larger normalized trend slope than "
                    f"{reliable_grower.language}, but with weaker reliability evidence."
                ),
            )
        )

    highest_attention = _top(summaries, lambda s: float(s.total_views))
    if highest_attention is not None:
        findings.append(
            ComparisonFinding(
                code="highest_absolute_attention",
                languages=(highest_attention.language,),
                metric="total_views",
                detail=(
                    f"{highest_attention.language} has the highest total article pageviews "
                    "in this Wikipedia edition over the compared period."
                ),
            )
        )

    contradicted = tuple(
        sorted(
            s.language for s in summaries if "recent_trend_contradicts_growth" in s.evidence_codes
        )
    )
    if contradicted:
        findings.append(
            ComparisonFinding(
                code="recent_trend_contradicts_long_term",
                languages=contradicted,
                metric="recent_confirmation",
                detail=(
                    "The most recent comparable window moved against the long-term trend "
                    "for these languages."
                ),
            )
        )

    insufficient = tuple(
        sorted(
            s.language
            for s in summaries
            if s.candidate_classification is CandidateClassification.INCONCLUSIVE
        )
    )
    if insufficient:
        findings.append(
            ComparisonFinding(
                code="insufficient_evidence",
                languages=insufficient,
                metric=None,
                detail=(
                    "Available evidence does not support a trend conclusion for these "
                    "languages (unclear trend, insufficient data, or insufficient reliability)."
                ),
            )
        )

    return tuple(findings)


def compare_languages(comparison_input: ComparisonInput) -> ComparisonResult:
    """Pure, deterministic comparison. Rejects rather than silently comparing the incomparable."""
    analyses = comparison_input.analyses
    topic_ids = {a.topic_id for a in analyses}
    if len(topic_ids) > 1:
        return ComparisonRejected(
            code=ComparisonRejectionCode.TOPIC_MISMATCH,
            message="Analyses reference different Wikidata topics and cannot be compared.",
        )

    mismatched = sorted(
        a.language for a in analyses if a.analysis.period != comparison_input.period
    )
    if mismatched:
        return ComparisonRejected(
            code=ComparisonRejectionCode.PERIOD_MISMATCH,
            message=(
                f"Language(s) {mismatched} were analyzed over a period different from the "
                "requested comparison period."
            ),
        )

    if len(analyses) < MIN_LANGUAGES_FOR_COMPARISON:
        return ComparisonRejected(
            code=ComparisonRejectionCode.INSUFFICIENT_LANGUAGES,
            message=(
                f"At least {MIN_LANGUAGES_FOR_COMPARISON} successfully analyzed languages are "
                f"required to compare; {len(analyses)} available."
            ),
        )

    total_views = sum(a.analysis.metrics.total_views for a in analyses)
    summaries = tuple(
        sorted(
            (_summarize(a, total_views) for a in analyses),
            key=lambda s: s.language,
        )
    )
    return ComparisonReport(
        topic_id=next(iter(topic_ids)),
        period=comparison_input.period,
        languages=summaries,
        unavailable=comparison_input.unavailable,
        findings=_findings(summaries),
    )
