"""Thin glue: resolve topic -> fetch per-language pageviews -> analyze -> compare.

No business logic lives here. Each step is delegated to its own module
(`resolver`, `wikimedia`, `analysis`, `comparison`); this module only wires them
together and translates outcomes between their contracts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from wikipedia_interest.analysis import (
    AnalysisReport,
    OutlierDependency,
    RecentConfirmation,
    Reliability,
    TrendDirection,
    analyze_pageview_series,
)
from wikipedia_interest.comparison import (
    CandidateClassification,
    ComparisonInput,
    ComparisonResult,
    LanguageAnalysis,
    LanguageUnavailability,
    UnavailabilityCategory,
    classify_candidate,
    compare_languages,
)
from wikipedia_interest.contracts import (
    AnalysisIntent,
    AnalysisRequest,
    AnalysisState,
    ArtifactKind,
    ArtifactMetadata,
    ClarificationResult,
    Finding,
    InvalidInputResult,
    PageviewSeries,
    Period,
    ResolvedArticle,
    ResolvedTopic,
    RunAnalysisRequest,
    TopicResolutionRequest,
    UnsupportedResult,
    UpstreamFailureResult,
)
from wikipedia_interest.reporting import generate_chart, generate_report
from wikipedia_interest.resolver import TopicResolver
from wikipedia_interest.wikimedia import WikimediaClient


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ComparisonRun(_Model):
    """Everything produced by one orchestration pass, including languages the comparison
    engine could not use (e.g. only one language succeeded). The single-language analyses
    remain available here even when `comparison` is a rejection."""

    status: Literal["success"] = "success"
    resolved: ResolvedTopic
    analyses: tuple[LanguageAnalysis, ...]
    unavailable: tuple[LanguageUnavailability, ...]
    comparison: ComparisonResult


OrchestrationResult = (
    ComparisonRun
    | ClarificationResult
    | UnsupportedResult
    | InvalidInputResult
    | UpstreamFailureResult
)


_FetchedLanguages = tuple[
    list[LanguageAnalysis], dict[str, PageviewSeries], list[LanguageUnavailability]
]


def _fetch_all(
    resolved: ResolvedTopic, period: Period, client: WikimediaClient
) -> _FetchedLanguages | InvalidInputResult:
    """Fetch + analyze pageviews for every verified article. Shared by both entry points.

    Raw `PageviewSeries` are returned separately from `LanguageAnalysis` (which only carries
    the Iteration 4 descriptive report) so callers that don't need them for charting never see
    a per-language series at all.
    """
    unavailable: list[LanguageUnavailability] = [
        LanguageUnavailability(
            language=language,
            category=UnavailabilityCategory.DOMAIN_UNAVAILABLE,
            reason=reason.value,
            retryable=False,
        )
        for language, reason in resolved.unavailable_languages.items()
    ]
    analyses: list[LanguageAnalysis] = []
    series_by_language: dict[str, PageviewSeries] = {}
    for language, article in resolved.articles.items():
        fetched = client.fetch_monthly_pageviews(language, article.title, period)
        if isinstance(fetched, PageviewSeries):
            series_by_language[language] = fetched
            analyses.append(
                LanguageAnalysis(
                    language=language,
                    topic_id=resolved.wikidata_id,
                    analysis=analyze_pageview_series(fetched),
                )
            )
        elif isinstance(fetched, InvalidInputResult):
            # A malformed request (e.g. period predates Wikimedia's dataset) affects every
            # language identically; it is not a per-language outcome.
            return fetched
        elif isinstance(fetched, UpstreamFailureResult):
            unavailable.append(
                LanguageUnavailability(
                    language=language,
                    category=UnavailabilityCategory.UPSTREAM_FAILURE,
                    reason=fetched.code.value,
                    retryable=fetched.retryable,
                )
            )
        else:  # UnsupportedResult: no pageview data for this article
            unavailable.append(
                LanguageUnavailability(
                    language=language,
                    category=UnavailabilityCategory.DOMAIN_UNAVAILABLE,
                    reason=fetched.code.value,
                    retryable=False,
                )
            )
    return analyses, series_by_language, unavailable


def run_comparison(
    *,
    topic: str,
    query_language: str,
    languages: tuple[str, ...],
    period: Period,
    resolver: TopicResolver,
    client: WikimediaClient,
    previous_state: AnalysisState | None = None,
) -> OrchestrationResult:
    """Resolve, fetch, analyze, then compare. Returns the first structural failure verbatim."""
    resolution_request = TopicResolutionRequest(
        topic=topic,
        query_language=query_language,
        target_languages=languages,
        previous_state=previous_state,
    )
    resolved = resolver.resolve(resolution_request)
    if not isinstance(resolved, ResolvedTopic):
        return resolved

    fetched = _fetch_all(resolved, period, client)
    if isinstance(fetched, InvalidInputResult):
        return fetched
    analyses, _series_by_language, unavailable = fetched

    comparison = compare_languages(
        ComparisonInput(period=period, analyses=tuple(analyses), unavailable=tuple(unavailable))
    )
    return ComparisonRun(
        resolved=resolved,
        analyses=tuple(analyses),
        unavailable=tuple(unavailable),
        comparison=comparison,
    )


# --- V1 end-to-end orchestration --------------------------------------------------------------


_ANALYSIS_ID_PREFIX = "wi"


class LanguageResult(_Model):
    """Everything a cheap LLM needs to explain one language's result: no raw pageviews."""

    language: str
    article: ResolvedArticle
    analysis: AnalysisReport
    candidate_classification: CandidateClassification
    candidate_evidence_codes: tuple[str, ...] = ()


class RunResult(_Model):
    """The V1 top-level success envelope: compact per-language results, an optional
    cross-language comparison, deterministic findings, artifact outcomes, and canonical
    follow-up state. Unavailable/failed languages are always listed, never dropped."""

    status: Literal["success"] = "success"
    resolved: ResolvedTopic
    request: AnalysisRequest
    requested_languages: tuple[str, ...]
    languages: tuple[LanguageResult, ...]
    unavailable: tuple[LanguageUnavailability, ...]
    comparison: ComparisonResult | None = None
    findings: tuple[Finding, ...] = ()
    artifacts: tuple[ArtifactMetadata, ...] = ()
    next_state: AnalysisState


RunAnalysisResult = (
    RunResult | ClarificationResult | UnsupportedResult | InvalidInputResult | UpstreamFailureResult
)


def _next_state(resolved: ResolvedTopic, request: AnalysisRequest) -> AnalysisState:
    """Stable across follow-ups (keyed by Wikidata QID); version increments per re-run."""
    previous = request.previous_state
    version = previous.version + 1 if previous is not None else 1
    return AnalysisState(
        analysis_id=f"{_ANALYSIS_ID_PREFIX}-{resolved.wikidata_id.lower()}",
        version=version,
        topic_id=resolved.wikidata_id,
        topic_label=resolved.label,
        languages=request.languages,
        period=request.period,
        analysis=request.analysis,
    )


_SINGLE_LANGUAGE_FINDING_TEXT: dict[str, str] = {
    "sustained_growth": "{language} shows a sustained growing trend with supported reliability.",
    "sustained_decline": "{language} shows a sustained declining trend with supported reliability.",
    "stable_interest": "{language} shows a stable trend with supported reliability.",
    "recent_growth_confirmed": (
        "{language}'s most recent comparable window confirms the longer-term direction."
    ),
    "recent_trend_contradiction": (
        "{language}'s most recent comparable window moved against the longer-term direction."
    ),
    "high_outlier_dependency": (
        "{language}'s trend weakens materially once diagnostic outlier months are excluded."
    ),
    "insufficient_history": (
        "{language} does not yet have enough observed history to classify a trend."
    ),
}


def _single_language_findings(language: str, result: LanguageResult) -> list[Finding]:
    """Deterministic per-language findings, capped at two, reusing Iteration 4/5 evidence only."""
    report = result.analysis
    codes: list[str] = []
    direction, reliability = report.trend.direction, report.trend.reliability
    if reliability is Reliability.INSUFFICIENT or direction is TrendDirection.INSUFFICIENT_DATA:
        codes.append("insufficient_history")
    elif reliability in (Reliability.HIGH, Reliability.MEDIUM):
        if direction is TrendDirection.GROWING:
            codes.append("sustained_growth")
        elif direction is TrendDirection.DECLINING:
            codes.append("sustained_decline")
        elif direction is TrendDirection.STABLE:
            codes.append("stable_interest")
    if report.metrics.recent_confirmation is RecentConfirmation.CONFIRMED:
        codes.append("recent_growth_confirmed")
    elif report.metrics.recent_confirmation is RecentConfirmation.CONTRADICTED:
        codes.append("recent_trend_contradiction")
    if report.metrics.outlier_dependency is OutlierDependency.HIGH:
        codes.append("high_outlier_dependency")
    return [
        Finding(code=code, text=_SINGLE_LANGUAGE_FINDING_TEXT[code].format(language=language))
        for code in codes[:2]
    ]


def _findings(
    languages: tuple[LanguageResult, ...], comparison: ComparisonResult | None
) -> tuple[Finding, ...]:
    """3-6 deterministic findings: the comparison engine's when available, otherwise per-language
    findings derived only from Iteration 4 evidence already attached to each result."""
    if comparison is not None and comparison.status == "success":
        return tuple(Finding(code=f.code, text=f.detail) for f in comparison.findings[:6])
    findings: list[Finding] = []
    for result in languages:
        findings.extend(_single_language_findings(result.language, result))
    return tuple(findings[:6])


def run_analysis(
    request: RunAnalysisRequest,
    *,
    resolver: TopicResolver,
    client: WikimediaClient,
) -> RunAnalysisResult:
    """The single V1 entry point: resolve -> fetch -> analyze -> (optionally) compare ->
    findings -> artifacts -> follow-up state. Stops before any pageview fetch on ambiguity."""
    analysis_request = request.request
    resolution_request = TopicResolutionRequest(
        topic=analysis_request.topic,
        query_language=request.query_language,
        target_languages=analysis_request.languages,
        previous_state=analysis_request.previous_state,
    )
    resolved = resolver.resolve(resolution_request)
    if not isinstance(resolved, ResolvedTopic):
        return resolved

    fetched = _fetch_all(resolved, analysis_request.period, client)
    if isinstance(fetched, InvalidInputResult):
        return fetched
    analyses, series_by_language, unavailable = fetched

    languages: list[LanguageResult] = []
    for language_analysis in analyses:
        article = resolved.articles[language_analysis.language]
        classification, codes = classify_candidate(language_analysis.analysis)
        languages.append(
            LanguageResult(
                language=language_analysis.language,
                article=article,
                analysis=language_analysis.analysis,
                candidate_classification=classification,
                candidate_evidence_codes=codes,
            )
        )
    languages.sort(key=lambda r: r.language)

    comparison: ComparisonResult | None = None
    if analysis_request.analysis is AnalysisIntent.COMPARE:
        comparison = compare_languages(
            ComparisonInput(
                period=analysis_request.period,
                analyses=tuple(analyses),
                unavailable=tuple(unavailable),
            )
        )

    next_state = _next_state(resolved, analysis_request)
    run = RunResult(
        resolved=resolved,
        request=analysis_request,
        requested_languages=analysis_request.languages,
        languages=tuple(languages),
        unavailable=tuple(unavailable),
        comparison=comparison,
        findings=_findings(tuple(languages), comparison),
        next_state=next_state,
    )

    if request.artifacts:
        artifacts: list[ArtifactMetadata] = []
        if ArtifactKind.CHART in request.artifacts:
            artifacts.append(generate_chart(run, series_by_language, request.output_dir))
        if ArtifactKind.REPORT in request.artifacts:
            artifacts.append(generate_report(run, series_by_language, request.output_dir))
        run = run.model_copy(update={"artifacts": tuple(artifacts)})

    return run
