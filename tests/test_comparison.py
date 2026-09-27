from datetime import date

from wikipedia_interest import (
    AnalysisReport,
    CandidateClassification,
    ComparisonInput,
    ComparisonRejected,
    ComparisonRejectionCode,
    ComparisonReport,
    LanguageAnalysis,
    LanguageUnavailability,
    Period,
    Reliability,
    TrendDirection,
    UnavailabilityCategory,
    analyze_pageview_series,
    compare_languages,
)
from wikipedia_interest.contracts import PageviewSeries

TOPIC_A = "Q1"
TOPIC_B = "Q2"
DEFAULT_PERIOD = Period(start="2024-01", end="2025-12")


def _series(
    values: list[int],
    *,
    start: str = "2024-01",
    missing: set[int] | None = None,
    article: str = "Example",
) -> PageviewSeries:
    missing = missing or set()
    start_year, start_month = (int(p) for p in start.split("-"))
    months: list[date] = []
    year, month = start_year, start_month
    for _ in values:
        months.append(date(year, month, 1))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    points = tuple(
        {"month": m.strftime("%Y-%m"), "views": v}
        for index, (m, v) in enumerate(zip(months, values, strict=True))
        if index not in missing
    )
    return PageviewSeries(
        project="en.wikipedia.org",
        article=article,
        period=Period(start=months[0].strftime("%Y-%m"), end=months[-1].strftime("%Y-%m")),
        points=points,
        missing_months=tuple(m.strftime("%Y-%m") for i, m in enumerate(months) if i in missing),
    )


def _report(
    values: list[int], *, start: str = "2024-01", missing: set[int] | None = None
) -> AnalysisReport:
    return analyze_pageview_series(_series(values, start=start, missing=missing))


def _lang(
    language: str,
    values: list[int],
    *,
    topic_id: str = TOPIC_A,
    start: str = "2024-01",
    missing: set[int] | None = None,
) -> LanguageAnalysis:
    return LanguageAnalysis(
        language=language, topic_id=topic_id, analysis=_report(values, start=start, missing=missing)
    )


GROWING_HIGH_REL = [100 + i * 10 for i in range(24)]
# A gently decaying, low-noise decline: Theil-Sen slope stays negative, consistency and
# coverage are full, and the flattening tail keeps recent_confirmation from contradicting.
DECLINING_HIGH_REL = [
    1519,
    1483,
    1452,
    1405,
    1371,
    1330,
    1299,
    1238,
    1204,
    1172,
    1132,
    1099,
    1090,
    1075,
    1048,
    1032,
    1000,
    958,
    904,
    869,
    854,
    857,
    856,
    837,
    846,
    824,
    824,
    799,
    795,
    806,
]
STABLE_HIGH_REL = [100] * 24
STABLE_HIGH_REL_30 = [100] * 30  # same length as DECLINING_HIGH_REL, for period-matched pairing
# Alternating spikes on top of a mild upward drift: growing direction, but high volatility
# forces reliability down to LOW regardless of the (near-perfect) segment consistency.
GROWING_LOW_REL = [100 + i * 5 + (250 if i % 2 == 0 else 0) for i in range(24)]


def _period_for(values: list[int], start: str = "2024-01") -> Period:
    return _report(values, start=start).period


# --- identity / period ------------------------------------------------------------------------


def test_same_topic_accepted() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)


def test_different_topic_ids_rejected() -> None:
    pl = _lang("pl", GROWING_HIGH_REL, topic_id=TOPIC_A)
    cs = _lang("cs", STABLE_HIGH_REL, topic_id=TOPIC_B)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonRejected)
    assert result.code is ComparisonRejectionCode.TOPIC_MISMATCH


def test_same_requested_period_accepted() -> None:
    period = _period_for(GROWING_HIGH_REL)
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    result = compare_languages(ComparisonInput(period=period, analyses=(pl, cs)))
    assert isinstance(result, ComparisonReport)
    assert result.period == period


def test_mismatched_periods_rejected() -> None:
    pl = _lang("pl", GROWING_HIGH_REL, start="2024-01")
    cs = _lang("cs", STABLE_HIGH_REL, start="2022-01")
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL, "2024-01"), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonRejected)
    assert result.code is ComparisonRejectionCode.PERIOD_MISMATCH


def test_duplicate_languages_deduplicated_deterministically() -> None:
    first = _lang("pl", GROWING_HIGH_REL)
    duplicate = _lang("pl", STABLE_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    comparison_input = ComparisonInput(
        period=_period_for(GROWING_HIGH_REL), analyses=(first, duplicate, cs)
    )
    assert [a.language for a in comparison_input.analyses] == ["pl", "cs"]
    assert comparison_input.analyses[0].analysis.trend.direction is TrendDirection.GROWING


# --- minimum comparison ------------------------------------------------------------------------


def test_two_successful_languages_compare() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    assert {s.language for s in result.languages} == {"pl", "cs"}


def test_one_successful_language_is_insufficient() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl,))
    )
    assert isinstance(result, ComparisonRejected)
    assert result.code is ComparisonRejectionCode.INSUFFICIENT_LANGUAGES


def test_unavailable_third_language_does_not_destroy_valid_comparison() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    sk_unavailable = LanguageUnavailability(
        language="sk",
        category=UnavailabilityCategory.UPSTREAM_FAILURE,
        reason="timeout",
        retryable=True,
    )
    result = compare_languages(
        ComparisonInput(
            period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs), unavailable=(sk_unavailable,)
        )
    )
    assert isinstance(result, ComparisonReport)
    assert len(result.languages) == 2
    assert result.unavailable == (sk_unavailable,)


# --- absolute attention ------------------------------------------------------------------------


def test_highest_total_views_identified() -> None:
    pl = _lang("pl", [1000] * 24)
    cs = _lang("cs", [10] * 24)
    result = compare_languages(ComparisonInput(period=_period_for([1000] * 24), analyses=(pl, cs)))
    assert isinstance(result, ComparisonReport)
    finding = next(f for f in result.findings if f.code == "highest_absolute_attention")
    assert finding.languages == ("pl",)
    assert finding.metric == "total_views"


def test_highest_median_attention_identified() -> None:
    pl = _lang("pl", [1000] * 24)
    cs = _lang("cs", [10] * 24)
    result = compare_languages(ComparisonInput(period=_period_for([1000] * 24), analyses=(pl, cs)))
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].median_monthly_views == 1000
    assert by_language["cs"].median_monthly_views == 10


def test_absolute_attention_does_not_affect_trend_classification() -> None:
    pl = _lang("pl", [1000] * 24)  # high attention, stable trend
    cs = _lang("cs", GROWING_HIGH_REL)  # low attention, growing trend
    result = compare_languages(ComparisonInput(period=_period_for([1000] * 24), analyses=(pl, cs)))
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].trend_direction is TrendDirection.STABLE
    assert by_language["cs"].trend_direction is TrendDirection.GROWING


def test_zero_view_language_handled_safely() -> None:
    pl = _lang("pl", [0] * 24)
    cs = _lang("cs", GROWING_HIGH_REL)
    result = compare_languages(ComparisonInput(period=_period_for([0] * 24), analyses=(pl, cs)))
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].total_views == 0
    assert by_language["pl"].comparison_pageview_share == 0.0
    assert "NaN" not in result.model_dump_json()
    assert "Infinity" not in result.model_dump_json()


# --- trend -------------------------------------------------------------------------------------


def test_stronger_normalized_growth_identified() -> None:
    fast = _lang("cs", [100 + i * 50 for i in range(24)])
    slow = _lang("pl", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(fast, slow))
    )
    assert isinstance(result, ComparisonReport)
    finding = next(f for f in result.findings if f.code == "strongest_reliable_growth")
    assert finding.languages == ("cs",)


def test_declining_language_remains_declining() -> None:
    pl = _lang("pl", DECLINING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL_30)
    result = compare_languages(
        ComparisonInput(period=_period_for(DECLINING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].trend_direction is TrendDirection.DECLINING


def test_stable_language_remains_stable() -> None:
    pl = _lang("pl", STABLE_HIGH_REL)
    cs = _lang("cs", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(STABLE_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].trend_direction is TrendDirection.STABLE


def test_unreliable_high_growth_not_described_as_strongly_supported() -> None:
    unreliable = _lang("cs", GROWING_LOW_REL)
    reliable = _lang("pl", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_LOW_REL), analyses=(unreliable, reliable))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    finding = next(f for f in result.findings if f.code == "strongest_reliable_growth")
    assert finding.languages == ("pl",)
    assert (
        by_language["cs"].candidate_classification is not CandidateClassification.STRONG_CANDIDATE
    )


def test_recent_contradiction_preserved() -> None:
    contradicted_values = [100 + i * 10 for i in range(18)] + [50] * 6
    pl = _lang("pl", contradicted_values)
    cs = _lang("cs", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(contradicted_values), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert "recent_trend_contradicts_growth" in by_language["pl"].evidence_codes
    finding = next(f for f in result.findings if f.code == "recent_trend_contradicts_long_term")
    assert "pl" in finding.languages


# --- reliability ---------------------------------------------------------------------------


def test_high_reliability_exposed_separately() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].reliability is Reliability.HIGH


def test_low_reliability_exposed_separately() -> None:
    cs = _lang("cs", GROWING_LOW_REL)
    pl = _lang("pl", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_LOW_REL), analyses=(cs, pl))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["cs"].reliability is Reliability.LOW


def test_larger_growth_low_reliability_not_auto_strong() -> None:
    cs = _lang("cs", GROWING_LOW_REL)
    result = compare_languages(
        ComparisonInput(
            period=_period_for(GROWING_LOW_REL), analyses=(cs, _lang("pl", STABLE_HIGH_REL))
        )
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert (
        by_language["cs"].candidate_classification is not CandidateClassification.STRONG_CANDIDATE
    )


# --- candidate classification -------------------------------------------------------------------


def test_growing_high_reliability_is_strong_candidate() -> None:
    report = _report(GROWING_HIGH_REL)
    assert report.trend.reliability is Reliability.HIGH
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", STABLE_HIGH_REL)))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].candidate_classification is CandidateClassification.STRONG_CANDIDATE


def test_growing_medium_reliability_is_strong_candidate() -> None:
    # 14 observed months keeps reliability below HIGH (needs >=24) but reaches MEDIUM; per
    # policy, growing + medium reliability (with no contradiction/outlier issue) still qualifies.
    values = [100 + i * 10 for i in range(14)]
    report = _report(values)
    assert report.trend.reliability is Reliability.MEDIUM
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", [100] * 14)))
    )
    assert isinstance(result, ComparisonReport)
    pl_summary = next(s for s in result.languages if s.language == "pl")
    assert pl_summary.candidate_classification is CandidateClassification.STRONG_CANDIDATE


def test_growing_low_reliability_is_possible_or_inconclusive_never_strong() -> None:
    report = _report(GROWING_LOW_REL)
    assert report.trend.reliability is Reliability.LOW
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", STABLE_HIGH_REL)))
    )
    assert isinstance(result, ComparisonReport)
    classification = next(
        s for s in result.languages if s.language == "pl"
    ).candidate_classification
    assert classification in (
        CandidateClassification.POSSIBLE_CANDIDATE,
        CandidateClassification.INCONCLUSIVE,
    )


def test_unclear_trend_is_inconclusive() -> None:
    # No closed-form shape produces UNCLEAR; this fixed sequence (24 months, irregular)
    # is verified to land on TrendDirection.UNCLEAR via analyze_pageview_series.
    erratic = [
        302,
        439,
        606,
        154,
        637,
        305,
        63,
        798,
        271,
        467,
        336,
        236,
        988,
        938,
        834,
        448,
        213,
        830,
        866,
        123,
        192,
        682,
        682,
        505,
    ]
    report = _report(erratic)
    assert report.trend.direction is TrendDirection.UNCLEAR
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", STABLE_HIGH_REL)))
    )
    assert isinstance(result, ComparisonReport)
    classification = next(
        s for s in result.languages if s.language == "pl"
    ).candidate_classification
    assert classification is CandidateClassification.INCONCLUSIVE


def test_insufficient_data_is_inconclusive() -> None:
    report = _report([10, 20, 30, 40, 50])
    assert report.trend.direction is TrendDirection.INSUFFICIENT_DATA
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", [10, 20, 30, 40, 50])))
    )
    assert isinstance(result, ComparisonReport)
    classification = next(
        s for s in result.languages if s.language == "pl"
    ).candidate_classification
    assert classification is CandidateClassification.INCONCLUSIVE


def test_reliable_decline_is_weak_current_signal() -> None:
    report = _report(DECLINING_HIGH_REL)
    assert report.trend.reliability is Reliability.HIGH
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", STABLE_HIGH_REL_30)))
    )
    assert isinstance(result, ComparisonReport)
    classification = next(
        s for s in result.languages if s.language == "pl"
    ).candidate_classification
    assert classification is CandidateClassification.WEAK_CURRENT_SIGNAL


def test_recent_decline_downgrades_long_term_positive_signal() -> None:
    values = [100 + i * 10 for i in range(18)] + [50] * 6
    report = _report(values)
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", GROWING_HIGH_REL)))
    )
    assert isinstance(result, ComparisonReport)
    classification = next(
        s for s in result.languages if s.language == "pl"
    ).candidate_classification
    assert classification is not CandidateClassification.STRONG_CANDIDATE


def test_high_outlier_dependency_prevents_strong_candidate() -> None:
    values = [100] * 23 + [100000]
    report = _report(values)
    pl = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=report)
    result = compare_languages(
        ComparisonInput(period=report.period, analyses=(pl, _lang("cs", STABLE_HIGH_REL)))
    )
    assert isinstance(result, ComparisonReport)
    classification = next(
        s for s in result.languages if s.language == "pl"
    ).candidate_classification
    assert classification is not CandidateClassification.STRONG_CANDIDATE


# --- data quality --------------------------------------------------------------------------


def test_missingness_visible_per_language() -> None:
    pl = _lang("pl", [100] * 24, missing={5, 6})
    cs = _lang("cs", STABLE_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(STABLE_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].missing_months == 2
    assert by_language["pl"].observed_months == 22


def test_weaker_coverage_cannot_improve_candidate_classification() -> None:
    sparse = _report([100 + i * 10 for i in range(24)], missing=set(range(6)))
    full = _report(GROWING_HIGH_REL)
    sparse_lang = LanguageAnalysis(language="pl", topic_id=TOPIC_A, analysis=sparse)
    full_lang = LanguageAnalysis(language="cs", topic_id=TOPIC_A, analysis=full)
    result = compare_languages(
        ComparisonInput(period=full.period, analyses=(sparse_lang, full_lang))
    )
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    if by_language["pl"].coverage_ratio < by_language["cs"].coverage_ratio:
        assert (
            by_language["pl"].candidate_classification
            is not CandidateClassification.STRONG_CANDIDATE
            or by_language["cs"].candidate_classification
            is CandidateClassification.STRONG_CANDIDATE
        )


def test_partial_upstream_failure_preserved() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    failure = LanguageUnavailability(
        language="sk",
        category=UnavailabilityCategory.UPSTREAM_FAILURE,
        reason="timeout",
        retryable=True,
    )
    result = compare_languages(
        ComparisonInput(
            period=_period_for(GROWING_HIGH_REL),
            analyses=(pl, _lang("cs", STABLE_HIGH_REL)),
            unavailable=(failure,),
        )
    )
    assert isinstance(result, ComparisonReport)
    assert failure in result.unavailable


# --- ordering / ties ------------------------------------------------------------------------


def test_deterministic_ordering_of_languages() -> None:
    zz = _lang("zz", STABLE_HIGH_REL)
    aa = _lang("aa", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(STABLE_HIGH_REL), analyses=(zz, aa))
    )
    assert isinstance(result, ComparisonReport)
    assert [s.language for s in result.languages] == ["aa", "zz"]


def test_equal_metric_values_use_language_code_tiebreak() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", GROWING_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    finding = next(f for f in result.findings if f.code == "strongest_reliable_growth")
    assert finding.languages == ("cs",)  # "cs" < "pl" lexicographically


def test_rounding_does_not_create_false_differences() -> None:
    values_a = [100 + i * 10 for i in range(24)]
    values_b = [v + 0 for v in values_a]  # identical series under a different language
    pl = _lang("pl", values_a)
    cs = _lang("cs", values_b)
    result = compare_languages(ComparisonInput(period=_period_for(values_a), analyses=(pl, cs)))
    assert isinstance(result, ComparisonReport)
    by_language = {s.language: s for s in result.languages}
    assert by_language["pl"].normalized_trend_slope == by_language["cs"].normalized_trend_slope


# --- findings -------------------------------------------------------------------------------


def test_strongest_reliable_growth_finding_present() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    codes = [f.code for f in result.findings]
    assert "strongest_reliable_growth" in codes


def test_highest_absolute_attention_finding_present() -> None:
    pl = _lang("pl", [5000] * 24)
    cs = _lang("cs", [10] * 24)
    result = compare_languages(ComparisonInput(period=_period_for([5000] * 24), analyses=(pl, cs)))
    assert isinstance(result, ComparisonReport)
    codes = [f.code for f in result.findings]
    assert "highest_absolute_attention" in codes


def test_insufficient_evidence_finding_present() -> None:
    pl = _lang("pl", [10, 20, 30, 40, 50])
    cs = _lang("cs", [10, 20, 30, 40, 50])
    result = compare_languages(
        ComparisonInput(period=_period_for([10, 20, 30, 40, 50]), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    finding = next(f for f in result.findings if f.code == "insufficient_evidence")
    assert set(finding.languages) == {"pl", "cs"}


def test_findings_remain_compact_for_many_languages() -> None:
    langs = tuple(
        _lang(code, GROWING_HIGH_REL)
        for code in ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"]
    )
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=langs)
    )
    assert isinstance(result, ComparisonReport)
    assert len(result.findings) <= 5


def test_no_pairwise_explosion_for_ten_languages() -> None:
    langs = tuple(
        _lang(code, GROWING_HIGH_REL if i % 2 == 0 else STABLE_HIGH_REL)
        for i, code in enumerate(["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"])
    )
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=langs)
    )
    assert isinstance(result, ComparisonReport)
    # O(n^2) pairwise prose for 10 languages would be 90 statements; findings must stay small.
    assert len(result.findings) < 10


# --- semantics ------------------------------------------------------------------------------


def test_no_forbidden_business_fields_anywhere() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    )
    assert isinstance(result, ComparisonReport)
    dumped = result.model_dump_json()
    for forbidden in ("market_size", "opportunity_score", "revenue", "best_market", "market_score"):
        assert forbidden not in dumped
    for summary in result.languages:
        assert "further_validation" not in summary.candidate_classification.value
        assert summary.candidate_classification.value in {
            "strong_candidate",
            "possible_candidate",
            "inconclusive",
            "weak_current_signal",
        }


# --- serialization --------------------------------------------------------------------------


def test_fully_typed_deterministic_serialization() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    comparison_input = ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs))
    first = compare_languages(comparison_input)
    second = compare_languages(comparison_input)
    assert isinstance(first, ComparisonReport)
    assert first.model_dump_json() == second.model_dump_json()
    assert "NaN" not in first.model_dump_json()
    assert "Infinity" not in first.model_dump_json()


def test_requested_unavailable_languages_preserved_in_output() -> None:
    pl = _lang("pl", GROWING_HIGH_REL)
    cs = _lang("cs", STABLE_HIGH_REL)
    sk = LanguageUnavailability(
        language="sk",
        category=UnavailabilityCategory.DOMAIN_UNAVAILABLE,
        reason="no_sitelink",
        retryable=False,
    )
    result = compare_languages(
        ComparisonInput(period=_period_for(GROWING_HIGH_REL), analyses=(pl, cs), unavailable=(sk,))
    )
    assert isinstance(result, ComparisonReport)
    assert sk in result.unavailable
