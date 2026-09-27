"""Deterministic descriptive analysis of one verified monthly pageview series.

This module accepts only a normalized :class:`PageviewSeries`.  It does not fetch
data, fill missing months, attribute causes, forecast, or make business claims.
All thresholds are policy choices for this evidence grade, not statistical laws.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from enum import StrEnum
from itertools import combinations, pairwise
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wikipedia_interest.contracts import (
    PageviewPoint,
    PageviewSeries,
    Period,
    ResolvedArticle,
    month_range,
)

# Policy thresholds are centralized so the deterministic methodology is inspectable.
MIN_OBSERVED_MONTHS = 6
MIN_COVERAGE_FOR_TREND = 0.75
LOW_HISTORY_MONTHS = 12
STRONG_HISTORY_MONTHS = 24
BOUNDARY_MONTHS = 3
COMPARISON_WINDOW_MONTHS = 12
RECENT_WINDOW_MONTHS = 6
MIN_COMPARISON_COVERAGE = 10
MIN_RECENT_COVERAGE = 5
STABLE_SLOPE_THRESHOLD = 0.01  # normalized robust views/month
MEANINGFUL_LOCAL_CHANGE = 0.05
OUTLIER_ROBUST_Z_THRESHOLD = 3.5
RECENT_CONFIRMATION_THRESHOLD = 0.05
LOW_VOLATILITY_THRESHOLD = 0.10
HIGH_VOLATILITY_THRESHOLD = 0.30


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TrendDirection(StrEnum):
    GROWING = "growing"
    DECLINING = "declining"
    STABLE = "stable"
    UNCLEAR = "unclear"
    INSUFFICIENT_DATA = "insufficient_data"


class Reliability(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class VolatilityBand(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNAVAILABLE = "unavailable"


class OutlierDirection(StrEnum):
    HIGH = "high"
    LOW = "low"


class OutlierDependency(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNAVAILABLE = "unavailable"


class RecentConfirmation(StrEnum):
    CONFIRMED = "confirmed"
    FLATTENED = "flattened"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    UNAVAILABLE = "unavailable"


class EvidenceSeverity(StrEnum):
    SUPPORTING = "supporting"
    LIMITING = "limiting"
    NEUTRAL = "neutral"


class AnalysisEvidence(_Model):
    code: str = Field(min_length=1)
    severity: EvidenceSeverity
    detail: str = Field(min_length=1)


class DataQuality(_Model):
    observed_months: int = Field(ge=0)
    requested_months: int = Field(ge=1)
    missing_months: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)


class OutlierMonth(_Model):
    month: date
    views: int = Field(ge=0)
    direction: OutlierDirection
    robust_deviation: float | None = None


class AnalysisMetrics(_Model):
    observed_months: int = Field(ge=0)
    requested_months: int = Field(ge=1)
    missing_months: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)
    total_views: int = Field(ge=0)
    mean_monthly_views: float | None = Field(default=None, ge=0)
    median_monthly_views: float | None = Field(default=None, ge=0)
    start_level: float | None = Field(default=None, ge=0)
    end_level: float | None = Field(default=None, ge=0)
    period_change: float | None = None
    year_over_year_change: float | None = None
    robust_trend_slope: float | None = None
    normalized_trend_slope: float | None = None
    trend_consistency: float | None = Field(default=None, ge=0, le=1)
    volatility: float | None = Field(default=None, ge=0)
    volatility_band: VolatilityBand
    outlier_months: tuple[OutlierMonth, ...] = ()
    outlier_dependency: OutlierDependency
    recent_change: float | None = None
    recent_confirmation: RecentConfirmation


class TrendAssessment(_Model):
    direction: TrendDirection
    reliability: Reliability


class AnalysisReport(_Model):
    """Typed result for one verified article's descriptive pageview analysis."""

    status: Literal["success"] = "success"
    article: ResolvedArticle
    period: Period
    data_quality: DataQuality
    metrics: AnalysisMetrics
    trend: TrendAssessment
    evidence: tuple[AnalysisEvidence, ...] = ()


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _month_count(series: PageviewSeries) -> int:
    return len(month_range(series.period))


def _median(values: Sequence[int]) -> float:
    return float(median(values))


def _safe_change(new: float, old: float) -> float | None:
    return None if old == 0 else (new - old) / old


def _window(points: Sequence[PageviewPoint], months: Iterable[date]) -> list[int]:
    wanted = set(months)
    return [p.views for p in points if p.month in wanted]


def _split(values: Sequence[int], count: int) -> list[list[int]]:
    return [
        list(values[i * len(values) // count : (i + 1) * len(values) // count])
        for i in range(count)
    ]


def _theil_sen(points: Sequence[PageviewPoint]) -> float | None:
    slopes = [
        (right.views - left.views)
        / (right.month.year * 12 + right.month.month - left.month.year * 12 - left.month.month)
        for left, right in combinations(points, 2)
    ]
    return None if not slopes else float(median(slopes))


def _outliers(points: Sequence[PageviewPoint]) -> tuple[OutlierMonth, ...]:
    values = [p.views for p in points]
    centre = median(values)
    deviations = [abs(value - centre) for value in values]
    mad = median(deviations)
    result: list[OutlierMonth] = []
    for point in points:
        robust_z = None if mad == 0 else 0.6745 * (point.views - centre) / mad
        is_outlier = (
            abs(point.views - centre) > 0
            if mad == 0
            else abs(robust_z or 0) >= OUTLIER_ROBUST_Z_THRESHOLD
        )
        if is_outlier:
            result.append(
                OutlierMonth(
                    month=point.month,
                    views=point.views,
                    direction=OutlierDirection.HIGH
                    if point.views > centre
                    else OutlierDirection.LOW,
                    robust_deviation=_round(robust_z),
                )
            )
    return tuple(result)


def _comparison_change(
    points: Sequence[PageviewPoint], end: date, months: int, minimum: int
) -> float | None:
    latest_months = _calendar_window(end, months)
    first_latest = latest_months[0]
    previous_end = (
        date(first_latest.year - 1, 12, 1)
        if first_latest.month == 1
        else date(first_latest.year, first_latest.month - 1, 1)
    )
    previous_months = _calendar_window(previous_end, months)
    latest = _window(points, latest_months)
    previous = _window(points, previous_months)
    if len(latest) < minimum or len(previous) < minimum:
        return None
    return _safe_change(float(sum(latest)), float(sum(previous)))


def _calendar_window(end: date, length: int) -> list[date]:
    y, m = end.year, end.month
    out: list[date] = []
    for _ in range(length):
        out.append(date(y, m, 1))
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return list(reversed(out))


def _recent(
    points: Sequence[PageviewPoint], end: date, direction: TrendDirection
) -> tuple[float | None, RecentConfirmation]:
    """Whether the most recent comparable window moved with or against `direction`.

    For a declining trend, a further negative recent change *confirms* it (and a positive one
    contradicts it) -- the opposite sign mapping from a growing trend. Stable/unclear/other
    directions keep the plain sign-of-change mapping, since there is no long-term direction to
    confirm or contradict against.
    """
    latest_months = _calendar_window(end, RECENT_WINDOW_MONTHS)
    previous_end = latest_months[0]
    y, m = previous_end.year, previous_end.month - 1
    if m == 0:
        y, m = y - 1, 12
    previous_months = _calendar_window(date(y, m, 1), RECENT_WINDOW_MONTHS)
    latest, previous = _window(points, latest_months), _window(points, previous_months)
    if len(latest) < MIN_RECENT_COVERAGE or len(previous) < MIN_RECENT_COVERAGE:
        return None, RecentConfirmation.UNAVAILABLE
    change = _safe_change(_median(latest), _median(previous))
    if change is None:
        return None, RecentConfirmation.MIXED
    if direction is TrendDirection.DECLINING:
        if change <= -RECENT_CONFIRMATION_THRESHOLD:
            return change, RecentConfirmation.CONFIRMED
        if change >= RECENT_CONFIRMATION_THRESHOLD:
            return change, RecentConfirmation.CONTRADICTED
        return change, RecentConfirmation.FLATTENED
    if change >= RECENT_CONFIRMATION_THRESHOLD:
        return change, RecentConfirmation.CONFIRMED
    if change <= -RECENT_CONFIRMATION_THRESHOLD:
        return change, RecentConfirmation.CONTRADICTED
    return change, RecentConfirmation.FLATTENED


def _consistency(
    points: Sequence[PageviewPoint], direction: TrendDirection, baseline: float
) -> float:
    segments = _split([p.views for p in points], 4 if len(points) >= 12 else 3)
    levels = [_median(segment) for segment in segments if segment]
    if len(levels) < 2:
        return 0.0
    tolerance = max(baseline, 1.0) * MEANINGFUL_LOCAL_CHANGE
    moves = [right - left for left, right in pairwise(levels)]
    if direction is TrendDirection.GROWING:
        return sum(move > tolerance for move in moves) / len(moves)
    if direction is TrendDirection.DECLINING:
        return sum(move < -tolerance for move in moves) / len(moves)
    if direction is TrendDirection.STABLE:
        return sum(abs(move) <= tolerance for move in moves) / len(moves)
    return max(
        sum(move > tolerance for move in moves), sum(move < -tolerance for move in moves)
    ) / len(moves)


def _volatility(points: Sequence[PageviewPoint]) -> tuple[float | None, VolatilityBand]:
    changes: list[float] = []
    for previous, current in pairwise(points):
        if previous.views == 0:
            if current.views == 0:
                changes.append(0.0)
            continue
        changes.append(abs(current.views - previous.views) / previous.views)
    if not changes:
        return None, VolatilityBand.UNAVAILABLE
    value = float(median(changes))
    band = (
        VolatilityBand.LOW
        if value <= LOW_VOLATILITY_THRESHOLD
        else (
            VolatilityBand.MODERATE if value <= HIGH_VOLATILITY_THRESHOLD else VolatilityBand.HIGH
        )
    )
    return value, band


def _dependency(
    points: Sequence[PageviewPoint], outliers: Sequence[OutlierMonth], slope: float | None
) -> OutlierDependency:
    if slope is None or not outliers or len(points) - len(outliers) < 3:
        return OutlierDependency.UNAVAILABLE if slope is None else OutlierDependency.LOW
    excluded = {item.month for item in outliers}
    filtered = [point for point in points if point.month not in excluded]
    filtered_slope = _theil_sen(filtered)
    if filtered_slope is None:
        return OutlierDependency.UNAVAILABLE
    if (slope > 0 > filtered_slope) or (slope < 0 < filtered_slope):
        return OutlierDependency.HIGH
    if abs(slope) > 0 and abs(filtered_slope) < abs(slope) * 0.5:
        return OutlierDependency.HIGH
    if abs(slope) > 0 and abs(filtered_slope) < abs(slope) * 0.75:
        return OutlierDependency.MODERATE
    return OutlierDependency.LOW


def analyze_pageview_series(series: PageviewSeries) -> AnalysisReport:
    """Analyze one normalized series without mutating it or fabricating observations."""
    points = series.points
    requested = _month_count(series)
    observed = len(points)
    missing = len(series.missing_months)
    coverage = observed / requested
    quality = DataQuality(
        observed_months=observed,
        requested_months=requested,
        missing_months=missing,
        coverage_ratio=_round(coverage),
    )
    total = sum(point.views for point in points)
    mean = total / observed if observed else None
    med = _median([point.views for point in points]) if points else None
    start_values = [point.views for point in points[:BOUNDARY_MONTHS]]
    end_values = [point.views for point in points[-BOUNDARY_MONTHS:]]
    start = _median(start_values) if start_values else None
    end = _median(end_values) if end_values else None
    period_change = _safe_change(end, start) if start is not None and end is not None else None
    slope = _theil_sen(points)
    normalized = None if slope is None or med is None or med == 0 else slope / med
    preliminary = TrendDirection.INSUFFICIENT_DATA
    if observed >= MIN_OBSERVED_MONTHS and coverage >= MIN_COVERAGE_FOR_TREND:
        if med == 0 and slope is not None:
            preliminary = (
                TrendDirection.GROWING
                if slope > 0
                else TrendDirection.DECLINING
                if slope < 0
                else TrendDirection.STABLE
            )
        elif normalized is not None and abs(normalized) <= STABLE_SLOPE_THRESHOLD:
            preliminary = TrendDirection.STABLE
        elif normalized is not None and normalized > 0:
            preliminary = TrendDirection.GROWING
        elif normalized is not None:
            preliminary = TrendDirection.DECLINING
    consistency = (
        None
        if preliminary is TrendDirection.INSUFFICIENT_DATA
        else _consistency(points, preliminary, med or 0)
    )
    if (
        preliminary in (TrendDirection.GROWING, TrendDirection.DECLINING)
        and consistency is not None
        and consistency < 0.5
    ):
        direction = TrendDirection.UNCLEAR
    else:
        direction = preliminary
    yoy = (
        _comparison_change(
            points, series.period.end, COMPARISON_WINDOW_MONTHS, MIN_COMPARISON_COVERAGE
        )
        if requested >= 24
        else None
    )
    vol, vol_band = _volatility(points)
    outlier_list = _outliers(points)
    dependency = _dependency(points, outlier_list, slope)
    recent_change, recent_confirmation = (
        _recent(points, series.period.end, direction)
        if requested >= 12
        else (None, RecentConfirmation.UNAVAILABLE)
    )
    reliability = _reliability(
        observed,
        coverage,
        direction,
        consistency,
        dependency,
        vol_band,
        recent_confirmation,
        bool(outlier_list),
    )
    metrics = AnalysisMetrics(
        observed_months=observed,
        requested_months=requested,
        missing_months=missing,
        coverage_ratio=_round(coverage),
        total_views=total,
        mean_monthly_views=_round(mean),
        median_monthly_views=_round(med),
        start_level=_round(start),
        end_level=_round(end),
        period_change=_round(period_change),
        year_over_year_change=_round(yoy),
        robust_trend_slope=_round(slope),
        normalized_trend_slope=_round(normalized),
        trend_consistency=_round(consistency),
        volatility=_round(vol),
        volatility_band=vol_band,
        outlier_months=outlier_list,
        outlier_dependency=dependency,
        recent_change=_round(recent_change),
        recent_confirmation=recent_confirmation,
    )
    evidence = _evidence(metrics, direction, reliability)
    return AnalysisReport(
        article=ResolvedArticle(title=series.article, project=series.project),
        period=series.period,
        data_quality=quality,
        metrics=metrics,
        trend=TrendAssessment(direction=direction, reliability=reliability),
        evidence=evidence,
    )


def _reliability(
    observed: int,
    coverage: float,
    direction: TrendDirection,
    consistency: float | None,
    dependency: OutlierDependency,
    volatility: VolatilityBand,
    recent: RecentConfirmation,
    has_outliers: bool,
) -> Reliability:
    if direction is TrendDirection.INSUFFICIENT_DATA:
        return Reliability.INSUFFICIENT
    if observed < LOW_HISTORY_MONTHS or coverage < MIN_COVERAGE_FOR_TREND:
        return Reliability.LOW
    contradiction = recent is RecentConfirmation.CONTRADICTED
    if (
        observed >= STRONG_HISTORY_MONTHS
        and coverage >= 0.9
        and consistency is not None
        and consistency >= 0.75
        and dependency is OutlierDependency.LOW
        and not has_outliers
        and volatility is not VolatilityBand.HIGH
        and not contradiction
        and recent in (RecentConfirmation.CONFIRMED, RecentConfirmation.FLATTENED)
    ):
        return Reliability.HIGH
    if (
        consistency is not None
        and consistency >= 0.5
        and dependency is not OutlierDependency.HIGH
        and volatility is not VolatilityBand.HIGH
        and not contradiction
    ):
        return Reliability.MEDIUM
    return Reliability.LOW


def _evidence(
    metrics: AnalysisMetrics, direction: TrendDirection, reliability: Reliability
) -> tuple[AnalysisEvidence, ...]:
    result: list[AnalysisEvidence] = []
    if direction is TrendDirection.GROWING:
        result.append(
            AnalysisEvidence(
                code="positive_robust_slope",
                severity=EvidenceSeverity.SUPPORTING,
                detail="Theil-Sen slope is positive.",
            )
        )
    elif direction is TrendDirection.DECLINING:
        result.append(
            AnalysisEvidence(
                code="negative_robust_slope",
                severity=EvidenceSeverity.SUPPORTING,
                detail="Theil-Sen slope is negative.",
            )
        )
    elif direction is TrendDirection.STABLE:
        result.append(
            AnalysisEvidence(
                code="slope_near_zero",
                severity=EvidenceSeverity.SUPPORTING,
                detail="Normalized robust slope is within the stable threshold.",
            )
        )
    elif direction is TrendDirection.UNCLEAR:
        result.append(
            AnalysisEvidence(
                code="inconsistent_direction",
                severity=EvidenceSeverity.LIMITING,
                detail="Local segment movement does not consistently support the robust slope.",
            )
        )
    if metrics.recent_confirmation is RecentConfirmation.CONFIRMED:
        result.append(
            AnalysisEvidence(
                code="recent_change_confirmed",
                severity=EvidenceSeverity.SUPPORTING,
                detail="The latest comparable six-month window moved in the same direction.",
            )
        )
    elif metrics.recent_confirmation is RecentConfirmation.CONTRADICTED:
        result.append(
            AnalysisEvidence(
                code="recent_change_contradicts",
                severity=EvidenceSeverity.LIMITING,
                detail=(
                    "The latest comparable six-month window moved against the long-term direction."
                ),
            )
        )
    if metrics.missing_months:
        result.append(
            AnalysisEvidence(
                code="missing_months",
                severity=EvidenceSeverity.LIMITING,
                detail=f"{metrics.missing_months} requested months have no reported observations.",
            )
        )
    if metrics.volatility_band is VolatilityBand.HIGH:
        result.append(
            AnalysisEvidence(
                code="high_volatility",
                severity=EvidenceSeverity.LIMITING,
                detail="Robust month-to-month movement is high.",
            )
        )
    if metrics.outlier_dependency is OutlierDependency.HIGH:
        result.append(
            AnalysisEvidence(
                code="outlier_dependent",
                severity=EvidenceSeverity.LIMITING,
                detail="The trend weakens materially after diagnostic outlier removal.",
            )
        )
    if reliability is Reliability.INSUFFICIENT:
        result.append(
            AnalysisEvidence(
                code="insufficient_data",
                severity=EvidenceSeverity.LIMITING,
                detail="The available observed coverage cannot support trend classification.",
            )
        )
    return tuple(result)
