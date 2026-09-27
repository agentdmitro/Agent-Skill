from datetime import date

import pytest

from wikipedia_interest import (
    OutlierDependency,
    PageviewSeries,
    Period,
    RecentConfirmation,
    Reliability,
    TrendDirection,
    analyze_pageview_series,
)


def series(values: list[int], *, missing: set[int] | None = None) -> PageviewSeries:
    missing = missing or set()
    months: list[date] = []
    year, month = 2024, 1
    for _ in values:
        months.append(date(year, month, 1))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    points = tuple(
        {"month": month.strftime("%Y-%m"), "views": value}
        for index, (month, value) in enumerate(zip(months, values, strict=True))
        if index not in missing
    )
    return PageviewSeries(
        project="en.wikipedia.org",
        article="Example",
        period=Period(start=months[0].strftime("%Y-%m"), end=months[-1].strftime("%Y-%m")),
        points=points,
        missing_months=tuple(
            month.strftime("%Y-%m") for index, month in enumerate(months) if index in missing
        ),
    )


def test_insufficient_data_and_missing_months_are_explicit() -> None:
    result = analyze_pageview_series(series([10, 20, 30, 40, 50]))
    assert result.trend.direction is TrendDirection.INSUFFICIENT_DATA
    assert result.trend.reliability is Reliability.INSUFFICIENT

    result = analyze_pageview_series(series([100] * 12, missing={2, 7}))
    assert result.data_quality.missing_months == 2
    assert result.data_quality.observed_months == 10
    assert result.metrics.total_views == 1000
    assert result.trend.direction is TrendDirection.STABLE


@pytest.mark.parametrize(
    ("values", "direction"),
    [
        ([100 + i * 10 for i in range(24)], TrendDirection.GROWING),
        ([300 - i * 10 for i in range(24)], TrendDirection.DECLINING),
        ([100] * 24, TrendDirection.STABLE),
    ],
)
def test_clear_directions_and_full_metrics(values: list[int], direction: TrendDirection) -> None:
    result = analyze_pageview_series(series(values))
    assert result.trend.direction is direction
    assert result.metrics.year_over_year_change is not None
    assert result.metrics.trend_consistency == 1.0
    assert result.metrics.volatility_band.value == "low"


def test_boundary_medians_do_not_use_single_anomalous_month() -> None:
    result = analyze_pageview_series(series([10, 100, 10, 10, 10, 10, 10, 10, 10]))
    assert result.metrics.start_level == 10
    assert result.metrics.end_level == 10
    assert result.metrics.period_change == 0


def test_outlier_and_dependency_are_diagnostic_only() -> None:
    raw = series([100] * 23 + [10000])
    result = analyze_pageview_series(raw)
    assert result.metrics.outlier_months[-1].views == 10000
    assert result.metrics.outlier_dependency is OutlierDependency.LOW
    assert raw.points[-1].views == 10000
    assert result.trend.reliability is not Reliability.HIGH


def test_yoy_and_recent_confirmation_are_calendar_aligned() -> None:
    values = [100] * 12 + [200] * 6 + [300] * 6
    result = analyze_pageview_series(series(values))
    assert result.metrics.year_over_year_change == pytest.approx(1.5)
    assert result.metrics.recent_confirmation is RecentConfirmation.CONFIRMED
    assert result.metrics.recent_change == pytest.approx(0.5)


def test_zero_values_never_emit_invalid_numbers() -> None:
    zero = analyze_pageview_series(series([0] * 12))
    assert zero.trend.direction is TrendDirection.STABLE
    assert zero.metrics.period_change is None
    assert zero.metrics.normalized_trend_slope is None
    assert "NaN" not in zero.model_dump_json()
    assert "Infinity" not in zero.model_dump_json()

    from_zero = analyze_pageview_series(series([0, 0, 0, 10, 20, 30]))
    assert from_zero.trend.direction is TrendDirection.GROWING
    assert from_zero.metrics.period_change is None


def test_json_is_deterministic_and_precision_is_bounded() -> None:
    values = [100 + i * 7 for i in range(12)]
    result = analyze_pageview_series(series(values))
    assert result.model_dump_json() == analyze_pageview_series(series(values)).model_dump_json()
    assert len(str(result.metrics.normalized_trend_slope).split(".")[-1]) <= 4
