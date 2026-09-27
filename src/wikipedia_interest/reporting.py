"""Deterministic chart and one-page PDF artifact generation.

Presentation only: nothing here computes a statistic. All numbers are read from
Iteration 4/5 outputs (`AnalysisReport`, `ComparisonReport`) and the raw
`PageviewSeries` kept only for plotting. Report copy is template text filled with
those numbers, never LLM-written and never a business claim.

Artifact generation failure (bad font, unwritable path, malformed label) must never
be reported as an analysis failure; callers catch broadly and return `ArtifactMetadata`
with `success=False` instead of raising.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless: never touch a display server

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from wikipedia_interest.analysis import Reliability, TrendDirection
from wikipedia_interest.comparison import CandidateClassification
from wikipedia_interest.contracts import ArtifactKind, ArtifactMetadata, PageviewSeries

if TYPE_CHECKING:
    from wikipedia_interest.orchestration import RunResult

_AnyFigure = Figure | SubFigure

CHART_MIME = "image/png"
REPORT_MIME = "application/pdf"
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FIGSIZE_A4 = (8.27, 11.69)

_RELIABILITY_TEXT = {
    Reliability.HIGH: "high evidence reliability",
    Reliability.MEDIUM: "medium evidence reliability",
    Reliability.LOW: "low evidence reliability",
    Reliability.INSUFFICIENT: "insufficient evidence to assess reliability",
}
_TREND_TEXT = {
    TrendDirection.GROWING: "growing",
    TrendDirection.DECLINING: "declining",
    TrendDirection.STABLE: "stable",
    TrendDirection.UNCLEAR: "unclear",
    TrendDirection.INSUFFICIENT_DATA: "not yet classifiable (insufficient data)",
}
_CANDIDATE_TEXT = {
    CandidateClassification.STRONG_CANDIDATE: "a strong candidate for further validation",
    CandidateClassification.POSSIBLE_CANDIDATE: "a possible candidate for further validation",
    CandidateClassification.WEAK_CURRENT_SIGNAL: "a weak current Wikipedia interest signal",
    CandidateClassification.INCONCLUSIVE: "inconclusive based on Wikipedia interest alone",
}
_LIMITATIONS = (
    "Wikipedia pageviews measure attention, not purchase intent or demand.",
    "Language editions differ in audience size and usage habits; raw pageviews "
    "across editions are not a market-size comparison.",
    "Correlation in the data does not establish a cause for any change.",
)


class ArtifactError(Exception):
    """Raised internally to unify chart/report failure handling; never escapes this module."""


def sanitize_slug(text: str, max_length: int = 40) -> str:
    """Lowercase ASCII slug safe for use as a filename component. Never empty."""
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return (slug or "artifact")[:max_length].strip("-") or "artifact"


def safe_artifact_path(output_dir: str, filename: str) -> Path:
    """Resolve `filename` under `output_dir`, rejecting any attempt to escape it."""
    base = Path(output_dir).resolve()
    candidate = (base / filename).resolve()
    if candidate.parent != base:
        raise ArtifactError(f"unsafe artifact filename: {filename!r}")
    base.mkdir(parents=True, exist_ok=True)
    return candidate


def _analysis_slug(run: RunResult) -> str:
    slug = sanitize_slug(run.resolved.label)
    return f"{run.next_state.analysis_id}-v{run.next_state.version}-{slug}"


def _chart_filename(run: RunResult) -> str:
    return f"{_analysis_slug(run)}-chart.png"


def _report_filename(run: RunResult) -> str:
    return f"{_analysis_slug(run)}-report.pdf"


def _normalized_index(series: PageviewSeries) -> list[tuple[str, float]] | None:
    """Robust-baseline index (first 3 observed months' median = 100). None if undefined."""
    points = sorted(series.points, key=lambda p: p.month)
    if len(points) < 3:
        baseline = points[0].views if points else 0
    else:
        values = sorted(p.views for p in points[:3])
        baseline = values[1]
    if baseline == 0:
        return None
    return [(p.month.strftime("%Y-%m"), 100.0 * p.views / baseline) for p in points]


def _plot_series(ax: Axes, language: str, series: PageviewSeries) -> None:
    """Plot observed months only; a gap in the line marks a missing month (never zero)."""
    points = sorted(series.points, key=lambda p: p.month)
    months = [p.month.strftime("%Y-%m") for p in points]
    views = [p.views for p in points]
    ax.plot(months, views, marker="o", markersize=3, linewidth=1.5, label=language)


def _draw_single_language_chart(fig: _AnyFigure, language: str, series: PageviewSeries) -> None:
    ax = fig.add_subplot(111)
    _plot_series(ax, language, series)
    ax.set_title(f"{series.article} — {language} Wikipedia monthly pageviews")
    ax.set_xlabel("Month")
    ax.set_ylabel("Pageviews (all-access, user)")
    if series.missing_months:
        ax.text(
            0.01,
            0.99,
            f"{len(series.missing_months)} month(s) have no reported data (not shown as zero).",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="0.4",
        )
    _thin_xticks(ax)


def _thin_xticks(ax: Axes, max_labels: int = 12) -> None:
    labels = ax.get_xticklabels()
    if len(labels) > max_labels:
        step = max(1, len(labels) // max_labels)
        for i, label in enumerate(labels):
            if i % step != 0:
                label.set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)


def _draw_multi_language_chart(
    fig: _AnyFigure, series_by_language: dict[str, PageviewSeries], primary_normalized: bool
) -> None:
    axes = fig.subplots(2, 1)
    absolute_ax, normalized_ax = (axes[1], axes[0]) if primary_normalized else (axes[0], axes[1])

    for language, series in sorted(series_by_language.items()):
        _plot_series(absolute_ax, language, series)
    absolute_ax.set_title("Absolute monthly pageviews by language edition")
    absolute_ax.set_ylabel("Pageviews")
    absolute_ax.legend(fontsize=8)
    _thin_xticks(absolute_ax)

    excluded: list[str] = []
    for language, series in sorted(series_by_language.items()):
        index = _normalized_index(series)
        if index is None:
            excluded.append(language)
            continue
        normalized_ax.plot(
            [m for m, _ in index], [v for _, v in index], marker="o", markersize=3, label=language
        )
    normalized_ax.axhline(100, color="0.6", linewidth=0.8, linestyle="--")
    title = "Normalized trend index (first observed months = 100) — presentation only"
    if excluded:
        title += f"\nExcluded (undefined baseline): {', '.join(sorted(excluded))}"
    normalized_ax.set_title(title, fontsize=9)
    normalized_ax.set_ylabel("Index")
    normalized_ax.legend(fontsize=8)
    _thin_xticks(normalized_ax)
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()


def generate_chart(
    run: RunResult, series_by_language: dict[str, PageviewSeries], output_dir: str
) -> ArtifactMetadata:
    """One PNG: a single trend line, or a comparable multi-language panel."""
    try:
        if not series_by_language:
            raise ArtifactError("no successfully analyzed language has pageview data to chart")
        path = safe_artifact_path(output_dir, _chart_filename(run))
        fig = plt.figure(figsize=(9, 5) if len(series_by_language) == 1 else (9, 8))
        try:
            if len(series_by_language) == 1:
                ((language, series),) = series_by_language.items()
                _draw_single_language_chart(fig, language, series)
            else:
                primary_normalized = run.request.analysis.value == "compare"
                _draw_multi_language_chart(fig, series_by_language, primary_normalized)
            fig.savefig(path, dpi=150)
        finally:
            plt.close(fig)
        return ArtifactMetadata(
            kind=ArtifactKind.CHART, path=str(path), mime_type=CHART_MIME, success=True
        )
    except Exception as exc:
        return ArtifactMetadata(
            kind=ArtifactKind.CHART, success=False, failure_reason=str(exc)[:200]
        )


def _report_lead_sentence(run: RunResult) -> str:
    languages = run.languages
    if not languages:
        return "No language edition produced usable evidence for this period."
    if len(languages) == 1:
        lang = languages[0]
        trend = _TREND_TEXT[lang.analysis.trend.direction]
        reliability = _RELIABILITY_TEXT[lang.analysis.trend.reliability]
        return f"Interest in {lang.language} Wikipedia is {trend}, with {reliability}."
    grower = next(
        (
            lang
            for lang in languages
            if lang.analysis.trend.direction is TrendDirection.GROWING
            and lang.analysis.trend.reliability in (Reliability.HIGH, Reliability.MEDIUM)
        ),
        None,
    )
    if grower is not None:
        return (
            f"{grower.language} Wikipedia shows the strongest reliable growth among the "
            f"{len(languages)} analyzed editions."
        )
    return f"{len(languages)} Wikipedia editions were analyzed; no edition shows reliable growth."


def _report_candidate_sentence(run: RunResult) -> str:
    ranked = sorted(
        run.languages,
        key=lambda lang: (
            0
            if lang.candidate_classification is CandidateClassification.STRONG_CANDIDATE
            else 1
            if lang.candidate_classification is CandidateClassification.POSSIBLE_CANDIDATE
            else 2,
            lang.language,
        ),
    )
    if not ranked:
        return "No language edition has enough evidence to name a validation candidate."
    top = ranked[0]
    verdict = _CANDIDATE_TEXT[top.candidate_classification]
    return f"Based on Wikipedia attention alone, {top.language} is {verdict}."


def generate_report(
    run: RunResult, series_by_language: dict[str, PageviewSeries], output_dir: str
) -> ArtifactMetadata:
    """One-page deterministic PDF research brief. A single matplotlib Figure is one PDF page."""
    fig = None
    try:
        if not run.languages:
            raise ArtifactError("no successfully analyzed language data")
        path = safe_artifact_path(output_dir, _report_filename(run))
        fig = plt.figure(figsize=_FIGSIZE_A4)
        grid = fig.add_gridspec(nrows=4, height_ratios=[1.1, 2.6, 1.6, 0.9], hspace=0.55)

        head_ax = fig.add_subplot(grid[0])
        head_ax.axis("off")
        editions = ", ".join(sorted(lang.language for lang in run.languages)) or "none analyzed"
        head_ax.text(0, 1.0, "Wikipedia Interest Brief", fontsize=18, weight="bold", va="top")
        head_ax.text(
            0,
            0.55,
            f"Topic: {run.resolved.label}\n"
            f"Period: {run.request.period.start.strftime('%Y-%m')} to "
            f"{run.request.period.end.strftime('%Y-%m')}\n"
            f"Wikipedia editions analyzed: {editions}",
            fontsize=10,
            va="top",
        )
        key_finding = "\n".join(f"• {finding.text}" for finding in run.findings[:3]) or (
            f"• {_report_lead_sentence(run)}"
        )
        head_ax.text(0, -0.15, "KEY FINDING", fontsize=9, weight="bold", va="top")
        head_ax.text(0, -0.35, key_finding, fontsize=8.5, va="top", wrap=True)

        chart_ax_slot = grid[1]
        if series_by_language:
            inner = fig.add_subfigure(chart_ax_slot)
            if len(series_by_language) == 1:
                ((language, series),) = series_by_language.items()
                _draw_single_language_chart(inner, language, series)
            else:
                _draw_multi_language_chart(
                    inner, series_by_language, run.request.analysis.value == "compare"
                )
        else:
            empty_ax = fig.add_subplot(chart_ax_slot)
            empty_ax.axis("off")
            empty_ax.text(0.5, 0.5, "No chartable data.", ha="center", va="center")

        table_ax = fig.add_subplot(grid[2])
        table_ax.axis("off")
        table_ax.text(0, 1.0, "LANGUAGE SUMMARY", fontsize=9, weight="bold", va="top")
        rows = [
            (
                lang.language,
                f"{lang.analysis.metrics.median_monthly_views:,.0f}"
                if lang.analysis.metrics.median_monthly_views is not None
                else "n/a",
                _TREND_TEXT[lang.analysis.trend.direction],
                lang.analysis.trend.reliability.value,
                lang.candidate_classification.value,
            )
            for lang in run.languages
        ]
        if rows:
            table = table_ax.table(
                cellText=rows,
                colLabels=["Lang", "Median monthly views", "Trend", "Reliability", "Candidate"],
                loc="upper left",
                bbox=(0.0, 0.0, 1.0, 0.85),  # type: ignore[arg-type]  # tuple accepted at runtime
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
        else:
            table_ax.text(0, 0.6, "No language produced a usable analysis.", fontsize=8.5)

        bottom_ax = fig.add_subplot(grid[3])
        bottom_ax.axis("off")
        bottom_ax.text(0, 1.0, "NEXT VALIDATION", fontsize=9, weight="bold", va="top")
        bottom_ax.text(0, 0.75, _report_candidate_sentence(run), fontsize=8.5, va="top", wrap=True)
        bottom_ax.text(0, 0.4, "LIMITATIONS", fontsize=9, weight="bold", va="top")
        bottom_ax.text(
            0, 0.15, "\n".join(f"• {line}" for line in _LIMITATIONS), fontsize=7.5, va="top"
        )

        fig.savefig(path, format="pdf")
        return ArtifactMetadata(
            kind=ArtifactKind.REPORT, path=str(path), mime_type=REPORT_MIME, success=True
        )
    except Exception as exc:
        return ArtifactMetadata(
            kind=ArtifactKind.REPORT, success=False, failure_reason=str(exc)[:200]
        )
    finally:
        if fig is not None:
            plt.close(fig)
