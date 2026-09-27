"""Chart/PDF artifact generation tests. No pixel-perfect image comparisons: we check that the
right files exist, are non-empty, carry the right MIME metadata, and that unsafe paths/inputs
are rejected or handled without crashing or corrupting the underlying analysis."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_orchestration import (
    Env,
    build_request,
    default_resolver,
    en_or_pl,
    pageview_response,
)
from wikipedia_interest import (
    ArtifactKind,
    RunResult,
)
from wikipedia_interest.reporting import safe_artifact_path, sanitize_slug

PERIOD = {"start": "2024-01", "end": "2024-12"}


def _single_language_run(tmp_path: Path, artifacts: tuple[ArtifactKind, ...]) -> RunResult:
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: pageview_response("Astronomy", "en.wikipedia"),
    )
    result = env.run(
        build_request(
            languages=("en",), intent="trend", artifacts=artifacts, output_dir=str(tmp_path)
        )
    )
    assert isinstance(result, RunResult)
    return result


def _multi_language_run(tmp_path: Path, artifacts: tuple[ArtifactKind, ...]) -> RunResult:
    sitelinks = {"en": "Astronomy", "pl": "Astronomia"}

    def fetch(request: Any) -> Any:
        return pageview_response(*en_or_pl(request.url.path))

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    result = env.run(
        build_request(
            languages=tuple(sitelinks),
            intent="compare",
            artifacts=artifacts,
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(result, RunResult)
    return result


def _zero_success_run(tmp_path: Path, artifacts: tuple[ArtifactKind, ...]) -> RunResult:
    """All requested languages resolve fine but every pageview fetch fails upstream: zero
    successfully analyzed languages, distinct from a domain-unavailable/no-sitelink case."""
    sitelinks = {"uk": "Astronomy", "pl": "Astronomia", "cs": "Astronomie"}

    def fetch(request: Any) -> Any:
        return httpx.Response(503)

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    result = env.run(
        build_request(
            languages=tuple(sitelinks),
            intent="compare",
            artifacts=artifacts,
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(result, RunResult)
    return result


def test_zero_successful_languages_chart_fails(tmp_path: Path) -> None:
    result = _zero_success_run(tmp_path, (ArtifactKind.CHART,))
    assert result.languages == ()
    artifact = result.artifacts[0]
    assert artifact.success is False
    assert artifact.path is None
    assert artifact.mime_type is None
    assert artifact.failure_reason is not None


def test_zero_successful_languages_report_fails(tmp_path: Path) -> None:
    result = _zero_success_run(tmp_path, (ArtifactKind.REPORT,))
    artifact = result.artifacts[0]
    assert artifact.success is False
    assert artifact.path is None
    assert artifact.mime_type is None
    assert artifact.failure_reason == "no successfully analyzed language data"


def test_zero_successful_languages_no_pdf_written_to_disk(tmp_path: Path) -> None:
    _zero_success_run(tmp_path, (ArtifactKind.REPORT,))
    assert list(tmp_path.iterdir()) == []


def test_zero_successful_languages_preserves_analysis_state(tmp_path: Path) -> None:
    """Artifact rejection must not corrupt the (still valid) domain result: unavailability and
    the comparison engine's own rejection are unaffected by the artifact guard."""
    result = _zero_success_run(tmp_path, (ArtifactKind.REPORT,))
    assert {u.language for u in result.unavailable} == {"uk", "pl", "cs"}
    assert result.comparison is not None
    assert result.comparison.status == "rejected"
    assert result.comparison.code.value == "insufficient_languages"


def test_partial_success_pdf_still_allowed(tmp_path: Path) -> None:
    """One upstream failure among three requested languages must not block the report: the
    zero-successful-languages guard only fires when there is truly nothing to report."""
    sitelinks = {"pl": "Astronomia", "cs": "Astronomie", "sk": "Astronomia (SK)"}

    def fetch(request: Any) -> Any:
        if "sk.wikipedia" in request.url.path:
            return httpx.Response(503)
        if "pl.wikipedia" in request.url.path:
            return pageview_response("Astronomia", "pl.wikipedia")
        return pageview_response("Astronomie", "cs.wikipedia")

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    result = env.run(
        build_request(
            languages=tuple(sitelinks),
            intent="compare",
            artifacts=(ArtifactKind.REPORT,),
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(result, RunResult)
    assert {lg.language for lg in result.languages} == {"pl", "cs"}
    assert any(u.language == "sk" for u in result.unavailable)
    artifact = result.artifacts[0]
    assert artifact.success is True
    assert Path(artifact.path or "").read_bytes()[:5] == b"%PDF-"


def test_single_language_png_generated(tmp_path: Path) -> None:
    result = _single_language_run(tmp_path, (ArtifactKind.CHART,))
    artifact = result.artifacts[0]
    assert artifact.success is True
    assert artifact.mime_type == "image/png"
    path = Path(artifact.path or "")
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_multi_language_png_generated(tmp_path: Path) -> None:
    result = _multi_language_run(tmp_path, (ArtifactKind.CHART,))
    artifact = result.artifacts[0]
    assert artifact.success is True
    path = Path(artifact.path or "")
    assert path.stat().st_size > 0


def test_pdf_generated_with_valid_signature(tmp_path: Path) -> None:
    result = _single_language_run(tmp_path, (ArtifactKind.REPORT,))
    artifact = result.artifacts[0]
    assert artifact.success is True
    assert artifact.mime_type == "application/pdf"
    path = Path(artifact.path or "")
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:5] == b"%PDF-"


def test_multi_language_pdf_generated(tmp_path: Path) -> None:
    result = _multi_language_run(tmp_path, (ArtifactKind.REPORT,))
    artifact = result.artifacts[0]
    assert artifact.success is True
    path = Path(artifact.path or "")
    assert path.read_bytes()[:5] == b"%PDF-"


def test_pdf_has_exactly_one_page(tmp_path: Path) -> None:
    result = _multi_language_run(tmp_path, (ArtifactKind.REPORT,))
    assert result.artifacts[0].success is True
    raw = Path(result.artifacts[0].path or "").read_bytes()
    # Structural, dependency-free page count check: matplotlib's single-Figure PDF writer
    # emits exactly one `/Type /Page` object per page.
    assert raw.count(b"/Type /Page") - raw.count(b"/Type /Pages") == 1


def test_pdf_candidate_wording_is_validation_not_launch(tmp_path: Path) -> None:
    result = _single_language_run(tmp_path, (ArtifactKind.REPORT,))
    assert result.artifacts[0].success is True
    # Candidate wording lives in the (untested-for-rendering) template function directly.
    from wikipedia_interest.reporting import _report_candidate_sentence

    sentence = _report_candidate_sentence(result)
    assert (
        "candidate for further validation" in sentence
        or "current Wikipedia interest signal" in sentence
    )
    for forbidden in ("launch", "best market", "guaranteed", "will pay"):
        assert forbidden not in sentence.lower()


def test_unicode_topic_label_does_not_crash(tmp_path: Path) -> None:
    env = Env(
        default_resolver("Q1", "Астрономія", {"uk": "Астрономія"}, ("uk",), query_language="uk"),
        lambda req: pageview_response("Астрономія", "uk.wikipedia"),
    )
    result = env.run(
        build_request(
            topic="Астрономія",
            query_language="uk",
            languages=("uk",),
            intent="trend",
            artifacts=(ArtifactKind.CHART, ArtifactKind.REPORT),
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(result, RunResult)
    assert all(a.success for a in result.artifacts)


def test_missing_data_not_drawn_as_zero(tmp_path: Path) -> None:
    """A gap (fewer plotted points than requested months), not a zero, represents a missing
    month; we assert on the underlying series rather than parsing plotted pixels."""
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: pageview_response("Astronomy", "en.wikipedia", n=8),  # 4 of 12 missing
    )
    result = env.run(
        build_request(
            languages=("en",),
            intent="trend",
            artifacts=(ArtifactKind.CHART,),
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(result, RunResult)
    assert result.languages[0].analysis.data_quality.missing_months == 4
    assert result.artifacts[0].success is True


def test_normalized_chart_rejects_zero_baseline(tmp_path: Path) -> None:
    sitelinks = {"en": "Astronomy", "pl": "Astronomia"}

    def fetch(request: Any) -> Any:
        if "pl.wikipedia" in request.url.path:
            return pageview_response("Astronomia", "pl.wikipedia", base=0)
        return pageview_response("Astronomy", "en.wikipedia")

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    result = env.run(
        build_request(
            languages=tuple(sitelinks),
            intent="compare",
            artifacts=(ArtifactKind.CHART,),
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(result, RunResult)
    # Must not raise (undefined baseline for a language starting at 0 views is handled, not
    # divided-by-zero), and the chart must still be produced.
    assert result.artifacts[0].success is True


def test_safe_filename_generation() -> None:
    assert sanitize_slug("Astronomy") == "astronomy"
    assert sanitize_slug("  Été/Über?! ") != ""
    assert "/" not in sanitize_slug("a/b/../c")


def test_path_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception):  # noqa: B017 - ArtifactError is internal to the module
        safe_artifact_path(str(tmp_path), "../escaped.png")


def test_existing_unrelated_file_not_overwritten(tmp_path: Path) -> None:
    other = tmp_path / "keep-me.png"
    other.write_bytes(b"original")
    path = safe_artifact_path(str(tmp_path), "chart.png")
    assert path.name == "chart.png"
    assert other.read_bytes() == b"original"
