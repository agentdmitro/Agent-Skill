"""The agent-facing compact result must never carry raw monthly pageviews, must serialize to
plain, deterministic, Unicode-safe JSON with no NaN/Infinity, and must still carry everything
needed to answer ordinary questions without recomputing anything."""

import json
import math

import httpx

from tests.test_orchestration import Env, build_request, default_resolver, pageview_response
from wikipedia_interest import RunResult


def _run(languages: tuple[str, ...], intent: str) -> RunResult:
    sitelinks = {lang: lang.capitalize() + "topic" for lang in languages}

    def fetch(request: httpx.Request) -> httpx.Response:
        for lang, title in sitelinks.items():
            if f"{lang}.wikipedia" in request.url.path:
                return pageview_response(title, f"{lang}.wikipedia")
        raise AssertionError("unexpected fetch")

    env = Env(default_resolver("Q1", "Topic", sitelinks, languages), fetch)
    result = env.run(build_request(topic="Topic", languages=languages, intent=intent))
    assert isinstance(result, RunResult)
    return result


def _walk(value: object) -> list[object]:
    if isinstance(value, dict):
        out: list[object] = []
        for v in value.values():
            out.extend(_walk(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_walk(v))
        return out
    return [value]


def test_compact_result_excludes_raw_monthly_observations() -> None:
    result = _run(("en",), "trend")
    dumped = result.model_dump(mode="json")
    assert "points" not in json.dumps(dumped)
    assert "missing_months" in json.dumps(dumped)  # count/list of gaps is fine, not the series


def test_single_language_compact_output_has_explainable_fields() -> None:
    result = _run(("en",), "trend")
    lang = result.languages[0]
    assert lang.analysis.trend.direction is not None
    assert lang.analysis.trend.reliability is not None
    assert lang.candidate_classification is not None
    assert lang.analysis.data_quality.coverage_ratio is not None


def test_multi_language_compact_output_has_comparison() -> None:
    result = _run(("en", "pl"), "compare")
    assert result.comparison is not None
    assert result.comparison.status == "success"
    assert len(result.comparison.languages) == 2


def test_unavailable_languages_retained_in_compact_output() -> None:
    result = _run(("en",), "trend")
    dumped = result.model_dump(mode="json")
    assert "unavailable" in dumped
    assert dumped["unavailable"] == []


def test_json_is_deterministic_across_dumps() -> None:
    result = _run(("en", "pl"), "compare")
    assert result.model_dump_json() == result.model_dump_json()


def test_json_serialization_preserves_unicode() -> None:
    def fetch(request: httpx.Request) -> httpx.Response:
        return pageview_response("Астрономіятопик", "uk.wikipedia")

    env = Env(
        default_resolver(
            "Q1", "Астрономія", {"uk": "Астрономіятопик"}, ("uk",), query_language="uk"
        ),
        fetch,
    )
    result = env.run(
        build_request(topic="Астрономія", query_language="uk", languages=("uk",), intent="trend")
    )
    assert isinstance(result, RunResult)
    text = result.model_dump_json()
    assert "Астрономія" in text or "\\u0410\\u0441\\u0442" in text.replace("\\\\", "\\")


def test_no_nan_or_infinity_in_json_output() -> None:
    result = _run(("en", "pl"), "compare")
    dumped = result.model_dump(mode="json")
    for value in _walk(dumped):
        if isinstance(value, float):
            assert math.isfinite(value)
    text = result.model_dump_json()
    assert "NaN" not in text
    assert "Infinity" not in text
