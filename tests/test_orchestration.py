"""End-to-end orchestration tests: resolve -> fetch -> analyze -> compare -> findings -> state.

Mocks only the two HTTP boundaries (Wikidata/Wikipedia via TopicResolver, Wikimedia pageviews
via WikimediaClient); everything else runs for real, exactly as scripts/run_analysis.py does.
"""

import os
from collections.abc import Callable
from typing import Any

import httpx

from wikipedia_interest import (
    AnalysisRequest,
    AnalysisState,
    ArtifactKind,
    ClarificationResult,
    InvalidInputResult,
    Period,
    RunAnalysisRequest,
    RunResult,
    TopicResolver,
    UnsupportedResult,
    WikimediaClient,
    run_analysis,
)

PERIOD = {"start": "2024-01", "end": "2024-12"}
Handler = Callable[[httpx.Request], httpx.Response]


def matrix(*languages: str) -> dict[str, Any]:
    return {
        "sitematrix": {
            str(i): {"code": lang, "site": [{"code": "wiki"}]} for i, lang in enumerate(languages)
        }
    }


def entity(
    qid: str, label: str, sitelinks: dict[str, str], query_language: str = "en"
) -> dict[str, Any]:
    return {
        "entities": {
            qid: {
                "labels": {query_language: {"language": query_language, "value": label}},
                "descriptions": {query_language: {"language": query_language, "value": "a topic"}},
                "sitelinks": {
                    f"{lang}wiki": {"site": f"{lang}wiki", "title": title}
                    for lang, title in sitelinks.items()
                },
            }
        }
    }


def page(title: str, ns: int = 0) -> dict[str, Any]:
    return {"query": {"pages": [{"pageid": 1, "ns": ns, "title": title}]}}


def pageview_response(article: str, project: str, n: int = 12, base: int = 1000) -> httpx.Response:
    items = []
    y, m = 2024, 1
    for i in range(n):
        items.append(
            {
                "project": project,
                "article": article,
                "granularity": "monthly",
                "timestamp": f"{y}{m:02d}0100",
                "access": "all-access",
                "agent": "user",
                "views": base + i * 5,
            }
        )
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return httpx.Response(200, json={"items": items})


class Env:
    """One resolver + one Wikimedia client, each backed by a scriptable MockTransport."""

    def __init__(
        self,
        resolve: Handler,
        fetch: Handler,
    ) -> None:
        self.resolve_requests: list[httpx.Request] = []
        self.fetch_requests: list[httpx.Request] = []

        def resolve_transport(request: httpx.Request) -> httpx.Response:
            self.resolve_requests.append(request)
            return resolve(request)

        def fetch_transport(request: httpx.Request) -> httpx.Response:
            self.fetch_requests.append(request)
            return fetch(request)

        self.resolver = TopicResolver(
            transport=httpx.MockTransport(resolve_transport), sleep=lambda _: None
        )
        self.client = WikimediaClient(
            transport=httpx.MockTransport(fetch_transport), sleep=lambda _: None
        )

    def run(self, request: RunAnalysisRequest) -> Any:
        return run_analysis(request, resolver=self.resolver, client=self.client)


def default_resolver(
    qid: str,
    label: str,
    sitelinks: dict[str, str],
    langs: tuple[str, ...],
    query_language: str = "en",
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        action = params.get("action")
        if action == "wbsearchentities":
            return httpx.Response(200, json={"search": [{"id": qid, "label": label}]})
        if action == "wbgetentities":
            return httpx.Response(200, json=entity(qid, label, sitelinks, query_language))
        if action == "sitematrix":
            return httpx.Response(200, json=matrix(*langs))
        for lang in langs:
            if request.url.host.startswith(f"{lang}."):
                return httpx.Response(200, json=page(sitelinks.get(lang, "")))
        return httpx.Response(200, json=page("Unknown"))

    return handler


def en_or_pl(path: str) -> tuple[str, str]:
    """(article, project) for a pageview fetch path, given the en/pl fixtures above."""
    if "pl.wikipedia.org" in path:
        return "Astronomia", "pl.wikipedia.org"
    return "Astronomy", "en.wikipedia.org"


def build_request(**overrides: Any) -> RunAnalysisRequest:
    data: dict[str, Any] = {
        "request": AnalysisRequest(
            topic=overrides.pop("topic", "Astronomy"),
            languages=overrides.pop("languages", ("en",)),
            period=Period.model_validate(overrides.pop("period", PERIOD)),
            analysis=overrides.pop("intent", "trend"),
            previous_state=overrides.pop("previous_state", None),
        ),
        "query_language": overrides.pop("query_language", "en"),
        "artifacts": overrides.pop("artifacts", ()),
        "output_dir": overrides.pop("output_dir", "output"),
    }
    return RunAnalysisRequest.model_validate(data | overrides)


def test_successful_single_language_end_to_end() -> None:
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: pageview_response("Astronomy", "en.wikipedia.org"),
    )
    result = env.run(build_request(languages=("en",), intent="trend"))
    assert isinstance(result, RunResult)
    assert [lg.language for lg in result.languages] == ["en"]
    assert result.comparison is None
    assert result.next_state.version == 1
    assert result.next_state.topic_id == "Q1"


def test_successful_multi_language_comparison() -> None:
    sitelinks = {"en": "Astronomy", "pl": "Astronomia", "cs": "Astronomie"}
    projects = {"en": "en.wikipedia.org", "pl": "pl.wikipedia.org", "cs": "cs.wikipedia.org"}

    def fetch(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for lang, title in sitelinks.items():
            if f"/{projects[lang]}/" in path:
                return pageview_response(title, projects[lang])
        raise AssertionError(f"unexpected fetch path: {path}")

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    result = env.run(build_request(languages=tuple(sitelinks), intent="compare"))
    assert isinstance(result, RunResult)
    assert {lg.language for lg in result.languages} == set(sitelinks)
    assert result.comparison is not None and result.comparison.status == "success"
    assert 1 <= len(result.findings) <= 6


def test_ambiguous_topic_stops_before_pageview_fetch() -> None:
    def resolve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "search": [
                    {"id": "Q1", "label": "Mercury", "description": "planet"},
                    {"id": "Q2", "label": "Mercury", "description": "element"},
                ]
            },
        )

    env = Env(resolve, lambda req: (_ for _ in ()).throw(AssertionError("must not fetch")))
    result = env.run(build_request(topic="Mercury", languages=("en",), intent="trend"))
    assert isinstance(result, ClarificationResult)
    assert len(env.fetch_requests) == 0


def test_direct_qid_bypasses_search() -> None:
    calls = {"search": 0}

    def resolve(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("action") == "wbsearchentities":
            calls["search"] += 1
            return httpx.Response(200, json={"search": []})
        return default_resolver("Q333", "Mercury", {"en": "Mercury"}, ("en",))(request)

    env = Env(resolve, lambda req: pageview_response("Mercury", "en.wikipedia.org"))
    req = build_request(topic="Q333", languages=("en",), intent="trend")
    result = env.run(req)
    assert isinstance(result, RunResult)
    assert result.next_state.topic_id == "Q333"
    assert calls["search"] == 0


def test_one_unavailable_language_two_successful() -> None:
    # "cs" has no sitelink -> domain-unavailable, never fetched.
    sitelinks = {"en": "Astronomy", "pl": "Astronomia"}

    def fetch(request: httpx.Request) -> httpx.Response:
        title, project = en_or_pl(request.url.path)
        return pageview_response(title, project)

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, ("en", "pl", "cs")), fetch)
    result = env.run(build_request(languages=("en", "pl", "cs"), intent="compare"))
    assert isinstance(result, RunResult)
    assert {lg.language for lg in result.languages} == {"en", "pl"}
    assert [u.language for u in result.unavailable] == ["cs"]
    assert result.unavailable[0].category.value == "domain_unavailable"


def test_one_upstream_failure_two_successful() -> None:
    sitelinks = {"en": "Astronomy", "pl": "Astronomia", "cs": "Astronomie"}

    def fetch(request: httpx.Request) -> httpx.Response:
        if "cs.wikipedia.org" in request.url.path:
            return httpx.Response(503)
        title, project = en_or_pl(request.url.path)
        return pageview_response(title, project)

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    result = env.run(build_request(languages=tuple(sitelinks), intent="compare"))
    assert isinstance(result, RunResult)
    assert {lg.language for lg in result.languages} == {"en", "pl"}
    cs = next(u for u in result.unavailable if u.language == "cs")
    assert cs.category.value == "upstream_failure"
    assert cs.retryable is True


def test_all_languages_unavailable_is_unsupported() -> None:
    def resolve(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("action") == "sitematrix":
            return httpx.Response(200, json=matrix())  # no active project at all
        return default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",))(request)

    env = Env(resolve, lambda req: (_ for _ in ()).throw(AssertionError("must not fetch")))
    result = env.run(build_request(languages=("en",), intent="trend"))
    assert isinstance(result, UnsupportedResult)
    assert result.code.value == "article_unavailable"


def test_only_one_language_survives_requested_comparison() -> None:
    sitelinks = {"en": "Astronomy", "pl": "Astronomia"}

    def fetch(request: httpx.Request) -> httpx.Response:
        if "pl.wikipedia.org" in request.url.path:
            return httpx.Response(503)
        return pageview_response("Astronomy", "en.wikipedia.org")

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    result = env.run(build_request(languages=tuple(sitelinks), intent="compare"))
    assert isinstance(result, RunResult)
    assert len(result.languages) == 1
    assert result.comparison is not None and result.comparison.status == "rejected"
    assert result.comparison.code.value == "insufficient_languages"
    assert 1 <= len(result.findings) <= 6  # falls back to single-language findings


def test_pre_2015_period_is_invalid_input() -> None:
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: (_ for _ in ()).throw(AssertionError("must not be reached")),
    )
    period = {"start": "2010-01", "end": "2010-06"}
    result = env.run(build_request(languages=("en",), intent="trend", period=period))
    assert isinstance(result, InvalidInputResult)
    assert any(i.field == "period.start" for i in result.issues)


def test_previous_state_reuses_qid_without_research() -> None:
    calls = {"search": 0}

    def resolve(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("action") == "wbsearchentities":
            calls["search"] += 1
        return default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",))(request)

    env = Env(resolve, lambda req: pageview_response("Astronomy", "en.wikipedia.org"))
    previous = AnalysisState(
        analysis_id="wi-q1",
        version=1,
        topic_id="Q1",
        topic_label="Astronomy",
        languages=("en",),
        period=Period.model_validate(PERIOD),
        analysis="trend",
    )
    result = env.run(build_request(languages=("en",), intent="trend", previous_state=previous))
    assert isinstance(result, RunResult)
    assert calls["search"] == 0
    assert result.next_state.version == 2
    assert result.next_state.topic_id == "Q1"


def test_follow_up_adding_language() -> None:
    sitelinks = {"en": "Astronomy", "pl": "Astronomia"}

    def fetch(request: httpx.Request) -> httpx.Response:
        title, project = en_or_pl(request.url.path)
        return pageview_response(title, project)

    env = Env(default_resolver("Q1", "Astronomy", sitelinks, tuple(sitelinks)), fetch)
    previous = AnalysisState(
        analysis_id="wi-q1",
        version=1,
        topic_id="Q1",
        topic_label="Astronomy",
        languages=("en",),
        period=Period.model_validate(PERIOD),
        analysis="trend",
    )
    result = env.run(
        build_request(languages=("en", "pl"), intent="compare", previous_state=previous)
    )
    assert isinstance(result, RunResult)
    assert {lg.language for lg in result.languages} == {"en", "pl"}
    assert result.next_state.languages == ("en", "pl")
    assert result.next_state.version == 2


def test_inconsistent_state_rejected() -> None:
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    previous = AnalysisState(
        analysis_id="wi-q1",
        version=1,
        topic_id="Q1",
        topic_label="Astronomy",
        languages=("en",),
        period=Period.model_validate(PERIOD),
        analysis="trend",
    )
    req = build_request(
        topic="Something else", languages=("en",), intent="trend", previous_state=previous
    )
    result = env.run(req)
    assert isinstance(result, UnsupportedResult)
    assert result.code.value == "inconsistent_state"
    assert len(env.fetch_requests) == 0


def test_artifacts_disabled_by_default() -> None:
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: pageview_response("Astronomy", "en.wikipedia.org"),
    )
    result = env.run(build_request(languages=("en",), intent="trend"))
    assert isinstance(result, RunResult)
    assert result.artifacts == ()


def test_artifacts_enabled_produce_files(tmp_path: Any) -> None:
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: pageview_response("Astronomy", "en.wikipedia.org"),
    )
    result = env.run(
        build_request(
            languages=("en",),
            intent="trend",
            artifacts=(ArtifactKind.CHART, ArtifactKind.REPORT),
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(result, RunResult)
    assert len(result.artifacts) == 2
    for artifact in result.artifacts:
        assert artifact.success is True
        assert artifact.path is not None
        path_obj = tmp_path / os.path.basename(artifact.path)
        assert path_obj.exists()
        assert path_obj.stat().st_size > 0


def test_artifact_failure_does_not_destroy_valid_analysis(tmp_path: Any) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    env = Env(
        default_resolver("Q1", "Astronomy", {"en": "Astronomy"}, ("en",)),
        lambda req: pageview_response("Astronomy", "en.wikipedia.org"),
    )
    result = env.run(
        build_request(
            languages=("en",),
            intent="trend",
            artifacts=(ArtifactKind.CHART,),
            output_dir=str(blocked / "output"),
        )
    )
    assert isinstance(result, RunResult)
    assert len(result.languages) == 1
    assert result.artifacts[0].success is False
    assert result.artifacts[0].failure_reason is not None
