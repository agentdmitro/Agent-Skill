"""Deterministic Mode-A fixtures: real orchestration code, mocked HTTP boundary.

Reuses the same pattern as ``tests/test_orchestration.py`` (httpx.MockTransport in front of
the real ``TopicResolver`` / ``WikimediaClient``) so evaluation exercises the production
``run_analysis`` engine end-to-end, never a hand-rolled stand-in.

This module is evaluation-only. Nothing here is imported by ``src/wikipedia_interest``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from wikipedia_interest import (
    AnalysisRequest,
    AnalysisState,
    ArtifactKind,
    Period,
    RunAnalysisRequest,
    TopicResolver,
    WikimediaClient,
    run_analysis,
)
from wikipedia_interest.orchestration import RunAnalysisResult

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


def pageview_response(
    article: str, project: str, n: int = 24, base: int = 1000, step: int = 15
) -> httpx.Response:
    """A clean, reliably-growing 24-month series by default (base + step per month).

    Wikimedia echoes `article` underscore-encoded (matching the request URL), not with raw
    spaces; encoding it here matters for any multi-word title (e.g. "Przerywany post").
    """
    encoded_article = article.replace(" ", "_")
    items = []
    y, m = 2024, 1
    for i in range(n):
        items.append(
            {
                "project": project,
                "article": encoded_article,
                "granularity": "monthly",
                "timestamp": f"{y}{m:02d}0100",
                "access": "all-access",
                "agent": "user",
                "views": max(0, base + i * step),
            }
        )
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return httpx.Response(200, json={"items": items})


def pageview_response_between(
    article: str,
    project: str,
    start_ym: tuple[int, int],
    end_ym: tuple[int, int],
    *,
    base: int = 1000,
    step: int = 15,
) -> httpx.Response:
    """Like `pageview_response`, but bounded to exactly the requested (start, end) months -
    needed so a follow-up that changes the period doesn't get months outside its new window
    (which the real engine correctly rejects as a malformed upstream response)."""
    encoded_article = article.replace(" ", "_")
    items = []
    y, m = start_ym
    i = 0
    while (y, m) <= end_ym:
        items.append(
            {
                "project": project,
                "article": encoded_article,
                "granularity": "monthly",
                "timestamp": f"{y}{m:02d}0100",
                "access": "all-access",
                "agent": "user",
                "views": max(0, base + i * step),
            }
        )
        i += 1
        m += 1
        if m == 13:
            m, y = 1, y + 1
    return httpx.Response(200, json={"items": items})


def sparse_pageview_response(article: str, project: str) -> httpx.Response:
    """Only 5 valid months of data: exercises `insufficient_data`."""
    encoded_article = article.replace(" ", "_")
    items = []
    for i, (y, m) in enumerate([(2024, 1), (2024, 2), (2024, 3), (2024, 4), (2024, 5)]):
        items.append(
            {
                "project": project,
                "article": encoded_article,
                "granularity": "monthly",
                "timestamp": f"{y}{m:02d}0100",
                "access": "all-access",
                "agent": "user",
                "views": 500 + i * 10,
            }
        )
    return httpx.Response(200, json={"items": items})


def default_resolver(
    qid: str,
    label: str,
    sitelinks: dict[str, str],
    langs: tuple[str, ...],
    query_language: str = "en",
    search_candidates: list[dict[str, Any]] | None = None,
) -> Handler:
    """A resolver handler for one unambiguous entity (or, with `search_candidates`, an
    ambiguous one whose clarification the caller must stop at)."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        action = params.get("action")
        if action == "wbsearchentities":
            if search_candidates is not None:
                return httpx.Response(200, json={"search": search_candidates})
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


class Env:
    """One resolver + one Wikimedia client, each backed by a scriptable MockTransport."""

    def __init__(self, resolve: Handler, fetch: Handler) -> None:
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

    def run(self, request: RunAnalysisRequest) -> RunAnalysisResult:
        return run_analysis(request, resolver=self.resolver, client=self.client)

    def close(self) -> None:
        self.resolver.close()
        self.client.close()


def build_request(
    *,
    topic: str = "Astronomy",
    languages: tuple[str, ...] = ("en",),
    period: dict[str, str] | None = None,
    intent: str = "trend",
    query_language: str = "en",
    artifacts: tuple[ArtifactKind, ...] = (),
    output_dir: str = "output",
    previous_state: AnalysisState | None = None,
) -> RunAnalysisRequest:
    return RunAnalysisRequest(
        request=AnalysisRequest(
            topic=topic,
            languages=languages,
            period=Period.model_validate(period or {"start": "2024-01", "end": "2025-12"}),
            analysis=intent,
            previous_state=previous_state,
        ),
        query_language=query_language,
        artifacts=artifacts,
        output_dir=output_dir,
    )


def multi_language_env(
    qid: str, label: str, sitelinks: dict[str, str], *, trend: str = "growing"
) -> Env:
    """A resolver + fetch pair serving every language in `sitelinks` with a distinct,
    deterministic pageview series so cross-language comparisons have real variation."""
    projects = {lang: f"{lang}.wikipedia.org" for lang in sitelinks}
    step_by_language = {
        lang: (20 if trend == "growing" else -10 if trend == "declining" else 0) + (i * 3)
        for i, lang in enumerate(sitelinks)
    }
    base_by_language = {lang: 800 + i * 400 for i, lang in enumerate(sitelinks)}

    def fetch(request: httpx.Request) -> httpx.Response:
        for lang, title in sitelinks.items():
            if f"/{projects[lang]}/" in request.url.path:
                return pageview_response(
                    title,
                    projects[lang],
                    base=base_by_language[lang],
                    step=step_by_language[lang],
                )
        raise AssertionError(f"unexpected fetch path: {request.url.path}")

    return Env(default_resolver(qid, label, sitelinks, tuple(sitelinks)), fetch)
