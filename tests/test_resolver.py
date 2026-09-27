from collections.abc import Callable
from typing import Any

import httpx

from wikipedia_interest import (
    ClarificationResult,
    ResolvedTopic,
    TopicResolutionRequest,
    TopicResolver,
    UnsupportedResult,
)


def matrix(*languages: str) -> dict[str, Any]:
    return {
        "sitematrix": {
            str(i): {"code": lang, "site": [{"code": "wiki"}]} for i, lang in enumerate(languages)
        }
    }


def entity(qid: str = "Q1") -> dict[str, Any]:
    return {
        "entities": {
            qid: {
                "labels": {"en": {"language": "en", "value": "Astronomy"}},
                "descriptions": {"en": {"language": "en", "value": "science"}},
                "sitelinks": {
                    "enwiki": {"site": "enwiki", "title": "Astronomy"},
                    "plwiki": {"site": "plwiki", "title": "Astronomia"},
                },
            }
        }
    }


def page(title: str, ns: int = 0) -> dict[str, Any]:
    return {"query": {"pages": [{"pageid": 1, "ns": ns, "title": title}]}}


def resolver_for(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[TopicResolver, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return TopicResolver(transport=httpx.MockTransport(transport), sleep=lambda _: None), requests


def test_exact_single_candidate_hydrates_and_verifies_canonical_titles() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("action") == "wbsearchentities":
            return httpx.Response(200, json={"search": [{"id": "Q1", "label": "Astronomy"}]})
        if params.get("action") == "wbgetentities":
            return httpx.Response(200, json=entity())
        if params.get("action") == "sitematrix":
            return httpx.Response(200, json=matrix("en", "pl"))
        title = "Astronomia" if request.url.host.startswith("pl.") else "Astronomy"
        return httpx.Response(200, json=page(title))

    resolver, requests = resolver_for(handler)
    result = resolver.resolve(
        TopicResolutionRequest(
            topic=" Astronomy ", query_language="en", target_languages=["en", "pl"]
        )
    )
    assert isinstance(result, ResolvedTopic)
    assert result.articles["pl"].title == "Astronomia"
    assert [r.url.params.get("action") for r in requests].count("wbsearchentities") == 1


def test_ambiguous_results_are_not_resolved_by_rank() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "search": [
                    {"id": "Q1", "label": "Mercury", "description": "planet"},
                    {"id": "Q2", "label": "Mercury", "description": "element"},
                ]
            },
        )

    resolver, requests = resolver_for(handler)
    result = resolver.resolve(
        TopicResolutionRequest(topic="Mercury", query_language="en", target_languages=["en"])
    )
    assert isinstance(result, ClarificationResult)
    assert len(requests) == 1


def test_direct_qid_bypasses_search_and_missing_sitelink_is_explicit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("action") == "wbgetentities":
            return httpx.Response(200, json=entity("Q333"))
        if params.get("action") == "sitematrix":
            return httpx.Response(200, json=matrix("en", "cs"))
        return httpx.Response(200, json=page("Astronomy"))

    resolver, requests = resolver_for(handler)
    result = resolver.resolve(
        TopicResolutionRequest(topic="Q333", query_language="en", target_languages=["en", "cs"])
    )
    assert isinstance(result, ResolvedTopic)
    assert result.unavailable_languages["cs"] == "no_sitelink"
    assert all(dict(r.url.params).get("action") != "wbsearchentities" for r in requests)


def test_no_candidates_is_not_found() -> None:
    resolver, _ = resolver_for(lambda _: httpx.Response(200, json={"search": []}))
    result = resolver.resolve(
        TopicResolutionRequest(topic="does not exist", query_language="en", target_languages=["en"])
    )
    assert isinstance(result, UnsupportedResult)
    assert result.code.value == "topic_not_found"
