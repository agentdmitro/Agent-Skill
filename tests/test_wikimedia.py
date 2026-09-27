from typing import Any

import httpx
import pytest

from wikipedia_interest import (
    PageviewSeries,
    Period,
    WikimediaClient,
    pageviews_project,
    wikipedia_domain,
)
from wikipedia_interest.contracts import (
    InvalidInputResult,
    UnsupportedCode,
    UnsupportedResult,
    UpstreamFailureCode,
    UpstreamFailureResult,
)
from wikipedia_interest.wikimedia import DEFAULT_USER_AGENT, api_timestamps, build_url

PERIOD = Period.model_validate({"start": "2024-01", "end": "2024-03"})


def item(month: str, views: Any = 10, **over: Any) -> dict[str, Any]:
    d = {
        "project": "uk.wikipedia",
        "article": "Астрономія",
        "granularity": "monthly",
        "timestamp": month.replace("-", "") + "0100",
        "access": "all-access",
        "agent": "user",
        "views": views,
    }
    return d | over


def ok(*items: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json={"items": list(items)})


FULL = [item("2024-01"), item("2024-02"), item("2024-03")]


class Harness:
    def __init__(self, *responses: httpx.Response | Exception, **kw: Any) -> None:
        self.queue = list(responses)
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self.client = WikimediaClient(
            transport=httpx.MockTransport(self._handle), sleep=self.sleeps.append, **kw
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        nxt = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def fetch(
        self, article: str = "Астрономія", period: Period = PERIOD, language: str = "uk"
    ) -> Any:
        return self.client.fetch_monthly_pageviews(language, article, period)


def failure(result: Any, code: UpstreamFailureCode) -> UpstreamFailureResult:
    assert isinstance(result, UpstreamFailureResult)
    assert result.code is code
    return result


def test_wikipedia_domain_mapping() -> None:
    assert wikipedia_domain("uk") == "uk.wikipedia.org"
    assert wikipedia_domain("simple") == "simple.wikipedia.org"
    with pytest.raises(ValueError):
        wikipedia_domain("uk.evil.com/")


def test_pageviews_project_mapping() -> None:
    assert pageviews_project("uk") == "uk.wikipedia"
    assert pageviews_project("simple") == "simple.wikipedia"
    with pytest.raises(ValueError):
        pageviews_project("uk.evil.com/")


def test_domain_and_project_are_distinct_for_all_known_languages() -> None:
    # The MediaWiki domain and the Pageviews project id are related but never interchangeable.
    for lang in ("pl", "cs", "en", "uk", "simple"):
        domain = wikipedia_domain(lang)
        project = pageviews_project(lang)
        assert domain == f"{project}.org"
        assert domain != project


def test_endpoint_construction() -> None:
    h = Harness(ok(*FULL))
    h.fetch()
    url = h.requests[0].url
    assert url.host == "wikimedia.org"
    assert url.path.startswith("/api/rest_v1/metrics/pageviews/per-article/uk.wikipedia/")
    assert url.raw_path.decode().endswith(
        "/all-access/user/%D0%90%D1%81%D1%82%D1%80%D0%BE%D0%BD%D0%BE%D0%BC%D1%96%D1%8F/monthly/2024010100/2024033100"
    )


def test_title_with_spaces_and_punctuation_cannot_escape_segment() -> None:
    url = build_url("en.wikipedia.org", "AC/DC ? #x & y%", PERIOD)
    assert "/AC%2FDC_%3F_%23x_%26_y%25/monthly/" in url
    assert url.startswith("https://wikimedia.org/api/rest_v1/")
    h = Harness(ok())
    h.fetch("AC/DC ? #x")
    req = h.requests[0]
    assert req.url.query == b"" and req.url.fragment == ""
    assert req.url.host == "wikimedia.org"


def test_inclusive_period_timestamps() -> None:
    assert api_timestamps(PERIOD) == ("2024010100", "2024033100")
    leap = Period.model_validate({"start": "2024-01", "end": "2024-02"})
    assert api_timestamps(leap)[1] == "2024022900"
    dec = Period.model_validate({"start": "2024-01", "end": "2025-12"})
    assert api_timestamps(dec)[1] == "2025123100"


def test_pre_2015_07_rejected_not_clamped() -> None:
    h = Harness(ok(*FULL))
    early = Period.model_validate({"start": "2014-01", "end": "2016-01"})
    result = h.fetch(period=early)
    assert isinstance(result, InvalidInputResult)
    assert result.issues[0].field == "period.start"
    assert h.requests == []
    boundary = Period.model_validate({"start": "2015-07", "end": "2015-08"})
    Harness(ok(item("2015-07"), item("2015-08"))).fetch(period=boundary)


def test_success_and_chronology_and_zero_views() -> None:
    result = Harness(ok(item("2024-01", 5), item("2024-02", 0), item("2024-03", 7))).fetch()
    assert isinstance(result, PageviewSeries)
    assert [p.views for p in result.points] == [5, 0, 7]
    assert result.missing_months == ()
    assert result.period == PERIOD
    dumped = result.model_dump(mode="json")
    assert dumped["points"][0] == {"month": "2024-01", "views": 5}
    assert dumped["period"] == {"start": "2024-01", "end": "2024-03"}


def test_missing_months_not_fabricated() -> None:
    result = Harness(ok(item("2024-01"), item("2024-03"))).fetch()
    assert isinstance(result, PageviewSeries)
    assert [p.month.month for p in result.points] == [1, 3]
    assert [m.month for m in result.missing_months] == [2]


def test_article_with_spaces_matches_underscored_upstream() -> None:
    h = Harness(
        ok(
            *[
                item(m, article="Intermittent_fasting", project="en.wikipedia")
                for m in ("2024-01", "2024-02", "2024-03")
            ]
        )
    )
    result = h.fetch("Intermittent fasting", language="en")
    assert isinstance(result, PageviewSeries)
    assert result.article == "Intermittent fasting"
    assert "/Intermittent_fasting/" in h.requests[0].url.path


def test_unknown_fields_ignored() -> None:
    assert isinstance(
        Harness(ok(*[item(m, extra="x") for m in ("2024-01",)])).fetch(), PageviewSeries
    )


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, json={"nope": []}),
        httpx.Response(200, json={"items": "x"}),
        ok("garbage"),  # type: ignore[arg-type]
        ok(item("2024-01", -1)),
        ok(item("2024-01", "1234")),
        ok(item("2024-01", True)),
        ok(item("2024-01"), item("2024-01")),
        ok(item("2024-02"), item("2024-01")),
        ok(item("2024-01", timestamp="2024-01")),
        ok(item("2024-01", timestamp="2024130100")),
        ok(item("2024-01", article="Other")),
        ok(item("2024-01", project="pl.wikipedia")),
        ok(item("2024-01", granularity="daily")),
        ok(item("2023-12")),
        ok({k: v for k, v in item("2024-01").items() if k != "views"}),
    ],
)
def test_malformed_responses_rejected(response: httpx.Response) -> None:
    failure(Harness(response).fetch(), UpstreamFailureCode.MALFORMED_RESPONSE)


def test_empty_items_explicit() -> None:
    result = Harness(ok()).fetch()
    assert isinstance(result, UnsupportedResult)
    assert result.code is UnsupportedCode.ARTICLE_UNAVAILABLE


def test_404() -> None:
    h = Harness(httpx.Response(404, json={}))
    result = h.fetch()
    assert isinstance(result, UnsupportedResult)
    assert result.code is UnsupportedCode.ARTICLE_UNAVAILABLE
    assert len(h.requests) == 1


def test_timeout_and_transport_mapping() -> None:
    h = Harness(httpx.ReadTimeout("slow"))
    assert failure(h.fetch(), UpstreamFailureCode.TIMEOUT).retryable
    h = Harness(httpx.ConnectError("down"))
    failure(h.fetch(), UpstreamFailureCode.UNAVAILABLE)
    assert len(h.requests) == 3


def test_429_then_success_and_exhaustion() -> None:
    h = Harness(httpx.Response(429), ok(*FULL))
    assert isinstance(h.fetch(), PageviewSeries)
    assert len(h.requests) == 2 and h.sleeps == [1.0]
    h = Harness(httpx.Response(429))
    failure(h.fetch(), UpstreamFailureCode.RATE_LIMITED)
    assert len(h.requests) == 3 and h.sleeps == [1.0, 2.0]


def test_500_retried_then_unavailable() -> None:
    h = Harness(httpx.Response(500), httpx.Response(503), ok(*FULL))
    assert isinstance(h.fetch(), PageviewSeries)
    h = Harness(httpx.Response(500))
    failure(h.fetch(), UpstreamFailureCode.UNAVAILABLE)
    assert len(h.requests) == 3


@pytest.mark.parametrize("status", [400, 403])
def test_non_retryable_status(status: int) -> None:
    h = Harness(httpx.Response(status))
    assert not failure(h.fetch(), UpstreamFailureCode.UNAVAILABLE).retryable
    assert len(h.requests) == 1 and h.sleeps == []


def test_retry_after_honoured_and_capped() -> None:
    h = Harness(httpx.Response(429, headers={"Retry-After": "4"}), ok(*FULL))
    h.fetch()
    assert h.sleeps == [4.0]
    h = Harness(httpx.Response(429, headers={"Retry-After": "999999"}), ok(*FULL))
    h.fetch()
    assert h.sleeps == [10.0]
    h = Harness(httpx.Response(429, headers={"Retry-After": "soon"}), ok(*FULL))
    h.fetch()
    assert h.sleeps == [1.0]


def test_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIKIPEDIA_INTEREST_USER_AGENT", raising=False)
    h = Harness(ok(*FULL))
    h.fetch()
    assert h.requests[0].headers["User-Agent"] == DEFAULT_USER_AGENT
    monkeypatch.setenv("WIKIPEDIA_INTEREST_USER_AGENT", "me/1.0 (a@b.c)")
    h = Harness(ok(*FULL))
    h.fetch()
    assert h.requests[0].headers["User-Agent"] == "me/1.0 (a@b.c)"


def test_invalid_language_rejected() -> None:
    assert isinstance(Harness(ok(*FULL)).fetch(language="U K/"), InvalidInputResult)
