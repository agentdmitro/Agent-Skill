"""Mode-A (offline/deterministic) scenario execution: no model, no network.

Runs the real `run_analysis` engine (via `evals.fixtures`) against each scenario's mocked
Wikidata/Wikimedia responses, then checks the declarative `expected` assertions from the
scenario JSON. Also runs the graders' forbidden-claim / reliability-language regression check
against each adversarial scenario's `bad_answer` / `good_answer` pair.

This module contains NO model calls. Categories/assertions that inherently require a model
in the loop (`requires_model: true`, or `skill_should_activate` on a no-tool-call turn) are
recorded as `NOT_APPLICABLE_OFFLINE`, not pass/fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from evals import graders
from evals.fixtures import (
    Env,
    build_request,
    default_resolver,
    pageview_response_between,
    sparse_pageview_response,
)
from wikipedia_interest.contracts import (
    AnalysisState,
    ClarificationResult,
    UnsupportedResult,
    invalid_input_from,
)
from wikipedia_interest.orchestration import RunResult

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

Verdict = str  # "pass" | "fail" | "not_applicable_offline"


@dataclass
class CheckResult:
    name: str
    verdict: Verdict
    detail: str = ""


@dataclass
class TurnReport:
    turn_index: int
    checks: list[CheckResult] = field(default_factory=list)
    status: str | None = None

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.verdict == "fail"]


@dataclass
class ScenarioReport:
    scenario_id: str
    category: str
    turns: list[TurnReport] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        return all(not t.failed for t in self.turns)

    @property
    def all_checks(self) -> list[CheckResult]:
        return [c for t in self.turns for c in t.checks]


def load_scenarios(categories: list[str] | None = None) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario in data:
            if categories is None or scenario["category"] in categories:
                scenarios.append(scenario)
    return scenarios


# --- Fixture -> Env -------------------------------------------------------------------------


def _period_from_url(path: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Wikimedia pageview URLs end .../monthly/<start:YYYYMMDDHH>/<end:YYYYMMDDHH>."""
    start_raw, end_raw = path.rstrip("/").split("/")[-2:]
    start = (int(start_raw[:4]), int(start_raw[4:6]))
    end = (int(end_raw[:4]), int(end_raw[4:6]))
    return start, end


def _trend_step(trend: str | None, index: int) -> int:
    base = {"growing": 20, "declining": -10, "stable": 0}.get(trend or "growing", 20)
    return base + index * 3 if base else 0


def build_env(fixture: dict[str, Any] | None) -> Env:
    if not fixture:
        raise AssertionError("scenario turn requires a fixture to run offline")

    if fixture.get("qid") is None and "search_candidates" in fixture:
        candidates = fixture["search_candidates"]

        def resolve(request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            if params.get("action") == "wbsearchentities":
                return httpx.Response(200, json={"search": candidates})
            raise AssertionError("must not hydrate/fetch an entity before clarification")

        def fetch_before_clarification(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not fetch pageviews before clarification")

        return Env(resolve, fetch_before_clarification)

    if fixture.get("resolver_always_503"):

        def resolve_503(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        def fetch_unreached(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not fetch pageviews when topic resolution failed upstream")

        return Env(resolve_503, fetch_unreached)

    qid, label = fixture["qid"], fixture["label"]
    sitelinks: dict[str, str] = fixture["sitelinks"]
    query_language = fixture.get("query_language", "en")
    timeout_languages = set(fixture.get("timeout_languages", ()))
    sparse = bool(fixture.get("sparse", False))
    trend = fixture.get("trend", "growing")
    projects = {lang: f"{lang}.wikipedia" for lang in sitelinks}

    def fetch(request: httpx.Request) -> httpx.Response:
        for i, (lang, title) in enumerate(sitelinks.items()):
            if f"/{projects[lang]}/" not in request.url.path:
                continue
            if lang in timeout_languages:
                raise httpx.TimeoutException("simulated upstream timeout", request=request)
            if sparse:
                return sparse_pageview_response(title, projects[lang])
            start_ym, end_ym = _period_from_url(request.url.path)
            return pageview_response_between(
                title,
                projects[lang],
                start_ym,
                end_ym,
                base=800 + i * 400,
                step=_trend_step(trend, i),
            )
        raise AssertionError(f"unexpected fetch path: {request.url.path}")

    return Env(default_resolver(qid, label, sitelinks, tuple(sitelinks), query_language), fetch)


# --- Assertions ------------------------------------------------------------------------------


def _finding_texts(result: Any) -> str:
    if isinstance(result, RunResult):
        return " ".join(f.text for f in result.findings).lower()
    return ""


def _check(name: str, condition: bool, detail: str = "") -> CheckResult:
    return CheckResult(name, "pass" if condition else "fail", detail)


def evaluate_turn(
    expected: dict[str, Any],
    result: Any,
    *,
    resolve_requests: list[httpx.Request],
    fetch_requests: list[httpx.Request],
    previous_next_state: AnalysisState | None,
    turn_request_period: dict[str, str] | None,
) -> list[CheckResult]:
    checks: list[CheckResult] = []

    if expected.get("requires_model") and expected.get("simulated_request_is_null", False):
        checks.append(CheckResult("skill_should_activate", "not_applicable_offline"))
        return checks

    if "status" in expected:
        checks.append(_check("status", result.status == expected["status"], f"got {result.status}"))

    if "requires_clarification" in expected:
        is_clarification = isinstance(result, ClarificationResult)
        checks.append(
            _check("requires_clarification", is_clarification == expected["requires_clarification"])
        )

    if "candidates_at_least" in expected and isinstance(result, ClarificationResult):
        checks.append(
            _check("candidates_at_least", len(result.candidates) >= expected["candidates_at_least"])
        )

    if expected.get("no_fetch_before_clarification"):
        checks.append(_check("no_fetch_before_clarification", len(fetch_requests) == 0))

    if expected.get("no_hidden_first_result_selection"):
        checks.append(
            _check("no_hidden_first_result_selection", isinstance(result, ClarificationResult))
        )

    if not isinstance(result, RunResult):
        # Remaining assertions below only apply to success results.
        if "unsupported_code" in expected and isinstance(result, UnsupportedResult):
            checks.append(
                _check("unsupported_code", result.code.value == expected["unsupported_code"])
            )
        if "retryable" in expected and hasattr(result, "retryable"):
            checks.append(_check("retryable", result.retryable == expected["retryable"]))
        if expected.get("previous_qid_not_reused"):
            checks.append(_check("previous_qid_not_reused", isinstance(result, UnsupportedResult)))
        return checks

    if "expected_topic_id" in expected:
        checks.append(
            _check("expected_topic_id", result.next_state.topic_id == expected["expected_topic_id"])
        )

    if "expected_languages" in expected:
        got = {lg.language for lg in result.languages}
        checks.append(
            _check("expected_languages", got == set(expected["expected_languages"]), f"got {got}")
        )

    if "expected_intent" in expected:
        checks.append(
            _check("expected_intent", result.request.analysis.value == expected["expected_intent"])
        )

    if "min_languages_analyzed" in expected:
        checks.append(
            _check(
                "min_languages_analyzed",
                len(result.languages) >= expected["min_languages_analyzed"],
            )
        )

    if "comparison_present" in expected:
        present = result.comparison is not None and result.comparison.status == "success"
        checks.append(_check("comparison_present", present == expected["comparison_present"]))

    if "comparison_rejected_reason" in expected:
        comparison = result.comparison
        matches = (
            comparison is not None
            and comparison.status == "rejected"
            and (comparison.code.value == expected["comparison_rejected_reason"])
        )
        checks.append(_check("comparison_rejected_reason", bool(matches)))

    if expected.get("no_topic_search_performed"):
        no_search = not any(
            dict(r.url.params).get("action") == "wbsearchentities" for r in resolve_requests
        )
        checks.append(_check("no_topic_search_performed", no_search))

    if expected.get("titles_are_utf8"):
        try:
            json.dumps(
                {lg.language: lg.article.title for lg in result.languages}, ensure_ascii=False
            ).encode("utf-8")
            ok = True
        except UnicodeError:
            ok = False
        checks.append(_check("titles_are_utf8", ok))

    if "artifact_requested" in expected:
        kinds = {a.kind.value for a in result.artifacts}
        checks.append(_check("artifact_requested", expected["artifact_requested"] in kinds))

    if expected.get("analysis_present_regardless_of_artifact_outcome"):
        checks.append(
            _check("analysis_present_regardless_of_artifact_outcome", len(result.languages) > 0)
        )

    if "unavailable_languages" in expected:
        got = {u.language for u in result.unavailable}
        checks.append(
            _check(
                "unavailable_languages", got == set(expected["unavailable_languages"]), f"got {got}"
            )
        )

    if "unavailable_category" in expected:
        matches = all(
            u.category.value == expected["unavailable_category"]
            for u in result.unavailable
            if u.language
            in expected.get("unavailable_languages", [u.language for u in result.unavailable])
        )
        checks.append(_check("unavailable_category", matches))

    if "unavailable_retryable" in expected:
        matches = all(u.retryable == expected["unavailable_retryable"] for u in result.unavailable)
        checks.append(_check("unavailable_retryable", matches))

    if "trend_direction" in expected:
        checks.append(
            _check(
                "trend_direction",
                result.languages[0].analysis.trend.direction.value == expected["trend_direction"],
            )
        )

    if "reliability" in expected:
        checks.append(
            _check(
                "reliability",
                result.languages[0].analysis.trend.reliability.value == expected["reliability"],
            )
        )

    if "forbidden_claims_in_findings" in expected or expected.get("forbidden_in_findings") is False:
        hits = graders.forbidden_claims(_finding_texts(result))
        checks.append(_check("forbidden_in_findings", not hits, f"hits={hits}"))

    if "next_state_version" in expected:
        checks.append(
            _check(
                "next_state_version", result.next_state.version == expected["next_state_version"]
            )
        )

    if expected.get("topic_reused_not_reresolved"):
        no_search = not any(
            dict(r.url.params).get("action") == "wbsearchentities" for r in resolve_requests
        )
        checks.append(_check("topic_reused_not_reresolved", no_search))

    if (
        expected.get("same_period_as_previous")
        and turn_request_period is not None
        and previous_next_state is not None
    ):
        same = turn_request_period["start"] == previous_next_state.period.start.strftime(
            "%Y-%m"
        ) and turn_request_period["end"] == previous_next_state.period.end.strftime("%Y-%m")
        checks.append(_check("same_period_as_previous", same))

    if (
        expected.get("period_changed_from_previous")
        and turn_request_period is not None
        and previous_next_state is not None
    ):
        changed = turn_request_period["start"] != previous_next_state.period.start.strftime(
            "%Y-%m"
        ) or (turn_request_period["end"] != previous_next_state.period.end.strftime("%Y-%m"))
        checks.append(_check("period_changed_from_previous", changed))

    if expected.get("topic_freshly_resolved"):
        fresh = result.next_state.version == 1 and (
            previous_next_state is None
            or result.next_state.topic_id != previous_next_state.topic_id
        )
        checks.append(_check("topic_freshly_resolved", fresh))

    return checks


def _adversarial_grader_checks(
    expected: dict[str, Any], turn: dict[str, Any], compact_result: dict[str, Any]
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    omission_style = bool(expected.get("required_caveat_keywords"))
    if "bad_answer" in turn and not omission_style:
        report = graders.grade_answer(turn["bad_answer"], compact_result)
        caught = bool(report.forbidden_claims or report.reliability_violations)
        checks.append(
            CheckResult(
                "grader_catches_bad_answer",
                "pass" if caught else "fail",
                f"forbidden={report.forbidden_claims} reliability={report.reliability_violations}",
            )
        )
    if "good_answer" in turn:
        report = graders.grade_answer(turn["good_answer"], compact_result)
        clean = not (report.forbidden_claims or report.reliability_violations)
        checks.append(
            CheckResult(
                "grader_clears_good_answer",
                "pass" if clean else "fail",
                f"forbidden={report.forbidden_claims} reliability={report.reliability_violations}",
            )
        )
    keywords = expected.get("required_caveat_keywords")
    if keywords and "bad_answer" in turn and "good_answer" in turn:
        bad_missing = not graders.contains_any(turn["bad_answer"], keywords)
        checks.append(
            CheckResult("grader_catches_missing_caveat", "pass" if bad_missing else "fail")
        )
        good_present = graders.contains_any(turn["good_answer"], keywords)
        checks.append(
            CheckResult("grader_clears_caveat_present", "pass" if good_present else "fail")
        )
    return checks


def run_scenario(scenario: dict[str, Any]) -> ScenarioReport:
    report = ScenarioReport(scenario_id=scenario["id"], category=scenario["category"])
    env: Env | None = None
    previous_next_state: AnalysisState | None = None
    turn_results: dict[str, RunResult] = {}

    try:
        for i, turn in enumerate(scenario["turns"]):
            simulated_request = turn.get("simulated_request")
            if simulated_request is None:
                turn_report = TurnReport(turn_index=i)
                for name, verdict in expected_bool_as_offline_na(turn.get("expected", {})):
                    turn_report.checks.append(CheckResult(name, verdict))
                report.turns.append(turn_report)
                continue

            fixture = turn.get("fixture")
            if fixture is not None:
                if env is not None:
                    env.close()
                env = build_env(fixture)
            if env is None:
                raise AssertionError(f"turn {i} has no fixture and no prior turn to reuse")

            previous_state = None
            ps_marker = simulated_request.get("previous_state")
            if ps_marker is not None:
                previous_state = (
                    turn_results[ps_marker].next_state if ps_marker in turn_results else None
                )

            artifacts = tuple(_artifact_kind(a) for a in simulated_request.get("artifacts", []))
            resolve_start, fetch_start = len(env.resolve_requests), len(env.fetch_requests)
            expected = turn.get("expected", {})
            try:
                request = build_request(
                    topic=simulated_request["topic"],
                    languages=tuple(simulated_request["languages"]),
                    period=simulated_request["period"],
                    intent=simulated_request["intent"],
                    query_language=simulated_request["query_language"],
                    artifacts=artifacts,
                    previous_state=previous_state,
                )
            except ValidationError as exc:
                # Mirrors scripts/run_analysis.py: a malformed request is reported as
                # `invalid_input`, never silently corrected or raised past the caller.
                result: Any = invalid_input_from(exc)
            else:
                result = env.run(request)

            turn_report = TurnReport(turn_index=i, status=result.status)
            turn_report.checks.extend(
                evaluate_turn(
                    expected,
                    result,
                    resolve_requests=env.resolve_requests[resolve_start:],
                    fetch_requests=env.fetch_requests[fetch_start:],
                    previous_next_state=previous_next_state,
                    turn_request_period=simulated_request.get("period"),
                )
            )
            if isinstance(result, RunResult):
                turn_results[f"turn_{i}"] = result
                previous_next_state = result.next_state
                compact = result.model_dump(mode="json")
                turn_report.checks.extend(_adversarial_grader_checks(expected, turn, compact))
            report.turns.append(turn_report)
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()

    return report


def expected_bool_as_offline_na(expected: dict[str, Any]) -> list[tuple[str, str]]:
    """For turns with no simulated_request (activation/ambiguity-detection is model-only),
    record every expectation as not-applicable-offline rather than silently skipping it."""
    return [(key, "not_applicable_offline") for key in expected]


def _artifact_kind(name: str) -> Any:
    from wikipedia_interest.contracts import ArtifactKind

    return ArtifactKind(name)
