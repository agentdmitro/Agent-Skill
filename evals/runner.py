#!/usr/bin/env python
"""Iteration 7 evaluation CLI.

    python evals/runner.py --mode offline
    python evals/runner.py --mode offline --categories basic ambiguity
    python evals/runner.py --mode model --model "$EVAL_MODEL" --categories basic ambiguity

Offline mode needs no network access, no API key, and is the default CI command. Model mode
is explicitly opt-in (see evals/model_client.py) and is skipped with a clear message if
OPENROUTER_API_KEY / EVAL_MODEL are not set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python evals/runner.py` as well as `python -m evals.runner`
    sys.path.insert(0, str(ROOT))

from evals.scenario_engine import ScenarioReport, load_scenarios, run_scenario  # noqa: E402

RESULTS_DIR = ROOT / "eval-results"


# --- Token / context-size measurement (character + JSON-byte counts; see <token_measurement>) --


def _size(text: str) -> dict[str, int]:
    return {"characters": len(text), "utf8_bytes": len(text.encode("utf-8"))}


def measure_context_sizes() -> dict[str, Any]:
    skill_md = (ROOT / "SKILL.md").read_text(encoding="utf-8")

    from evals.scenario_engine import build_env as build_scenario_env

    env = build_scenario_env(
        {
            "qid": "Q1360827",
            "label": "Intermittent fasting",
            "sitelinks": {"pl": "Przerywany post", "cs": "Prerušovaný půst"},
            "trend": "growing",
        }
    )
    try:
        from evals.fixtures import build_request

        request = build_request(
            topic="Intermittent fasting",
            languages=("pl", "cs"),
            period={"start": "2024-01", "end": "2025-12"},
            intent="compare",
            query_language="en",
        )
        result = env.run(request)
        if result.status != "success":
            raise AssertionError(f"context-size fixture unexpectedly returned {result.status}")

        # What SKILL.md actually tells the model to read from a `success` result: languages,
        # comparison, findings, unavailable, artifacts - not the full envelope (request/resolved
        # echo, next_state) it also carries for orchestration bookkeeping.
        full_dump = result.model_dump(mode="json")
        compact_for_model = {
            k: full_dump[k]
            for k in ("languages", "comparison", "findings", "unavailable", "artifacts")
        }
        compact_json = json.dumps(compact_for_model, ensure_ascii=False)
        next_state_json = json.dumps(result.next_state.model_dump(mode="json"), ensure_ascii=False)

        # The naive alternative this design avoids: handing the model the raw monthly Wikimedia
        # pageview items for every language directly (see <raw_vs_compact_test>).
        raw_items = {
            lang: [
                {
                    "project": f"{lang}.wikipedia.org",
                    "article": title.replace(" ", "_"),
                    "granularity": "monthly",
                    "timestamp": f"2024{m:02d}0100" if m <= 12 else f"2025{m - 12:02d}0100",
                    "access": "all-access",
                    "agent": "user",
                    "views": 800 + i * 20,
                }
                for m in range(1, 25)
            ]
            for i, (lang, title) in enumerate(
                {"pl": "Przerywany post", "cs": "Prerušovaný půst"}.items()
            )
        }
        raw_json = json.dumps(raw_items, ensure_ascii=False)
    finally:
        env.close()

    initial_user_request = (
        "Compare growth in interest in intermittent fasting in Polish and Czech Wikipedia "
        "over the last two years."
    )

    return {
        "skill_md": _size(skill_md),
        "initial_user_request": _size(initial_user_request),
        "compact_deterministic_result": _size(compact_json),
        "previous_state_followup_context": _size(next_state_json),
        "raw_monthly_series_equivalent": _size(raw_json),
        "compact_vs_raw_reduction_ratio": round(1 - len(compact_json) / len(raw_json), 4),
        "note": (
            "Character/byte counts, not billed tokens (no tokenizer dependency added; see "
            "<token_measurement>). compact_deterministic_result is exactly the subset SKILL.md "
            "tells the model to read (languages/comparison/findings/unavailable/artifacts); "
            "raw_monthly_series_equivalent is the naive alternative this design avoids: the raw "
            "monthly Wikimedia items for every requested language, same scenario, 24 months each."
        ),
    }


# --- Scoring dimensions (see <scoring>: kept separate, never collapsed into one score) ---------


def _dimension_rate(reports: list[ScenarioReport], check_name: str) -> dict[str, Any]:
    relevant = [c for r in reports for t in r.turns for c in t.checks if c.name == check_name]
    applicable = [c for c in relevant if c.verdict != "not_applicable_offline"]
    if not applicable:
        return {
            "pass_rate": None,
            "n": 0,
            "note": "no offline-applicable checks for this dimension",
        }
    passed = sum(1 for c in applicable if c.verdict == "pass")
    return {"pass_rate": round(passed / len(applicable), 4), "n": len(applicable)}


def compute_scores(reports: list[ScenarioReport]) -> dict[str, Any]:
    by_category: dict[str, list[ScenarioReport]] = {}
    for r in reports:
        by_category.setdefault(r.category, []).append(r)

    total_checks = [c for r in reports for t in r.turns for c in t.checks]
    applicable_checks = [c for c in total_checks if c.verdict != "not_applicable_offline"]
    scenario_pass_rate = (
        round(sum(1 for r in reports if r.passed) / len(reports), 4) if reports else None
    )

    return {
        "overall_scenario_pass_rate": scenario_pass_rate,
        "overall_check_pass_rate": (
            round(
                sum(1 for c in applicable_checks if c.verdict == "pass") / len(applicable_checks), 4
            )
            if applicable_checks
            else None
        ),
        "ambiguity_handling_pass_rate": _dimension_rate(
            by_category.get("ambiguity", []), "requires_clarification"
        ),
        "follow_up_topic_identity_pass_rate": _dimension_rate(
            by_category.get("follow_up", []), "expected_topic_id"
        ),
        "partial_failure_handling_pass_rate": _dimension_rate(
            by_category.get("failures", []), "unavailable_languages"
        ),
        "grader_catches_bad_answer_rate": _dimension_rate(reports, "grader_catches_bad_answer"),
        "grader_clears_good_answer_rate": _dimension_rate(reports, "grader_clears_good_answer"),
        "scenarios_per_category": {cat: len(rs) for cat, rs in sorted(by_category.items())},
        "note": (
            "Dimensions are reported separately, per <scoring>; there is no single opaque "
            "'AI quality score'."
        ),
    }


# --- Critical pass-gate check (see <pass_policy>) -----------------------------------------------


def critical_gate_failures(reports: list[ScenarioReport]) -> list[str]:
    failures: list[str] = []
    for r in reports:
        if r.error:
            failures.append(f"{r.scenario_id}: scenario execution error: {r.error}")
            continue
        for t in r.turns:
            for c in t.checks:
                if c.verdict != "fail":
                    continue
                if c.name in (
                    "requires_clarification",
                    "no_fetch_before_clarification",
                    "no_hidden_first_result_selection",
                    "expected_topic_id",
                    "topic_reused_not_reresolved",
                    "previous_qid_not_reused",
                    "grader_catches_bad_answer",
                    "grader_clears_good_answer",
                ):
                    failures.append(
                        f"{r.scenario_id} turn {t.turn_index}: critical check "
                        f"'{c.name}' failed ({c.detail})"
                    )
    return failures


# --- Report writing --------------------------------------------------------------------------


def report_to_dict(r: ScenarioReport) -> dict[str, Any]:
    return {
        "scenario_id": r.scenario_id,
        "category": r.category,
        "passed": r.passed,
        "error": r.error,
        "turns": [
            {
                "turn_index": t.turn_index,
                "status": t.status,
                "checks": [
                    {"name": c.name, "verdict": c.verdict, "detail": c.detail} for c in t.checks
                ],
            }
            for t in r.turns
        ],
    }


def run_offline(categories: list[str] | None) -> int:
    scenarios = load_scenarios(categories)
    reports = [run_scenario(s) for s in scenarios]

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps(report_to_dict(r), ensure_ascii=False) + "\n")

    context_sizes = measure_context_sizes()
    scores = compute_scores(reports)
    gate_failures = critical_gate_failures(reports)

    summary = {
        "mode": "offline",
        "total_scenarios": len(reports),
        "scenarios_passed": sum(1 for r in reports if r.passed),
        "scenarios_failed": sum(1 for r in reports if not r.passed),
        "scores": scores,
        "context_sizes": context_sizes,
        "critical_gate_failures": gate_failures,
        "critical_gate_passed": not gate_failures,
        "failing_scenarios": [r.scenario_id for r in reports if not r.passed],
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not gate_failures and summary["scenarios_failed"] == 0 else 1


def run_model(categories: list[str] | None, model: str | None) -> int:
    from evals.model_client import is_configured

    if not is_configured():
        print(
            json.dumps(
                {
                    "mode": "model",
                    "executed": False,
                    "reason": "OPENROUTER_API_KEY is not set; real-model evaluation is opt-in.",
                }
            )
        )
        return 0
    print(
        "Model-mode wiring (evals/model_client.py) is implemented but real-model scenario "
        "execution against it was not run in this environment. See evals/README.md."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["offline", "model"], default="offline")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.mode == "offline":
        return run_offline(args.categories)
    return run_model(args.categories, args.model)


if __name__ == "__main__":
    sys.exit(main())
