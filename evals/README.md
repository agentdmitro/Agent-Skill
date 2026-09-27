# Iteration 7: cheap-model reliability evaluation

This directory evaluates whether the Wikipedia Interest skill (Iterations 1-6) works reliably
when driven by a small, cheap, tool-capable model — not whether the underlying engine is
correct (that's `tests/`). Nothing here changes `src/wikipedia_interest/`; it is a separate
evaluation layer, per the iteration brief's `<evaluation_architecture>`.

## Layout

```
evals/
├── scenarios/          declarative scenario fixtures (basic, ambiguity, follow_up, adversarial, failures)
├── fixtures/            reusable mocked-HTTP Env, matching tests/test_orchestration.py's pattern
├── scenario_engine.py   Mode A: runs scenarios against the real run_analysis() engine, offline
├── graders.py           forbidden-claim / reliability-language / numeric-faithfulness checks
├── model_client.py      Mode B: optional OpenRouter tool-calling loop (opt-in, unexecuted here)
├── runner.py            CLI entry point, scoring, and report writing
└── README.md
```

## Modes

**Mode A - offline (default, no network, no API key):**

```
python evals/runner.py --mode offline
python evals/runner.py --mode offline --categories basic ambiguity
```

Builds a mocked Wikidata/Wikimedia HTTP layer per scenario (`evals/fixtures`, the same
`httpx.MockTransport` pattern `tests/test_orchestration.py` uses) and runs the *real*
`wikipedia_interest.orchestration.run_analysis` against it — never a stand-in. It checks:

- orchestration behavior (activation-adjacent structural checks, ambiguity stopping before
  any pageview fetch, partial-failure reporting, follow-up QID/period reuse);
- the compact output's shape;
- a grader regression suite: for every adversarial scenario's hand-written `bad_answer` /
  `good_answer` pair, confirms `graders.py` flags the bad one and clears the good one;
- context-size measurement (SKILL.md, compact result, follow-up state, and a raw-monthly-series
  counterfactual, to make the compact-output design's payoff visible as a number, not a claim).

Scenario turns with `"simulated_request": null` (an activation-negative prompt, or ambiguity in
missing *query_language*) have no fixture to run against — they are recorded as
`not_applicable_offline`, never silently skipped and never scored as a pass.

**Mode B - real model (opt-in, requires `OPENROUTER_API_KEY` + `EVAL_MODEL`):**

```
export OPENROUTER_API_KEY=...
export EVAL_MODEL=anthropic/claude-haiku-4.5   # or any cheap/free OpenRouter model id
python evals/runner.py --mode model --categories basic ambiguity
```

`evals/model_client.py` gives the model exactly one tool
(`run_wikipedia_interest_analysis`, mirroring `scripts/run_analysis.py`'s flags) at
`temperature=0`, executes `run_analysis` in-process against the scenario's mocked fixture when
the model calls it, and returns the compact JSON as the tool result. The model's final answer is
then graded with `evals/graders.py`.

**This mode was not executed in this iteration** (no `OPENROUTER_API_KEY` was available in this
environment). The wiring is real and unit-testable, but no real-model runs, latency, token
counts, or pass rates exist yet — do not infer any from this repository. Running it is the next
step before claiming cheap-model reliability end-to-end, not just at the orchestration layer.

## Why offline mode can still say a lot

Most of what makes a *small* model reliable here is structural, not linguistic: does ambiguity
literally stop the code path before a fetch, does a follow-up literally reuse the QID without a
new search, does a partial failure literally show up as `unavailable` rather than vanishing.
Those are properties of `run_analysis` and the scenario fixtures around it, checkable without any
model in the loop — Mode A checks exactly these, plus the grader's ability to catch bad language
if a small model ever produces it. What Mode A *cannot* check is whether a given small model
actually chooses to call the tool, extracts the right parameters from an ambiguous sentence, or
resists an adversarial prompt in its own words — that's exactly what Mode B is for.

## Scoring

`evals/runner.py` reports dimensions separately (`<scoring>`: no single opaque quality score):
`overall_scenario_pass_rate`, `overall_check_pass_rate`, per-category pass rates, and the
grader's own catch/clear rates. A small set of checks are treated as release-blocking
(`critical_gate_failures` / `critical_gate_passed` in the summary): clarification behavior,
follow-up topic identity, and the grader's bad/good discrimination. See `runner.py`'s
`critical_gate_failures()` for the exact list.

## Adding a scenario

Add an entry to the right file in `evals/scenarios/`. Minimum shape:

```json
{
  "id": "unique_id",
  "category": "basic|ambiguity|follow_up|adversarial|failures",
  "user_messages": ["what the user said, for documentation and Mode B"],
  "turns": [
    {
      "simulated_request": {"topic": "...", "query_language": "en", "languages": ["en"], "period": {"start": "2024-01", "end": "2025-12"}, "intent": "trend"},
      "fixture": {"qid": "Q...", "label": "...", "sitelinks": {"en": "..."}, "trend": "growing"},
      "expected": {"status": "success", "expected_topic_id": "Q..."}
    }
  ]
}
```

See `scenario_engine.py`'s `evaluate_turn` for every supported `expected` key. For a follow-up
turn, set `simulated_request.previous_state` to `"turn_<n>"` and omit `fixture` to reuse the
prior turn's mocked environment. For an adversarial scenario, add `bad_answer` / `good_answer`
strings to get automatic grader regression coverage for free.
