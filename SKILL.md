---
name: wikipedia-interest
description: Analyze Wikipedia pageview interest in a topic across languages and time (trend, cross-language comparison), with optional chart/PDF artifacts. Use when the user asks how public interest in a topic changed or differs between Wikipedia language editions. Not for purchase intent, sales, market-size, or forecasting questions.
---

# Wikipedia Interest

## Use for

- "How did interest in X change between <month> and <month>?" (`trend`)
- "Compare interest in X across <languages>." (`compare`)
- Follow-ups that change the period, or add/remove a language.
- Requests to also produce a chart (`--chart`) or a one-page PDF brief (`--pdf`).

## Not for

- Purchase intent, demand, market size, revenue, or "should I launch in X" questions.
- Forecasts ("predict next year", "how many users will buy"). Say plainly this skill does not answer that, then offer the supported Wikipedia-interest analysis for the same topic instead.

## Entry point

Run `scripts/run_analysis.py`. Do not call the resolver/analysis/comparison modules directly and do not compute anything yourself — every number in the result came from that script.

Always execute repository Python entry points through `uv run`. Never invoke them with bare `python`.

Example: `uv run python scripts/run_analysis.py ...`

Required: `--topic`, `--query-language`, `--languages` (one or more), `--start`, `--end` (both `YYYY-MM`), `--intent` (`trend` or `compare`). Optional: `--chart`, `--pdf`, `--output-dir`, `--previous-state <path to a saved next_state JSON>`.

## Extracting parameters

1. `topic`: the user's subject as free text, or a Wikidata QID (e.g. `Q333`) if the user gave one.
2. `query_language`: the language to resolve the topic's label in. If the user provides a natural-language topic, infer it from the language of that topic phrase (for example, `astronomy` → `en`, `астрономія` → `uk`, and `astronomia` → `pl`). Do not infer it from the requested Wikipedia editions. Ask the user only if the topic language is genuinely ambiguous or cannot be identified confidently. If the topic is already a Wikidata QID, do not ask for `query_language` merely to resolve the entity.
3. `languages`: the target Wikipedia editions (ISO codes, e.g. `en`, `pl`, `uk`).
4. `period`: `start`/`end` months. If unstated, ask; do not guess a default range.
5. `intent`: `trend` for one language, `compare` for two or more.

## Reading the result (`status` field)

- `success`: explain only `languages`, `comparison`, `findings`, `unavailable`, and `artifacts` from the JSON. Never restate raw numbers you weren't given, and never open or re-derive from the underlying pageview series.
- `clarification_required`: the topic is ambiguous. Ask the user **one** short question listing `candidates` by label/description. Never pick one yourself, and never proceed to analysis.
- `unsupported`: the request is well-formed but cannot be served (e.g. no verified article in any requested language, or `previous_state` doesn't match the new topic). Report the reason plainly; do not retry with a guessed workaround.
- `invalid_input`: a validation error (e.g. period before 2015-07, malformed language code). Report the listed `issues`; do not silently correct them.
- `upstream_failure`: Wikimedia/Wikidata could not be reached. State whether `retryable` is true; do not fabricate data in its place.

## Partial success

A `success` result can still have `unavailable` languages (domain-unavailable, e.g. no sitelink, or upstream-failed, e.g. timeout) alongside successfully analyzed ones. Always report the full picture: how many languages were requested, how many were analyzed, which were unavailable and why, and whether a retry might help (`retryable` on each unavailable entry). Never present the result as if only the successful languages were requested.

## Follow-ups

Every `success` result carries `next_state`. To continue the same investigation (add a language, change the period, switch trend/compare), pass the previous `next_state` back as `--previous-state`; do not re-type the original free-text topic search. The engine reuses the same Wikidata entity and rejects the follow-up if the new topic text doesn't match it.

## Anti-hallucination rules

1. Never invent a Wikipedia title, a Wikidata ID, a number, or a cause for a spike or drop.
2. Never recompute or restate a trend, growth rate, or reliability grade yourself — use exactly what the result gives you.
3. `reliability` is an evidence grade, never a probability.
4. `candidate_classification` (e.g. `strong_candidate`) means only "this language edition's Wikipedia interest signal may be worth further validation" — never a launch or market recommendation, and never based on pageview size alone.
5. In a comparison, treat absolute attention (pageviews), trend, and reliability as three separate dimensions. A language with more pageviews or a bigger percentage change is never a "better market" — raw pageviews across editions are not comparable as market size (audience size, usage habits, and article coverage differ by edition).
6. Never invent an overall "best language" or cross-dimension ranking. If a finding names a specific metric (e.g. `total_views`, `normalized_trend_slope`), report only that metric.
7. When multiple languages share the same `candidate_classification` (for example, all are `weak_current_signal`), do not invent a first/second/best validation priority between them. You may report metric-specific differences such as highest absolute attention, least severe decline, or higher reliability, but keep those dimensions separate and do not turn them into an overall recommendation unless the deterministic result explicitly provides one.
8. If data is missing, say so plainly; `insufficient_data` and low/insufficient reliability must be reported honestly, never smoothed over.
9. Do not explain why a reliability grade is high, medium, or low unless the deterministic result explicitly provides that reason in its evidence. Report related metrics such as volatility or outliers separately rather than inferring causation.

## Limitations

Wikipedia pageviews are an interest signal, not purchase intent, demand, or market size. Language editions differ in audience size and usage habits, so raw pageviews across editions are not comparable as market size. Correlation in the data never establishes a cause.
