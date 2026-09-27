"""Primary V1 CLI entry point: end-to-end Wikipedia interest analysis.

No business logic lives here; it parses arguments, builds the typed request, and
delegates to `wikipedia_interest.orchestration.run_analysis`.
"""

import argparse
import json
import sys

from pydantic import ValidationError

from wikipedia_interest import (
    AnalysisRequest,
    AnalysisState,
    ArtifactKind,
    Period,
    RunAnalysisRequest,
    TopicResolver,
    WikimediaClient,
    invalid_input_from,
    run_analysis,
)

_EXIT_BY_STATUS = {
    "success": 0,
    "clarification_required": 0,
    "unsupported": 1,
    "invalid_input": 2,
    "upstream_failure": 3,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", required=True, help="Free text, or a Wikidata QID (e.g. Q333)")
    ap.add_argument("--query-language", required=True)
    ap.add_argument("--languages", nargs="+", required=True, help="Target Wikipedia editions")
    ap.add_argument("--start", required=True, help="YYYY-MM (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM (inclusive)")
    ap.add_argument("--intent", choices=["trend", "compare"], required=True)
    ap.add_argument("--chart", action="store_true", help="Generate a PNG chart artifact")
    ap.add_argument("--pdf", action="store_true", help="Generate a one-page PDF report artifact")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument(
        "--previous-state",
        help="Path to a JSON file holding a previously returned `next_state` (follow-ups)",
    )
    args = ap.parse_args()

    artifacts: list[ArtifactKind] = []
    if args.chart:
        artifacts.append(ArtifactKind.CHART)
    if args.pdf:
        artifacts.append(ArtifactKind.REPORT)

    try:
        previous_state = None
        if args.previous_state:
            with open(args.previous_state, encoding="utf-8") as handle:
                previous_state = AnalysisState.model_validate(json.load(handle))
        request = RunAnalysisRequest(
            request=AnalysisRequest(
                topic=args.topic,
                languages=tuple(args.languages),
                period=Period.model_validate({"start": args.start, "end": args.end}),
                analysis=args.intent,
                previous_state=previous_state,
            ),
            query_language=args.query_language,
            artifacts=tuple(artifacts),
            output_dir=args.output_dir,
        )
    except ValidationError as exc:
        print(json.dumps(invalid_input_from(exc).model_dump(mode="json"), indent=2))
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        issue = {"field": "previous_state", "message": str(exc)}
        print(json.dumps({"status": "invalid_input", "issues": [issue]}))
        return 2

    resolver = TopicResolver()
    client = WikimediaClient()
    try:
        result = run_analysis(request, resolver=resolver, client=client)
    finally:
        resolver.close()
        client.close()

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return _EXIT_BY_STATUS[result.status]


if __name__ == "__main__":
    sys.exit(main())
