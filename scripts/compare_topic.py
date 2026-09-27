"""Developer check: resolve a topic, fetch pageviews for each language, analyze, compare.

No business logic lives here; it wires resolver -> client -> orchestration.run_comparison.
"""

import argparse
import json
import sys

from pydantic import ValidationError

from wikipedia_interest import Period, TopicResolver, WikimediaClient, invalid_input_from
from wikipedia_interest.orchestration import run_comparison


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--query-language", required=True)
    ap.add_argument("--languages", nargs="+", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM (inclusive)")
    args = ap.parse_args()
    try:
        period = Period.model_validate({"start": args.start, "end": args.end})
    except ValidationError as exc:
        print(json.dumps(invalid_input_from(exc).model_dump(mode="json"), indent=2))
        return 2

    resolver = TopicResolver()
    client = WikimediaClient()
    try:
        result = run_comparison(
            topic=args.topic,
            query_language=args.query_language,
            languages=tuple(args.languages),
            period=period,
            resolver=resolver,
            client=client,
        )
    finally:
        resolver.close()
        client.close()

    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.status in {"success", "clarification_required"} else 1


if __name__ == "__main__":
    sys.exit(main())
