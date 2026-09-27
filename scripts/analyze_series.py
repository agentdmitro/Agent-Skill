"""Developer check: fetch and analyze one verified Wikipedia article."""

import argparse
import sys

from pydantic import ValidationError

from wikipedia_interest import Period, WikimediaClient, analyze_pageview_series, invalid_input_from
from wikipedia_interest.contracts import PageviewSeries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--language", required=True)
    ap.add_argument("--article", required=True, help="Verified Wikipedia article title")
    ap.add_argument("--start", required=True, help="YYYY-MM (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM (inclusive)")
    args = ap.parse_args()
    try:
        period = Period.model_validate({"start": args.start, "end": args.end})
    except ValidationError as exc:
        print(invalid_input_from(exc).model_dump_json(indent=2), file=sys.stderr)
        return 2
    client = WikimediaClient()
    try:
        result = client.fetch_monthly_pageviews(args.language, args.article, period)
    finally:
        client.close()
    if not isinstance(result, PageviewSeries):
        print(result.model_dump_json(indent=2), file=sys.stderr)
        return 1
    print(analyze_pageview_series(result).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
