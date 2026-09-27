#!/usr/bin/env python3
"""Resolve a topic to verified Wikimedia article titles."""

import argparse
import json
import sys

from pydantic import ValidationError

from wikipedia_interest import TopicResolutionRequest, TopicResolver, invalid_input_from


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--query-language", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    args = parser.parse_args()
    try:
        request = TopicResolutionRequest(
            topic=args.topic, query_language=args.query_language, target_languages=args.languages
        )
    except ValidationError as exc:
        print(
            json.dumps(
                invalid_input_from(exc).model_dump(mode="json"), ensure_ascii=False, indent=2
            )
        )
        return 2
    resolver = TopicResolver()
    try:
        result = resolver.resolve(request)
    finally:
        resolver.close()
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.status in {"success", "clarification_required"} else 1


if __name__ == "__main__":
    sys.exit(main())
