"""Optional Mode-B runner: drive a real, cheap tool-capable model through OpenRouter.

Opt-in only. Requires OPENROUTER_API_KEY; EVAL_MODEL selects the model (any OpenRouter model
id, e.g. "anthropic/claude-haiku-4.5" or a free/cheap model such as
"meta-llama/llama-3.1-8b-instruct:free"). OPENROUTER_BASE_URL overrides the API base
(default "https://openrouter.ai/api/v1") for compatible providers.

The model is given exactly one tool, `run_wikipedia_interest_analysis`, whose schema mirrors
`scripts/run_analysis.py`'s CLI flags. When the model calls it, this module executes the real
deterministic `run_analysis` engine in-process (against the scenario's mocked fixture, never
live Wikimedia/Wikidata) and returns the compact JSON result as the tool response. The model
then produces its final natural-language answer, which the runner grades with `evals.graders`.

No network calls happen here unless OPENROUTER_API_KEY is set. No user/private data is used -
scenario prompts are synthetic and public.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.0  # tool reliability, not creative diversity; see <temperature> in the brief

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_wikipedia_interest_analysis",
        "description": (
            "Deterministically analyze or compare Wikipedia pageview interest in a topic. "
            "Never estimate this yourself; always call this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Free text, or a Wikidata QID like Q333",
                },
                "query_language": {"type": "string"},
                "languages": {"type": "array", "items": {"type": "string"}},
                "start": {"type": "string", "description": "YYYY-MM"},
                "end": {"type": "string", "description": "YYYY-MM"},
                "intent": {"type": "string", "enum": ["trend", "compare"]},
                "previous_state": {
                    "type": "object",
                    "description": "Pass back a prior next_state verbatim for a follow-up.",
                },
            },
            "required": ["topic", "query_language", "languages", "start", "end", "intent"],
        },
    },
}


def is_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


@dataclass
class ModelRunResult:
    model: str
    final_text: str
    tool_calls: list[dict[str, Any]]
    activated: bool
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class OpenRouterClient:
    def __init__(self, *, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.environ.get("EVAL_MODEL", "").strip()
        if not self.model:
            raise RuntimeError("EVAL_MODEL is not set")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._http = httpx.Client(
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(60.0),
        )

    def close(self) -> None:
        self._http.close()

    def run(
        self,
        *,
        system_prompt: str,
        user_messages: list[str],
        tool_executor: Any,
        max_tool_rounds: int = 3,
    ) -> ModelRunResult:
        """One conversation: user turns are sent in order; the tool loop resolves each model
        tool call via `tool_executor(arguments) -> dict` before asking for the final answer."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        tool_calls_seen: list[dict[str, Any]] = []
        start = time.monotonic()
        prompt_tokens = completion_tokens = None

        for user_message in user_messages:
            messages.append({"role": "user", "content": user_message})
            for _ in range(max_tool_rounds):
                response = self._complete(messages)
                usage = response.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                choice = response["choices"][0]["message"]
                messages.append(choice)
                calls = choice.get("tool_calls") or []
                if not calls:
                    break
                for call in calls:
                    args = json.loads(call["function"]["arguments"] or "{}")
                    tool_calls_seen.append({"name": call["function"]["name"], "arguments": args})
                    result = tool_executor(args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

        final_text = next(
            (
                m["content"]
                for m in reversed(messages)
                if m.get("role") == "assistant" and m.get("content")
            ),
            "",
        )
        return ModelRunResult(
            model=self.model,
            final_text=final_text,
            tool_calls=tool_calls_seen,
            activated=bool(tool_calls_seen),
            latency_seconds=time.monotonic() - start,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": [TOOL_SCHEMA],
            "temperature": TEMPERATURE,
        }
        response = self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body
