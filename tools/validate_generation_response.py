#!/usr/bin/env python3
"""Validate one saved OpenAI chat completion against a deterministic text oracle.

This offline check does not prove HTTP status, listener ownership, cache hits,
model accuracy on a dataset, or service cleanup. Preserve those separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Sequence


def _has_controls(text: str) -> bool:
    return any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in text)


def validate_response(
    response: Any,
    expected_final_texts: Sequence[str],
    *,
    allow_think_prefix: bool = False,
    expected_model: str | None = None,
) -> dict:
    """Check a single plain-text answer; inline reasoning requires explicit opt-in."""
    if isinstance(expected_final_texts, str) or not expected_final_texts:
        raise ValueError("supply a nonempty sequence of expected final texts")
    if any(not isinstance(s, str) or not s.strip() or _has_controls(s)
           for s in expected_final_texts):
        raise ValueError("expected final texts must be nonempty text without control characters")
    if expected_model is not None and not expected_model.strip():
        raise ValueError("expected model must be nonempty")
    expected = [s.strip() for s in expected_final_texts]
    result = {
        "status": "failed", "expected_final_texts": expected,
        "expected_model": expected_model, "allow_think_prefix": allow_think_prefix,
        "model": None, "finish_reason": None, "final_text": None,
        "separate_reasoning": False, "errors": [],
    }

    def fail(field, reason):
        result["errors"].append({"field": field, "reason": reason})

    if not isinstance(response, dict):
        fail("response", "expected a chat-completion object")
        return result
    model = response.get("model")
    result["model"] = model
    if not isinstance(model, str) or not model.strip():
        fail("model", "missing model identity")
    if expected_model is not None and model != expected_model:
        fail("model", "response is from a different model")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        fail("choices", "expected exactly one non-streaming completion")
        return result
    choice = choices[0]
    result["finish_reason"] = choice.get("finish_reason")
    if result["finish_reason"] != "stop":
        fail("finish_reason", "expected normal stop; truncation is not a correct answer")
    message = choice.get("message")
    if not isinstance(message, dict):
        fail("message", "missing assistant message")
        return result
    if message.get("role") != "assistant":
        fail("message.role", "expected assistant")
    if message.get("tool_calls") or message.get("refusal"):
        fail("message", "expected a plain answer, not a tool call or refusal")
    reasoning = message.get("reasoning_content")
    result["separate_reasoning"] = isinstance(reasoning, str) and bool(reasoning)
    if reasoning is not None and (not isinstance(reasoning, str) or _has_controls(reasoning)):
        fail("message.reasoning_content", "expected text without control characters")
    text = message.get("content")
    if not isinstance(text, str):
        fail("message.content", "expected text content")
        return result
    if _has_controls(text):
        fail("message.content", "unexpected control characters")
    text = text.strip()
    if "<think>" in text or "</think>" in text:
        if not allow_think_prefix:
            fail("message.content", "inline reasoning requires explicit --allow-think-prefix")
        elif (not text.startswith("<think>") or text.count("<think>") != 1
              or text.count("</think>") != 1):
            fail("message.content", "incomplete or ambiguous inline reasoning")
        else:
            text = text.split("</think>", 1)[1].strip()
    result["final_text"] = text
    if text not in expected:
        fail("message.content", "final text does not exactly match any expected answer")
    if not result["errors"]:
        result["status"] = "passed"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True, help="saved non-streaming chat-completion JSON")
    parser.add_argument("--expected-final-text", action="append", required=True,
                        help="exact accepted final text; repeat for explicitly accepted alternatives")
    parser.add_argument("--expected-model", help="require this exact response model identity")
    parser.add_argument("--allow-think-prefix", action="store_true",
                        help="allow one closed leading <think>...</think> block before the final answer")
    parser.add_argument("--output", type=Path, help="write a separate JSON report")
    args = parser.parse_args(argv)
    if args.output is not None and (args.output.resolve() == args.response.resolve()
            or (args.output.exists() and args.response.exists()
                and args.output.samefile(args.response))):
        parser.error("output must not overwrite the raw response")
    try:
        # Validate caller configuration before classifying input failures.
        validate_response({}, args.expected_final_text,
                          expected_model=args.expected_model)
    except ValueError as exc:
        parser.error(str(exc))
    digest = None
    try:
        raw = args.response.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        result = validate_response(json.loads(raw.decode("utf-8")), args.expected_final_text,
                                   allow_think_prefix=args.allow_think_prefix,
                                   expected_model=args.expected_model)
    except (OSError, UnicodeError, ValueError) as exc:
        result = {"status": "failed", "errors": [{"field": "response", "reason": str(exc)}]}
    result.update(response=str(args.response.resolve()), response_sha256=digest)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        except OSError as exc:
            print(f"Cannot write report: {exc}", file=sys.stderr)
            return 2
    print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
