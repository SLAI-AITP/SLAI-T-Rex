from __future__ import annotations

import ast
import json
import re
from typing import Any

_FENCED_CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", re.DOTALL)
_STRUCTURED_CODE_KEYS = ("code", "snippet", "text")


def unwrap_code_payload(value: Any) -> str | None:
    if value is None:
        return None

    current = value
    for _ in range(3):
        if isinstance(current, dict):
            nested = _first_code_value(current)
            if nested is None:
                return _stringify_code_value(current)
            current = nested
            continue

        if isinstance(current, str):
            stripped = current.strip()
            if not stripped:
                return None
            parsed = _parse_structured_code_string(stripped)
            if isinstance(parsed, dict):
                nested = _first_code_value(parsed)
                if nested is not None:
                    current = nested
                    continue
            return stripped

        return _stringify_code_value(current)

    return _stringify_code_value(current)


def extract_executable_code(text: Any) -> str | None:
    unwrapped = unwrap_code_payload(text)
    if unwrapped is None:
        return None

    blocks = [match.strip() for match in _FENCED_CODE_RE.findall(unwrapped) if match.strip()]
    if blocks:
        return "\n\n".join(blocks)
    return unwrapped


def _parse_structured_code_string(text: str) -> dict[str, Any] | None:
    if not text.startswith("{") or not text.endswith("}"):
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            payload = loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _first_code_value(payload: dict[str, Any]) -> Any:
    for key in _STRUCTURED_CODE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _stringify_code_value(value: Any) -> str | None:
    text = str(value).strip()
    return text or None
