from __future__ import annotations

import json
from typing import Any

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


def _default_serializer(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def dumps(payload: Any) -> bytes:
    if _orjson is not None:
        return _orjson.dumps(payload)
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_default_serializer,
    ).encode("utf-8")


def loads(payload: bytes | str) -> Any:
    if _orjson is not None:
        return _orjson.loads(payload)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)
