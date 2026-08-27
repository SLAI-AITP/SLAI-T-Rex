from __future__ import annotations

from or_cpt_engine.adapters.optmath_adapter import OptMATHGeneratorAdapter


def build_adapter(registry_record: dict):
    source = registry_record.get("source", "optmath")
    if source in {"optmath", "optmath_internal"}:
        return OptMATHGeneratorAdapter(registry_record)
    raise ValueError(f"unsupported generator source: {source}")
