from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from cpt_cleaner.utils import json_compat


def iter_jsonl(path: str | Path):
    file_path = Path(path)
    if not file_path.exists():
        return
    with file_path.open("rb") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json_compat.loads(stripped)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path) or [])


def write_jsonl(path: str | Path, rows: Iterable[dict | BaseModel]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json_compat.dumps(payload))
            handle.write(b"\n")


def append_jsonl(path: str | Path, row: dict | BaseModel) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
    with file_path.open("ab") as handle:
        handle.write(json_compat.dumps(payload))
        handle.write(b"\n")


def count_jsonl_records(path: str | Path) -> int:
    """Count non-empty lines in a JSONL file without parsing JSON."""
    file_path = Path(path)
    if not file_path.exists():
        return 0
    count = 0
    with file_path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_text(path: str | Path, text: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
