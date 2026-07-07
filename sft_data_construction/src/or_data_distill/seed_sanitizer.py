from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_json, write_jsonl
from .ir_schema import infer_data_interface, normalize_mode


QUESTION_MARKERS = (
    "Answer the following mathematical modeling question:",
    "Mathematical modeling question:",
    "Optimization problem:",
    "Problem:",
    "Question:",
)


def _last_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def _last_user(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def strip_wrapped_prompt(text: str) -> str:
    text = text.replace("\r\n", "\n")
    for marker in QUESTION_MARKERS:
        if marker in text:
            text = text.split(marker)[-1]
    lines = []
    skip_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\[[A-Z_ ]+\]", stripped):
            skip_block = True
            continue
        if skip_block and not stripped:
            skip_block = False
            continue
        if skip_block:
            continue
        if stripped.startswith("[") and "]" in stripped[:80]:
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def sanitize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    messages = row.get("messages")
    if isinstance(messages, list):
        problem = strip_wrapped_prompt(_last_user(messages))
        answer = _last_assistant(messages)
    else:
        problem = strip_wrapped_prompt(str(row.get("problem") or row.get("question") or row.get("input") or ""))
        answer = str(row.get("answer") or row.get("output") or row.get("response") or "")

    old_meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    mode = normalize_mode(str(row.get("mode") or row.get("problem_mode") or old_meta.get("mode") or "DP"))
    metadata = {
        "mode": mode,
        "domain": str(old_meta.get("domain") or row.get("domain") or "generic"),
        "structure": str(old_meta.get("structure") or row.get("structure") or "generic_lp_mip"),
        "difficulty": str(old_meta.get("difficulty") or row.get("difficulty") or "medium"),
        "data_interface": str(
            old_meta.get("data_interface") or row.get("data_interface") or infer_data_interface(mode, problem)
        ),
        "answer_style": str(old_meta.get("answer_style") or row.get("answer_style") or "math_model_with_code"),
        "seed_stage": "sanitized",
    }
    payload = {"problem": problem, "answer": answer, "metadata": metadata, "index": index}
    return {
        "id": f"seed_{sha256_json(payload)[:16]}",
        "problem": problem,
        "answer": answer,
        "metadata": metadata,
    }


def sanitize_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    clean_rows = [sanitize_row(row, idx) for idx, row in enumerate(rows)]
    write_jsonl(output_path, clean_rows)
    return {"input_rows": len(rows), "output_rows": len(clean_rows), "output": str(output_path)}

