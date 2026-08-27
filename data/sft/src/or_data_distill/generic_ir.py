from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_json, write_jsonl
from .ir_schema import infer_data_interface, normalize_mode


def seed_to_ir(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    problem = str(row.get("problem") or "")
    answer = str(row.get("answer") or "")
    mode = normalize_mode(str(metadata.get("mode") or "DP"))
    ir = {
        "mode": mode,
        "domain": str(metadata.get("domain") or "generic"),
        "structure": str(metadata.get("structure") or "generic_lp_mip"),
        "difficulty": str(metadata.get("difficulty") or "medium"),
        "data_interface": str(metadata.get("data_interface") or infer_data_interface(mode, problem)),
        "answer_style": str(metadata.get("answer_style") or "math_model_with_code"),
        "source_problem": problem,
        "source_answer": answer,
        "objective": metadata.get("objective") or "infer from source problem",
        "decision_variables": metadata.get("decision_variables") or ["infer from source answer"],
        "parameters": metadata.get("parameters") or ["infer from source problem"],
        "constraints": metadata.get("constraints") or ["infer from source answer"],
    }
    ir["id"] = f"ir_{sha256_json(ir)[:16]}"
    return ir


def extract_ir_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    ir_rows = [seed_to_ir(row) for row in rows]
    write_jsonl(output_path, ir_rows)
    return {"input_rows": len(rows), "ir_rows": len(ir_rows), "output": str(output_path)}

