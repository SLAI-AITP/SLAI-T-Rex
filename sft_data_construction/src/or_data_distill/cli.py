from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline
from .quality_gate import validate_sft_record
from .seed_sanitizer import sanitize_file
from .io_utils import read_jsonl
from .generic_ir import extract_ir_file


def _validation_probe(row: dict) -> dict:
    if isinstance(row.get("messages"), list):
        return row
    if "problem" in row or "answer" in row:
        return {
            "id": row.get("id"),
            "messages": [
                {"role": "user", "content": str(row.get("problem") or "")},
                {"role": "assistant", "content": str(row.get("answer") or "")},
            ],
            "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic OR modeling data distillation toolkit.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sanitize-seeds", help="Convert existing problem-answer data into clean seed JSONL.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("run", help="Run the distillation pipeline.")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("extract-ir", help="Convert sanitized seeds into generic modeling IR.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("validate-sft", help="Validate exported SFT JSONL.")
    p.add_argument("--input", required=True)

    args = parser.parse_args()
    if args.cmd == "sanitize-seeds":
        print(json.dumps(sanitize_file(args.input, args.output), ensure_ascii=False, indent=2))
    elif args.cmd == "extract-ir":
        print(json.dumps(extract_ir_file(args.input, args.output), ensure_ascii=False, indent=2))
    elif args.cmd == "run":
        print(json.dumps(run_pipeline(args.config, dry_run=args.dry_run), ensure_ascii=False, indent=2))
    elif args.cmd == "validate-sft":
        rows = read_jsonl(Path(args.input))
        issue_count = 0
        for row in rows:
            issues = validate_sft_record(_validation_probe(row))
            if issues:
                issue_count += 1
                print(json.dumps({"id": row.get("id"), "issues": issues}, ensure_ascii=False))
        print(json.dumps({"rows": len(rows), "rows_with_issues": issue_count}, ensure_ascii=False, indent=2))
