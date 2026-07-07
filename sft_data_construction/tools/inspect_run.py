#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    print(json.dumps({
        "run_dir": str(run_dir),
        "manifest": manifest,
        "requests": count_jsonl(run_dir / "requests.jsonl"),
        "attempts": count_jsonl(run_dir / "attempts.jsonl"),
        "synthetic_ir": count_jsonl(run_dir / "synthetic_ir.jsonl"),
        "accepted_synthetic_pool": count_jsonl(run_dir / "accepted_synthetic_pool.jsonl"),
        "sft": count_jsonl(run_dir / "sft.jsonl"),
        "surplus_sft": count_jsonl(run_dir / "surplus_sft.jsonl"),
        "surplus_synthetic_pool": count_jsonl(run_dir / "surplus_synthetic_pool.jsonl"),
        "rejected": count_jsonl(run_dir / "rejected.jsonl"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
