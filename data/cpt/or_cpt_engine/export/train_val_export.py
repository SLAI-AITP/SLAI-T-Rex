from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from cpt_cleaner.utils import json_compat

from or_cpt_engine.schemas.common import EngineConfig
from or_cpt_engine.utils.io import append_jsonl, read_jsonl, write_text

REQUIRED_TRAIN_METADATA_FIELDS = (
    "generator_id",
    "task_family",
    "sub_family",
    "doc_type",
    "difficulty_level",
    "scenario_id",
    "scenario_variant_id",
    "business_trigger",
    "token_count",
    "pair_id",
    "seed_instance_id",
    "solver_objective_value",
    "forward_eval_status",
)

FORBIDDEN_TRAIN_PHRASES = (
    "downstream cpt training data",
    "suitable for training",
    "as an ai",
    "user:",
    "assistant:",
    "chatml",
    "<|im_start|>",
    "<|im_end|>",
)


def export_train_val_test(
    input_path: str | Path,
    output_dir: str | Path,
    config: EngineConfig,
    *,
    exact_dedup: bool = True,
) -> dict[str, Any]:
    """Export CPT documents as train-only data.

    The function name is kept for CLI/import compatibility, but OR-CPT
    production now writes only `train.jsonl`. Validation/test sets should come
    from separate curated benchmarks rather than from synthetic production data.
    """
    output = Path(output_dir)
    exporter = StreamingTrainOnlyExporter(output, exact_dedup=exact_dedup)
    for row in read_jsonl(input_path):
        exporter.append_document(row)
    return exporter.finalize()


class StreamingTrainOnlyExporter:
    """Append accepted CPT documents to train.jsonl as soon as they are ready."""

    def __init__(self, output_dir: str | Path, *, exact_dedup: bool = True) -> None:
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.train_path = self.output / "train.jsonl"
        self.exact_dedup = exact_dedup
        self.seen_text_hashes: set[str] = set()
        self.train_count = 0
        self.duplicate_count = 0
        self.total_token_count = 0
        self.doc_type_counts: Counter[str] = Counter()
        self.generator_counts: Counter[str] = Counter()
        self.task_family_counts: Counter[str] = Counter()
        self.sub_family_counts: Counter[str] = Counter()
        self.difficulty_counts: Counter[str] = Counter()
        self.scenario_counts: Counter[str] = Counter()
        self.business_trigger_counts: Counter[str] = Counter()
        self.style_counts: Counter[str] = Counter()
        self.variant_counts: Counter[str] = Counter()
        self.concept_counts: Counter[str] = Counter()
        self.data_quality_counts: Counter[str] = Counter()
        self.metadata_present_counts: Counter[str] = Counter()
        self.metadata_missing_counts: Counter[str] = Counter()
        self._reset_outputs()

    def append_document(self, row: dict[str, Any]) -> bool:
        if self.exact_dedup:
            digest = hashlib.sha1(_normalize_text(row.get("text") or "").encode("utf-8")).hexdigest()
            if digest in self.seen_text_hashes:
                self.duplicate_count += 1
                return False
            self.seen_text_hashes.add(digest)
        export_row = _export_row(row)
        append_jsonl(self.train_path, export_row)
        self.train_count += 1
        self._update_counters(row)
        self._update_metadata_coverage(export_row)
        self._update_data_quality(export_row)
        return True

    def finalize(self) -> dict[str, Any]:
        manifest = self._build_manifest()
        (self.output / "manifest.json").write_bytes(json_compat.dumps(manifest))
        (self.output / "train_manifest.json").write_bytes(json_compat.dumps(manifest))
        write_text(self.output / "export_report.md", render_export_report(manifest))
        return {
            "train": self.train_count,
            "duplicates_removed": self.duplicate_count,
            "mode": "train_only_streaming",
        }

    def _reset_outputs(self) -> None:
        for filename in (
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
            "manifest.json",
            "train_manifest.json",
            "export_report.md",
            "train_all.jsonl",
            "train_reasoning_heavy.jsonl",
            "train_code_heavy.jsonl",
            "train_validation_heavy.jsonl",
            "train_high_value_or.jsonl",
            "train_balanced_by_generator.jsonl",
        ):
            path = self.output / filename
            if path.exists():
                path.unlink()

    def _update_counters(self, row: dict[str, Any]) -> None:
        metadata = row.get("metadata") or {}
        self.total_token_count += int(row.get("token_count") or 0)
        self.doc_type_counts[str(row.get("doc_type") or "unknown")] += 1
        self.generator_counts[str(row.get("generator_id") or "unknown")] += 1
        self.task_family_counts[str(metadata.get("task_family") or "unknown")] += 1
        self.sub_family_counts[str(metadata.get("sub_family") or "unknown")] += 1
        self.difficulty_counts[str(row.get("difficulty_level") or "unknown")] += 1
        self.scenario_counts[str(metadata.get("scenario_id") or "unknown")] += 1
        self.business_trigger_counts[str(metadata.get("business_trigger") or "unknown")] += 1
        self.style_counts[str(metadata.get("writing_style") or "unknown")] += 1
        self.variant_counts[str(metadata.get("variant_id") or "unknown")] += 1
        for concept in metadata.get("concept_tags") or []:
            self.concept_counts[str(concept)] += 1

    def _update_metadata_coverage(self, row: dict[str, Any]) -> None:
        metadata = row.get("metadata") or {}
        for field in REQUIRED_TRAIN_METADATA_FIELDS:
            if metadata.get(field) is None or metadata.get(field) == "":
                self.metadata_missing_counts[field] += 1
            else:
                self.metadata_present_counts[field] += 1

    def _update_data_quality(self, row: dict[str, Any]) -> None:
        text = str(row.get("text") or "")
        lowered = text.lower()
        for phrase in FORBIDDEN_TRAIN_PHRASES:
            if phrase in lowered:
                self.data_quality_counts[f"forbidden_phrase:{phrase}"] += 1
        if "```python" not in text:
            self.data_quality_counts["missing_code_block"] += 1
        if not any(marker in text for marker in ("## Validation", "## Objective Check", "## Solver Result", "## Verified Outcome")):
            self.data_quality_counts["missing_solver_validation_section"] += 1

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "export_mode": "train_only_streaming",
            "total_documents": self.train_count,
            "duplicates_removed": self.duplicate_count,
            "total_token_count": self.total_token_count,
            "split_counts": {"train": self.train_count},
            "doc_type_counts": dict(self.doc_type_counts),
            "generator_counts": dict(self.generator_counts),
            "task_family_counts": dict(self.task_family_counts),
            "sub_family_counts": dict(self.sub_family_counts),
            "difficulty_counts": dict(self.difficulty_counts),
            "scenario_counts": dict(self.scenario_counts),
            "business_trigger_counts": dict(self.business_trigger_counts),
            "style_counts": dict(self.style_counts),
            "variant_counts": dict(self.variant_counts),
            "concept_counts": dict(self.concept_counts),
            "metadata_coverage": {
                field: {
                    "present": self.metadata_present_counts[field],
                    "missing": self.metadata_missing_counts[field],
                }
                for field in REQUIRED_TRAIN_METADATA_FIELDS
            },
            "data_quality_counts": dict(self.data_quality_counts),
            "notes": "Synthetic OR-CPT production exports train.jsonl only. Exact text dedup is applied before append.",
        }


def render_export_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Train-Only Export Report",
        "",
        f"- Export mode: {manifest.get('export_mode', 'train_only_streaming')}",
        f"- Total documents: {manifest['total_documents']}",
        f"- Duplicates removed: {manifest['duplicates_removed']}",
        f"- Total approx tokens: {manifest['total_token_count']}",
        "",
        "## Output",
        "",
        "| File | Count |",
        "|---|---:|",
    ]
    for split, count in manifest["split_counts"].items():
        lines.append(f"| {split}.jsonl | {count} |")
    lines.extend(["", "## Doc Types", "", "| Type | Count |", "|---|---:|"])
    for doc_type, count in manifest["doc_type_counts"].items():
        lines.append(f"| {doc_type} | {count} |")
    lines.extend(_render_count_section("Task Families", manifest.get("task_family_counts") or {}, limit=30))
    lines.extend(_render_count_section("Generator Families", manifest.get("generator_counts") or {}, limit=30))
    lines.extend(_render_count_section("Sub Families", manifest.get("sub_family_counts") or {}, limit=30))
    lines.extend(_render_count_section("Difficulty Levels", manifest.get("difficulty_counts") or {}, limit=20))
    lines.extend(_render_count_section("Scenarios", manifest.get("scenario_counts") or {}, limit=30))
    lines.extend(_render_count_section("Business Triggers", manifest.get("business_trigger_counts") or {}, limit=30))
    lines.extend(_render_count_section("Writing Styles", manifest.get("style_counts") or {}, limit=30))
    lines.extend(_render_count_section("Variants", manifest.get("variant_counts") or {}, limit=20))
    lines.extend(_render_count_section("Concept Tags", manifest.get("concept_counts") or {}, limit=30))
    lines.extend(_render_metadata_coverage(manifest.get("metadata_coverage") or {}))
    lines.extend(_render_count_section("Data Quality Flags", manifest.get("data_quality_counts") or {}, limit=30))
    return "\n".join(lines) + "\n"


def _export_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "id": row["doc_id"],
        "text": row["text"],
        "source_id": row["instance_id"],
        "metadata": {
            **metadata,
            "pair_id": row.get("pair_id"),
            "generator_id": row.get("generator_id"),
            "doc_type": row.get("doc_type"),
            "language": row.get("language"),
            "token_count": row.get("token_count"),
            "difficulty_level": row.get("difficulty_level") or metadata.get("difficulty_level"),
            "seed_instance_id": row.get("instance_id") or metadata.get("seed_instance_id"),
            "scenario_id": metadata.get("scenario_id"),
            "scenario_variant_id": metadata.get("scenario_variant_id"),
            "business_trigger": metadata.get("business_trigger"),
            "solver_objective_value": metadata.get("solver_objective_value"),
            "forward_eval_status": metadata.get("forward_eval_status"),
        },
    }

def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _render_count_section(title: str, counts: dict[str, int], *, limit: int) -> list[str]:
    lines = ["", f"## {title}", "", "| Value | Count |", "|---|---:|"]
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        lines.append(f"| {value} | {count} |")
    if len(counts) > limit:
        lines.append(f"| ... | {len(counts) - limit} more values omitted |")
    return lines


def _render_metadata_coverage(coverage: dict[str, dict[str, int]]) -> list[str]:
    lines = ["", "## Required Metadata Coverage", "", "| Field | Present | Missing |", "|---|---:|---:|"]
    for field, counts in coverage.items():
        lines.append(f"| {field} | {counts.get('present', 0)} | {counts.get('missing', 0)} |")
    return lines
