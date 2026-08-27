from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from cpt_cleaner.utils import json_compat

from or_cpt_engine.contracts import resolve_family_contract
from or_cpt_engine.forward_modeling.generate_forward_model import _static_signature_issues
from or_cpt_engine.quality.semantic_coherence import semantic_coherence_issues
from or_cpt_engine.rendering.cpt_renderer import FORBIDDEN_RENDER_PHRASES, _check_rendered_text
from or_cpt_engine.utils.io import iter_jsonl, write_text


STAGE_COUNT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("seed_count", "04b_instance_quality/quality_validated_instances.jsonl", "04b_instance_quality"),
    ("bt_candidates", "05_backtranslation/backtranslation_candidates.jsonl", "05_backtranslation"),
    ("nl_validated", "06_nl_quality_filter/nl_validated_candidates.jsonl", "06_nl_quality_filter"),
    ("fm_outputs", "07_forward_modeling/forward_modeling_outputs.jsonl", "07_forward_modeling"),
    ("eval_accepted", "08_forward_eval/accepted_pairs.jsonl", "08_forward_eval"),
    ("rendered", "09_cpt_rendering/cpt_documents.jsonl", "09_cpt_rendering"),
    ("train", "10_train_val_export/train.jsonl", "10_train_export"),
)

REJECTION_SPECS: tuple[tuple[str, str], ...] = (
    ("05_backtranslation", "05_backtranslation/backtranslation_rejected.jsonl"),
    ("06_nl_quality_filter", "06_nl_quality_filter/nl_rejected.jsonl"),
    ("07_forward_modeling", "07_forward_modeling/forward_modeling_rejected.jsonl"),
    ("08_forward_eval", "08_forward_eval/rejected_pairs.jsonl"),
    ("09_cpt_rendering", "09_cpt_rendering/cpt_rendering_rejected.jsonl"),
)

GENERATOR_FUNNEL_FIELDS: tuple[str, ...] = (
    "seed_count",
    "bt_candidates",
    "bt_rejected",
    "nl_validated",
    "nl_rejected",
    "fm_outputs",
    "fm_rejected",
    "eval_accepted",
    "eval_rejected",
    "rendered",
    "render_rejected",
    "train",
)

TOKEN_BUCKETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("<1k", None, 1000),
    ("1k~2k", 1000, 2000),
    ("2k~3k", 2000, 3000),
    ("3k~4k", 3000, 4000),
    (">=4k", 4000, None),
)


def analyze_run_quality(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir)
    output = Path(output_dir) if output_dir is not None else run / "reports"
    output.mkdir(parents=True, exist_ok=True)

    stage_counts, funnel_rows = _collect_stage_counts(run)
    rejection_rows = _collect_rejections(run, funnel_rows)
    train_stats = _collect_train_distribution(run)
    throughput = _collect_throughput(run)
    endpoint_reliability = _collect_endpoint_reliability(run)
    model_failures = _collect_model_failures(run)
    forward_static_replay = _collect_forward_static_replay(run)
    data_quality = _collect_data_quality(run)
    quality_retries = _collect_quality_retries(run)
    repair_attempts = _collect_repair_attempts(run)

    summary_rows = _build_summary_rows(
        stage_counts,
        rejection_rows,
        train_stats,
        throughput,
        endpoint_reliability,
        model_failures,
        forward_static_replay,
        data_quality,
        quality_retries,
        repair_attempts,
    )
    summary = {
        "run_dir": str(run),
        "stage_counts": stage_counts,
        "rejection_count": sum(int(row["count"]) for row in rejection_rows),
        "train_stats": train_stats,
        "throughput": throughput,
        "endpoint_reliability": endpoint_reliability,
        "model_failures": model_failures,
        "forward_static_replay": forward_static_replay,
        "data_quality": data_quality,
        "quality_retries": quality_retries,
        "repair_attempts": repair_attempts,
        "outputs": {
            "dashboard_md": str(output / "run_quality_dashboard.md"),
            "dashboard_csv": str(output / "run_quality_dashboard.csv"),
            "generator_stage_funnel_csv": str(output / "generator_stage_funnel.csv"),
            "rejection_reason_by_generator_csv": str(output / "rejection_reason_by_generator.csv"),
            "train_distribution_md": str(output / "train_distribution.md"),
        },
    }

    _write_csv(output / "run_quality_dashboard.csv", summary_rows)
    _write_csv(output / "generator_stage_funnel.csv", _finalize_funnel_rows(funnel_rows))
    _write_csv(output / "rejection_reason_by_generator.csv", rejection_rows)
    write_text(output / "train_distribution.md", _render_train_distribution(train_stats))
    write_text(output / "run_quality_dashboard.md", _render_dashboard(summary, rejection_rows))
    (output / "run_quality_summary.json").write_bytes(json_compat.dumps(summary))
    return {
        "stage_counts": stage_counts,
        "rejection_rows": len(rejection_rows),
        "train_count": stage_counts.get("train", 0),
        "output_dir": str(output),
    }


def _collect_stage_counts(run: Path) -> tuple[dict[str, int], dict[str, Counter[str]]]:
    counts: dict[str, int] = {}
    funnel: dict[str, Counter[str]] = defaultdict(Counter)
    seed_override_path = _resolve_seed_input_from_manifest(run)

    for field, relative_path, _stage_name in STAGE_COUNT_SPECS:
        path = run / relative_path
        if field == "seed_count" and (not path.exists()) and seed_override_path is not None:
            path = seed_override_path
        row_count = 0
        for row in iter_jsonl(path) or []:
            row_count += 1
            funnel[_generator_id(row)][field] += 1
        counts[field] = row_count

    if counts.get("seed_count", 0) == 0:
        inferred_seed_to_generator: dict[str, str] = {}
        for relative_path in (
            "05_backtranslation/backtranslation_candidates.jsonl",
            "05_backtranslation/backtranslation_rejected.jsonl",
        ):
            for row in iter_jsonl(run / relative_path) or []:
                if row.get("instance_id"):
                    inferred_seed_to_generator.setdefault(str(row["instance_id"]), _generator_id(row))
        for generator_id in inferred_seed_to_generator.values():
            funnel[generator_id]["seed_count"] += 1
        counts["seed_count"] = len(inferred_seed_to_generator)
    return counts, funnel


def _collect_rejections(run: Path, funnel: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall_rejections = 0
    stage_totals: Counter[str] = Counter()
    grouped: Counter[tuple[str, str, str, str]] = Counter()
    stage_to_funnel_field = {
        "05_backtranslation": "bt_rejected",
        "06_nl_quality_filter": "nl_rejected",
        "07_forward_modeling": "fm_rejected",
        "08_forward_eval": "eval_rejected",
        "09_cpt_rendering": "render_rejected",
    }

    for stage, relative_path in REJECTION_SPECS:
        for row in iter_jsonl(run / relative_path) or []:
            generator_id = _generator_id(row)
            stage_totals[stage] += 1
            overall_rejections += 1
            if stage in stage_to_funnel_field:
                funnel[generator_id][stage_to_funnel_field[stage]] += 1
            reasons = _split_rejection_reasons(row.get("rejection_reason") or row.get("error") or row.get("errors"))
            for reason_detail in reasons:
                base_reason = _base_reason(reason_detail)
                grouped[(stage, generator_id, base_reason, reason_detail)] += 1

    for (stage, generator_id, base_reason, reason_detail), count in grouped.most_common():
        rows.append(
            {
                "stage": stage,
                "generator_id": generator_id,
                "base_reason": base_reason,
                "reason_detail": reason_detail,
                "count": count,
                "stage_rejection_share": _rate(count, stage_totals[stage]),
                "overall_rejection_share": _rate(count, overall_rejections),
            }
        )
    return rows


def _collect_train_distribution(run: Path) -> dict[str, Any]:
    train_path = run / "10_train_val_export" / "train.jsonl"
    counters = {
        "generator_id": Counter(),
        "task_family": Counter(),
        "sub_family": Counter(),
        "doc_type": Counter(),
        "difficulty_level": Counter(),
        "scenario_id": Counter(),
        "business_trigger": Counter(),
        "token_bucket": Counter(),
    }
    token_total = 0
    token_min: int | None = None
    token_max: int | None = None
    row_count = 0

    for row in iter_jsonl(train_path) or []:
        metadata = row.get("metadata") or {}
        row_count += 1
        token_count = _int_or_zero(metadata.get("token_count") or row.get("token_count"))
        token_total += token_count
        token_min = token_count if token_min is None else min(token_min, token_count)
        token_max = token_count if token_max is None else max(token_max, token_count)
        counters["generator_id"][_generator_id(row)] += 1
        counters["task_family"][_field(row, "task_family")] += 1
        counters["sub_family"][_field(row, "sub_family")] += 1
        counters["doc_type"][_field(row, "doc_type")] += 1
        counters["difficulty_level"][_field(row, "difficulty_level")] += 1
        counters["scenario_id"][_field(row, "scenario_id")] += 1
        counters["business_trigger"][_field(row, "business_trigger")] += 1
        counters["token_bucket"][_token_bucket(token_count)] += 1

    return {
        "row_count": row_count,
        "token_total": token_total,
        "token_avg": (token_total / row_count) if row_count else 0.0,
        "token_min": token_min or 0,
        "token_max": token_max or 0,
        "distributions": {name: dict(counter.most_common()) for name, counter in counters.items()},
    }


def _collect_throughput(run: Path) -> dict[str, Any]:
    throughput_path = run / "reports" / "throughput_summary.json"
    payload: dict[str, Any] = {}
    if throughput_path.exists():
        try:
            payload = json_compat.loads(throughput_path.read_bytes())
        except Exception:
            payload = {}
    overall = payload.get("overall") or {}
    stage_wall_seconds = _parse_stage_wall_seconds(run)
    endpoint_health_events = _count_endpoint_health_events(run)
    api_attempts = int(overall.get("api_attempts") or 0)
    successful_api_attempts = int(overall.get("successful_api_attempts") or 0)
    return {
        "wall_seconds": float(payload.get("wall_seconds") or 0.0),
        "api_attempts": api_attempts,
        "successful_api_attempts": successful_api_attempts,
        "retry_or_failed_attempts": max(0, api_attempts - successful_api_attempts),
        "total_tokens": int(overall.get("total_tokens") or 0),
        "completion_tokens": int(overall.get("completion_tokens") or 0),
        "tokens_per_second": float(overall.get("tokens_per_second") or 0.0),
        "by_prompt": payload.get("by_prompt") or {},
        "stage_wall_seconds": stage_wall_seconds,
        "endpoint_health_events": endpoint_health_events,
        # Backward-compatible alias. This used to include model failure rows,
        # but the clearer meaning is endpoint quarantine/unavailable events.
        "endpoint_failure_count": endpoint_health_events,
    }


def _collect_model_failures(run: Path) -> dict[str, Any]:
    logs_dir = run / "logs"
    instance_to_generator = _collect_instance_generator_map(run)
    failure_counter: Counter[tuple[str, str, str, str, str]] = Counter()
    prompt_counter: Counter[str] = Counter()
    endpoint_counter: Counter[str] = Counter()
    generator_counter: Counter[str] = Counter()
    error_type_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    seen_events: set[tuple[str, str]] = set()

    if logs_dir.exists():
        for path in sorted(logs_dir.glob("*.log")):
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "model_api_call " not in line or " status=error" not in line:
                    continue
                payload = line.split("model_api_call ", 1)[1]
                fields = _parse_log_kv_fields(payload)
                if not _mark_model_failure_event_seen(seen_events, fields):
                    continue
                instance_id = fields.get("instance_id", "")
                prompt_name = fields.get("prompt_name") or "unknown"
                endpoint_name = fields.get("endpoint_name") or "unknown"
                generator_id = instance_to_generator.get(instance_id, "unknown")
                error_type = _model_failure_error_type(fields)
                _record_model_failure(
                    failure_counter,
                    prompt_counter,
                    endpoint_counter,
                    generator_counter,
                    error_type_counter,
                    source_counter,
                    source="model_api_call_log",
                    prompt_name=prompt_name,
                    generator_id=generator_id,
                    endpoint_name=endpoint_name,
                    error_type=error_type,
                )

        for path in sorted(logs_dir.glob("*model_failures.jsonl")):
            for row in iter_jsonl(path) or []:
                if not _mark_model_failure_event_seen(seen_events, row):
                    continue
                prompt_name = str(row.get("prompt_name") or "unknown")
                instance_id = str(row.get("instance_id") or row.get("sample_id") or "")
                endpoint_name = str(row.get("endpoint_name") or "unknown")
                generator_id = str(row.get("generator_id") or instance_to_generator.get(instance_id) or "unknown")
                error_type = _model_failure_error_type(row)
                _record_model_failure(
                    failure_counter,
                    prompt_counter,
                    endpoint_counter,
                    generator_counter,
                    error_type_counter,
                    source_counter,
                    source="model_failures_jsonl",
                    prompt_name=prompt_name,
                    generator_id=generator_id,
                    endpoint_name=endpoint_name,
                    error_type=error_type,
                )

    return {
        "total_failures": sum(failure_counter.values()),
        "by_prompt": dict(prompt_counter.most_common()),
        "by_endpoint": dict(endpoint_counter.most_common()),
        "by_generator": dict(generator_counter.most_common()),
        "by_error_type": dict(error_type_counter.most_common()),
        "by_source": dict(source_counter.most_common()),
        "top_failures": [
            {
                "prompt_name": prompt_name,
                "generator_id": generator_id,
                "endpoint_name": endpoint_name,
                "error_type": error_type,
                "source": source,
                "count": count,
            }
            for (prompt_name, generator_id, endpoint_name, error_type, source), count in failure_counter.most_common(20)
        ],
    }


def _collect_endpoint_reliability(run: Path) -> dict[str, Any]:
    logs_dir = run / "logs"
    endpoint_rows: dict[str, dict[str, Any]] = {}
    latencies: dict[str, list[float]] = defaultdict(list)
    prompt_counter: Counter[tuple[str, str]] = Counter()

    if logs_dir.exists():
        for path in sorted(logs_dir.glob("*.log")):
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "model_api_call " not in line:
                    continue
                payload = line.split("model_api_call ", 1)[1]
                fields = _parse_log_kv_fields(payload)
                endpoint_name = fields.get("endpoint_name") or "unknown"
                row = endpoint_rows.setdefault(
                    endpoint_name,
                    {
                        "endpoint_name": endpoint_name,
                        "api_attempts": 0,
                        "successful_api_attempts": 0,
                        "failed_api_attempts": 0,
                        "total_latency_seconds": 0.0,
                        "max_latency_seconds": 0.0,
                    },
                )
                status = str(fields.get("status") or "").lower()
                latency = _float_or_zero(fields.get("latency_seconds"))
                row["api_attempts"] += 1
                if status in {"ok", "success"}:
                    row["successful_api_attempts"] += 1
                else:
                    row["failed_api_attempts"] += 1
                row["total_latency_seconds"] += latency
                row["max_latency_seconds"] = max(float(row["max_latency_seconds"]), latency)
                latencies[endpoint_name].append(latency)
                prompt_counter[(endpoint_name, fields.get("prompt_name") or "unknown")] += 1

    rows: list[dict[str, Any]] = []
    for endpoint_name, row in endpoint_rows.items():
        attempts = int(row["api_attempts"])
        latency_values = sorted(latencies.get(endpoint_name) or [])
        row["failure_rate"] = _rate(int(row["failed_api_attempts"]), attempts)
        row["success_rate"] = _rate(int(row["successful_api_attempts"]), attempts)
        row["avg_latency_seconds"] = (float(row["total_latency_seconds"]) / attempts) if attempts else 0.0
        row["p95_latency_seconds"] = _percentile(latency_values, 0.95)
        row["top_prompts"] = [
            {"prompt_name": prompt_name, "count": count}
            for (candidate_endpoint, prompt_name), count in prompt_counter.most_common()
            if candidate_endpoint == endpoint_name
        ][:5]
        rows.append(row)

    rows.sort(key=lambda item: (-float(item.get("failure_rate") or 0.0), -int(item.get("failed_api_attempts") or 0), str(item.get("endpoint_name") or "")))
    return {
        "endpoint_count": len(rows),
        "rows": rows,
        "worst_endpoint": rows[0] if rows else None,
    }


def _collect_forward_static_replay(run: Path) -> dict[str, Any]:
    path = run / "07_forward_modeling" / "forward_modeling_rejected.jsonl"
    family_contract_config = _load_current_family_contract_config()
    total_rows = 0
    replayed_rows = 0
    old_static_rows = 0
    rows_with_current_static_issues = 0
    issue_counter: Counter[tuple[str, str]] = Counter()

    for row in iter_jsonl(path) or []:
        total_rows += 1
        old_reason = str(row.get("rejection_reason") or "")
        if _is_static_rejection_reason(old_reason):
            old_static_rows += 1
        payload = row.get("generated_answer") if isinstance(row.get("generated_answer"), dict) else {}
        code = str(payload.get("gurobipy_code") or "")
        if not payload or not code:
            continue
        replayed_rows += 1
        family_contract = row.get("family_contract") if isinstance(row.get("family_contract"), dict) else None
        if family_contract_config:
            try:
                family_contract = resolve_family_contract(row, family_contract_config)
            except Exception:
                family_contract = row.get("family_contract") if isinstance(row.get("family_contract"), dict) else None
        issues = _static_signature_issues(row, payload, code, family_contract=family_contract)
        if not issues:
            continue
        rows_with_current_static_issues += 1
        generator_id = _generator_id(row)
        for issue in issues:
            issue_counter[(generator_id, issue)] += 1

    return {
        "total_forward_rejected_rows": total_rows,
        "old_static_rejection_rows": old_static_rows,
        "replayed_rows": replayed_rows,
        "rows_with_current_static_issues": rows_with_current_static_issues,
        "rows_cleared_by_current_static_checker": max(0, replayed_rows - rows_with_current_static_issues),
        "current_static_issue_rate": _rate(rows_with_current_static_issues, replayed_rows),
        "top_current_static_issues": [
            {"generator_id": generator_id, "issue": issue, "count": count}
            for (generator_id, issue), count in issue_counter.most_common(20)
        ],
    }


def _load_current_family_contract_config() -> dict[str, Any] | None:
    path = Path("engine_configs/family_contracts.yaml")
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _is_static_rejection_reason(reason: str) -> bool:
    return reason.startswith("STATIC_SIGNATURE_MISMATCH") or reason.startswith("CONTRACT_")


def _record_model_failure(
    failure_counter: Counter[tuple[str, str, str, str, str]],
    prompt_counter: Counter[str],
    endpoint_counter: Counter[str],
    generator_counter: Counter[str],
    error_type_counter: Counter[str],
    source_counter: Counter[str],
    *,
    source: str,
    prompt_name: str,
    generator_id: str,
    endpoint_name: str,
    error_type: str,
) -> None:
    failure_counter[(prompt_name, generator_id, endpoint_name, error_type, source)] += 1
    prompt_counter[prompt_name] += 1
    endpoint_counter[endpoint_name] += 1
    generator_counter[generator_id] += 1
    error_type_counter[error_type] += 1
    source_counter[source] += 1


def _collect_instance_generator_map(run: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    relative_paths = [
        "04b_instance_quality/quality_validated_instances.jsonl",
        "05_backtranslation/backtranslation_candidates.jsonl",
        "05_backtranslation/backtranslation_rejected.jsonl",
        "06_nl_quality_filter/nl_validated_candidates.jsonl",
        "06_nl_quality_filter/nl_rejected.jsonl",
        "07_forward_modeling/forward_modeling_outputs.jsonl",
        "07_forward_modeling/forward_modeling_rejected.jsonl",
        "08_forward_eval/accepted_pairs.jsonl",
        "08_forward_eval/rejected_pairs.jsonl",
        "09_cpt_rendering/cpt_documents.jsonl",
        "09_cpt_rendering/cpt_rendering_rejected.jsonl",
        "10_train_val_export/train.jsonl",
    ]
    seed_override_path = _resolve_seed_input_from_manifest(run)
    if seed_override_path is not None:
        relative_or_absolute_paths: list[Path] = [seed_override_path]
    else:
        relative_or_absolute_paths = []
    relative_or_absolute_paths.extend(run / relative_path for relative_path in relative_paths)

    for path in relative_or_absolute_paths:
        for row in iter_jsonl(path) or []:
            generator_id = _generator_id(row)
            if generator_id == "unknown":
                continue
            for key in ("instance_id", "source_id", "seed_instance_id"):
                value = row.get(key)
                if value not in (None, ""):
                    mapping.setdefault(str(value), generator_id)
    return mapping


def _parse_log_kv_fields(payload: str) -> dict[str, str]:
    fields = {match.group(1): match.group(2) for match in re.finditer(r"(\w+)=([^\s]+)", payload)}
    error_match = re.search(r"\berror=(.*)$", payload)
    if error_match:
        fields["error"] = error_match.group(1).strip()
    return fields


def _mark_model_failure_event_seen(seen_events: set[tuple[str, str]], row: dict[str, Any]) -> bool:
    request_id = str(row.get("request_id") or "")
    attempt = str(row.get("attempt") or "")
    if request_id and attempt:
        key = (request_id, attempt)
    else:
        key = (f"fallback:{len(seen_events)}", "")
    if key in seen_events:
        return False
    seen_events.add(key)
    return True


def _model_failure_error_type(row: dict[str, Any]) -> str:
    explicit_error_type = row.get("error_type")
    if explicit_error_type not in (None, ""):
        return str(explicit_error_type)
    text_parts = [
        str(row.get("error") or ""),
        str(row.get("error_message") or ""),
        str(row.get("message") or ""),
        str(row.get("reason") or ""),
    ]
    text = " ".join(part for part in text_parts if part).lower()
    http_status = str(row.get("http_status") or "")
    if "timeout" in text or "timed out" in text or "readtimeout" in text:
        return "TIMEOUT"
    if "rate limit" in text or http_status == "429":
        return "RATE_LIMIT"
    if http_status.startswith("5"):
        return "HTTP_5XX"
    if http_status.startswith("4"):
        return "HTTP_4XX"
    try:
        latency_seconds = float(row.get("latency_seconds") or 0.0)
    except (TypeError, ValueError):
        latency_seconds = 0.0
    if latency_seconds >= 120.0:
        return "LIKELY_TIMEOUT"
    if "connection" in text or "connect" in text or "network" in text:
        return "CONNECTIVITY"
    if "endpoint_unavailable" in text or "unavailable" in text:
        return "ENDPOINT_UNAVAILABLE"
    if text:
        return _base_reason(text_parts[0] or text_parts[1] or text_parts[2] or text_parts[3])[:80]
    return "UNKNOWN"


def _collect_data_quality(run: Path) -> dict[str, Any]:
    train_path = run / "10_train_val_export" / "train.jsonl"
    text_hashes: Counter[str] = Counter()
    forbidden_phrase_hits: Counter[str] = Counter()
    missing_section_hits: Counter[str] = Counter()
    semantic_issue_hits: Counter[str] = Counter()
    semantic_generator_hits: Counter[tuple[str, str]] = Counter()
    missing_code_block_count = 0
    semantic_row_count = 0
    row_count = 0

    for row in iter_jsonl(train_path) or []:
        text = str(row.get("text") or "")
        row_count += 1
        text_hashes[_normalize_text(text)] += 1
        lowered = text.lower()
        for phrase in FORBIDDEN_RENDER_PHRASES:
            if phrase in lowered:
                forbidden_phrase_hits[phrase] += 1
        doc_type = _field(row, "doc_type")
        quality = _check_rendered_text(text, doc_type)
        for issue in quality.get("issues") or []:
            if issue.startswith("MISSING_SECTION:"):
                missing_section_hits[issue.replace("MISSING_SECTION:", "", 1)] += 1
            if issue == "MISSING_CODE_BLOCK":
                missing_code_block_count += 1
        semantic_issues = semantic_coherence_issues(_generator_id(row), text)
        if semantic_issues:
            semantic_row_count += 1
        for issue in semantic_issues:
            semantic_issue_hits[issue] += 1
            semantic_generator_hits[(_generator_id(row), issue)] += 1

    duplicate_rows = sum(count - 1 for count in text_hashes.values() if count > 1)
    return {
        "row_count": row_count,
        "exact_duplicate_rows": duplicate_rows,
        "forbidden_phrase_count": sum(forbidden_phrase_hits.values()),
        "forbidden_phrase_hits": dict(forbidden_phrase_hits.most_common()),
        "missing_section_count": sum(missing_section_hits.values()),
        "missing_section_hits": dict(missing_section_hits.most_common()),
        "missing_code_block_count": missing_code_block_count,
        "semantic_coherence_issue_count": sum(semantic_issue_hits.values()),
        "semantic_coherence_row_count": semantic_row_count,
        "semantic_coherence_hits": dict(semantic_issue_hits.most_common()),
        "semantic_coherence_top_generators": [
            {"generator_id": generator_id, "issue": issue, "count": count}
            for (generator_id, issue), count in semantic_generator_hits.most_common(20)
        ],
    }


def _collect_quality_retries(run: Path) -> dict[str, Any]:
    stage_specs = (
        ("05_backtranslation", "candidate", "05_backtranslation/backtranslation_candidates.jsonl"),
        ("05_backtranslation", "rejected", "05_backtranslation/backtranslation_rejected.jsonl"),
        ("07_forward_modeling", "output", "07_forward_modeling/forward_modeling_outputs.jsonl"),
        ("07_forward_modeling", "rejected", "07_forward_modeling/forward_modeling_rejected.jsonl"),
    )
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counter: Counter[tuple[str, str, str, str]] = Counter()
    total_rows_with_retries = 0
    total_quality_retries = 0

    for stage, outcome, relative_path in stage_specs:
        for row in iter_jsonl(run / relative_path) or []:
            metadata = row.get("llm_metadata") or {}
            if not isinstance(metadata, dict):
                continue
            retry_count = _int_or_zero(metadata.get("quality_retries"))
            if retry_count <= 0:
                continue
            generator_id = _generator_id(row)
            total_rows_with_retries += 1
            total_quality_retries += retry_count
            by_stage[stage]["rows_with_retries"] += 1
            by_stage[stage]["total_quality_retries"] += retry_count
            by_stage[stage][f"{outcome}_rows_with_retries"] += 1
            reasons = [str(value) for value in metadata.get("quality_retry_reasons") or []]
            events = metadata.get("quality_retry_events") or []
            if not reasons and isinstance(events, list):
                reasons = [str(event.get("reason")) for event in events if isinstance(event, dict) and event.get("reason")]
            for reason in reasons or ["UNKNOWN_RETRY_REASON"]:
                reason_counter[(stage, generator_id, outcome, reason)] += 1

    top_reasons = [
        {
            "stage": stage,
            "generator_id": generator_id,
            "outcome": outcome,
            "reason": reason,
            "base_reason": _base_reason(reason),
            "count": count,
        }
        for (stage, generator_id, outcome, reason), count in reason_counter.most_common(20)
    ]
    return {
        "total_rows_with_retries": total_rows_with_retries,
        "total_quality_retries": total_quality_retries,
        "by_stage": {stage: dict(counter) for stage, counter in by_stage.items()},
        "top_reasons": top_reasons,
    }


def _collect_repair_attempts(run: Path) -> dict[str, Any]:
    path = run / "08_forward_eval" / "forward_repair_attempts.jsonl"
    status_counter: Counter[str] = Counter()
    generator_counter: Counter[tuple[str, str, str, str]] = Counter()
    transition_counter: Counter[tuple[str, str, str]] = Counter()
    total_attempts = 0

    for row in iter_jsonl(path) or []:
        total_attempts += 1
        status = str(row.get("status") or "unknown")
        generator_id = _repair_field(row, "generator_id") or "unknown"
        original_family = _repair_field(row, "original_failure_family") or "unknown"
        final_family = _repair_field(row, "final_failure_family") or ("recovered" if status == "recovered" else "unknown")
        status_counter[status] += 1
        generator_counter[(generator_id, status, original_family, final_family)] += 1
        transition_counter[(status, original_family, final_family)] += 1

    return {
        "total_attempts": total_attempts,
        "recovered": int(status_counter.get("recovered", 0)),
        "failed_eval": int(status_counter.get("failed_eval", 0)),
        "failed_generation": int(status_counter.get("failed_generation", 0)),
        "by_status": dict(status_counter.most_common()),
        "top_generators": [
            {
                "generator_id": generator_id,
                "status": status,
                "original_failure_family": original_family,
                "final_failure_family": final_family,
                "count": count,
            }
            for (generator_id, status, original_family, final_family), count in generator_counter.most_common(20)
        ],
        "top_failure_transitions": [
            {
                "status": status,
                "original_failure_family": original_family,
                "final_failure_family": final_family,
                "count": count,
            }
            for (status, original_family, final_family), count in transition_counter.most_common(20)
        ],
    }


def _build_summary_rows(
    stage_counts: dict[str, int],
    rejection_rows: list[dict[str, Any]],
    train_stats: dict[str, Any],
    throughput: dict[str, Any],
    endpoint_reliability: dict[str, Any],
    model_failures: dict[str, Any],
    forward_static_replay: dict[str, Any],
    data_quality: dict[str, Any],
    quality_retries: dict[str, Any],
    repair_attempts: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in stage_counts.items():
        rows.append({"section": "stage_funnel", "metric": key, "value": value})
    rows.extend(
        [
            {"section": "stage_funnel", "metric": "seed_to_train_rate", "value": _rate(stage_counts.get("train", 0), stage_counts.get("seed_count", 0))},
            {"section": "stage_funnel", "metric": "nl_accept_rate", "value": _rate(stage_counts.get("nl_validated", 0), stage_counts.get("bt_candidates", 0))},
            {"section": "stage_funnel", "metric": "forward_eval_accept_rate", "value": _rate(stage_counts.get("eval_accepted", 0), stage_counts.get("fm_outputs", 0))},
        ]
    )
    for row in rejection_rows[:20]:
        rows.append(
            {
                "section": "top_rejection",
                "metric": f"{row['stage']}|{row['generator_id']}|{row['base_reason']}",
                "value": row["count"],
            }
        )
    for key in ("wall_seconds", "api_attempts", "successful_api_attempts", "retry_or_failed_attempts", "total_tokens", "tokens_per_second", "endpoint_health_events"):
        rows.append({"section": "throughput", "metric": key, "value": throughput.get(key, 0)})
    for endpoint_row in endpoint_reliability.get("rows") or []:
        endpoint_name = endpoint_row.get("endpoint_name") or "unknown"
        for key in ("api_attempts", "failed_api_attempts", "failure_rate", "avg_latency_seconds", "p95_latency_seconds"):
            rows.append({"section": "endpoint_reliability", "metric": f"{endpoint_name}|{key}", "value": endpoint_row.get(key, 0)})
    rows.append({"section": "model_failures", "metric": "total_failures", "value": model_failures.get("total_failures", 0)})
    for prompt_name, value in list((model_failures.get("by_prompt") or {}).items())[:10]:
        rows.append({"section": "model_failures", "metric": f"prompt|{prompt_name}", "value": value})
    for endpoint_name, value in list((model_failures.get("by_endpoint") or {}).items())[:10]:
        rows.append({"section": "model_failures", "metric": f"endpoint|{endpoint_name}", "value": value})
    for error_type, value in list((model_failures.get("by_error_type") or {}).items())[:10]:
        rows.append({"section": "model_failures", "metric": f"error_type|{error_type}", "value": value})
    for key in ("total_forward_rejected_rows", "old_static_rejection_rows", "replayed_rows", "rows_with_current_static_issues", "rows_cleared_by_current_static_checker", "current_static_issue_rate"):
        rows.append({"section": "forward_static_replay", "metric": key, "value": forward_static_replay.get(key, 0)})
    for row in forward_static_replay.get("top_current_static_issues") or []:
        rows.append(
            {
                "section": "forward_static_replay",
                "metric": f"{row.get('generator_id')}|{row.get('issue')}",
                "value": row.get("count", 0),
            }
        )
    for key in (
        "exact_duplicate_rows",
        "forbidden_phrase_count",
        "missing_section_count",
        "missing_code_block_count",
        "semantic_coherence_row_count",
        "semantic_coherence_issue_count",
    ):
        rows.append({"section": "data_quality", "metric": key, "value": data_quality.get(key, 0)})
    for phrase, value in list((data_quality.get("forbidden_phrase_hits") or {}).items())[:20]:
        rows.append({"section": "data_quality", "metric": f"forbidden_phrase|{phrase}", "value": value})
    for key in ("total_rows_with_retries", "total_quality_retries"):
        rows.append({"section": "quality_retries", "metric": key, "value": quality_retries.get(key, 0)})
    for stage, values in (quality_retries.get("by_stage") or {}).items():
        for key, value in values.items():
            rows.append({"section": "quality_retries", "metric": f"{stage}|{key}", "value": value})
    for key in ("total_attempts", "recovered", "failed_eval", "failed_generation"):
        rows.append({"section": "forward_repair", "metric": key, "value": repair_attempts.get(key, 0)})
    for status, value in (repair_attempts.get("by_status") or {}).items():
        rows.append({"section": "forward_repair", "metric": f"status|{status}", "value": value})
    rows.append({"section": "train_distribution", "metric": "token_avg", "value": round(float(train_stats.get("token_avg") or 0.0), 3)})
    return rows


def _finalize_funnel_rows(funnel: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generator_id, counts in sorted(funnel.items()):
        row = {"generator_id": generator_id}
        for field in GENERATOR_FUNNEL_FIELDS:
            row[field] = int(counts.get(field, 0))
        row["seed_to_train_rate"] = _rate(row["train"], row["seed_count"] or row["bt_candidates"])
        row["nl_accept_rate"] = _rate(row["nl_validated"], row["bt_candidates"])
        row["eval_accept_rate"] = _rate(row["eval_accepted"], row["fm_outputs"])
        rows.append(row)
    return rows


def _render_dashboard(summary: dict[str, Any], rejection_rows: list[dict[str, Any]]) -> str:
    stage_counts = summary["stage_counts"]
    train_stats = summary["train_stats"]
    throughput = summary["throughput"]
    endpoint_reliability = summary.get("endpoint_reliability") or {}
    model_failures = summary.get("model_failures") or {}
    forward_static_replay = summary.get("forward_static_replay") or {}
    data_quality = summary["data_quality"]
    quality_retries = summary.get("quality_retries") or {}
    repair_attempts = summary.get("repair_attempts") or {}
    lines = [
        "# OR-CPT Run Quality Dashboard",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Train samples: `{stage_counts.get('train', 0)}`",
        f"- Seed-to-train rate: `{_pct(_rate(stage_counts.get('train', 0), stage_counts.get('seed_count', 0)))}`",
        f"- NL accept rate: `{_pct(_rate(stage_counts.get('nl_validated', 0), stage_counts.get('bt_candidates', 0)))}`",
        f"- Forward-eval accept rate: `{_pct(_rate(stage_counts.get('eval_accepted', 0), stage_counts.get('fm_outputs', 0)))}`",
        "",
        "## Stage Funnel",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]
    for field, _path, _stage in STAGE_COUNT_SPECS:
        lines.append(f"| `{field}` | {stage_counts.get(field, 0)} |")
    lines.extend(
        [
            "",
            "## Top Rejection Reasons",
            "",
            "| Stage | Generator | Reason | Count | Stage Share |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in rejection_rows[:30]:
        lines.append(
            f"| `{row['stage']}` | `{row['generator_id']}` | `{row['base_reason']}` | "
            f"{row['count']} | {_pct(float(row['stage_rejection_share']))} |"
        )
    if not rejection_rows:
        lines.append("| n/a | n/a | n/a | 0 | 0.00% |")
    lines.extend(
        [
            "",
            "## Forward Static Replay",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| 07 rejected rows | {forward_static_replay.get('total_forward_rejected_rows', 0)} |",
            f"| Old static/contract rejection rows | {forward_static_replay.get('old_static_rejection_rows', 0)} |",
            f"| Replayed rows with code | {forward_static_replay.get('replayed_rows', 0)} |",
            f"| Rows still failing current static checker | {forward_static_replay.get('rows_with_current_static_issues', 0)} |",
            f"| Rows cleared by current static checker | {forward_static_replay.get('rows_cleared_by_current_static_checker', 0)} |",
            f"| Current static issue rate | {_pct(float(forward_static_replay.get('current_static_issue_rate') or 0.0))} |",
            "",
            "| Generator | Current Static Issue | Count |",
            "|---|---|---:|",
        ]
    )
    replay_issues = forward_static_replay.get("top_current_static_issues") or []
    if replay_issues:
        for row in replay_issues[:10]:
            lines.append(f"| `{row.get('generator_id')}` | `{row.get('issue')}` | {row.get('count', 0)} |")
    else:
        lines.append("| n/a | n/a | 0 |")
    lines.extend(
        [
            "",
            "## Throughput",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Wall seconds | {_fmt(throughput.get('wall_seconds'))} |",
            f"| API attempts | {throughput.get('api_attempts', 0)} |",
            f"| Successful API attempts | {throughput.get('successful_api_attempts', 0)} |",
            f"| Retry or failed attempts | {throughput.get('retry_or_failed_attempts', 0)} |",
            f"| Endpoint quarantine/unavailable events | {throughput.get('endpoint_health_events', 0)} |",
            f"| Total tokens | {throughput.get('total_tokens', 0)} |",
            f"| Tokens per second | {_fmt(throughput.get('tokens_per_second'))} |",
            "",
            "## Endpoint Reliability",
            "",
            "| Endpoint | Attempts | Success | Failed | Failure Rate | Avg Latency | P95 Latency | Max Latency |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    endpoint_rows = endpoint_reliability.get("rows") or []
    if endpoint_rows:
        for row in endpoint_rows:
            lines.append(
                f"| `{row.get('endpoint_name')}` | {row.get('api_attempts', 0)} | "
                f"{row.get('successful_api_attempts', 0)} | {row.get('failed_api_attempts', 0)} | "
                f"{_pct(float(row.get('failure_rate') or 0.0))} | {_fmt(row.get('avg_latency_seconds'))} | "
                f"{_fmt(row.get('p95_latency_seconds'))} | {_fmt(row.get('max_latency_seconds'))} |"
            )
    else:
        lines.append("| n/a | 0 | 0 | 0 | 0.00% | 0.000 | 0.000 | 0.000 |")
    lines.extend(
        [
            "",
            "## Model Failures",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total model failures | {model_failures.get('total_failures', 0)} |",
            "",
            "| Prompt | Generator | Endpoint | Error Type | Source | Count |",
            "|---|---|---|---|---|---:|",
        ]
    )
    top_model_failures = model_failures.get("top_failures") or []
    if top_model_failures:
        for row in top_model_failures[:15]:
            lines.append(
                f"| `{row.get('prompt_name')}` | `{row.get('generator_id')}` | `{row.get('endpoint_name')}` | "
                f"`{row.get('error_type')}` | `{row.get('source')}` | {row.get('count', 0)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | 0 |")
    lines.extend(
        [
            "",
            "## Quality Retries",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Rows with quality retries | {quality_retries.get('total_rows_with_retries', 0)} |",
            f"| Total quality retry attempts | {quality_retries.get('total_quality_retries', 0)} |",
            "",
            "| Stage | Rows With Retries | Total Retries | Recovered/Output Rows | Rejected Rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    retry_by_stage = quality_retries.get("by_stage") or {}
    if retry_by_stage:
        for stage, values in sorted(retry_by_stage.items()):
            recovered_rows = int(values.get("candidate_rows_with_retries", 0)) + int(values.get("output_rows_with_retries", 0))
            rejected_rows = int(values.get("rejected_rows_with_retries", 0))
            lines.append(
                f"| `{stage}` | {values.get('rows_with_retries', 0)} | {values.get('total_quality_retries', 0)} | "
                f"{recovered_rows} | {rejected_rows} |"
            )
    else:
        lines.append("| n/a | 0 | 0 | 0 | 0 |")
    lines.extend(
        [
            "",
            "### Top Quality Retry Reasons",
            "",
            "| Stage | Generator | Outcome | Reason | Count |",
            "|---|---|---|---|---:|",
        ]
    )
    top_retry_reasons = quality_retries.get("top_reasons") or []
    if top_retry_reasons:
        for row in top_retry_reasons[:10]:
            lines.append(
                f"| `{row.get('stage')}` | `{row.get('generator_id')}` | `{row.get('outcome')}` | "
                f"`{row.get('base_reason')}` | {row.get('count', 0)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | 0 |")
    lines.extend(
        [
            "",
            "## Forward Repair Attempts",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total attempts | {repair_attempts.get('total_attempts', 0)} |",
            f"| Recovered | {repair_attempts.get('recovered', 0)} |",
            f"| Failed eval | {repair_attempts.get('failed_eval', 0)} |",
            f"| Failed generation | {repair_attempts.get('failed_generation', 0)} |",
            "",
            "| Generator | Status | Original Family | Final Family | Count |",
            "|---|---|---|---|---:|",
        ]
    )
    top_repair_generators = repair_attempts.get("top_generators") or []
    if top_repair_generators:
        for row in top_repair_generators[:15]:
            lines.append(
                f"| `{row.get('generator_id')}` | `{row.get('status')}` | "
                f"`{row.get('original_failure_family')}` | `{row.get('final_failure_family')}` | {row.get('count', 0)} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | 0 |")
    lines.extend(
        [
            "",
            "## Data Quality Gates",
            "",
            "| Check | Count |",
            "|---|---:|",
            f"| Exact duplicate rows | {data_quality.get('exact_duplicate_rows', 0)} |",
            f"| Forbidden phrase hits | {data_quality.get('forbidden_phrase_count', 0)} |",
            f"| Missing section hits | {data_quality.get('missing_section_count', 0)} |",
            f"| Missing code block rows | {data_quality.get('missing_code_block_count', 0)} |",
            f"| Semantic coherence warning rows | {data_quality.get('semantic_coherence_row_count', 0)} |",
            f"| Semantic coherence warning issues | {data_quality.get('semantic_coherence_issue_count', 0)} |",
            "",
            "### Top Forbidden Phrase Hits",
            "",
            "| Phrase | Count |",
            "|---|---:|",
        ]
    )
    forbidden_phrase_hits = data_quality.get("forbidden_phrase_hits") or {}
    if forbidden_phrase_hits:
        for phrase, count in list(forbidden_phrase_hits.items())[:10]:
            lines.append(f"| `{_escape_table_cell(str(phrase))}` | {count} |")
    else:
        lines.append("| n/a | 0 |")
    lines.extend(
        [
            "",
            "### Top Semantic Coherence Warnings",
            "",
            "| Generator | Issue | Count |",
            "|---|---|---:|",
        ]
    )
    semantic_rows = data_quality.get("semantic_coherence_top_generators") or []
    if semantic_rows:
        for row in semantic_rows[:10]:
            lines.append(f"| `{row.get('generator_id')}` | `{row.get('issue')}` | {row.get('count', 0)} |")
    else:
        lines.append("| n/a | n/a | 0 |")
    lines.extend(
        [
            "",
            "## Train Distribution Snapshot",
            "",
            f"- Average token count: `{_fmt(train_stats.get('token_avg'))}`",
            f"- Token min/max: `{train_stats.get('token_min', 0)}` / `{train_stats.get('token_max', 0)}`",
            "",
        ]
    )
    for field in ("generator_id", "task_family", "doc_type", "difficulty_level", "token_bucket"):
        lines.extend(_render_counter_table(f"Top {field}", train_stats["distributions"].get(field) or {}, limit=10))
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `run_quality_dashboard.csv`: machine-readable summary metrics.",
            "- `generator_stage_funnel.csv`: per-generator pass/reject funnel.",
            "- `rejection_reason_by_generator.csv`: per-stage rejection breakdown.",
            "- `train_distribution.md`: train-set distribution details.",
            "- `run_quality_summary.json`: structured dashboard payload.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_train_distribution(train_stats: dict[str, Any]) -> str:
    lines = [
        "# Train Distribution",
        "",
        f"- Rows: `{train_stats.get('row_count', 0)}`",
        f"- Total approx tokens: `{train_stats.get('token_total', 0)}`",
        f"- Average approx tokens: `{_fmt(train_stats.get('token_avg'))}`",
        f"- Token min/max: `{train_stats.get('token_min', 0)}` / `{train_stats.get('token_max', 0)}`",
        "",
    ]
    for field, values in (train_stats.get("distributions") or {}).items():
        lines.extend(_render_counter_table(field, values, limit=30))
    return "\n".join(lines)


def _render_counter_table(title: str, values: dict[str, int], *, limit: int) -> list[str]:
    total = sum(int(value) for value in values.values())
    lines = [f"## {title}", "", "| Value | Count | Share |", "|---|---:|---:|"]
    for value, count in list(values.items())[:limit]:
        lines.append(f"| `{value}` | {count} | {_pct(_rate(count, total))} |")
    if len(values) > limit:
        lines.append(f"| ... | {len(values) - limit} more values omitted |  |")
    lines.append("")
    return lines


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_seed_input_from_manifest(run: Path) -> Path | None:
    manifest_path = run / "reports" / "cpt_run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json_compat.loads(manifest_path.read_bytes())
    except Exception:
        return None
    seed_input = manifest.get("seed_input")
    if not seed_input:
        return None
    path = Path(seed_input)
    if not path.is_absolute():
        path = run / path
    return path if path.exists() else None


def _parse_stage_wall_seconds(run: Path) -> dict[str, float]:
    logs_dir = run / "logs"
    if not logs_dir.exists():
        return {}
    started: dict[str, str] = {}
    durations: dict[str, float] = {}
    pattern = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*stage=(?P<stage>\S+) status=(?P<status>started|completed)")
    for path in sorted(logs_dir.glob("*.log")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = pattern.search(line)
            if not match:
                continue
            key = match.group("stage")
            if match.group("status") == "started":
                started[key] = match.group("ts")
                continue
            start_ts = started.get(key)
            if start_ts:
                durations[key] = max(0.0, _timestamp_to_seconds(match.group("ts")) - _timestamp_to_seconds(start_ts))
    return durations


def _count_endpoint_health_events(run: Path) -> int:
    logs_dir = run / "logs"
    if not logs_dir.exists():
        return 0
    count = 0
    for path in sorted(logs_dir.glob("*.log")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "status=quarantined" in line or "endpoint_unavailable" in line:
                count += 1
    return count


def _timestamp_to_seconds(value: str) -> float:
    from datetime import datetime

    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def _split_rejection_reasons(value: Any) -> list[str]:
    if value is None:
        return ["UNKNOWN"]
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_split_rejection_reasons(item))
        return flattened or ["UNKNOWN"]
    text = str(value).strip()
    if not text:
        return ["UNKNOWN"]
    parts = [part.strip() for part in re.split(r";\s*", text) if part.strip()]
    return parts or [text]


def _base_reason(reason_detail: str) -> str:
    return reason_detail.split(":", 1)[0].strip() or "UNKNOWN"


def _generator_id(row: dict[str, Any]) -> str:
    return _field(row, "generator_id")


def _repair_field(row: dict[str, Any], key: str) -> str:
    nested_key_map = {
        "generator_id": (("original", "generator_id"), ("final", "generator_id")),
        "original_failure_family": (("original", "failure_family"),),
        "final_failure_family": (("final", "failure_family"),),
    }
    value = row.get(key)
    if value not in (None, ""):
        return str(value)
    for container_name, nested_key in nested_key_map.get(key, ()):
        container = row.get(container_name)
        if isinstance(container, dict):
            value = container.get(nested_key)
            if value not in (None, ""):
                return str(value)
    return ""


def _field(row: dict[str, Any], key: str) -> str:
    metadata = row.get("metadata") or {}
    source_metadata = row.get("source_metadata") or {}
    metadata_source = metadata.get("source_metadata") if isinstance(metadata.get("source_metadata"), dict) else {}
    llm_metadata = row.get("llm_metadata") or {}
    for container in (row, metadata, source_metadata, metadata_source, llm_metadata):
        value = container.get(key) if isinstance(container, dict) else None
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _token_bucket(token_count: int) -> str:
    for label, lower, upper in TOKEN_BUCKETS:
        if lower is not None and token_count < lower:
            continue
        if upper is not None and token_count >= upper:
            continue
        return label
    return "unknown"


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    clamped = max(0.0, min(1.0, percentile))
    index = int(round((len(values) - 1) * clamped))
    return values[index]


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if not denominator else float(numerator) / float(denominator)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.3f}"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")
